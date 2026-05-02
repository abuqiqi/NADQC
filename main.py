import sys
import time
import datetime
import traceback
from pprint import pprint
import numpy as np

from src.utils import log, get_args, get_config, select_circuit, write_compiler_results_to_csv, Backend, Network
from src.compiler import Compiler, CompilerFactory

def _build_compile_config(global_config: dict, compiler: Compiler, circuit_name: str) -> dict:
    return {
        "circuit_name": circuit_name,
        **global_config.get("compile_config", {}),
        **global_config.get("compiler_config", {}).get(compiler.compiler_id, {}),
    }


def _print_flush_stats(compiler_name: str, result) -> None:
    total_costs = result.total_costs
    print(
        f"[FLUSH] [{compiler_name}] "
        f"flush_calls={total_costs.flush_calls}, "
        f"nonempty_flushes={total_costs.nonempty_flushes}, "
        f"local_transpile_calls={total_costs.local_transpile_calls}"
    )


def _ordered_result_info(result_info: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    preferred_order = [
        "Static OEE",
        "FGP-rOEE",
        "WBCP",
        "AutoComm",
        "NAVI Hybrid TD Direct",
        "NAVI Hybrid+direct mapper",
        "NAVI Hybrid+boundeddp neigh mapper",
        "NAVI Hybrid Direct",
    ]
    ordered: dict[str, dict[str, float]] = {}
    for name in preferred_order:
        if name in result_info:
            ordered[name] = result_info[name]
    for name, metrics in result_info.items():
        if name not in ordered:
            ordered[name] = metrics
    return ordered


def _run_shared_prefix_navi_variants(
    compiler_by_id: dict[str, Compiler],
    global_config: dict,
    circuit_name: str,
    trans_circ,
    network: Network,
    task_info: dict,
    result_info: dict[str, dict[str, float]],
) -> set[str]:
    handled_compiler_ids: set[str] = set()

    shared_variant_ids = [
        "navihybridtd",
        "navihybriddirectmapper",
        "navihybrid",
        "navihybriddirect",
    ]
    available_variant_ids = [cid for cid in shared_variant_ids if cid in compiler_by_id]
    if len(available_variant_ids) < 2:
        return handled_compiler_ids

    variant_plan: list[tuple[Compiler, dict, str]] = []
    signatures: list[tuple[str, tuple]] = []
    for compiler_id in available_variant_ids:
        compiler = compiler_by_id[compiler_id]
        compile_config = _build_compile_config(global_config, compiler, circuit_name)
        signatures.append((compiler_id, compiler.get_shared_prefix_signature(compile_config)))
        if compiler_id == "navihybridtd":
            mode = "teledata_direct"
        elif compiler_id == "navihybriddirectmapper":
            mode = "beam_direct"
        elif compiler_id == "navihybrid":
            mode = "beam_neighbor"
        else:
            mode = "direct_noise_aware"
        variant_plan.append((compiler, compile_config, mode))

    base_signature = signatures[0][1]
    if any(sig != base_signature for _, sig in signatures[1:]):
        print("[INFO] NAVI Hybrid shared prefix configs differ; fallback to independent runs.")
        return handled_compiler_ids

    print("[INFO] Reusing NAVI Hybrid shared prefix through Step 5 for hybrid-family compilers")
    base_compiler = variant_plan[0][0]
    try:
        shared_ctx, shared_prefix_time = base_compiler.build_shared_prefix_context(
            trans_circ,
            network,
            variant_plan[0][1],
        )
    except Exception as exc:
        print(f"[ERROR] Failed to build shared NAVI Hybrid prefix context: {exc}")
        traceback.print_exc(file=sys.stdout)
        return handled_compiler_ids

    teledata_records = None
    teledata_records_time = 0.0
    beam_records = None
    beam_records_time = 0.0
    direct_noise_aware_records = None
    direct_noise_aware_records_time = 0.0

    try:
        for compiler, compile_config, mode in variant_plan:
            if mode == "teledata_direct" and teledata_records is None:
                td_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                records_start_time = time.time()
                teledata_records = compiler._step_construct_teledata_only_records(td_ctx)
                teledata_records_time = time.time() - records_start_time
            elif mode in {"beam_direct", "beam_neighbor"} and beam_records is None:
                beam_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                beam_records, beam_records_time = compiler._construct_records_from_ctx(
                    beam_ctx,
                    use_direct_noise_aware=False,
                )
            elif mode == "direct_noise_aware" and direct_noise_aware_records is None:
                dna_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                direct_noise_aware_records, direct_noise_aware_records_time = compiler._construct_records_from_ctx(
                    dna_ctx,
                    use_direct_noise_aware=True,
                )
    except Exception as exc:
        print(f"[ERROR] Failed while preparing shared NAVI Hybrid records: {exc}")
        traceback.print_exc(file=sys.stdout)
        return handled_compiler_ids

    for compiler, compile_config, mode in variant_plan:
        print(f"Compiler: [{compiler.name}]")
        print(compile_config)
        print(compile_config, file=sys.stdout)
        try:
            if mode == "teledata_direct":
                td_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                result = compiler._map_records_with_mapper(
                    td_ctx,
                    teledata_records,
                    mapper_id="direct",
                    shared_prefix_time=shared_prefix_time,
                    records_time=teledata_records_time,
                )
            elif mode == "beam_direct":
                beam_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                result = compiler._map_records_with_mapper(
                    beam_ctx,
                    beam_records,
                    mapper_id="direct",
                    shared_prefix_time=shared_prefix_time,
                    records_time=beam_records_time,
                )
            elif mode == "beam_neighbor":
                beam_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                result = compiler._map_records_with_mapper(
                    beam_ctx,
                    beam_records,
                    mapper_id=str(compile_config.get("multi_record_mapper", "boundeddp_neighbor")).lower(),
                    shared_prefix_time=shared_prefix_time,
                    records_time=beam_records_time,
                )
            else:
                dna_ctx = compiler._prepare_ctx_from_shared_prefix(shared_ctx, compile_config)
                selected_mapper_id = "single_record_greedy" if len(direct_noise_aware_records.records) == 1 else "direct"
                result = compiler._map_records_with_mapper(
                    dna_ctx,
                    direct_noise_aware_records,
                    mapper_id=selected_mapper_id,
                    shared_prefix_time=shared_prefix_time,
                    records_time=direct_noise_aware_records_time,
                )
            pprint(result.total_costs)
            result_info[compiler.name] = {
                "F_eff": np.exp(result.total_costs.total_fidelity_log_sum / task_info["#Depth"]),
                **result.total_costs.to_dict()
            }
            handled_compiler_ids.add(compiler.compiler_id)
        except Exception as exc:
            print(f"[ERROR] Compiler [{compiler.name}] failed: {exc}")
            traceback.print_exc(file=sys.stdout)

    return handled_compiler_ids


def main(args):

    global_config = get_config(args.global_config_path)
    network_config = {
        **global_config.get('network_config', {}),
        'type': args.network,
    }
    comm_slot_reserve = int(network_config.get('comm_slot_reserve', global_config.get('comm_slot_reserve', 0)) or 0)

    backend_list = []
    for i in range(args.core_count):
        sampled_capacity = args.core_capacity[i] + comm_slot_reserve
        # 将字符串列表转换为datetime对象列表
        backend_config = {
            'backend_name': f'{args.backend_name[i]}_sampled_{sampled_capacity}q',
            'date': datetime.datetime.strptime(args.date[i], "%Y-%m-%d")
        }
        backend = Backend(global_config, backend_config)
        backend_list.append(backend)
        # pprint(backend.gate_dict)

    fidelity_range = [0.90, 0.93]
    if args.core_count == 2:
        fidelity_range = [0.90, 0.90]

    network_config['fidelity_range'] = fidelity_range

    print(f"[INFO] Reserved comm slots per QPU (network_config): {comm_slot_reserve}")
    print(f"[INFO] Core sampled QPU capacities: {args.core_capacity}")
    print(f"[INFO] Full QPU capacities (core + reserve): {[c + comm_slot_reserve for c in args.core_capacity]}")

    network = Network(network_config, backend_list)
    network.print_info()

    # 获取basis gate set
    basis_gates = network.basis_gates
    two_qubit_gates = network.two_qubit_gates

    # 调用量子线路
    circ, trans_circ, task_info = select_circuit(args.circuit_name,
                                                 args.qubit_count,
                                                 args.core_count,
                                                 args.core_capacity,
                                                #  args.gate_set
                                                 basis_gates,
                                                 two_qubit_gates
                                                 )

    # print(circ)
    # print(trans_circ)

    # 调用不同的compiler
    result_info = {}
    output_path = f"{global_config.get('output_folder')}260501-scalability.csv"

    compiler_ids = CompilerFactory.register_compilers(global_config.get("compiler_modules"))
    # compiler_ids = ["staticoee", "fgproee", "wbcp", "autocomm", "navi", "navihybrid"] # , "navinew"
    # compiler_ids = ["autocomm", "navihybrid"] # "navi", 
    compiler_ids = ["staticoee", "fgproee", "wbcp", "autocomm", "navihybridtd", "navihybriddirectmapper", "navihybrid"] # , "navihybriddirect"
    # compiler_ids = ["fgproee", "navihybrid"]
    # compiler_ids = ["wbcp"]
    # compiler_ids = ["staticoee", "autocomm", "navihybrid"]
    print(f"Registered compiler IDs: {compiler_ids}")
    compilers: list[Compiler] = []
    for compiler_id in compiler_ids:
        compilers.append(CompilerFactory.get_compiler(compiler_id)())

    circuit_name = f"{args.circuit_name}{trans_circ.num_qubits}"
    compiler_by_id = {compiler.compiler_id: compiler for compiler in compilers}
    handled_compiler_ids = _run_shared_prefix_navi_variants(
        compiler_by_id,
        global_config,
        circuit_name,
        trans_circ,
        network,
        task_info,
        result_info,
    )

    for compiler in compilers:
        if compiler.compiler_id in handled_compiler_ids:
            continue
        print(f"Compiler: [{compiler.name}]")
        compile_config = _build_compile_config(global_config, compiler, circuit_name)
        log(f"{compile_config}")
        try:
            result = compiler.compile(trans_circ, network, compile_config)
            pprint(result.total_costs)
            result_info[compiler.name] = {
                "F_eff": np.exp(result.total_costs.total_fidelity_log_sum / task_info["#Depth"]),
                **result.total_costs.to_dict()
            }
        except Exception as exc:
            print(f"[ERROR] Compiler [{compiler.name}] failed: {exc}")
            traceback.print_exc(file=sys.stdout)

    if result_info:
        write_compiler_results_to_csv(
            {**task_info, **network.info()},
            _ordered_result_info(result_info),
            output_path,
        )
    else:
        print("[WARN] No compiler succeeded; skipping CSV write.")

    return

if __name__ == "__main__":
    # 获取全局配置
    args = get_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}_{timestamp}"# 
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'w', buffering=1) as f:
        sys.stdout = f
        start_time = time.time()
        main(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout
