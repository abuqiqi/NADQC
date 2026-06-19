import sys
import time
import datetime
import traceback
import io
from pprint import pprint
from pathlib import Path
import numpy as np

from src.utils import log, get_args, get_config, select_circuit, write_compiler_results_to_csv, Backend, Network
from src.compiler import Compiler, CompilerFactory
from src.navi import evaluate_navi_checkpoints_from_config, navi_checkpoint_enabled

def _build_compile_config(global_config: dict, compiler: Compiler, circuit_name: str) -> dict:
    compiler_config = global_config.get("compiler_config", {})
    return {
        "circuit_name": circuit_name,
        **global_config.get("compile_config", {}),
        **compiler_config.get("common", {}),
        **compiler_config.get(compiler.compiler_id, {}),
    }

def _ordered_result_info(result_info: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    preferred_order = [
        "Static OEE",
        "FGP-rOEE",
        "AutoComm",
        "WBCP",
        "NAVI",
    ]
    ordered: dict[str, dict[str, float]] = {}
    for name in preferred_order:
        if name in result_info:
            ordered[name] = result_info[name]
    for name, metrics in result_info.items():
        if name not in ordered:
            ordered[name] = metrics
    return ordered


def _build_log_path(global_config: dict, circuit_name: str, args, timestamp: str) -> Path:
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}_{timestamp}.txt"
    return Path(str(global_config.get("output_folder", "outputs"))) / circuit_name / filename


def _build_output_csv_path(global_config: dict, args, timestamp: str | None = None) -> str:
    if timestamp is None:
        date_prefix = datetime.datetime.now().strftime("%y%m%d")
    else:
        date_prefix = datetime.datetime.strptime(timestamp[:8], "%Y%m%d").strftime("%y%m%d")
    capacities = [str(capacity) for capacity in args.core_capacity]
    if len(set(capacities)) == 1:
        capacity_label = capacities[0]
    else:
        capacity_label = "-".join(capacities)
    filename = f"{date_prefix}-{args.core_count}x{capacity_label}-{args.network}.csv"
    return str(Path(str(global_config.get("output_folder", "outputs"))) / filename)


def _switch_stdout_to_log_file(
    log_path: Path,
    initial_stdout_buffer: io.StringIO | None = None,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    if initial_stdout_buffer is not None:
        log_file.write(initial_stdout_buffer.getvalue())
    sys.stdout = log_file
    return log_file


def main(args, timestamp: str | None = None, initial_stdout_buffer: io.StringIO | None = None):

    global_config = get_config(args.global_config_path)
    network_config = {
        **global_config.get('network_config', {}),
        'type': args.network,
    }
    comm_slot_reserve = Network.resolve_comm_slot_reserve(network_config, global_config)

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

    if 'fidelity_range' not in network_config:
        fidelity_range = [0.90, 0.93]
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
                                                 basis_gates,
                                                 two_qubit_gates
                                                 )

    # print(circ)
    # print(trans_circ)

    # 调用不同的compiler
    result_info = {}
    output_path = _build_output_csv_path(global_config, args, timestamp)
    output_path = "./outputs/260615.csv"

    registered_compiler_ids = CompilerFactory.register_compilers(global_config.get("compiler_modules"))
    default_compiler_ids = ["staticoee", "fgproee", "wbcp", "autocomm", "navi"]
    requested_compiler_ids = global_config.get("compiler_ids", default_compiler_ids)
    missing_compiler_ids = [compiler_id for compiler_id in requested_compiler_ids if compiler_id not in registered_compiler_ids]
    if missing_compiler_ids:
        print(f"[WARN] Requested compiler IDs not registered and will be skipped: {missing_compiler_ids}")
    compiler_ids = [compiler_id for compiler_id in requested_compiler_ids if compiler_id in registered_compiler_ids]
    print(f"Registered compiler IDs: {compiler_ids}")
    compilers: list[Compiler] = []
    for compiler_id in compiler_ids:
        compilers.append(CompilerFactory.get_compiler(compiler_id)())

    circuit_name = f"{args.circuit_name}{trans_circ.num_qubits}"
    log_file = None
    if timestamp is not None and initial_stdout_buffer is not None:
        log_path = _build_log_path(global_config, circuit_name, args, timestamp)
        log_file = _switch_stdout_to_log_file(log_path, initial_stdout_buffer)

    checkpoint_enabled = navi_checkpoint_enabled(global_config)
    checkpoint_dir: Path | None = None
    navi_compile_config = None
    navi_evaluator_policy = None
    if checkpoint_enabled:
        checkpoint_dir = (
            Path(str(global_config.get("output_folder", "outputs")))
            / circuit_name
            / f"navi_checkpoints_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    for compiler in compilers:
        log(f"Compiler: [{compiler.name}]")
        compile_config = _build_compile_config(global_config, compiler, circuit_name)
        if checkpoint_enabled and compiler.compiler_id == "navi" and checkpoint_dir is not None:
            compile_config["navi_checkpoint_enabled"] = True
            compile_config["navi_checkpoint_dir"] = str(checkpoint_dir)
            navi_compile_config = dict(compile_config)
            navi_evaluator_policy = compile_config.get("evaluator_policy")
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

    try:
        checkpoint_result = evaluate_navi_checkpoints_from_config(
            global_config,
            checkpoint_dir,
            network,
            task_info,
            default_policy_name=navi_evaluator_policy,
            config=navi_compile_config,
        )
        for compiler_name, metrics in checkpoint_result.result_info.items():
            if compiler_name in result_info:
                print(
                    f"[WARN] NAVI checkpoint result [{compiler_name}] duplicates an existing "
                    "compiler result; keeping the primary compiler result."
                )
                continue
            result_info[compiler_name] = metrics
    except Exception as exc:
        print(f"[ERROR] NAVI checkpoint evaluation failed: {exc}")
        traceback.print_exc(file=sys.stdout)

    if result_info:
        write_compiler_results_to_csv(
            {**task_info, **network.info()},
            _ordered_result_info(result_info),
            output_path,
        )
    else:
        print("[WARN] No compiler succeeded; skipping CSV write.")

    return log_file

if __name__ == "__main__":
    # 获取全局配置
    args = get_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stdout = sys.stdout
    initial_stdout_buffer = io.StringIO()
    sys.stdout = initial_stdout_buffer
    log_file = None
    try:
        start_time = time.time()
        log_file = main(args, timestamp=timestamp, initial_stdout_buffer=initial_stdout_buffer)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
    finally:
        current_stdout = sys.stdout
        sys.stdout = original_stdout
        if current_stdout is not original_stdout and current_stdout is not initial_stdout_buffer:
            current_stdout.close()
        elif current_stdout is initial_stdout_buffer:
            original_stdout.write(initial_stdout_buffer.getvalue())
