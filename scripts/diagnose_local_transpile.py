import argparse
import copy
import datetime
import json
import sys
from collections import Counter
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit

from src.baselines.autocomm import QAutoComm
from src.baselines.static_oee import StaticOEE
from src.compiler import CompilerUtils
import src.compiler.compiler_utils as compiler_utils_mod
from src.compiler.compiler_utils import CommOp
from src.utils import Backend, Network, get_config, select_circuit


@contextmanager
def quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open("/dev/null", "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class TranspileTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._orig_transpile = compiler_utils_mod.transpile

    def install(self) -> None:
        trace = self

        def normalize_initial_layout(initial_layout: Any, circuit: QuantumCircuit) -> dict[int, int]:
            normalized: dict[int, int] = {}
            if isinstance(initial_layout, dict):
                for qobj, phy in initial_layout.items():
                    try:
                        local_idx = circuit.qubits.index(qobj)
                    except ValueError:
                        continue
                    normalized[int(local_idx)] = int(phy)
            return dict(sorted(normalized.items()))

        def traced_transpile(circuit: QuantumCircuit, *args: Any, **kwargs: Any):
            out = trace._orig_transpile(circuit, *args, **kwargs)
            try:
                final_local_to_phy = CompilerUtils.get_local_to_physical_map(out)
            except Exception as exc:
                final_local_to_phy = {"error": str(exc)}
            trace.records.append(
                {
                    "input_size": int(circuit.size()),
                    "input_depth": int(circuit.depth() or 0),
                    "input_width": int(circuit.num_qubits),
                    "input_ops": dict(circuit.count_ops()),
                    "output_size": int(out.size()),
                    "output_depth": int(out.depth() or 0),
                    "output_width": int(out.num_qubits),
                    "output_ops": dict(out.count_ops()),
                    "initial_layout": str(kwargs.get("initial_layout")),
                    "initial_local_to_phy": normalize_initial_layout(kwargs.get("initial_layout"), circuit),
                    "final_local_to_phy": final_local_to_phy,
                    "layout_changed": normalize_initial_layout(kwargs.get("initial_layout"), circuit) != final_local_to_phy,
                    "basis_gates": list(kwargs.get("basis_gates") or []),
                }
            )
            return out

        compiler_utils_mod.transpile = traced_transpile

    def uninstall(self) -> None:
        compiler_utils_mod.transpile = self._orig_transpile


def build_bv100_context() -> tuple[QuantumCircuit, Network, dict[str, Any]]:
    global_config = get_config("config.json")
    network_config = {
        **global_config.get("network_config", {}),
        "type": "mesh_grid",
    }
    comm_slot_reserve = int(
        network_config.get("comm_slot_reserve", global_config.get("comm_slot_reserve", 0)) or 0
    )

    backend_list = []
    for _ in range(2):
        sampled_capacity = 50 + comm_slot_reserve
        backend_config = {
            "backend_name": f"ibm_marrakesh_sampled_{sampled_capacity}q",
            "date": datetime.datetime.strptime("2026-03-01", "%Y-%m-%d"),
        }
        backend_list.append(Backend(global_config, backend_config))

    network_config["fidelity_range"] = [0.90, 0.90]
    network = Network(network_config, backend_list)
    basis_gates = network.basis_gates
    two_qubit_gates = network.two_qubit_gates
    _, trans_circ, task_info = select_circuit(
        "BV",
        100,
        2,
        [50, 50],
        basis_gates,
        two_qubit_gates,
    )
    return trans_circ, network, task_info


def summarize_transpile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    total_in = sum(r["input_size"] for r in records)
    total_out = sum(r["output_size"] for r in records)
    out_ops = Counter()
    in_ops = Counter()
    for r in records:
        out_ops.update(r["output_ops"])
        in_ops.update(r["input_ops"])

    sorted_records = sorted(
        enumerate(records),
        key=lambda item: item[1]["output_size"],
        reverse=True,
    )
    top = []
    for idx, r in sorted_records[:8]:
        top.append(
            {
                "flush_idx": idx,
                "input_size": r["input_size"],
                "output_size": r["output_size"],
                "ratio": round(r["output_size"] / max(1, r["input_size"]), 3),
                "input_depth": r["input_depth"],
                "output_depth": r["output_depth"],
                "input_ops": r["input_ops"],
                "output_ops_top": dict(Counter(r["output_ops"]).most_common(8)),
            }
        )

    return {
        "transpile_calls": len(records),
        "total_input_size": total_in,
        "total_output_size": total_out,
        "overall_ratio": round(total_out / max(1, total_in), 3),
        "input_ops": dict(in_ops),
        "output_ops_top": dict(out_ops.most_common(12)),
        "top_flushes": top,
        "flushes": records,
    }


def analyze_autocomm_op_list(circuit: QuantumCircuit, network: Network) -> dict[str, Any]:
    compiler = QAutoComm()
    gate_list = compiler._to_autocomm_gate_list(circuit)
    qmap = compiler._generate_qubit_node_mapping(circuit.num_qubits, network.backend_sizes)
    partition = compiler._partition_from_mapping(circuit.num_qubits, qmap, network.num_backends)

    from src.baselines.AutoComm.autocomm import autocomm_full

    num_q = len(qmap)
    qb_per_node = max(network.backend_sizes) if network.backend_sizes else 1
    with quiet(True):
        epr_cnt, latency, assigned_blocks, comm_costs, op_list = autocomm_full(
            gate_list,
            qmap,
            aggregate_iter_cnt=max(1, num_q // max(1, qb_per_node)),
            schedule_iter_cnt=max(1, num_q // max(1, qb_per_node)),
            return_ops=True,
        )

    comm_blocks = []
    normal_ops = Counter()
    comm_payload_ops = Counter()
    for inst_idx, inst in enumerate(op_list):
        op = inst.operation
        if isinstance(op, CommOp):
            payload_ops = Counter(g.name for g in op.gate_list)
            comm_payload_ops.update(payload_ops)
            comm_blocks.append(
                {
                    "inst_idx": inst_idx,
                    "comm_type": op.comm_type,
                    "source": int(op.source_qubit),
                    "src_qpu": int(op.src_qpu),
                    "dst_qpu": int(op.dst_qpu),
                    "payload_size": len(op.gate_list),
                    "involved_qubits": [int(q) for q in op.involved_qubits],
                    "payload_ops": dict(payload_ops),
                }
            )
        else:
            normal_ops.update([op.name])

    comm_blocks_sorted = sorted(comm_blocks, key=lambda b: b["payload_size"], reverse=True)
    return {
        "partition": partition,
        "epr_cnt": epr_cnt,
        "latency": latency,
        "assigned_block_count": len(assigned_blocks),
        "op_list_size": int(op_list.size()),
        "comm_block_count": len(comm_blocks),
        "normal_op_count": sum(normal_ops.values()),
        "normal_ops": dict(normal_ops),
        "comm_payload_total": sum(b["payload_size"] for b in comm_blocks),
        "comm_payload_ops": dict(comm_payload_ops),
        "top_comm_blocks": comm_blocks_sorted[:12],
        "comm_block_payload_hist": dict(Counter(b["payload_size"] for b in comm_blocks_sorted)),
    }


def run_compiler_with_trace(name: str, compiler: Any, circuit: QuantumCircuit, network: Network) -> dict[str, Any]:
    trace = TranspileTrace()
    network.layout_trace_records = []
    trace.install()
    try:
        with quiet(True):
            result = compiler.compile(
                copy.deepcopy(circuit),
                network,
                {
                    "circuit_name": "BV100_diagnose",
                    "save_records": False,
                },
            )
    finally:
        trace.uninstall()

    return {
        "compiler": name,
        "total_costs": result.total_costs.to_dict(),
        "num_records": result.num_records,
        "partitions": [r.partition for r in result.records],
        "transpile": summarize_transpile_records(trace.records),
        "layout_trace": copy.deepcopy(getattr(network, "layout_trace_records", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/diagnose_bv100_local_transpile.json")
    args = parser.parse_args()

    circuit, network, task_info = build_bv100_context()
    static_summary = run_compiler_with_trace("Static OEE", StaticOEE(), circuit, network)
    autocomm_summary = run_compiler_with_trace("AutoComm", QAutoComm(), circuit, network)
    autocomm_ops = analyze_autocomm_op_list(circuit, network)

    report = {
        "task_info": task_info,
        "network": network.info(),
        "static_oee": static_summary,
        "autocomm": autocomm_summary,
        "autocomm_op_list": autocomm_ops,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for key in ("static_oee", "autocomm"):
        item = report[key]
        costs = item["total_costs"]
        tr = item["transpile"]
        print(
            f"{item['compiler']}: local_gate_num={costs['local_gate_num']}, "
            f"payload_gate_num={costs['payload_gate_num']}, epairs={costs['epairs']}, "
            f"transpile_calls={tr['transpile_calls']}, input_size={tr['total_input_size']}, "
            f"output_size={tr['total_output_size']}, ratio={tr['overall_ratio']}"
        )
        print(f"  input_ops={tr['input_ops']}")
        print(f"  output_ops_top={tr['output_ops_top']}")
        print(f"  top_flushes={tr['top_flushes'][:3]}")

    print(
        "AutoComm op_list: "
        f"comm_blocks={autocomm_ops['comm_block_count']}, "
        f"comm_payload_total={autocomm_ops['comm_payload_total']}, "
        f"normal_op_count={autocomm_ops['normal_op_count']}, "
        f"payload_hist={autocomm_ops['comm_block_payload_hist']}"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
