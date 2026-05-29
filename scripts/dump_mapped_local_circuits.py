import argparse
import copy
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit
from qiskit.qpy import dump as qpy_dump

import src.compiler.evaluator as evaluator_mod
from scripts.compare_local_gate_shapes import (
    build_context,
    compile_config,
    load_record_list_from_json,
    parse_record_json_specs,
    parse_str_list,
)
from src.compiler import CompilerFactory, CompilerUtils, MappingRecordList


class LocalCircuitDumpTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._pending_contexts: list[dict[str, Any]] = []
        self._orig_flush_local_ops = evaluator_mod.MappingEvaluator.flush_local_ops
        self._orig_transpile = evaluator_mod.transpile

    def install(self) -> None:
        trace = self

        def traced_flush_local_ops(
            evaluator_self,
            state,
            partition,
            network,
            policy,
            qpu_ids=None,
            final=False,
        ):
            targets = range(network.num_backends) if qpu_ids is None else [int(q) for q in qpu_ids]
            will_transpile = (not evaluator_self._is_deferred(policy)) or bool(final)
            if will_transpile:
                for qpu_id in targets:
                    qpu_id = int(qpu_id)
                    buffer = state.routed_buffers[qpu_id]
                    if buffer.size() == 0:
                        continue
                    trace._pending_contexts.append(
                        {
                            "qpu_id": qpu_id,
                            "final": bool(final),
                            "input_size": int(buffer.size()),
                            "input_depth": int(buffer.depth() or 0),
                            "input_width": int(buffer.num_qubits),
                            "input_ops": dict(buffer.count_ops()),
                        }
                    )
            return trace._orig_flush_local_ops(
                evaluator_self,
                state,
                partition,
                network,
                policy,
                qpu_ids=qpu_ids,
                final=final,
            )

        def traced_transpile(circuit: QuantumCircuit, *args: Any, **kwargs: Any):
            context = (
                trace._pending_contexts.pop(0)
                if trace._pending_contexts
                else {
                    "qpu_id": None,
                    "final": None,
                    "input_size": int(circuit.size()),
                    "input_depth": int(circuit.depth() or 0),
                    "input_width": int(circuit.num_qubits),
                    "input_ops": dict(circuit.count_ops()),
                }
            )
            input_circuit = copy.deepcopy(circuit)
            output_circuit = trace._orig_transpile(circuit, *args, **kwargs)
            trace.records.append(
                {
                    **context,
                    "output_size": int(output_circuit.size()),
                    "output_depth": int(output_circuit.depth() or 0),
                    "output_width": int(output_circuit.num_qubits),
                    "output_ops": dict(output_circuit.count_ops()),
                    "added_gates": int(output_circuit.size()) - int(input_circuit.size()),
                    "pre": input_circuit,
                    "post": copy.deepcopy(output_circuit),
                }
            )
            return output_circuit

        evaluator_mod.MappingEvaluator.flush_local_ops = traced_flush_local_ops
        evaluator_mod.transpile = traced_transpile

    def uninstall(self) -> None:
        evaluator_mod.MappingEvaluator.flush_local_ops = self._orig_flush_local_ops
        evaluator_mod.transpile = self._orig_transpile


def sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "item"


def first_n_ops(circuit: QuantumCircuit, n_ops: int) -> QuantumCircuit:
    out = QuantumCircuit(circuit.num_qubits)
    for inst in list(circuit.data)[: max(0, int(n_ops))]:
        qids = [circuit.find_bit(q).index for q in inst.qubits]
        cids = [circuit.find_bit(c).index for c in inst.clbits]
        out.append(copy.deepcopy(inst.operation), qids, cids)
    return out


def write_circuit_artifacts(
    circuit: QuantumCircuit,
    out_base: Path,
    first_ops: int,
    fold: int,
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    full_qpy_path = out_base.with_suffix(".full.qpy")
    with full_qpy_path.open("wb") as f:
        qpy_dump(circuit, f)

    partial = first_n_ops(circuit, first_ops)
    text = partial.draw(output="text", fold=fold)
    out_base.with_suffix(".txt").write_text(str(text), encoding="utf-8")

    try:
        fig = partial.draw(output="mpl", fold=fold, idle_wires=False)
        fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except Exception as exc:
        out_base.with_suffix(".png.error.txt").write_text(str(exc), encoding="utf-8")


def write_manifest(out_dir: Path, compiler: str, selected: list[dict[str, Any]]) -> None:
    lines = [f"# {compiler}", ""]
    for idx, rec in enumerate(selected, start=1):
        lines.append(
            f"{idx}. qpu={rec['qpu_id']} "
            f"pre_size={rec['input_size']} pre_depth={rec['input_depth']} "
            f"post_size={rec['output_size']} added={rec['added_gates']} "
            f"pre_ops={rec['input_ops']} post_ops={rec['output_ops']}"
        )
    (out_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_top_circuits(
    compiler_label: str,
    trace_records: list[dict[str, Any]],
    out_dir: Path,
    top_n: int,
    first_ops: int,
    fold: int,
) -> None:
    selected = sorted(trace_records, key=lambda r: int(r["input_size"]), reverse=True)[:top_n]
    compiler_dir = out_dir / sanitize(compiler_label)
    compiler_dir.mkdir(parents=True, exist_ok=True)
    for idx, rec in enumerate(selected, start=1):
        prefix = (
            compiler_dir
            / f"top{idx}_qpu{rec['qpu_id']}_pre{rec['input_size']}_post{rec['output_size']}"
        )
        write_circuit_artifacts(rec["pre"], prefix.with_name(prefix.name + "_pre_first"), first_ops, fold)
        write_circuit_artifacts(rec["post"], prefix.with_name(prefix.name + "_post_first"), first_ops, fold)
    write_manifest(compiler_dir, compiler_label, selected)


def run_compiler_with_trace(
    compiler_id: str,
    global_config: dict[str, Any],
    circuit: QuantumCircuit,
    network,
    quiet: bool,
) -> tuple[str, list[dict[str, Any]]]:
    compiler_cls = CompilerFactory.get_compiler(compiler_id)
    compiler = compiler_cls()
    trace = LocalCircuitDumpTrace()
    original_save_records = MappingRecordList.save_records

    def noop_save_records(self, filename: str):
        del self, filename
        return None

    MappingRecordList.save_records = noop_save_records
    trace.install()
    try:
        if quiet:
            with open("/dev/null", "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    compiler.compile(
                        copy.deepcopy(circuit),
                        network,
                        compile_config(global_config, compiler, f"dump_{compiler_id}"),
                    )
        else:
            compiler.compile(
                copy.deepcopy(circuit),
                network,
                compile_config(global_config, compiler, f"dump_{compiler_id}"),
            )
    finally:
        trace.uninstall()
        MappingRecordList.save_records = original_save_records
    return compiler.name, trace.records


def replay_json_with_trace(
    label: str,
    path: Path,
    circuit: QuantumCircuit,
    network,
) -> tuple[str, list[dict[str, Any]]]:
    record_list = load_record_list_from_json(path)
    trace = LocalCircuitDumpTrace()
    trace.install()
    try:
        CompilerUtils.evaluate_raw_mapping_records(
            record_list,
            network,
            policy_name=None,
        )
    finally:
        trace.uninstall()
    return f"{label} (replayed)", trace.records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config-path", default="config.json")
    parser.add_argument("--circuit-name", default="QFT")
    parser.add_argument("--qubit-count", type=int, default=200)
    parser.add_argument("--core-count", type=int, default=4)
    parser.add_argument("--core-capacity", default="50")
    parser.add_argument("--backend-name", default="ibm_marrakesh")
    parser.add_argument("--date", default="2026-03-01")
    parser.add_argument("--network", default="all_to_all")
    parser.add_argument("--compilers", default="navihybrid")
    parser.add_argument("--record-json", action="append", default=[])
    parser.add_argument("--record-jsons", default="")
    parser.add_argument("--local-eval-mode", choices=["immediate", "deferred"], default=None)
    parser.add_argument("--deferred-initial-layout", choices=["fixed", "free"], default=None)
    parser.add_argument("--deferred-route-local-gates", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--out-dir", default="outputs/mapped_local_circuit_diagrams_qft200")
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--first-ops", type=int, default=120)
    parser.add_argument("--fold", type=int, default=80)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    global_config, circuit, network, _task_info = build_context(args)
    CompilerFactory.register_compilers(global_config.get("compiler_modules", ["src.baselines", "src.navi"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for compiler_id in parse_str_list(args.compilers):
        if not compiler_id:
            continue
        label, records = run_compiler_with_trace(compiler_id, global_config, circuit, network, args.quiet)
        save_top_circuits(label, records, out_dir, args.top_n, args.first_ops, args.fold)

    record_specs = list(args.record_json or []) + parse_record_json_specs(args.record_jsons)
    for spec in record_specs:
        if "=" in spec:
            label, path_text = spec.split("=", 1)
        else:
            path_text = spec
            label = Path(path_text).stem
        label, records = replay_json_with_trace(label.strip(), Path(path_text.strip()), circuit, network)
        save_top_circuits(label, records, out_dir, args.top_n, args.first_ops, args.fold)

    print(f"Wrote circuit diagrams to {out_dir}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
