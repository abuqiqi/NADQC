import argparse
import copy
import datetime
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit

import src.compiler.evaluator as evaluator_mod
from src.compiler import CompilerFactory, CompilerUtils, MappingRecord, MappingRecordList
from src.compiler.evaluator import LocalGateKind
from src.utils import Backend, Network, get_config, select_circuit


@contextmanager
def quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open("/dev/null", "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_record_json_specs(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";;") if item.strip()]


def normalize_repeated(values: list[Any], count: int) -> list[Any]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"expected 1 or {count} values, got {len(values)}")
    return values


def counter_to_top(counter: Counter, limit: int = 20) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def coupling_edges(coupling_map: Any) -> list[tuple[int, int]]:
    if coupling_map is None:
        return []
    if hasattr(coupling_map, "get_edges"):
        return [(int(a), int(b)) for a, b in coupling_map.get_edges()]
    return [(int(a), int(b)) for a, b in coupling_map]


def shortest_path_distances(coupling_map: Any) -> dict[tuple[int, int], int]:
    edges = coupling_edges(coupling_map)
    if not edges:
        return {}
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    distances: dict[tuple[int, int], int] = {}
    for src in graph:
        seen = {src: 0}
        frontier = [src]
        for node in frontier:
            for nxt in graph[node]:
                if nxt in seen:
                    continue
                seen[nxt] = seen[node] + 1
                frontier.append(nxt)
        for dst, dist in seen.items():
            distances[(int(src), int(dst))] = int(dist)
    return distances


def local_qubit_indices(circuit: QuantumCircuit, qargs: Any) -> list[int]:
    return [int(circuit.qubits.index(q)) for q in qargs]


class LocalShapeTrace:
    def __init__(self) -> None:
        self.local_ops: list[dict[str, Any]] = []
        self.flushes: list[dict[str, Any]] = []
        self._pending_flush_contexts: list[dict[str, Any]] = []
        self._orig_add_local_ops = evaluator_mod.MappingEvaluator.add_local_ops
        self._orig_flush_local_ops = evaluator_mod.MappingEvaluator.flush_local_ops
        self._orig_transpile = evaluator_mod.transpile

    def install(self) -> None:
        trace = self

        def traced_add_local_ops(
            evaluator_self,
            state,
            partition,
            network,
            policy,
            qpu_id,
            ops,
            kind,
        ):
            qpu_id_int = int(qpu_id)
            for gate, wires in ops:
                wire_list = [int(w) for w in wires]
                owners = []
                for wire in wire_list:
                    owner = state.wire_owners[qpu_id_int][wire]
                    if owner is None:
                        owners.append({"kind": "free", "logical_qid": None, "label": None})
                    else:
                        owners.append(
                            {
                                "kind": owner.kind,
                                "logical_qid": owner.logical_qid,
                                "label": owner.label,
                            }
                        )
                gate_name = getattr(gate, "name", str(gate))
                trace.local_ops.append(
                    {
                        "qpu_id": qpu_id_int,
                        "kind": str(kind.value if isinstance(kind, LocalGateKind) else kind),
                        "gate": str(gate_name),
                        "wires": wire_list,
                        "owners": owners,
                    }
                )
            return trace._orig_add_local_ops(
                evaluator_self,
                state,
                partition,
                network,
                policy,
                qpu_id,
                ops,
                kind,
            )

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
            if not evaluator_self._is_deferred(policy) or final:
                for qpu_id in targets:
                    buffer = state.routed_buffers[int(qpu_id)]
                    if buffer.size() == 0:
                        continue
                    trace._pending_flush_contexts.append(
                        {
                            "qpu_id": int(qpu_id),
                            "final": bool(final),
                            "buffer_kind": (
                                None
                                if state.active_buffer_kind[int(qpu_id)] is None
                                else str(state.active_buffer_kind[int(qpu_id)].value)
                            ),
                            "use_coupling_map": state.active_buffer_use_coupling_map[int(qpu_id)],
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
                trace._pending_flush_contexts.pop(0)
                if trace._pending_flush_contexts
                else {"qpu_id": None, "final": None, "buffer_kind": None, "use_coupling_map": None}
            )
            coupling_map = kwargs.get("coupling_map")
            distances = shortest_path_distances(coupling_map)
            input_pair_counter: Counter[str] = Counter()
            input_pair_distance_hist: Counter[str] = Counter()
            input_gate_counter: Counter[str] = Counter()
            twoq_count = 0
            oneq_count = 0
            weighted_pair_distance = 0
            first_twoq_ops = []
            first_ops = []

            for instruction in circuit:
                gate_name = instruction.operation.name
                if gate_name in {"barrier", "delay"} or len(instruction.qubits) == 0:
                    continue
                qids = local_qubit_indices(circuit, instruction.qubits)
                input_gate_counter[gate_name] += 1
                if len(first_ops) < 40:
                    first_ops.append({"gate": gate_name, "qids": qids})
                if len(qids) == 1:
                    oneq_count += 1
                elif len(qids) == 2:
                    twoq_count += 1
                    pair = tuple(sorted(qids))
                    input_pair_counter[f"{pair[0]}-{pair[1]}"] += 1
                    dist = distances.get((qids[0], qids[1]))
                    if dist is None:
                        input_pair_distance_hist["unknown"] += 1
                    else:
                        input_pair_distance_hist[str(dist)] += 1
                        weighted_pair_distance += int(dist)
                    if len(first_twoq_ops) < 40:
                        first_twoq_ops.append({"gate": gate_name, "qids": qids, "distance": dist})
                else:
                    input_pair_counter[f"{len(qids)}q"] += 1

            out = trace._orig_transpile(circuit, *args, **kwargs)
            output_ops = Counter(dict(out.count_ops()))
            trace.flushes.append(
                {
                    **context,
                    "input_size": int(circuit.size()),
                    "input_depth": int(circuit.depth() or 0),
                    "input_width": int(circuit.num_qubits),
                    "input_ops": dict(input_gate_counter),
                    "input_oneq_count": int(oneq_count),
                    "input_twoq_count": int(twoq_count),
                    "input_top_pairs": counter_to_top(input_pair_counter, 20),
                    "input_pair_distance_hist": dict(input_pair_distance_hist),
                    "input_weighted_pair_distance": int(weighted_pair_distance),
                    "first_ops": first_ops,
                    "first_twoq_ops": first_twoq_ops,
                    "output_size": int(out.size()),
                    "output_depth": int(out.depth() or 0),
                    "output_width": int(out.num_qubits),
                    "output_ops_top": counter_to_top(output_ops, 20),
                    "added_gates": int(out.size()) - int(circuit.size()),
                }
            )
            return out

        evaluator_mod.MappingEvaluator.add_local_ops = traced_add_local_ops
        evaluator_mod.MappingEvaluator.flush_local_ops = traced_flush_local_ops
        evaluator_mod.transpile = traced_transpile

    def uninstall(self) -> None:
        evaluator_mod.MappingEvaluator.add_local_ops = self._orig_add_local_ops
        evaluator_mod.MappingEvaluator.flush_local_ops = self._orig_flush_local_ops
        evaluator_mod.transpile = self._orig_transpile


def owner_key(owner: dict[str, Any]) -> str:
    logical_qid = owner.get("logical_qid")
    if logical_qid is not None:
        return f"q{int(logical_qid)}"
    label = owner.get("label")
    kind = owner.get("kind")
    return f"{kind}:{label}" if label else str(kind)


def summarize_local_ops(local_ops: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind_gate: Counter[str] = Counter()
    by_gate: Counter[str] = Counter()
    by_qpu: dict[int, Counter[str]] = defaultdict(Counter)
    kind_counts: Counter[str] = Counter()
    payload_logical_pairs: Counter[str] = Counter()
    all_wire_pairs: Counter[str] = Counter()
    protocol_pairs: Counter[str] = Counter()
    arity_hist: Counter[str] = Counter()

    for op in local_ops:
        qpu_id = int(op["qpu_id"])
        kind = str(op["kind"])
        gate = str(op["gate"])
        wires = [int(w) for w in op["wires"]]
        owners = list(op["owners"])
        by_kind_gate[f"{kind}:{gate}"] += 1
        by_gate[gate] += 1
        by_qpu[qpu_id][f"{kind}:{gate}"] += 1
        kind_counts[kind] += 1
        arity_hist[str(len(wires))] += 1
        if len(wires) == 2:
            wire_pair = tuple(sorted(wires))
            all_wire_pairs[f"qpu{qpu_id}:w{wire_pair[0]}-w{wire_pair[1]}"] += 1
            owner_pair = "-".join(sorted(owner_key(owner) for owner in owners))
            if kind == "payload":
                payload_logical_pairs[owner_pair] += 1
            else:
                protocol_pairs[f"{kind}:{owner_pair}"] += 1

    return {
        "total_local_ops_added": len(local_ops),
        "arity_hist": dict(arity_hist),
        "kind_counts": dict(kind_counts),
        "gate_counts": counter_to_top(by_gate, 30),
        "kind_gate_counts": counter_to_top(by_kind_gate, 40),
        "qpu_kind_gate_counts": {
            str(qpu_id): counter_to_top(counter, 30)
            for qpu_id, counter in sorted(by_qpu.items())
        },
        "top_payload_logical_pairs": counter_to_top(payload_logical_pairs, 30),
        "top_wire_pairs": counter_to_top(all_wire_pairs, 30),
        "top_protocol_pairs": counter_to_top(protocol_pairs, 30),
        "first_local_ops": local_ops[:80],
    }


def summarize_flushes(flushes: list[dict[str, Any]]) -> dict[str, Any]:
    if not flushes:
        return {}
    input_ops = Counter()
    output_ops = Counter()
    distance_hist = Counter()
    qpu_summary: dict[int, dict[str, Any]] = {}
    qpu_acc: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "input_size": 0,
            "output_size": 0,
            "added_gates": 0,
            "input_twoq_count": 0,
            "distance_hist": Counter(),
            "top_pairs": Counter(),
        }
    )

    for flush in flushes:
        input_ops.update(flush["input_ops"])
        output_ops.update(flush["output_ops_top"])
        distance_hist.update(flush["input_pair_distance_hist"])
        qpu_id = flush.get("qpu_id")
        if qpu_id is None:
            continue
        acc = qpu_acc[int(qpu_id)]
        acc["input_size"] += int(flush["input_size"])
        acc["output_size"] += int(flush["output_size"])
        acc["added_gates"] += int(flush["added_gates"])
        acc["input_twoq_count"] += int(flush["input_twoq_count"])
        acc["distance_hist"].update(flush["input_pair_distance_hist"])
        acc["top_pairs"].update(flush["input_top_pairs"])

    for qpu_id, acc in sorted(qpu_acc.items()):
        qpu_summary[str(qpu_id)] = {
            "input_size": int(acc["input_size"]),
            "output_size": int(acc["output_size"]),
            "added_gates": int(acc["added_gates"]),
            "ratio": round(acc["output_size"] / max(1, acc["input_size"]), 3),
            "input_twoq_count": int(acc["input_twoq_count"]),
            "distance_hist": dict(acc["distance_hist"]),
            "top_pairs": counter_to_top(acc["top_pairs"], 20),
        }

    top_by_added = sorted(flushes, key=lambda item: item["added_gates"], reverse=True)[:12]
    top_by_ratio = sorted(
        flushes,
        key=lambda item: item["output_size"] / max(1, item["input_size"]),
        reverse=True,
    )[:12]
    total_input = sum(int(f["input_size"]) for f in flushes)
    total_output = sum(int(f["output_size"]) for f in flushes)
    return {
        "transpile_calls": len(flushes),
        "total_input_size": int(total_input),
        "total_output_size": int(total_output),
        "total_added_gates": int(total_output - total_input),
        "overall_ratio": round(total_output / max(1, total_input), 3),
        "input_ops": counter_to_top(input_ops, 30),
        "output_ops_top": counter_to_top(output_ops, 30),
        "input_pair_distance_hist": dict(distance_hist),
        "qpu_summary": qpu_summary,
        "top_flushes_by_added": top_by_added,
        "top_flushes_by_ratio": top_by_ratio,
    }


def build_context(args: argparse.Namespace) -> tuple[dict[str, Any], QuantumCircuit, Network, dict[str, Any]]:
    global_config = get_config(args.global_config_path)
    network_config = {
        **global_config.get("network_config", {}),
        "type": args.network,
    }
    if args.local_eval_mode is not None:
        network_config["local_eval_mode"] = args.local_eval_mode
    if args.deferred_initial_layout is not None:
        network_config["deferred_initial_layout"] = args.deferred_initial_layout
    if args.deferred_route_local_gates is not None:
        network_config["deferred_route_local_gates"] = args.deferred_route_local_gates

    core_capacity = normalize_repeated(parse_int_list(args.core_capacity), args.core_count)
    backend_names = normalize_repeated(parse_str_list(args.backend_name), args.core_count)
    dates = normalize_repeated(parse_str_list(args.date), args.core_count)
    comm_slot_reserve = int(
        network_config.get("comm_slot_reserve", global_config.get("comm_slot_reserve", 0)) or 0
    )

    backends = []
    for idx in range(args.core_count):
        sampled_capacity = int(core_capacity[idx]) + comm_slot_reserve
        backend_config = {
            "backend_name": f"{backend_names[idx]}_sampled_{sampled_capacity}q",
            "date": datetime.datetime.strptime(dates[idx], "%Y-%m-%d"),
        }
        backends.append(Backend(global_config, backend_config))

    network = Network(network_config, backends)
    _, trans_circ, task_info = select_circuit(
        args.circuit_name,
        args.qubit_count,
        args.core_count,
        core_capacity,
        network.basis_gates,
        network.two_qubit_gates,
    )
    return global_config, trans_circ, network, task_info


def compile_config(global_config: dict[str, Any], compiler: Any, circuit_name: str) -> dict[str, Any]:
    return {
        "circuit_name": circuit_name,
        **global_config.get("compile_config", {}),
        **global_config.get("compiler_config", {}).get(compiler.compiler_id, {}),
    }


def run_one_compiler(
    compiler_id: str,
    global_config: dict[str, Any],
    circuit: QuantumCircuit,
    network: Network,
    quiet_enabled: bool,
) -> dict[str, Any]:
    compiler_cls = CompilerFactory.get_compiler(compiler_id)
    compiler = compiler_cls()
    trace = LocalShapeTrace()
    orig_save_records = MappingRecordList.save_records

    def noop_save_records(self, filename: str):
        del self, filename
        return None

    MappingRecordList.save_records = noop_save_records
    trace.install()
    try:
        with quiet(quiet_enabled):
            result = compiler.compile(
                copy.deepcopy(circuit),
                network,
                compile_config(global_config, compiler, f"diagnose_{compiler_id}"),
            )
    finally:
        trace.uninstall()
        MappingRecordList.save_records = orig_save_records

    return {
        "compiler_id": compiler_id,
        "compiler": compiler.name,
        "total_costs": result.total_costs.to_dict(),
        "num_records": result.num_records,
        "record_shapes": [
            {
                "layer_start": record.layer_start,
                "layer_end": record.layer_end,
                "mapping_type": record.mapping_type,
                "partition_sizes": [len(group) for group in record.partition],
            }
            for record in result.records
        ],
        "local_ops": summarize_local_ops(trace.local_ops),
        "flushes": summarize_flushes(trace.flushes),
    }


def load_record_list_from_json(path: Path) -> MappingRecordList:
    data = json.loads(path.read_text(encoding="utf-8"))
    record_list = MappingRecordList()
    for item in data.get("records", []):
        extra_info = item.get("extra_info")
        extra_info = MappingRecordList._deserialize_extra_info(extra_info)
        record = MappingRecord(
            layer_start=int(item.get("layer_start", -1)),
            layer_end=int(item.get("layer_end", -1)),
            partition=[[int(q) for q in group] for group in item.get("partition", [])],
            mapping_type=str(item.get("mapping_type", "")),
            extra_info=copy.deepcopy(extra_info) if extra_info is not None else None,
        )
        record_list.add_record(record)
    return record_list


def replay_record_json(
    label: str,
    path: Path,
    circuit: QuantumCircuit,
    network: Network,
    policy_name: str | None = None,
) -> dict[str, Any]:
    record_list = load_record_list_from_json(path)
    if any(record.mapping_type == "cat" and not ((record.extra_info or {}).get("ops")) for record in record_list.records):
        raise ValueError(
            f"{path} contains cat records without serialized extra_info['ops']; "
            "replay cannot reconstruct those local gates from JSON."
        )
    trace = LocalShapeTrace()
    trace.install()
    try:
        result = CompilerUtils.evaluate_raw_mapping_records(
            record_list,
            network,
            policy_name=policy_name,
        )
    finally:
        trace.uninstall()

    return {
        "compiler_id": label,
        "compiler": f"{label} (replayed)",
        "source_json": str(path),
        "total_costs": result.total_costs.to_dict(),
        "num_records": result.num_records,
        "record_shapes": [
            {
                "layer_start": record.layer_start,
                "layer_end": record.layer_end,
                "mapping_type": record.mapping_type,
                "partition_sizes": [len(group) for group in record.partition],
            }
            for record in result.records
        ],
        "local_ops": summarize_local_ops(trace.local_ops),
        "flushes": summarize_flushes(trace.flushes),
    }


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
    parser.add_argument("--compilers", default="wbcp,navihybrid")
    parser.add_argument(
        "--record-jsons",
        default="",
        help="Replay inputs separated by ';;', e.g. wbcp=outputs/.../WBCP.json. "
        "Useful when a compiler cannot be rerun locally.",
    )
    parser.add_argument(
        "--record-json",
        action="append",
        default=[],
        help="Single replay input, e.g. wbcp=outputs/.../WBCP.json. May be repeated.",
    )
    parser.add_argument("--local-eval-mode", choices=["immediate", "deferred"], default=None)
    parser.add_argument("--deferred-initial-layout", choices=["fixed", "free"], default=None)
    parser.add_argument("--deferred-route-local-gates", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--out", default="outputs/local_gate_shapes.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    global_config, circuit, network, task_info = build_context(args)
    CompilerFactory.register_compilers(global_config.get("compiler_modules", ["src.baselines", "src.navi"]))

    reports = []
    for compiler_id in parse_str_list(args.compilers):
        if not compiler_id:
            continue
        print(f"[local-shape] running {compiler_id}...", file=sys.stderr)
        reports.append(run_one_compiler(compiler_id, global_config, circuit, network, args.quiet))
    record_specs = list(args.record_json or []) + parse_record_json_specs(args.record_jsons)
    for spec in record_specs:
        if not spec:
            continue
        if "=" in spec:
            label, path_text = spec.split("=", 1)
        else:
            path_text = spec
            label = Path(path_text).stem
        print(f"[local-shape] replaying {label} from {path_text}...", file=sys.stderr)
        reports.append(
            replay_record_json(
                label.strip(),
                Path(path_text.strip()),
                circuit,
                network,
                policy_name=None,
            )
        )

    report = {
        "task_info": task_info,
        "network": network.info(),
        "network_local_eval_mode": getattr(network, "local_eval_mode", None),
        "network_deferred_initial_layout": getattr(network, "deferred_initial_layout", None),
        "compilers": reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for item in reports:
        costs = item["total_costs"]
        flushes = item["flushes"]
        local_ops = item["local_ops"]
        print(
            f"{item['compiler']}: epairs={costs['epairs']}, "
            f"local_pre={costs['local_pre_transpile_gate_num']}, "
            f"local_added={costs['local_transpile_added_gate_num']}, "
            f"local_total={costs['local_gate_num']}, "
            f"transpile_calls={flushes.get('transpile_calls')}, "
            f"ratio={flushes.get('overall_ratio')}"
        )
        print(f"  local kind counts: {local_ops['kind_counts']}")
        print(f"  local gate counts: {local_ops['gate_counts']}")
        print(f"  top payload logical pairs: {local_ops['top_payload_logical_pairs']}")
        print(f"  distance hist: {flushes.get('input_pair_distance_hist')}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
