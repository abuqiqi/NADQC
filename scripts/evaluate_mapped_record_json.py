from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler import CompilerUtils, MappingRecord, MappingRecordList
from src.navi import MapperFactory
from src.utils import Backend, Network, get_config, write_compiler_results_to_csv
from src.utils.io import parse_int_list, parse_str_list


OPS_ONLY_MAPPERS = {
    "direct",
    "single_record_debug",
    "single_record_greedy",
    "boundeddp_neighbor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a raw mapping-record JSON directly and after NAVI mapper remapping."
    )
    parser.add_argument("--record-json", required=True, help="Path to a self-contained raw mapping record JSON.")
    parser.add_argument("--mapper", default=None, help="Mapper override. Default: single_record_greedy for 1 record, boundeddp_neighbor otherwise.")
    parser.add_argument("--output-csv", default=None, help="CSV output path. Default: <output_folder>/raw_record_mapper_eval.csv.")
    parser.add_argument("--output-prefix", default=None, help="Output JSON path prefix. Default: main-style output prefix.")
    parser.add_argument("--policy", default=None, help="Evaluator policy override.")

    parser.add_argument("--global_config_path", "-gconf", type=str, default="config.json")
    parser.add_argument("--circuit_name", "-cname", type=str, required=True)
    parser.add_argument("--qubit_count", "-nq", type=int, required=True)
    parser.add_argument("--core_count", "-core", type=int, required=True)
    parser.add_argument("--core_capacity", "-cap", type=parse_int_list, required=True)
    parser.add_argument("--backend_name", "-bname", type=parse_str_list, default=["ibm_marrakesh"])
    parser.add_argument("--date", "-date", type=parse_str_list, default=["2026-03-01"])
    parser.add_argument("--network", "-net", type=str, default="all_to_all")
    parser.add_argument("--gate_set", "-gset", type=str, default="cu1,u3")

    args = parser.parse_args()

    if len(args.core_capacity) == 1:
        args.core_capacity = args.core_capacity * args.core_count
    if len(args.backend_name) == 1:
        args.backend_name = args.backend_name * args.core_count
    if len(args.date) == 1:
        args.date = args.date * args.core_count

    if len(args.core_capacity) != args.core_count:
        raise ValueError(
            f"The number of QPU capacities {args.core_capacity} must match core_count={args.core_count}."
        )
    if len(args.backend_name) != args.core_count:
        raise ValueError(
            f"The number of backend names {args.backend_name} must match core_count={args.core_count}."
        )
    if len(args.date) != args.core_count:
        raise ValueError(
            f"The number of backend dates {args.date} must match core_count={args.core_count}."
        )
    if not args.gate_set:
        raise ValueError("The basis gate set must not be empty.")
    args.gate_set = args.gate_set.split(",")

    return args


def load_record_list_from_json(path: Path) -> MappingRecordList:
    data = json.loads(path.read_text(encoding="utf-8"))
    record_list = MappingRecordList()
    for item in data.get("records", []):
        extra_info = MappingRecordList._deserialize_extra_info(item.get("extra_info"))
        record = MappingRecord(
            layer_start=int(item.get("layer_start", -1)),
            layer_end=int(item.get("layer_end", -1)),
            partition=[[int(q) for q in group] for group in item.get("partition", [])],
            mapping_type=str(item.get("mapping_type", "")),
            extra_info=copy.deepcopy(extra_info) if extra_info is not None else None,
        )
        record_list.add_record(record)
    record_list.num_records = len(record_list.records)
    return record_list


def require_ops(record_list: MappingRecordList) -> None:
    if not record_list.records:
        raise ValueError("record JSON contains no records")
    for idx, record in enumerate(record_list.records):
        if not isinstance(record.extra_info, dict) or "ops" not in record.extra_info:
            raise ValueError(
                f"record #{idx} is missing extra_info['ops']; this script only supports self-contained raw records."
            )
        if not isinstance(record.extra_info["ops"], QuantumCircuit):
            raise TypeError(
                f"record #{idx} extra_info['ops'] must deserialize to QuantumCircuit, got {type(record.extra_info['ops'])}"
            )


def build_network(args: argparse.Namespace, global_config: dict[str, Any]) -> Network:
    network_config = {
        **global_config.get("network_config", {}),
        "type": args.network,
    }
    comm_slot_reserve = int(network_config.get("comm_slot_reserve", global_config.get("comm_slot_reserve", 0)) or 0)

    backends = []
    for idx in range(args.core_count):
        sampled_capacity = int(args.core_capacity[idx]) + comm_slot_reserve
        backend_config = {
            "backend_name": f"{args.backend_name[idx]}_sampled_{sampled_capacity}q",
            "date": datetime.datetime.strptime(args.date[idx], "%Y-%m-%d"),
        }
        backends.append(Backend(global_config, backend_config))

    if "fidelity_range" not in network_config:
        network_config["fidelity_range"] = [0.90, 0.93]

    return Network(network_config, backends)


def infer_selected_mapper(record_list: MappingRecordList, mapper_override: Optional[str]) -> str:
    if mapper_override is not None:
        mapper_id = str(mapper_override).lower()
    elif len(record_list.records) == 1:
        mapper_id = "single_record_greedy"
    else:
        mapper_id = "boundeddp_neighbor"

    if mapper_id not in OPS_ONLY_MAPPERS:
        allowed = ", ".join(sorted(OPS_ONLY_MAPPERS))
        raise ValueError(
            f"mapper '{mapper_id}' is not supported by this ops-only script. "
            f"Allowed mappers: {allowed}"
        )
    return mapper_id


def iter_record_ops(record_list: MappingRecordList):
    for record in record_list.records:
        assert isinstance(record.extra_info, dict)
        yield record.extra_info["ops"]


def build_task_info(args: argparse.Namespace, record_list: MappingRecordList, network: Network) -> dict[str, Any]:
    total_gates = 0
    total_2q_gates = 0
    total_depth = 0
    two_qubit_names = set(network.two_qubit_gates)

    for ops in iter_record_ops(record_list):
        counts = ops.count_ops()
        total_gates += int(sum(counts.values()))
        total_depth += int(ops.depth() or 0)
        for instruction in ops.data:
            op = instruction.operation
            if getattr(op, "name", None) in two_qubit_names or int(getattr(op, "num_qubits", 0)) == 2:
                total_2q_gates += 1

    return {
        "Circuit": args.circuit_name,
        "#Qubits": args.qubit_count,
        "#Gates": total_gates,
        "#2Q Gates": total_2q_gates,
        "#Depth": total_depth,
        "#QPUs": args.core_count,
        "QPUs": args.core_capacity,
        "Basis Gates": network.basis_gates,
        "2Q Gate Names": network.two_qubit_gates,
    }


def result_metrics(result: MappingRecordList, depth: int) -> dict[str, Any]:
    safe_depth = max(1, int(depth))
    return {
        "F_eff": np.exp(result.total_costs.total_fidelity_log_sum / safe_depth),
        **result.total_costs.to_dict(),
    }


def extract_perm_sequence(record_list: MappingRecordList) -> list[Any]:
    perms = []
    for record in record_list.records:
        if isinstance(record.extra_info, dict) and "perm" in record.extra_info:
            perms.append(record.extra_info["perm"])
    return perms


def default_output_prefix(args: argparse.Namespace, record_path: Path, network: Network, mapper_id: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    circuit_name = f"{args.circuit_name}{args.qubit_count}"
    compiler_name = f"RawRecord_{record_path.stem}_{mapper_id}"
    return CompilerUtils.build_record_base_path(circuit_name, network, compiler_name, timestamp)


def ensure_has_parent_path(path_value: str) -> str:
    path = Path(path_value)
    if path.parent == Path(".") and not str(path_value).startswith("./"):
        return f"./{path_value}"
    return path_value


def main() -> None:
    args = parse_args()
    global_config = get_config(args.global_config_path)
    record_path = Path(args.record_json)

    network = build_network(args, global_config)
    raw_records = load_record_list_from_json(record_path)
    require_ops(raw_records)

    mapper_id = infer_selected_mapper(raw_records, args.mapper)
    task_info = build_task_info(args, raw_records, network)

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = str(Path(str(global_config.get("output_folder", "outputs"))) / "raw_record_mapper_eval.csv")
    output_csv = ensure_has_parent_path(output_csv)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = default_output_prefix(args, record_path, network, mapper_id)
    output_prefix = ensure_has_parent_path(output_prefix)

    common_config = {
        **global_config.get("compile_config", {}),
        **global_config.get("compiler_config", {}).get("common", {}),
    }
    if args.policy is not None:
        common_config["evaluator_policy"] = args.policy
    policy_name = common_config.get("evaluator_policy")

    direct_records = CompilerUtils.evaluate_raw_mapping_records(
        copy.deepcopy(raw_records),
        network,
        policy_name=policy_name,
        config=common_config,
    )
    direct_records.save_records(f"{output_prefix}_direct_evaluated.json", dump_type="evaluated")

    mapper = MapperFactory.create_mapper(mapper_id)
    placeholder_circuit = QuantumCircuit(args.qubit_count)
    mapped_raw_records = mapper.map(
        copy.deepcopy(raw_records),
        placeholder_circuit,
        [],
        network,
        config=common_config,
    )
    mapped_raw_records.num_records = len(mapped_raw_records.records)
    mapped_raw_records.save_records(f"{output_prefix}_mapped_raw.json", dump_type="raw")

    mapped_eval_records = CompilerUtils.evaluate_raw_mapping_records(
        copy.deepcopy(mapped_raw_records),
        network,
        policy_name=policy_name,
        config=common_config,
    )
    mapped_eval_records.save_records(f"{output_prefix}_mapped_evaluated.json", dump_type="evaluated")

    result_info = {
        "Raw Direct Eval": result_metrics(direct_records, task_info["#Depth"]),
        "Mapped Eval": result_metrics(mapped_eval_records, task_info["#Depth"]),
    }
    write_compiler_results_to_csv(
        {**task_info, **network.info()},
        result_info,
        output_csv,
    )

    print(
        f"[INFO] input_json={record_path} mapper={mapper_id} "
        f"records={len(raw_records.records)} perm_sequence={extract_perm_sequence(mapped_raw_records)}"
    )
    print(f"[INFO] output_csv={output_csv}")
    print(f"[INFO] output_prefix={output_prefix}")


if __name__ == "__main__":
    main()
