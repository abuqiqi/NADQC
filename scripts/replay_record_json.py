import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler import CompilerUtils, MappingRecord, MappingRecordList
from src.utils import Backend, Network, get_config


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_date_list(value: str) -> list[datetime.datetime]:
    dates: list[datetime.datetime] = []
    for item in parse_str_list(value):
        dates.append(datetime.datetime.strptime(item, "%Y-%m-%d"))
    return dates


def normalize_repeated(values: list[Any], count: int) -> list[Any]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"expected 1 or {count} values, got {len(values)}")
    return values


def build_context(args: argparse.Namespace) -> tuple[dict[str, Any], Network]:
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
    dates = normalize_repeated(parse_date_list(args.date), args.core_count)
    comm_slot_reserve = int(
        network_config.get("comm_slot_reserve", global_config.get("comm_slot_reserve", 0)) or 0
    )

    backends = []
    for idx in range(args.core_count):
        sampled_capacity = int(core_capacity[idx]) + comm_slot_reserve
        backend_config = {
            "backend_name": f"{backend_names[idx]}_sampled_{sampled_capacity}q",
            "date": dates[idx],
        }
        backends.append(Backend(global_config, backend_config))

    network = Network(network_config, backends)
    return global_config, network


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-json", required=True)
    parser.add_argument("--global-config-path", default="config.json")
    parser.add_argument("--circuit-name", default="QFT", help=argparse.SUPPRESS)
    parser.add_argument("--qubit-count", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--core-count", type=int, default=2)
    parser.add_argument("--core-capacity", default="5")
    parser.add_argument("--backend-name", default="ibm_marrakesh")
    parser.add_argument("--date", default="2026-03-01")
    parser.add_argument("--network", default="all_to_all")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--local-eval-mode", choices=["immediate", "deferred"], default=None)
    parser.add_argument("--deferred-initial-layout", choices=["fixed", "free"], default=None)
    parser.add_argument("--deferred-route-local-gates", type=lambda x: x.lower() == "true", default=None)
    args = parser.parse_args()

    _config, network = build_context(args)

    record_path = Path(args.record_json)
    record_list = load_record_list_from_json(record_path)

    result = CompilerUtils.evaluate_raw_mapping_records(
        record_list,
        network,
        policy_name=args.policy,
    )

    costs = result.total_costs.to_dict()
    print(
        f"replay={record_path.name} records={result.num_records} "
        f"epairs={costs.get('epairs')} "
        f"local_total={costs.get('local_gate_num')} "
        f"remote_hops={costs.get('remote_hops')} "
        f"remote_swaps={costs.get('remote_swaps')}"
    )


if __name__ == "__main__":
    main()
