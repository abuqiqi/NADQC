from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.navi import NAVI, evaluate_navi_checkpoints_from_config
from src.utils import Backend, Network, get_config, select_circuit, write_compiler_results_to_csv


def build_network(global_config: dict[str, Any], args: argparse.Namespace) -> Network:
    network_config = {
        **global_config.get("network_config", {}),
        "type": args.network,
        "fidelity_range": [0.90, 0.90],
        "comm_slot_reserve": args.comm_slot_reserve,
    }

    backends = []
    for _ in range(args.core_count):
        sampled_capacity = args.core_capacity + args.comm_slot_reserve
        backend_config = {
            "backend_name": f"{args.backend_name}_sampled_{sampled_capacity}q",
            "date": datetime.datetime.strptime(args.date, "%Y-%m-%d"),
        }
        backends.append(Backend(global_config, backend_config))

    return Network(network_config, backends)


def build_smoke_config(global_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(global_config)
    compiler_config = copy.deepcopy(cfg.get("compiler_config", {}))
    compiler_config["common"] = {
        **compiler_config.get("common", {}),
        "local_eval_mode": "immediate",
        "deferred_initial_layout": "fixed",
        "deferred_route_local_gates": True,
        "sequential_initial_layout": True,
    }
    compiler_config["navi"] = {
        **compiler_config.get("navi", {}),
        "hybrid_beam_width": 1,
        "hybrid_max_merge_span": 3,
        "hybrid_merge_probe_budget": 3,
        "hybrid_pos_probe_budget": 3,
        "hybrid_prune_trigger_ratio": 2,
        "telegate_cache": True,
        "telegate_cache_mode": "range",
    }
    cfg["compiler_config"] = compiler_config
    cfg["navi_checkpoint"] = {
        "enabled": True,
        "evaluations": [
            {
                "id": "td_only",
                "name": "NAVI-Td",
                "source": "NAVI-Td",
            },
            {
                "id": "hybrid_direct",
                "name": "NAVI-Hy",
                "source": "NAVI-Hy",
            },
            {
                "id": "full",
                "name": "NAVI",
                "source": "NAVI",
            },
        ],
    }
    return cfg


def assert_raw_has_no_costs(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("dump_type") == "raw", f"{path} is not a raw dump"
    assert "total_costs" not in data, f"{path} unexpectedly contains total_costs"
    for idx, record in enumerate(data.get("records", [])):
        assert "costs" not in record, f"{path} record {idx} unexpectedly contains costs"
        assert "logical_phy_map" not in record, f"{path} record {idx} unexpectedly contains logical_phy_map"
        assert "comm_phy_map" not in record, f"{path} record {idx} unexpectedly contains comm_phy_map"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a narrow NAVI checkpoint smoke test.")
    parser.add_argument("--global-config-path", default="config.json")
    parser.add_argument("--circuit-name", default="test")
    parser.add_argument("--qubit-count", type=int, default=4)
    parser.add_argument("--core-count", type=int, default=2)
    parser.add_argument("--core-capacity", type=int, default=3)
    parser.add_argument("--comm-slot-reserve", type=int, default=1)
    parser.add_argument("--backend-name", default="ibm_marrakesh")
    parser.add_argument("--date", default="2026-03-01")
    parser.add_argument("--network", default="all_to_all")
    parser.add_argument("--local-eval-mode", choices=["immediate", "deferred"], default="immediate")
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    global_config = get_config(args.global_config_path)
    smoke_config = build_smoke_config(global_config)
    smoke_config["compiler_config"]["common"]["local_eval_mode"] = args.local_eval_mode
    network = build_network(smoke_config, args)

    _, trans_circ, task_info = select_circuit(
        args.circuit_name,
        args.qubit_count,
        args.core_count,
        [args.core_capacity] * args.core_count,
        network.basis_gates,
        network.two_qubit_gates,
    )
    circuit_name = f"{args.circuit_name}{trans_circ.num_qubits}"
    checkpoint_dir = (
        Path(str(smoke_config.get("output_folder", "outputs")))
        / circuit_name
        / f"navi_checkpoints_smoke_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    compile_config = {
        "circuit_name": circuit_name,
        **smoke_config.get("compile_config", {}),
        **smoke_config.get("compiler_config", {}).get("common", {}),
        **smoke_config.get("compiler_config", {}).get("navi", {}),
        "navi_checkpoint_enabled": True,
        "navi_checkpoint_dir": str(checkpoint_dir),
    }
    result = NAVI().compile(trans_circ, network, compile_config)
    pprint(result.total_costs)

    manifest_path = checkpoint_dir / "manifest.json"
    assert manifest_path.exists(), f"missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("NAVI-Td", "NAVI-Hy", "NAVI"):
        assert key in manifest, f"missing manifest key: {key}"
        raw_path = checkpoint_dir / manifest[key]
        assert raw_path.exists(), f"missing raw checkpoint: {raw_path}"
        assert_raw_has_no_costs(raw_path)

    eval_result = evaluate_navi_checkpoints_from_config(
        smoke_config,
        checkpoint_dir,
        network,
        task_info,
        default_policy_name=compile_config.get("evaluator_policy"),
        config=compile_config,
    )
    expected_names = {
        "NAVI-Td",
        "NAVI-Hy",
        "NAVI",
    }
    missing = expected_names - set(eval_result.result_info.keys())
    assert not missing, f"missing checkpoint evaluation results: {sorted(missing)}"

    disabled_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_disabled_probe"
    disabled_config = {
        **compile_config,
        "navi_checkpoint_enabled": False,
        "navi_checkpoint_dir": str(disabled_dir),
    }
    NAVI().compile(trans_circ, network, disabled_config)
    assert not (disabled_dir / "manifest.json").exists(), "disabled checkpoint unexpectedly wrote manifest"

    if args.write_csv:
        csv_path = Path(str(smoke_config.get("output_folder", "outputs"))) / "navi_checkpoint_smoke.csv"
        write_compiler_results_to_csv(
            {**task_info, **network.info()},
            eval_result.result_info,
            str(csv_path),
        )
        print(f"[SMOKE] csv={csv_path}")

    print("[SMOKE] NAVI checkpoint result_info:")
    pprint(eval_result.result_info)
    print(f"[SMOKE] checkpoints={checkpoint_dir}")
    print("[SMOKE] PASS")


if __name__ == "__main__":
    main()
