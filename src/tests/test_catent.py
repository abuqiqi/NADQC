import json
import datetime
from pathlib import Path

from src.utils import get_config, Backend, Network, select_circuit
from src.navi.navi_hybrid import NAVIHybrid
from src.navi.navi_compiler import CompilationContext
from src.compiler.compiler_utils import CompilerUtils
from src.baselines.autocomm import QAutoComm


def main():
    root = Path("/home/xls/project/NADQC")
    json_path = root / "outputs/CuccaroAdder80/CuccaroAdder80_net[20, 20, 20, 20]_NAVI Hybrid_20260419_113930.json"

    obj = json.load(open(json_path))
    cat_record = obj["records"][1]

    partition = cat_record["partition"]
    layer_start = cat_record["layer_start"]
    layer_end = cat_record["layer_end"]

    print("cat_record layers:", layer_start, layer_end)
    print(
        "cat_record stored costs:",
        {k: cat_record["costs"][k] for k in [
            "epairs",
            "num_comms",
            "remote_hops",
            "remote_swaps",
            "cat_ents",
            "local_gate_num",
            "total_fidelity_loss",
        ]}
    )

    cfg = get_config("config.json")
    network_config = {
        **cfg.get("network_config", {}),
        "type": "mesh_grid",
    }
    comm_slot_reserve = int(
        network_config.get("comm_slot_reserve", cfg.get("comm_slot_reserve", 0)) or 0
    )

    backend_list = []
    for _ in range(4):
        sampled_capacity = 20 + comm_slot_reserve
        backend_config = {
            "backend_name": f"ibm_marrakesh_sampled_{sampled_capacity}q",
            "date": datetime.datetime.strptime("2026-03-01", "%Y-%m-%d"),
        }
        backend_list.append(Backend(cfg, backend_config))

    network_config["fidelity_range"] = [0.90, 0.93]
    network = Network(network_config, backend_list)

    basis_gates = network.basis_gates
    two_qubit_gates = network.two_qubit_gates

    _, trans_circ, task_info = select_circuit(
        "CuccaroAdder",
        80,
        4,
        [20, 20, 20, 20],
        basis_gates,
        two_qubit_gates,
    )
    print("task_info:", task_info)

    nh = NAVIHybrid()
    ctx = CompilationContext(circuit=trans_circ, network=network, config={})
    nh._step_remove_single_qubit_gates(ctx)

    subcircuit = CompilerUtils.get_subcircuit_by_level(
        trans_circ.num_qubits,
        trans_circ,
        ctx.circuit_layers,
        layer_start,
        layer_end,
    )

    print(subcircuit)

    twoq_count = sum(v for k, v in subcircuit.count_ops().items() if k in two_qubit_gates)
    print("subcircuit depth:", subcircuit.depth())
    print("subcircuit size:", subcircuit.size())
    print("subcircuit 2q_count:", twoq_count)

    compiler = QAutoComm()

    for comm_only in (False, True):
        result = compiler.compile(
            subcircuit,
            network,
            {
                "circuit_name": "debug_subc",
                "partition": partition,
                "save_records": False,
                "comm_only_costs": comm_only,
            },
        )

        c = result.total_costs
        print(f"\nQAutoComm result comm_only_costs={comm_only}")
        print({
            "epairs": c.epairs,
            "num_comms": c.num_comms,
            "remote_hops": c.remote_hops,
            "remote_swaps": c.remote_swaps,
            "cat_ents": c.cat_ents,
            "local_gate_num": c.local_gate_num,
            "total_fidelity_loss": c.total_fidelity_loss,
            "remote_fidelity_loss": c.remote_fidelity_loss,
            "local_fidelity_loss": c.local_fidelity_loss,
        })

        rec = result.records[0]
        ei = rec.extra_info or {}
        print("extra_info:", {
            "autocomm_latency": ei.get("autocomm_latency"),
            "autocomm_assigned_block_count": ei.get("autocomm_assigned_block_count"),
            "comm_only_costs": ei.get("comm_only_costs"),
        })


if __name__ == "__main__":
    main()
