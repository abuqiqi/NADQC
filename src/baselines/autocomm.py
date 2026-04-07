from typing import Any, Optional
import time
import datetime

from qiskit import QuantumCircuit

from .AutoComm.gate_util import (
    build_gate,
    build_RZ_gate,
    build_CX_gate,
    build_CZ_gate,
    build_CRZ_gate,
    build_CU1_gate,
    build_H_gate,
)
from .AutoComm.autocomm import autocomm_full

from ..compiler import Compiler, MappingRecord, MappingRecordList, ExecCosts, CompilerUtils
from ..utils import Network


class QAutoComm(Compiler):
    """
    AutoComm baseline wrapped as NADQC Compiler interface.
    """

    compiler_id = "autocomm"

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "AutoComm"

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> MappingRecordList:
        print(f"Compiling with [{self.name}]...")
        cfg = config or {}
        circuit_name = cfg.get("circuit_name", "circ")

        start_time = time.time()

        gate_list = self._to_autocomm_gate_list(circuit)
        qubit_node_mapping = self._generate_qubit_node_mapping(circuit.num_qubits, network.backend_sizes)

        num_q = len(qubit_node_mapping)
        qb_per_node = max(network.backend_sizes) if network.backend_sizes else 1
        aggregate_iter_cnt = max(1, num_q // max(1, qb_per_node))
        schedule_iter_cnt = max(1, num_q // max(1, qb_per_node))

        result = autocomm_full(
            gate_list,
            qubit_node_mapping,
            aggregate_iter_cnt=aggregate_iter_cnt,
            schedule_iter_cnt=schedule_iter_cnt,
            return_comm_events=True,
        )
        if len(result) == 5:
            epr_cnt, all_latency, assigned_gate_blocks, comm_costs, comm_events = result
        else:
            epr_cnt, all_latency, assigned_gate_blocks, comm_costs = result
            comm_events = []

        exec_time = time.time() - start_time

        record = MappingRecord(
            layer_start=0,
            layer_end=max(0, circuit.depth() - 1),
            partition=self._partition_from_mapping(circuit.num_qubits, qubit_node_mapping, network.num_backends),
            mapping_type="autocomm",
        )

        cat_comm = int(comm_costs[0]) if len(comm_costs) > 0 else 0
        tp_comm = int(sum(comm_costs[1:])) if len(comm_costs) > 1 else 0
        total_remote_events = cat_comm + tp_comm

        costs = ExecCosts()
        costs.execution_time = exec_time

        if comm_events:
            for event in comm_events:
                src = int(event.get("src", -1))
                dst = int(event.get("dst", -1))
                if src < 0 or dst < 0:
                    continue
                model = str(event.get("model", "")).lower()
                if model == "swap":
                    costs = CompilerUtils.update_remote_swap_costs(costs, src, dst, 1, network)
                elif model in ("move", "cat"):
                    weight = max(1, int(event.get("epr", 1)))
                    costs = CompilerUtils.update_remote_move_costs(costs, src, dst, weight, network)
        elif total_remote_events > 0:
            costs.remote_hops = cat_comm
            costs.remote_swaps = tp_comm

            move_vals = []
            swap_vals = []
            for i in range(network.num_backends):
                for j in range(network.num_backends):
                    if i == j:
                        continue
                    move_vals.append(network.move_fidelity[i][j])
                    swap_vals.append(network.swap_fidelity[i][j])

            avg_move_fid = sum(move_vals) / len(move_vals) if move_vals else 1.0
            avg_swap_fid = sum(swap_vals) / len(swap_vals) if swap_vals else 1.0

            costs.remote_fidelity = (avg_move_fid ** cat_comm) * (avg_swap_fid ** tp_comm)
            costs.remote_fidelity_log_sum = 0.0
            if cat_comm > 0 and avg_move_fid > 0:
                import math
                costs.remote_fidelity_log_sum += cat_comm * math.log(avg_move_fid)
            if tp_comm > 0 and avg_swap_fid > 0:
                import math
                costs.remote_fidelity_log_sum += tp_comm * math.log(avg_swap_fid)
            costs.remote_fidelity_loss = 1.0 - costs.remote_fidelity

        costs.epairs = int(epr_cnt)

        costs.local_gate_num = self._count_local_gate_like_ops(gate_list, qubit_node_mapping)

        # AutoComm latency is a scheduler makespan proxy; keep total execution wall time in execution_time,
        # and expose scheduler latency in extra_info for analysis.
        record.costs = costs
        record.extra_info = {
            "autocomm_epr_cnt": int(epr_cnt),
            "autocomm_latency": float(all_latency),
            "autocomm_comm_costs": [int(x) for x in comm_costs],
            "autocomm_assigned_block_count": len(assigned_gate_blocks),
            "autocomm_event_count": len(comm_events),
        }

        result = MappingRecordList()
        result.add_record(record)
        result.summarize_total_costs()
        result.update_total_costs(execution_time=exec_time)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")
        return result

    def _to_autocomm_gate_list(self, circuit: QuantumCircuit) -> list[list[Any]]:
        gate_list: list[list[Any]] = []
        global_phase = circuit.global_phase

        for instruction in circuit:
            gate = instruction.operation
            gate_name = gate.name
            qubits = [q._index for q in instruction.qubits]
            if qubits and qubits[0] is None:
                qubits = [circuit.qubits.index(q) for q in instruction.qubits]
            params = list(gate.params) if gate.params else []

            if gate_name == "u3":
                g = build_gate("U3", qubits, params, global_phase=global_phase)
            elif gate_name == "rz":
                g = build_RZ_gate(qubits[0], angle=params[0], global_phase=global_phase)
            elif gate_name == "rx":
                g = build_gate("RX", [qubits[0]], [params[0]], global_phase=global_phase)
            elif gate_name == "x":
                g = build_gate("X", [qubits[0]], [], global_phase=global_phase)
            elif gate_name == "sx":
                g = build_gate("SX", [qubits[0]], [], global_phase=global_phase)
            elif gate_name == "id":
                g = build_gate("ID", [qubits[0]], [], global_phase=global_phase)
            elif gate_name == "reset":
                g = build_gate("RESET", [qubits[0]], [], global_phase=global_phase)
            elif gate_name == "cx":
                g = build_CX_gate(qubits[0], qubits[1])
            elif gate_name == "cz":
                g = build_CZ_gate(qubits[0], qubits[1])
            elif gate_name == "rzz":
                g = build_gate("RZZ", [qubits[0], qubits[1]], [params[0]], global_phase=global_phase)
            elif gate_name == "crz":
                g = build_CRZ_gate(qubits[0], qubits[1], angle=params[0])
            elif gate_name == "cu1":
                g = build_CU1_gate(qubits[0], qubits[1], angle=params[0])
            elif gate_name == "h":
                g = build_H_gate(qubits[0])
            elif gate_name in ("barrier", "measure"):
                continue
            else:
                raise ValueError(f"Unsupported gate type for AutoComm: {gate_name}")

            gate_list.append(g)
        return gate_list

    def _generate_qubit_node_mapping(self, num_qubits: int, qpus: list[int]) -> list[int]:
        qubit_node_mapping: list[int] = []
        node_index = 0
        remaining_capacity = qpus[node_index]

        for _ in range(num_qubits):
            if remaining_capacity == 0:
                node_index += 1
                remaining_capacity = qpus[node_index]
            qubit_node_mapping.append(node_index)
            remaining_capacity -= 1
        return qubit_node_mapping

    def _partition_from_mapping(self, num_qubits: int, qubit_node_mapping: list[int], n_nodes: int) -> list[list[int]]:
        partition = [[] for _ in range(n_nodes)]
        for q in range(num_qubits):
            partition[qubit_node_mapping[q]].append(q)
        return partition

    def _count_local_gate_like_ops(self, gate_list: list[list[Any]], qubit_node_mapping: list[int]) -> int:
        local_cnt = 0
        for g in gate_list:
            qids = g[1]
            if len(qids) <= 1:
                local_cnt += 1
            elif len(qids) == 2 and qubit_node_mapping[qids[0]] == qubit_node_mapping[qids[1]]:
                local_cnt += 1
        return local_cnt
