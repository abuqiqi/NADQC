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
        # print(f"Compiling with [{self.name}]...")
        cfg = config or {}
        circuit_name = cfg.get("circuit_name", "circ")

        # print(f"[DEBUG] [AutoComm] circuit:\n{circuit}")

        start_time = time.time()

        gate_list = self._to_autocomm_gate_list(circuit)
        init_partition = cfg.get("partition", None)
        if init_partition:
            partition = [list(part) for part in init_partition]
            qubit_node_mapping = self._mapping_from_partition(circuit.num_qubits, partition)
        else:
            qubit_node_mapping = self._generate_qubit_node_mapping(circuit.num_qubits, network.backend_sizes)
            partition = self._partition_from_mapping(circuit.num_qubits, qubit_node_mapping, network.num_backends)
        logical_phy_map = CompilerUtils.init_logical_phy_map(partition)
        
        num_q = len(qubit_node_mapping)
        qb_per_node = max(network.backend_sizes) if network.backend_sizes else 1
        aggregate_iter_cnt = max(1, num_q // max(1, qb_per_node))
        schedule_iter_cnt = max(1, num_q // max(1, qb_per_node))

        # print(f"[DEBUG] gate_list:\n")
        # for e in gate_list:
        #     print(e)
        # print(f"[DEBUG] qubit_node_mapping: {qubit_node_mapping}")
        # print(f"[DEBUG] aggregate_iter_cnt: {aggregate_iter_cnt}")
        # print(f"[DEBUG] schedule_iter_cnt: {schedule_iter_cnt}")

        mapping_record_list = autocomm_full(
            gate_list,
            qubit_node_mapping,
            aggregate_iter_cnt=aggregate_iter_cnt,
            schedule_iter_cnt=schedule_iter_cnt,
            return_ops=True,
        )
        if len(mapping_record_list) == 5:
            epr_cnt, all_latency, assigned_gate_blocks, comm_costs, op_list = mapping_record_list
        else:
            epr_cnt, all_latency, assigned_gate_blocks, comm_costs = mapping_record_list
            op_list = QuantumCircuit(circuit.num_qubits)

        # print(f"Transpiled circuit:\n")
        # print(circuit)
        # print(f"\n[DEBUG] op_list:\n")
        # print(op_list)
        # for op in op_list:
        #     print(op)

        record = MappingRecord(
            layer_start=0,
            layer_end=max(0, circuit.depth() - 1),
            partition=partition,
            mapping_type="cat",
            logical_phy_map=logical_phy_map,
            extra_info={"ops": op_list}
        )

        # print(f"[DEBUG] [AutoComm] op_list:\n{record.extra_info['ops']}")

        comm_only_costs = bool(cfg.get("comm_only_costs", False))
        if comm_only_costs:
            costs = CompilerUtils.evaluate_telegate_with_cat(
                record,
                op_list,
                network,
            )
        else:
            costs, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(
                record,
                op_list,
                network,
            )

        # AutoComm latency is a scheduler makespan proxy; keep total execution wall time in execution_time,
        # and expose scheduler latency in extra_info for analysis.
        extra_info = dict(record.extra_info or {})
        extra_info.update({
            "autocomm_latency": float(all_latency),
            "autocomm_assigned_block_count": len(assigned_gate_blocks),
            "comm_only_costs": comm_only_costs,
        })
        record.extra_info = extra_info

        mapping_record_list = MappingRecordList()
        mapping_record_list.add_record(record)
        mapping_record_list.summarize_total_costs()

        exec_time = time.time() - start_time
        mapping_record_list.update_total_costs(execution_time=exec_time)

        if bool(cfg.get("save_records", True)):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mapping_record_list.save_records(f"./outputs/{circuit_name}/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")
        return mapping_record_list

    def _to_autocomm_gate_list(self, circuit: QuantumCircuit) -> list[list[Any]]:
        gate_list: list[list[Any]] = []
        global_phase = circuit.global_phase

        for instruction in circuit:
            gate = instruction.operation
            gate_name = gate.name
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
        qubit_node_mapping: list[int] = []  # 最终返回的映射表：索引=量子比特编号，值=QPU编号
        node_index = 0                       # 当前正在使用的QPU索引（从0开始）
        remaining_capacity = qpus[node_index]  # 当前QPU剩余的“空位”

        for _ in range(num_qubits):          # 给每个量子比特分配QPU
            if remaining_capacity == 0:      # 如果当前QPU满了
                node_index += 1               # 切换到下一个QPU
                remaining_capacity = qpus[node_index]  # 更新剩余容量为新QPU的容量
            qubit_node_mapping.append(node_index)  # 记录当前量子比特属于哪个QPU
            remaining_capacity -= 1          # 当前QPU的剩余空位减1
        return qubit_node_mapping

    def _partition_from_mapping(self, num_qubits: int, qubit_node_mapping: list[int], n_nodes: int) -> list[list[int]]:
        partition = [[] for _ in range(n_nodes)]
        for q in range(num_qubits):
            partition[qubit_node_mapping[q]].append(q)
        return partition

    def _mapping_from_partition(self, num_qubits: int, partition: list[list[int]]) -> list[int]:
        qubit_node_mapping = [-1 for _ in range(num_qubits)]
        for node_idx, part in enumerate(partition):
            for q in part:
                if q < 0 or q >= num_qubits:
                    raise ValueError(f"Invalid qubit index in partition: {q}")
                if qubit_node_mapping[q] != -1:
                    raise ValueError(f"Qubit {q} appears in multiple partition blocks")
                qubit_node_mapping[q] = node_idx
        if any(node == -1 for node in qubit_node_mapping):
            raise ValueError("Partition does not cover all circuit qubits")
        return qubit_node_mapping

    # def _count_local_gate_like_ops(self, gate_list: list[list[Any]], qubit_node_mapping: list[int]) -> int:
    #     local_cnt = 0
    #     for g in gate_list:
    #         qids = g[1]
    #         if len(qids) <= 1:
    #             local_cnt += 1
    #         elif len(qids) == 2 and qubit_node_mapping[qids[0]] == qubit_node_mapping[qids[1]]:
    #             local_cnt += 1
    #     return local_cnt

    # def _estimate_local_gate_costs(
    #     self,
    #     gate_list: list[list[Any]],
    #     qubit_node_mapping: list[int],
    #     network: Network,
    # ) -> dict[str, int | float]:
    #     """
    #     统计AutoComm输入门表中可在本地执行的门的保真度损失。
    #     """
    #     costs = ExecCosts()
    #     for g in gate_list:
    #         if not isinstance(g, list) or len(g) < 2:
    #             continue
    #         if not isinstance(g[0], str) or not isinstance(g[1], list):
    #             continue

    #         gate_name = g[0].lower()
    #         qids = g[1]

    #         if len(qids) == 1:
    #             q = qids[0]
    #             if q < 0 or q >= len(qubit_node_mapping):
    #                 continue
    #             qpu = qubit_node_mapping[q]
    #             backend = network.backends[qpu]
    #             costs = CompilerUtils.update_local_gate_costs_by_name(costs, backend, gate_name, 1)
    #         elif len(qids) == 2:
    #             q0, q1 = qids[0], qids[1]
    #             if q0 < 0 or q1 < 0 or q0 >= len(qubit_node_mapping) or q1 >= len(qubit_node_mapping):
    #                 continue
    #             p0, p1 = qubit_node_mapping[q0], qubit_node_mapping[q1]
    #             if p0 != p1:
    #                 continue
    #             backend = network.backends[p0]
    #             costs = CompilerUtils.update_local_gate_costs_by_name(costs, backend, gate_name, 1)

    #     return {
    #         "local_gate_num": costs.local_gate_num,
    #         "local_fidelity_loss": costs.local_fidelity_loss,
    #         "local_fidelity_log_sum": costs.local_fidelity_log_sum,
    #         "local_fidelity": costs.local_fidelity,
    #     }

    # def _estimate_cat_localized_gate_costs(
    #     self,
    #     assigned_gate_blocks: list[Any],
    #     qubit_node_mapping: list[int],
    #     network: Network,
    # ) -> dict[str, int | float]:
    #     """
    #     统计CAT通信块中由远程telegate转成本地执行的双量子门损失。
    #     """
    #     local_gate_num = 0
    #     local_fidelity_loss = 0.0
    #     local_fidelity_log_sum = 0.0
    #     local_fidelity = 1.0

    #     for block in assigned_gate_blocks:
    #         if not isinstance(block, list) or len(block) < 2:
    #             continue
    #         tag = block[0]
    #         local_body = block[1]

    #         # CAT通信块标记形如: [[[source_qubit, target_node], 0], gate_block]
    #         if (
    #             not isinstance(tag, list)
    #             or len(tag) < 2
    #             or tag[1] != 0
    #             or not isinstance(tag[0], list)
    #             or len(tag[0]) < 2
    #         ):
    #             continue

    #         source_q = int(tag[0][0])
    #         target_node = int(tag[0][1])
    #         if target_node < 0 or target_node >= network.num_backends:
    #             continue

    #         backend = network.backends[target_node]
    #         twoq_name = self._pick_two_qubit_gate_name(backend)
    #         if not isinstance(local_body, list):
    #             continue

    #         for g in local_body:
    #             if not isinstance(g, list) or len(g) < 2:
    #                 continue
    #             if not isinstance(g[0], str) or not isinstance(g[1], list):
    #                 continue

    #             gate_name = g[0].lower()
    #             qids = g[1]
    #             # 仅统计“原跨分区门被CAT转本地”的门：双量子门且涉及source_q。
    #             if len(qids) != 2 or source_q not in qids:
    #                 continue

    #             src, dst = qids[0], qids[1]
    #             if src >= len(qubit_node_mapping) or dst >= len(qubit_node_mapping):
    #                 continue
    #             if qubit_node_mapping[src] == qubit_node_mapping[dst]:
    #                 # 原本本地门不应重复计入CAT转本地门统计。
    #                 continue

    #             gate_error = backend.gate_dict.get(gate_name, {}).get("gate_error_value", None)
    #             if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
    #                 gate_error = backend.gate_dict.get(twoq_name, {}).get("gate_error_value", 0.0)
    #             gate_error = float(gate_error)
    #             gate_error = min(max(gate_error, 0.0), 0.999999999)

    #             local_gate_num += 1
    #             local_fidelity_loss += gate_error
    #             local_fidelity *= (1 - gate_error)
    #             local_fidelity_log_sum += math.log(1 - gate_error)

    #     return {
    #         "local_gate_num": local_gate_num,
    #         "local_fidelity_loss": local_fidelity_loss,
    #         "local_fidelity_log_sum": local_fidelity_log_sum,
    #         "local_fidelity": local_fidelity,
    #     }

    # def _pick_two_qubit_gate_name(self, backend: Any) -> str:
    #     """
    #     选择可用于2Q误差估计的门名，优先使用后端真实basis中的2Q门。
    #     """
    #     if hasattr(backend, "two_qubit_gates") and backend.two_qubit_gates:
    #         for name in ("cz", "rzz", "ecr", "cx"):
    #             if name in backend.two_qubit_gates and name in backend.gate_dict:
    #                 return name
    #         for name in sorted(backend.two_qubit_gates):
    #             if name in backend.gate_dict:
    #                 return name

    #     for name in ("cz", "rzz", "ecr", "cx"):
    #         if name in backend.gate_dict:
    #             return name

    #     for name, info in backend.gate_dict.items():
    #         if not isinstance(name, str):
    #             continue
    #         if not isinstance(info, dict):
    #             continue
    #         if "gate_error_value" not in info:
    #             continue
    #         qubits = info.get("qubits")
    #         if isinstance(qubits, list) and len(qubits) == 2:
    #             return name

    #     raise ValueError(f"No available two-qubit gate for backend '{getattr(backend, 'name', 'unknown')}'.")
