import warnings
from typing import Any, Optional, Union, List, Dict, Tuple
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit

from backend import Backend
from network import Network


class NoiseEvaluator:
    """
    噪声评估器：根据量子比特划分、线路、后端噪声和网络信息，评估在给定硬件上执行线路的总体保真度。
    """

    def __init__(self, network: Network, backends: List[Backend]):
        """
        初始化评估器。

        Args:
            network: Network 实例，描述后端之间的连接关系。
            backends: 每个组对应的 Backend 实例列表，长度应与 partition 的组数一致。
        """
        self.network = network
        self.backends = backends
        self._validate_backends()

    def _validate_backends(self):
        """检查后端数量与网络后端数一致。"""
        if len(self.backends) != self.network.num_backends:
            raise ValueError(
                f"Number of backends ({len(self.backends)}) does not match network size ({self.network.num_backends})."
            )

    def evaluate(
        self,
        partition: List[List[int]],
        circuit: QuantumCircuit,
        initial_layouts: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        评估在给定划分和后端分配下执行线路的总体保真度。

        Args:
            partition: 量子比特划分，例如 [[0,1,2], [3,4]]，每个子列表为一组逻辑比特。
            circuit: 待执行的量子线路，逻辑比特索引为 0..N-1。
            initial_layouts: 每个组的初始映射，可选。长度应与 partition 相同，
                每个元素为列表，长度等于该组逻辑比特数，元素为物理比特索引。

        Returns:
            包含评估结果的字典，主要字段：
            - total_fidelity: 总体保真度
            - group_results: 每组详细结果列表
            - cross_group_gates: 跨组门列表
            - total_swaps: 所有组内显式 SWAP 门总数
            - total_cnots: 所有组内 CNOT 门总数
        """
        # 输入校验
        self._validate_partition(partition, circuit)

        # 构建逻辑比特到组 ID 的映射
        logical_to_group = {}
        for gid, group in enumerate(partition):
            for q in group:
                if q in logical_to_group:
                    raise ValueError(f"Logical qubit {q} appears in multiple groups.")
                logical_to_group[q] = gid

        # 检查每个逻辑比特是否都在划分中
        for q in range(circuit.num_qubits):
            if q not in logical_to_group:
                raise ValueError(f"Logical qubit {q} not assigned to any group.")

        # 为每个组构建本地索引映射
        group_local_map = {}  # group_id -> {global_qubit: local_index}
        for gid, group in enumerate(partition):
            group_local_map[gid] = {q: idx for idx, q in enumerate(group)}

        # 初始化各组子线路和跨组门列表
        group_circuits = {gid: QuantumCircuit(len(group)) for gid, group in enumerate(partition)}
        cross_gates = []  # 每个元素为 (gate_name, global_qubits, group_ids)

        # 遍历原线路的所有指令
        for instruction in circuit.data:
            op = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits

            # 忽略屏障、测量等非门操作（可根据需要扩展）
            if op.name in {"barrier", "measure", "reset"}:
                continue

            # 获取操作比特的全局索引
            global_qubits = [circuit.find_bit(q).index for q in qargs]

            # 获取这些比特所属的组
            groups = {logical_to_group[q] for q in global_qubits}

            if len(groups) == 1:
                # 组内门
                gid = next(iter(groups))
                local_qubits = [group_local_map[gid][q] for q in global_qubits]
                # 将指令添加到对应组的子线路中（保持顺序）
                group_circuits[gid].append(op, local_qubits, cargs)
            else:
                # 跨组门（假设最多两个组，否则需分解）
                if len(groups) > 2:
                    raise NotImplementedError(
                        f"Gate {op.name} acts on qubits from more than 2 groups. "
                        "Please decompose the circuit into 2-qubit gates first."
                    )
                cross_gates.append((op.name, global_qubits, sorted(groups)))

        # 处理每个组
        group_results = []
        total_swaps = 0
        total_cnots = 0

        for gid, group in enumerate(partition):
            backend = self.backends[gid]
            sub_circuit = group_circuits[gid]

            # 如果子线路为空，保真度为1
            if sub_circuit.size() == 0:
                group_results.append({
                    "group_id": gid,
                    "logical_qubits": group,
                    "physical_backend_name": backend.name,
                    "fidelity": 1.0,
                    "transpiled_circuit": None,
                    "gate_errors": [],
                    "num_swaps": 0,
                    "num_cnots": 0,
                    "depth": 0,
                })
                continue

            # 获取初始映射
            if initial_layouts is not None:
                init_layout = initial_layouts[gid]
                if len(init_layout) != len(group):
                    raise ValueError(
                        f"Initial layout for group {gid} has length {len(init_layout)}, "
                        f"but group has {len(group)} qubits."
                    )
                # 检查物理比特是否有效
                for p in init_layout:
                    if p < 0 or p >= backend.num_qubits:
                        raise ValueError(
                            f"Physical qubit {p} out of range for backend {backend.name} "
                            f"(has {backend.num_qubits} qubits)."
                        )
            else:
                # 默认映射：顺序取前 N 个物理比特
                init_layout = list(range(min(len(group), backend.num_qubits)))

            # 从后端提取编译所需信息
            coupling_list, basis_gates, error_map = self._extract_backend_info(backend)

            # 编译子线路
            transpiled = transpile(
                sub_circuit,
                coupling_map=coupling_list,
                basis_gates=basis_gates,
                initial_layout=init_layout,
                optimization_level=0,          # 最小优化，尽可能保留初始布局
                routing_method="sabre",        # 默认路由算法
            )

            # 计算该组保真度及门明细
            fidelity = 1.0
            gate_errors = []
            swap_count = 0
            cnot_count = 0

            for inst in transpiled.data:
                op_name = inst.operation.name
                phys_qubits = [transpiled.find_bit(q).index for q in inst.qubits]
                key = (op_name, tuple(phys_qubits))

                # 统计门类型
                if op_name == "swap":
                    swap_count += 1
                elif op_name == "cx":
                    cnot_count += 1

                # 获取错误率
                error = error_map.get(key)
                if error is None:
                    warnings.warn(
                        f"No error rate found for gate {op_name} on qubits {phys_qubits} "
                        f"in backend {backend.name}. Assuming error=0."
                    )
                    error = 0.0
                fidelity *= (1 - error)
                gate_errors.append({
                    "gate": op_name,
                    "physical_qubits": phys_qubits,
                    "error": error,
                })

            group_results.append({
                "group_id": gid,
                "logical_qubits": group,
                "physical_backend_name": backend.name,
                "fidelity": fidelity,
                "transpiled_circuit": transpiled,
                "gate_errors": gate_errors,
                "num_swaps": swap_count,
                "num_cnots": cnot_count,
                "depth": transpiled.depth(),
            })

            total_swaps += swap_count
            total_cnots += cnot_count

        # 处理跨组门
        cross_results = []
        for gate_name, global_qubits, groups in cross_gates:
            # 获取后端索引（与组ID相同）
            backend_ids = groups
            # 计算保真度：使用 network 中两后端间的有效保真度
            if len(backend_ids) == 2:
                i, j = backend_ids
                fid = self.network.get_effective_fidelity(i, j)
            else:
                # 理论上不会进入这里
                fid = 1.0
            cross_results.append({
                "gate": gate_name,
                "logical_qubits": global_qubits,
                "groups": groups,
                "physical_backends": backend_ids,
                "fidelity": fid,
            })

        # 总体保真度 = 组内保真度乘积 * 跨组门保真度乘积
        group_fidelities = [r["fidelity"] for r in group_results]
        cross_fidelities = [g["fidelity"] for g in cross_results]
        total_fidelity = np.prod(group_fidelities) * np.prod(cross_fidelities)

        return {
            "total_fidelity": total_fidelity,
            "group_results": group_results,
            "cross_group_gates": cross_results,
            "total_swaps": total_swaps,
            "total_cnots": total_cnots,
        }

    def _extract_backend_info(self, backend: Backend) -> Tuple[List[List[int]], List[str], Dict]:
        """
        从 Backend 实例中提取编译所需信息。

        Returns:
            coupling_list: 两比特门支持的物理比特对列表，例如 [[0,1], [1,0], [1,2], ...]
            basis_gates: 基门集列表
            error_map: 字典，键为 (gate_name, tuple(physical_qubits))，值为错误率
        """
        coupling_set = set()
        basis_set = set()
        error_map = {}

        for gate in backend.gate_info:
            gate_name = gate["gate"]
            qubits_str = gate["qubits"]
            # 解析量子比特列表
            try:
                qubits = [int(q) for q in qubits_str.split(",")]
            except:
                # 如果格式异常，跳过
                continue

            # 收集基门
            basis_set.add(gate_name)

            # 收集两比特门的耦合（假设两比特门为 'cx'，可根据需要扩展）
            if len(qubits) == 2 and gate_name in {"cx", "cz", "ecr"}:
                # 添加双向边以符合 Qiskit 耦合图要求
                coupling_set.add(tuple(qubits))
                coupling_set.add(tuple(reversed(qubits)))

            # 提取错误率
            error_val = gate.get("gate_error_value")
            if error_val is not None:
                try:
                    error = float(error_val)
                except:
                    error = 0.0
                key = (gate_name, tuple(qubits))
                error_map[key] = error

        coupling_list = [list(pair) for pair in coupling_set]
        basis_gates = list(basis_set)

        return coupling_list, basis_gates, error_map

    def _validate_partition(self, partition: List[List[int]], circuit: QuantumCircuit):
        """检查划分的基本合法性。"""
        if not partition:
            raise ValueError("Partition cannot be empty.")
        all_qubits = set()
        for group in partition:
            if not group:
                raise ValueError("Each group must contain at least one qubit.")
            for q in group:
                if q in all_qubits:
                    raise ValueError(f"Duplicate logical qubit {q} in partition.")
                all_qubits.add(q)
        # 检查线路量子比特数
        if max(all_qubits, default=-1) >= circuit.num_qubits:
            raise ValueError("Partition contains qubit index exceeding circuit's qubit count.")