import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import numpy as np
import copy
from collections import defaultdict
from typing import Any, Optional
import math
import os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from ..utils import Network


@dataclass
class ExecCosts:
    remote_hops: int = 0
    remote_swaps: int = 0
    cat_ents: int = 0
    epairs: int = 0
    remote_fidelity_loss: float = 0.0
    remote_fidelity_log_sum: float = 0.0
    remote_fidelity: float = 1.0
    local_fidelity_loss: float = 0.0
    local_fidelity_log_sum: float = 0.0
    local_fidelity: float = 1.0
    execution_time: float = 0.0

    local_gate_num: int = 0    # 本地量子门总个数

    @property
    def num_comms(self) -> int:
        return self.remote_hops + self.remote_swaps

    @property
    def total_fidelity_loss(self) -> float:
        return self.remote_fidelity_loss + self.local_fidelity_loss

    @property
    def total_fidelity_log_sum(self) -> float:
        return self.local_fidelity_log_sum + self.remote_fidelity_log_sum

    @property
    def total_fidelity(self) -> float:
        return self.remote_fidelity * self.local_fidelity

    @property
    def remote_geometric_mean_fidelity(self) -> float:
        if self.num_comms == 0:
            return 1.0
        rgeo_mean_fid = np.exp(self.remote_fidelity_log_sum / self.num_comms)
        return rgeo_mean_fid

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"ExecCosts 没有属性 {key}")

    def __str__(self) -> str:
        return (
            f"ExecCosts("
            f"comms={self.num_comms}, "
            f"cat_ents={self.cat_ents}, "
            f"epairs={self.epairs}, "
            f"rgeo_mean_fid={self.remote_geometric_mean_fidelity}, "
            f"fidelity={self.total_fidelity:.4f}, "
            f"loss={self.total_fidelity_loss:.4f}, "
            f"rhops={self.remote_hops}, "
            f"rswaps={self.remote_swaps}, "
            f"remote_fidelity_loss={self.remote_fidelity_loss}, "
            f"remote_fidelity_log_sum={self.remote_fidelity_log_sum}, "
            f"remote_fidelity={self.remote_fidelity}, "
            f"local_fidelity_loss={self.local_fidelity_loss}, "
            f"local_fidelity_log_sum={self.local_fidelity_log_sum}, "
            f"local_fidelity={self.local_fidelity}, "
            f"time={self.execution_time:.2f}), "
            f"local_gate_num={self.local_gate_num}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other: "ExecCosts") -> "ExecCosts":
        if not isinstance(other, ExecCosts):
            raise TypeError(f"不能与 {type(other)} 累加")

        self.remote_hops += other.remote_hops
        self.remote_swaps += other.remote_swaps
        self.cat_ents += other.cat_ents
        self.epairs += other.epairs
        self.remote_fidelity_loss += other.remote_fidelity_loss
        self.remote_fidelity_log_sum += other.remote_fidelity_log_sum
        self.local_fidelity_loss += other.local_fidelity_loss
        self.local_fidelity_log_sum += other.local_fidelity_log_sum
        self.execution_time += other.execution_time
        self.remote_fidelity *= other.remote_fidelity
        self.local_fidelity *= other.local_fidelity
        self.local_gate_num += other.local_gate_num
        return self

    def to_dict(self) -> dict:
        base_dict = dataclasses.asdict(self)
        base_dict.update({
            "num_comms": self.num_comms,
            "total_fidelity_log_sum": self.total_fidelity_log_sum,
            "total_fidelity_loss": self.total_fidelity_loss,
            "rgeo_mean_fid": self.remote_geometric_mean_fidelity,
            "total_fidelity": self.total_fidelity,
        })
        sorted_keys = [
            "total_fidelity_loss",
            "epairs",
            "execution_time",
            "num_comms",
            "rgeo_mean_fid",
            "total_fidelity_log_sum",
            "total_fidelity",
            "remote_hops",
            "remote_swaps",
            "cat_ents",
            "local_fidelity_loss",
            "remote_fidelity_loss",
            "local_fidelity_log_sum",
            "remote_fidelity_log_sum",
            "local_fidelity",
            "remote_fidelity",
            "local_gate_num",
        ]

        # 生成有序字典
        return {key: base_dict[key] for key in sorted_keys if key in base_dict}


class CommOp(Gate):
    """
    通信操作封装为Qiskit自定义Gate，便于直接插入QuantumCircuit。
    gate_list中保存通信块的Qiskit门（每个门额外携带_autocomm_qids元数据）。
    """

    def __init__(
        self,
        comm_type: str,
        source_qubit: int,
        src_qpu: int,
        dst_qpu: int,
        involved_qubits: list[int],
        gate_list: Optional[list[Gate]] = None,
    ):
        if comm_type not in {"cat", "rtp", "tp"}:
            raise ValueError(f"Unsupported comm_type: {comm_type}")
        if len(involved_qubits) == 0:
            raise ValueError("involved_qubits cannot be empty")

        self.comm_type = comm_type
        self.source_qubit = int(source_qubit)
        self.src_qpu = int(src_qpu)
        self.dst_qpu = int(dst_qpu)
        self.involved_qubits = [int(q) for q in involved_qubits]

        normalized: list[Gate] = []
        for g in gate_list or []:
            if not isinstance(g, Gate):
                raise TypeError(f"gate_list items must be qiskit Gate, got {type(g)}")
            normalized.append(g)
        self.gate_list = normalized

        super().__init__(
            name=f"comm_{comm_type}",
            num_qubits=len(self.involved_qubits),
            params=[],
        )

    def __repr__(self) -> str:
        return (
            f"CommOp(type={self.comm_type}, source={self.source_qubit}, "
            f"src_qpu={self.src_qpu}, dst_qpu={self.dst_qpu}, "
            f"involved_qubits={self.involved_qubits}, gates={len(self.gate_list)})"
        )


@dataclass
class MappingRecord:
    """
    映射记录类：记录线路层级范围、映射类型、开销及时间
    """
    # 必选字段：线路层级范围
    layer_start: int = -1          # 起始层级（第几层）
    layer_end: int = -1            # 结束层级（第几层）
    # 必选字段：量子比特划分
    partition: list[list[int]] = dataclasses.field(default_factory=list) # 划分结果，格式为 list of lists，每个子列表代表一个分区的量子比特索引
    # 必选字段：映射信息
    mapping_type: str = ""         # 映射类型（如 "teledata"、"telegate"）
    costs: ExecCosts = ExecCosts() # 执行成本，包含保真度损失、通信开销、执行时间等指标
    logical_phy_map: dict[int, tuple[int, int | None]] = dataclasses.field(default_factory=dict) # 量子比特映射信息，记录每个全局逻辑比特在物理上的位置（QPU编号和物理比特编号）
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 冻结模式下修改字段需用 object.__setattr__
        object.__setattr__(self, "partition", copy.deepcopy(self.partition))
        object.__setattr__(self, "costs", copy.deepcopy(self.costs))
        object.__setattr__(self, "logical_phy_map", copy.deepcopy(self.logical_phy_map))
        if self.extra_info is not None:
            object.__setattr__(self, "extra_info", copy.deepcopy(self.extra_info))

    def to_dict(self) -> dict:
        """将 MappingRecord 转为字典，包含嵌套的 ExecCosts 字典"""
        return {
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "partition": self.partition,
            "mapping_type": self.mapping_type,
            "costs": self.costs.to_dict(),  # 直接用自定义 to_dict
            "logical_phy_map": self.logical_phy_map,
            "extra_info": self.extra_info
        }

    def update(self, **kwargs):
        """批量更新字段"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"MappingRecord 没有属性 {key}")


# 辅助类：管理多条记录
@dataclass
class MappingRecordList:
    """
    映射记录管理器：批量存储、查询记录
    """
    total_costs: ExecCosts = dataclasses.field(default_factory=ExecCosts)
    num_records: int = 0
    records: list[MappingRecord] = dataclasses.field(default_factory=list)

    def add_record(self, record: MappingRecord):
        """添加一条记录"""
        self.records.append(record)

    def summarize_total_costs(self):
        """汇总所有记录的成本信息"""
        total_costs = ExecCosts()
        for record in self.records:
            total_costs += record.costs
        self.total_costs = total_costs
        self.num_records = len(self.records)
        return

    def update_total_costs(self, **kwargs):
        """批量更新total_costs指标"""
        self.total_costs.update(**kwargs)
        return

    # def add_cost(self, key: str, value: Any):
    #     """添加项目到total_costs"""
    #     self.total_costs[key] = value

    # def add_cost_sum(self, key: str):
    #     """添加求和项到total_costs"""
    #     sum = 0
    #     for record in self.records:
    #         sum += record.costs[key]
    #     self.total_costs[key] = sum

    # def add_cost_mul(self, key: str):
    #     mul = 1
    #     for record in self.records:
    #         mul *= record.costs[key]
    #     self.total_costs[key] = mul

    def get_records_by_layer_range(self, layer_start: int, layer_end: int) -> list[MappingRecord]:
        """按层级范围查询记录（包含交集）"""
        return [
            r for r in self.records
            if not (r.layer_end < layer_start or r.layer_start > layer_end)
        ]

    def save_records(self, filename: str):
        """
        将记录保存到文件，支持 JSON/CSV 格式
        Args:
            filename: 保存路径
        """
        if not self.records:
            print("⚠️ 无映射记录可保存")
            return

        # 统一序列化：将 dataclass 转为字典（兼容可选字段 extra_info）
        # 将total_costs转为字典
        total_costs_dict = self.total_costs.to_dict()
        # 将每条记录转为字典
        records_dict = [record.to_dict() for record in self.records]
        data_dict = {
            "total_costs": total_costs_dict,
            "num_records": self.num_records,
            "records": records_dict
        }
        # data_dict = dataclasses.asdict(self)
        data_dict = self._convert_numpy_types(data_dict)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 按格式保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                data_dict,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=False  # 保持字段顺序，更易读
            )
        print(f"✅ 成功保存 {len(data_dict['records'])} 条映射记录到 JSON 文件：{filename}")
        return

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """
        递归转换所有NumPy类型为原生Python类型
        支持：字典、列表、元组、np.int64/np.float64等
        """
        # 处理NumPy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # 处理列表/元组：递归转换每个元素
        elif isinstance(obj, (list, tuple)):
            return [MappingRecordList._convert_numpy_types(item) for item in obj]
        # 处理字典：递归转换每个键值对
        elif isinstance(obj, dict):
            return {k: MappingRecordList._convert_numpy_types(v) for k, v in obj.items()}
        # 其他类型直接返回（如str、int、float、None等）
        else:
            return obj


class CompilerUtils:
    """
    编译工具类
    """
    @staticmethod
    def allocate_qubits(num_qubits: int, network: Network) -> list[list[int]]:
        """
        Initialize the partition
        """
        partition = []
        cnt_qubits = 0
        for qpu_size in network.backend_sizes:
            remain = num_qubits - cnt_qubits
            if remain == 0:
                break
            end_index = min(cnt_qubits + qpu_size, num_qubits)
            part = list(range(cnt_qubits, end_index))
            partition.append(part)
            cnt_qubits = end_index
        assert(cnt_qubits == num_qubits)
        for _ in range(len(partition), network.num_backends):
            partition.append([])
        return partition

    @staticmethod
    def build_qubit_interaction_graph(circuit: QuantumCircuit) -> nx.Graph:
        """
        Construct the qubit interaction graph from the circuit
        """
        qig = nx.Graph()
        for qubit in range(circuit.num_qubits):
            qig.add_node(qubit)
        for instruction in circuit:
            # gate = instruction.operation
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) > 1:
                if instruction.name == "barrier":
                    continue
                assert len(qubits) == 2, f"instruction: {instruction}"
                if qig.has_edge(qubits[0], qubits[1]):
                    qig[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    qig.add_edge(qubits[0], qubits[1], weight=1)
        return qig

    @staticmethod
    def get_subcircuit_by_level(num_qubits: int, 
                                circuit: QuantumCircuit, 
                                circuit_layers: list[list[Any]], 
                                layer_start: int, 
                                layer_end: int) -> QuantumCircuit:
        """
        从DAGOpNode分层中提取子线路
        """
        subcircuit = QuantumCircuit(num_qubits)

        # 遍历指定层级的DAGOpNode
        for layer in circuit_layers[layer_start:layer_end + 1]:
            # print(f"[DEBUG] Processing layer with {len(layer)} nodes, layer_start: {layer_start}, layer_end: {layer_end}")
            for node in layer: # node 是 DAGOpNode 对象
                if node.op.name == "barrier":
                    continue
                gate_instruction = node.op  # 获取门指令（Instruction对象）
                # qubit_indices = [q._index for q in node.qargs]  # 提取量子比特索引
                # if qubit_indices[0] == None:
                qubit_indices = [circuit.qubits.index(q) for q in node.qargs]
                # print(f"[DEBUG] [{qubit_indices}] gate: {gate_instruction}")
                assert qubit_indices[0] is not None, f"无法找到量子比特索引，node.qargs: {node.qargs}"
                # 将门添加到子线路
                subcircuit.append(gate_instruction, qubit_indices)
        
        return subcircuit

    @staticmethod
    def evaluate_remote_hops(qig: nx.Graph, 
                           partition: list[list[int]], 
                           network: Any) -> int:
        """
        计算qubit interaction graph在partitions下的割
        """
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, part in enumerate(partition):
            for node in part:
                node_to_partition[node] = i
        remote_hops = 0
        for u, v in qig.edges(): # 遍历图中的每一条边，也就是双量子门
            qpu_u = node_to_partition[u]
            qpu_v = node_to_partition[v]
            if qpu_u != qpu_v:
                remote_hops += network.Hops[qpu_u][qpu_v] * qig[u][v]['weight']
                # fidelity_loss += (1 - network.move_fidelity[qpu_u][qpu_v]) * qig[u][v]['weight']
                # fidelity *= network.move_fidelity[qpu_u][qpu_v] ** qig[u][v]['weight']
        return remote_hops

    @staticmethod
    def evaluate_local_and_telegate(
        arg: MappingRecord | list[list[int]],  # 兼容两种类型：record / partition
        circuit: QuantumCircuit, 
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {},
        optimization_level: int = 3,
        enable_cat_telegate: bool = False,
        cat_controls: Optional[list[int]] = None,
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]]]:
        """
        评估在给定划分和后端分配下执行线路的总体保真度
        @param record: 包含划分和映射信息的记录
        @param circuit: 量子线路
        @param network: 网络信息
        @param logical_phy_map: 记录每个量子比特在特定QPU上的物理比特编号
        @param optimization_level: transpile优化级别，默认3
        @return: 更新后的记录，包含评估的成本信息
        """
        # print("[DEBUG] evaluate_local_and_telegate")
        partition = None
        
        if isinstance(arg, MappingRecord):
            record = arg
            partition = record.partition
            logical_phy_map = record.logical_phy_map # 获取专属于当前record的logical_phy_map，不需要copy()
        else:
            # arg is a list of lists representing the partition
            partition = arg
            # 检查logical_phy_map不为空
            if len(logical_phy_map) == 0:
                raise ValueError("当输入为partition时，必须提供非空的logical_phy_map")

        # --- 构建辅助字典 ---
        # 快速查比特归属QPU，直接从唯一字典取
        global_to_qpu: dict[int, int] = {q: logical_phy_map[q][0] for q in logical_phy_map}

        # 检查global_to_qpu和当前的partition是否一致
        global_to_local_lqid: dict[int, int] = {}
        for qpu_id, group in enumerate(partition):
            for local_lqid, global_lqid in enumerate(group):
                global_to_local_lqid[global_lqid] = local_lqid
                assert global_to_qpu[global_lqid] == qpu_id, f"全局逻辑比特 {global_lqid} 的QPU归属不一致，partition: {partition}, logical_phy_map: {logical_phy_map}"

        # --- 建立每个分区对应的子线路 ---
        subcircuits = [QuantumCircuit(len(group)) for group in partition]

        # 初始化噪声
        costs = ExecCosts()

        # 遍历circuit上的每个操作，如果操作完全属于某个group，则添加到对应的subcircuit；
        # 如果操作跨越多个group，则记录为telegate操作（支持可选CAT-aware计费）。
        for instruction in circuit:
            # 操作属于原始的全局量子比特编号，需要转换为子线路的局部编号
            global_qids = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # 检查操作涉及的量子比特属于哪个分区
            involved_qpus = set(global_to_qpu[q] for q in global_qids)

            # 本地操作，加入对应子线路（需要转换为局部编号）
            if len(involved_qpus) == 1:
                qpu_id = involved_qpus.pop()
                mapped_qubits = [global_to_local_lqid[q] for q in global_qids]
                subcircuits[qpu_id].append(instruction.operation, mapped_qubits)
            # 操作跨越多个分区，为telegate操作
            else:
                if not enable_cat_telegate:
                    for i in range(len(global_qids) - 1):
                        q1, q2 = global_qids[i], global_qids[i + 1]
                        p1, p2 = global_to_qpu[q1], global_to_qpu[q2]
                        if p1 != p2:
                            costs = CompilerUtils.update_remote_move_costs(
                                costs, p1, p2, 1, network
                            )

        if enable_cat_telegate:
            tg_costs = CompilerUtils.evaluate_telegate_with_cat(
                partition,
                circuit,
                network,
                cat_controls=cat_controls,
            )
            costs += tg_costs

        # 遍历每个子线路
        assert len(subcircuits) == len(network.backends)
        for qpu_id in range(len(subcircuits)):
            # 获取每个分区单独的保真度损失
            subcircuit = subcircuits[qpu_id]
            backend = network.backends[qpu_id]

            if subcircuit.size() == 0:
                continue

            initial_layout = CompilerUtils.get_initial_layout(
                subcircuit, partition[qpu_id], global_to_local_lqid, logical_phy_map
            )

            # print(f"[DEBUG] QPU[{qpu_id}] logical_phy_map: {logical_phy_map}, initial_layout: {initial_layout}")
            # print("===== logical circuit =====")
            # print(subcircuit)

            # 这里不能自由地transpile子线路
            # 要保证在不同子线路之间，量子比特在本地的位置是一致的，
            # 否则需要引入额外的SWAP操作来调整量子比特位置，导致评估不准确
            transpiled_circuit = transpile(
                subcircuit,
                coupling_map=backend.coupling_map,
                basis_gates=backend.basis_gates,
                initial_layout=initial_layout,
                optimization_level=optimization_level
            )

            # print("===== transpiled circuit =====")
            # print(transpiled_circuit)

            # 更新logical_phy_map：记录每个全局逻辑比特在物理上的位置（QPU编号和物理比特编号）
            logical_phy_map = CompilerUtils.get_logical_to_physical_map(
                transpiled_circuit, partition[qpu_id], global_to_local_lqid, logical_phy_map
            )

            # print(f"[DEBUG] logical_phy_map: {logical_phy_map}\n")

            # 统计每个量子门的保真度损失
            for instruction in transpiled_circuit:
                # 获取操作名字
                gate_name = instruction.operation.name
                qubits = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
                assert qubits[0] is not None, f"Qubit index is None for instruction: {instruction}"
                gate_key = f"{gate_name}{'_'.join(map(str, qubits))}"

                # 获取操作保真度
                gate_error = backend.gate_dict.get(gate_key, {}).get("gate_error_value", None)
                if gate_error == 1:
                    gate_error = 0.99 #
                    print(f"[WARNING] {gate_key}: {gate_error}")
                    # exit(1)
                
                if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
                    gate_error = backend.gate_dict[gate_name]["gate_error_value"]

                assert gate_error is not None, f"Gate error not found for gate_key: {gate_key} in backend.gate_dict"
                # print(f"[DEBUG] {gate_key}{qubits}: {gate_error}")
                costs.local_gate_num += 1
                costs.local_fidelity_loss += gate_error
                costs.local_fidelity *= (1 - gate_error)
                costs.local_fidelity_log_sum += np.log(1 - gate_error)

        if isinstance(arg, MappingRecord):
            # 更新record的costs
            arg.costs += costs
            arg.logical_phy_map = logical_phy_map

        return costs, logical_phy_map

    @staticmethod
    def evaluate_local_and_telegate_with_cat(
        arg: MappingRecord | list[list[int]],
        circuit: QuantumCircuit,
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {},
        optimization_level: int = 3
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]]]:
        """
        按CommOp + 门序列统一评估通信与本地门成本
        """
        partition = None

        if isinstance(arg, MappingRecord):
            partition = arg.partition
            logical_phy_map = arg.logical_phy_map
        else:
            partition = arg

        if len(logical_phy_map) == 0:
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        global_to_qpu: dict[int, int] = {q: logical_phy_map[q][0] for q in logical_phy_map}
        
        # 检查global_to_qpu和当前的partition是否一致
        global_to_local_lqid: dict[int, int] = {}
        for qpu_id, group in enumerate(partition):
            for local_lqid, global_lqid in enumerate(group):
                global_to_local_lqid[global_lqid] = local_lqid
                assert global_to_qpu[global_lqid] == qpu_id, f"全局逻辑比特 {global_lqid} 的QPU归属不一致，partition: {partition}, logical_phy_map: {logical_phy_map}"

        # --- 建立每个分区对应的子线路 ---
        subcircuits = [QuantumCircuit(len(group)) for group in partition]

        # 初始化噪声
        costs = ExecCosts()

        def _apply_comm_gate_errors(dst_qpu: int, gate_list: list[Gate]) -> None:
            """
            对通信块里的门采用门类型级估算：
            不再绑定具体物理位，直接按目标QPU上的门类型误差累计。
            """
            nonlocal costs
            backend = network.backends[dst_qpu]
            for gate_op in gate_list:
                costs = CompilerUtils.update_local_gate_costs_by_name(
                    costs,
                    backend,
                    gate_op.name,
                    1,
                )

        for instruction in circuit:
            op = instruction.operation
            global_qids = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # print(f"\n\n[DEBUG] Processing instruction: {instruction}")
            # print(f"[DEBUG] qids: {global_qids}")

            if isinstance(op, CommOp):
                if op.comm_type == "cat":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, op.src_qpu, op.dst_qpu, 1, network
                    )
                    costs.cat_ents += 1
                    _apply_comm_gate_errors(op.dst_qpu, op.gate_list)
                elif op.comm_type == "rtp":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, op.src_qpu, op.dst_qpu, 2, network
                    )
                    _apply_comm_gate_errors(op.dst_qpu, op.gate_list)
                elif op.comm_type == "tp":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, op.src_qpu, op.dst_qpu, 1, network
                    )
                    logical_phy_map[op.source_qubit] = (op.dst_qpu, None)
                    global_to_qpu[op.source_qubit] = op.dst_qpu
                    _apply_comm_gate_errors(op.dst_qpu, op.gate_list)

            else:
                # print(f"[DEBUG] Processing normal gate: {op}")
                
                # 检查操作涉及的量子比特属于哪个分区
                involved_qpus = set(global_to_qpu[q] for q in global_qids)

                # 本地操作，加入对应子线路（需要转换为局部编号）
                if len(involved_qpus) == 1:
                    qpu_id = involved_qpus.pop()
                    mapped_qubits = [global_to_local_lqid[q] for q in global_qids]
                    subcircuits[qpu_id].append(instruction.operation, mapped_qubits)
                # 操作跨越多个分区，为telegate操作
                else:
                    for i in range(len(global_qids) - 1):
                        q1, q2 = global_qids[i], global_qids[i + 1]
                        p1, p2 = global_to_qpu[q1], global_to_qpu[q2]
                        if p1 != p2:
                            costs = CompilerUtils.update_remote_move_costs(
                                costs, p1, p2, 1, network
                            )

        # 统一对子线路做transpile并统计本地门误差
        for qpu_id in range(len(subcircuits)):
            subcircuit = subcircuits[qpu_id]
            backend = network.backends[qpu_id]

            if subcircuit.size() == 0:
                continue

            initial_layout = CompilerUtils.get_initial_layout(
                subcircuit, partition[qpu_id], global_to_local_lqid, logical_phy_map
            )

            transpiled_circuit = transpile(
                subcircuit,
                coupling_map=backend.coupling_map,
                basis_gates=backend.basis_gates,
                initial_layout=initial_layout,
                optimization_level=optimization_level,
            )

            logical_phy_map = CompilerUtils.get_logical_to_physical_map(
                transpiled_circuit, partition[qpu_id], global_to_local_lqid, logical_phy_map
            )

            for instruction in transpiled_circuit:
                gate_name = instruction.operation.name
                qubits = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
                assert qubits[0] is not None, f"Qubit index is None for instruction: {instruction}"
                gate_key = f"{gate_name}{'_'.join(map(str, qubits))}"

                gate_error = backend.gate_dict.get(gate_key, {}).get("gate_error_value", None)
                if gate_error == 1:
                    gate_error = 0.99
                    print(f"[WARNING] {gate_key}: {gate_error}")

                if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
                    gate_error = backend.gate_dict[gate_name]["gate_error_value"]

                assert gate_error is not None, f"Gate error not found for gate_key: {gate_key} in backend.gate_dict"
                costs.local_gate_num += 1
                costs.local_fidelity_loss += gate_error
                costs.local_fidelity *= (1 - gate_error)
                costs.local_fidelity_log_sum += np.log(1 - gate_error)

        if isinstance(arg, MappingRecord):
            arg.costs += costs
            arg.logical_phy_map = logical_phy_map
            return arg.costs, logical_phy_map

        return costs, logical_phy_map

    @staticmethod
    def evaluate_telegate(
        arg: MappingRecord | list[list[int]],  # 兼容两种类型：record / partition
        circuit: QuantumCircuit, 
        network: Network
    ) -> ExecCosts:
        """
        评估在给定划分和后端分配下执行线路的总体保真度
        @param record: 包含划分和映射信息的记录
        @param circuit: 量子线路
        @param network: 网络信息
        @return: 更新后的记录，包含评估的成本信息
        """
        partition = None

        if isinstance(arg, MappingRecord):
            record = arg
            partition = record.partition
        else:
            # arg is a list of lists representing the partition
            partition = arg

        # 建立一个反向索引，用于快速查询每个量子比特属于哪个分区
        qubit_to_partition = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        # 初始化噪声
        costs = ExecCosts()

        # 遍历circuit上的每个操作，如果操作完全属于某个group，则添加到对应的subcircuit；如果操作跨越多个group，则记录为telegate操作
        for instruction in circuit:
            qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # 检查操作涉及的量子比特属于哪个分区
            involved_partitions = set()
            for qubit in qubits:
                involved_partitions.add(qubit_to_partition[qubit])

            if len(involved_partitions) > 1: # 操作跨越多个分区，为telegate操作
                # 对qubits里的每一对相邻qubits，算作一组remote_hop
                # 直接统计跨分区的telegate保真度损失
                for i in range(len(qubits) - 1):
                    q1, q2 = qubits[i], qubits[i + 1]
                    p1, p2 = qubit_to_partition[q1], qubit_to_partition[q2]
                    if p1 != p2:
                        costs = CompilerUtils.update_remote_move_costs(
                            costs, p1, p2, 1, network
                        )
                        # costs.remote_hops += network.Hops[p1][p2]
                        # costs.epairs += network.Hops[p1][p2]
                        # costs.remote_fidelity_loss += network.move_fidelity_loss[p1][p2]
                        # costs.remote_fidelity *= network.move_fidelity[p1][p2]
                        # costs.remote_fidelity_log_sum += np.log(network.move_fidelity[p1][p2])
                        # print(f"[DEBUG] evaluate_local_and_telegate remote: {costs}")

        if isinstance(arg, MappingRecord):
            # 更新record的costs
            arg.costs += costs

        return costs

    @staticmethod
    def evaluate_telegate_with_cat(
        arg: MappingRecord | list[list[int]],
        circuit: QuantumCircuit,
        network: Network,
        cat_controls: Optional[list[int]] = None,
        cat_gate_set: Optional[set[str]] = None,
    ) -> ExecCosts:
        """
        CAT-aware telegate估算：
        - 普通跨分区2Q门按一次remote move计费
        - 若同一控制位在同一远端QPU上命中多个不同目标位，可视为一次CAT setup复用，
          将该组通信权重由n折算为2（setup+teardown近似）
        """
        partition = None
        if isinstance(arg, MappingRecord):
            record = arg
            partition = record.partition
        else:
            partition = arg

        if cat_controls is None:
            cat_controls = CompilerUtils._extract_cat_controls_for_circuit(circuit, support=cat_gate_set)
        cat_control_set = set(cat_controls)
        support = cat_gate_set if cat_gate_set is not None else {"cx", "cz", "rzz"}
        symmetric_support = {"cz", "rzz"}

        qubit_to_partition: dict[int, int] = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        costs = ExecCosts()
        active_groups: dict[tuple[int, int, int], dict[str, Any]] = {}

        def _apply_group(ctrl: int, src_p: int, dst_p: int, info: dict[str, Any]) -> None:
            cnt = int(info["count"])
            n_targets = len(info["targets"])
            is_cat_group = n_targets >= 2 and cnt >= 2
            # CAT组按一次远程telegate通信计费；对应门在目标QPU本地执行并单独计入本地损失。
            effective_weight = 1 if is_cat_group else cnt
            nonlocal costs
            costs = CompilerUtils.update_remote_move_costs(
                costs, src_p, dst_p, effective_weight, network
            )
            if is_cat_group:
                # 单独统计CAT ent复用事件；每个有效CAT组记1。
                costs.cat_ents += 1
                backend = network.backends[dst_p]
                for gate_name, gate_cnt in info["gate_counts"].items():
                    costs = CompilerUtils.update_local_gate_costs_by_name(
                        costs,
                        backend,
                        gate_name,
                        int(gate_cnt),
                    )

        def _flush_control(ctrl: int) -> None:
            keys = [k for k in active_groups.keys() if k[0] == ctrl]
            for key in keys:
                info = active_groups.pop(key)
                _apply_group(key[0], key[1], key[2], info)

        for instruction in circuit:
            qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if not qubits:
                continue
            gate_name = instruction.operation.name

            touched_controls = [c for c in cat_control_set if c in qubits]

            is_cat_candidate = False
            selected_anchor: Optional[int] = None
            if len(qubits) == 2:
                q1, q2 = qubits
                p1 = qubit_to_partition[q1]
                p2 = qubit_to_partition[q2]

                if gate_name in support and p1 != p2:
                    anchor_candidates: list[int] = []
                    if gate_name in symmetric_support:
                        if q1 in cat_control_set:
                            anchor_candidates.append(q1)
                        if q2 in cat_control_set:
                            anchor_candidates.append(q2)
                    else:
                        if q1 in cat_control_set:
                            anchor_candidates.append(q1)

                    if anchor_candidates:
                        # 对称门只选择一个锚点，避免同一门被双计数。
                        selected_anchor = None
                        for cand in anchor_candidates:
                            other = q2 if cand == q1 else q1
                            src_p = qubit_to_partition[cand]
                            dst_p = qubit_to_partition[other]
                            if (cand, src_p, dst_p) in active_groups:
                                selected_anchor = cand
                                break
                        if selected_anchor is None:
                            selected_anchor = anchor_candidates[0]

                if selected_anchor is not None:
                    is_cat_candidate = True
                    other = q2 if selected_anchor == q1 else q1
                    src_p = qubit_to_partition[selected_anchor]
                    dst_p = qubit_to_partition[other]
                    key = (selected_anchor, src_p, dst_p)
                    if key not in active_groups:
                        active_groups[key] = {"count": 0, "targets": set(), "gate_counts": defaultdict(int)}
                    active_groups[key]["count"] += 1
                    active_groups[key]["targets"].add(other)
                    active_groups[key]["gate_counts"][gate_name] += 1
                elif p1 != p2:
                    costs = CompilerUtils.update_remote_move_costs(costs, p1, p2, 1, network)
            else:
                # 多比特门保守处理：相邻比特跨分区逐对计费
                for i in range(len(qubits) - 1):
                    q1, q2 = qubits[i], qubits[i + 1]
                    p1 = qubit_to_partition[q1]
                    p2 = qubit_to_partition[q2]
                    if p1 != p2:
                        costs = CompilerUtils.update_remote_move_costs(costs, p1, p2, 1, network)

            for ctrl in touched_controls:
                # 仅当当前门被选为该ctrl的CAT锚点时，才延续其片段。
                if not (is_cat_candidate and selected_anchor == ctrl):
                    _flush_control(ctrl)

        for ctrl in list(cat_control_set):
            _flush_control(ctrl)

        if isinstance(arg, MappingRecord):
            arg.costs += costs

        return costs

    @staticmethod
    def _extract_cat_controls_for_circuit(
        circuit: QuantumCircuit,
        support: Optional[set[str]] = None,
    ) -> list[int]:
        """
        提取具有CAT复用潜力的控制位：
        同一控制位在连续片段内作用到>=2个不同目标位；
        控制位被其他门触碰时会截断片段。
        """
        gate_support = support if support is not None else {"cx", "cz", "rzz"}
        symmetric_support = {"cz", "rzz"}
        active: dict[int, set[int]] = {}
        controls: set[int] = set()

        def _flush(ctrl: int) -> None:
            tgts = active.get(ctrl)
            if tgts is None:
                return
            if len(tgts) >= 2:
                controls.add(ctrl)
            active.pop(ctrl, None)

        for instruction in circuit:
            qids = [circuit.qubits.index(q) for q in instruction.qubits]
            if not qids:
                continue

            gate_name = instruction.operation.name
            is_supported_remote = (
                gate_name in gate_support and len(qids) == 2 and qids[0] != qids[1]
            )

            anchor_candidates: set[int] = set()
            if is_supported_remote:
                q1, q2 = qids
                if gate_name in symmetric_support:
                    anchor_candidates = {q1, q2}
                else:
                    anchor_candidates = {q1}

            touched = [ctrl for ctrl in list(active.keys()) if ctrl in qids]
            for ctrl in touched:
                if ctrl not in anchor_candidates:
                    _flush(ctrl)

            if is_supported_remote:
                q1, q2 = qids
                for ctrl in anchor_candidates:
                    tgt = q2 if ctrl == q1 else q1
                    if ctrl not in active:
                        active[ctrl] = set()
                    active[ctrl].add(tgt)

        for ctrl in list(active.keys()):
            _flush(ctrl)

        return sorted(controls)

    @staticmethod
    def evaluate_teledata(
        arg1: MappingRecord | list[list[int]],  # 兼容两种类型：prev_record / prev_partition
        arg2: MappingRecord | list[list[int]],  # 兼容两种类型：curr_record / curr_partition
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {}
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]]]:
        """
        计算切换划分的通信开销，支持两种输入格式：
        格式1：arg1=prev_record(MappingRecord), arg2=curr_record(MappingRecord), network
        格式2：arg1=prev_partition(list[list[int]]), arg2=curr_partition(list[list[int]]), network
        """
        # print(f"[DEBUG] evaluate_teledata")
        prev_record, curr_record = None, None
        prev_partition, curr_partition = None, None

        # ========== 第一步：类型判断 + 参数校验 ==========
        # 场景1：输入是 MappingRecord
        if isinstance(arg1, MappingRecord) and isinstance(arg2, MappingRecord):
            prev_record, curr_record = arg1, arg2
            # 提取 partition
            prev_partition = prev_record.partition
            curr_partition = curr_record.partition
            # 初始化logical_phy_map
            logical_phy_map = curr_record.logical_phy_map

        # 场景2：输入是 list[list[int]]
        elif isinstance(arg1, list) and isinstance(arg2, list):
            prev_partition, curr_partition = arg1, arg2
            # # 检查logical_phy_map不为空
            # if len(logical_phy_map) == 0:
            #     raise ValueError("当输入为partition时，必须提供非空的logical_phy_map")

        # 场景3：类型不匹配（抛错提示）
        else:
            raise TypeError(
                "输入参数类型错误！仅支持两种格式：\n"
                "1. arg1=MappingRecord, arg2=MappingRecord\n"
                "2. arg1=list[list[int]], arg2=list[list[int]]"
            )

        G = nx.DiGraph() # 初始化有向图
        G.add_nodes_from(range(len(prev_partition))) # 每个partition对应一个节点

        costs = ExecCosts()

        # 记录每个qubit在prev和curr的分区号
        qubit_mapping = {}
        for pno, part in enumerate(prev_partition):
            # print(f"{pno}: {partition}")
            for qubit in part:
                qubit_mapping[qubit] = [pno, -1]
        for pno, part in enumerate(curr_partition):
            # print(f"{pno}: {partition}")
            for qubit in part:
                qubit_mapping[qubit][1] = pno

        # ---------- 第三步：构建流量图 (记录具体 Qubit) ----------
        # 我们不再只记录 weight，而是把具体的 qubit 塞进边的属性里
        # 边属性结构: {'qubits': [], 'weight': int}
        
        for qubit, (p_part, c_part) in qubit_mapping.items():
            if p_part == c_part:
                continue
            
            u, v = p_part, c_part
            
            # 检查是否存在反向边 (v, u)，如果有，则可以配对做 Swap
            if G.has_edge(v, u):
                # 取出一个反向移动的 qubit 作为交换对象
                # 注意：这里我们从图的边属性里 pop 一个出来
                if len(G[v][u]['qubits']) > 0:
                    swap_partner = G[v][u]['qubits'].pop(0)
                    G[v][u]['weight'] -= 1
                    
                    if G[v][u]['weight'] == 0:
                        G.remove_edge(v, u)
                    
                    # 1. 计算开销
                    costs = CompilerUtils.update_remote_swap_costs(costs, u, v, 1, network)
                    
                    # 2. [核心] 更新 logical_phy_map：交换这两个 qubit 的物理位置
                    if logical_phy_map:
                        logical_phy_map[qubit], logical_phy_map[swap_partner] = \
                        logical_phy_map[swap_partner], logical_phy_map[qubit]

                    continue # 处理完了，不用加边了

            # 如果不能抵消，添加正向边 (u, v)
            if G.has_edge(u, v):
                G[u][v]['qubits'].append(qubit)
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1, qubits=[qubit])

        # ---------- 第四步：处理大环 (Length >= 3) ----------
        all_cycles = nx.simple_cycles(G)
        cycles_by_length = defaultdict(list)
        # 收集长度大于2的环
        for cycle in all_cycles:
            length = len(cycle)
            assert(3 <= length <= network.num_backends)
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            for cycle in cycles_by_length[length]:
                # 检查环是否还存在，并找出最小权重
                min_weight = float('inf')
                valid = True
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i+1) % length]
                    if not G.has_edge(u, v):
                        valid = False
                        break
                    min_weight = min(min_weight, G[u][v]['weight']) # 记录环的个数
                
                if not valid: # 当前环不存在了
                    continue

                # 执行 min_weight 次循环交换
                for _ in range(int(min_weight)):
                    # 从环的每条边取出一个 qubit
                    cycle_qubits = []
                    for i in range(length):
                        u, v = cycle[i], cycle[(i + 1) % length]
                        q = G[u][v]['qubits'].pop(0)
                        cycle_qubits.append(q)
                    
                    # 循环移动
                    # 先保存第一个的位置
                    first_q = cycle_qubits[0]
                    first_pos = (-1, -1)

                    if logical_phy_map:
                        first_pos = logical_phy_map[first_q]
                    
                    # 依次后移
                    for i in range(length - 1):
                        curr_q = cycle_qubits[i]
                        next_q = cycle_qubits[i + 1]
                        # 把next的位置给curr
                        if logical_phy_map:
                            logical_phy_map[curr_q] = logical_phy_map[next_q]
                    
                    # 把最初的位置给最后一个
                    last_q = cycle_qubits[-1]
                    if logical_phy_map:
                        logical_phy_map[last_q] = first_pos

                # 更新图权重
                for i in range(length): # 从G中移除这些环
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    G[u][v]['weight'] -= min_weight
                    if G[u][v]['weight'] == 0:
                        G.remove_edge(u, v)
                    # 对环中的每一条边，计算通信开销
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, u, v, int(min_weight), network
                    )

        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            # print(f"[DEBUG] QPU容量满，不应该有单向边，但发现了：{u} -> {v}，weight={data['weight']}")
            # exit(1)

            qubits_to_move = data['qubits']

            if logical_phy_map:
                for qubit in qubits_to_move:
                    logical_phy_map[qubit] = (v, None) # TODO: 移到了一个新的QPU上，物理位不固定

            costs = CompilerUtils.update_remote_move_costs(
                costs, u, v, data['weight'], network
            )

        if isinstance(arg2, MappingRecord):
            # 更新costs
            arg2.costs += costs
            arg2.logical_phy_map = logical_phy_map
            return arg2.costs, logical_phy_map

        return costs, logical_phy_map

    @staticmethod
    def update_remote_move_costs(costs: ExecCosts, src: int, dst: int, weight: int, network: Network):
        if src == dst:
            return costs

        hops = network.Hops[src][dst]
        
        costs.remote_hops += hops * weight
        costs.epairs += hops * weight
        costs.remote_fidelity_loss += network.move_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.move_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.move_fidelity[src][dst]) * weight

        return costs

    @staticmethod
    def update_local_gate_costs_by_name(costs: ExecCosts, backend: Any, gate_name: str, weight: int):
        """
        按门类型（不区分具体物理位）累计本地门误差。
        用于CAT将跨分区门转成本地门时的保真度统计。
        """
        if weight <= 0:
            return costs

        gname = gate_name.lower()
        gate_entry = backend.gate_dict.get(gname, {})
        gate_error = gate_entry.get("gate_error_value", None)

        if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
            raise ValueError(f"Gate error for '{gate_name}' is not available.")

        gate_error = float(gate_error)
        gate_error = min(max(gate_error, 0.0), 0.99)

        costs.local_gate_num += weight
        costs.local_fidelity_loss += gate_error * weight
        costs.local_fidelity *= (1 - gate_error) ** weight
        costs.local_fidelity_log_sum += np.log(1 - gate_error) * weight
        return costs

    @staticmethod
    def update_remote_swap_costs(costs: ExecCosts, src: int, dst: int, weight: int, network: Network):
        if src == dst:
            return costs
        
        hops = network.Hops[src][dst]
        rswaps = 2 * hops - 1
        
        costs.remote_swaps += rswaps * weight
        costs.epairs += 2 * rswaps * weight
        costs.remote_fidelity_loss += network.swap_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.swap_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.swap_fidelity[src][dst]) * weight

        return costs


    # 
    # 维护逻辑量子比特->QPU物理量子比特的稳定映射关系
    # 
    @staticmethod
    def init_logical_phy_map(partition: list[list[int]]) -> dict[int, tuple[int, int | None]]:
        """从初始分区初始化唯一字典，第一次transpile前用"""
        logical_phy_map = {}
        for qpu_id, qubits in enumerate(partition):
            for qubit in qubits:
                # 初始只赋值tuple[0]，也就是QPU id
                # tuple[1]会在第一次transpile后会更新为真实物理位
                logical_phy_map[qubit] = (qpu_id, None)
        return logical_phy_map
    
    @staticmethod
    def get_logical_to_physical_map(
        transpiled_circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]]
    ) -> dict[int, tuple[int, int | None]]:
        """
        从transpiled电路中提取稳定的逻辑→物理比特映射
        """
        # 初始化：假设所有本地比特暂时映射到 None
        # 这里的 key 是“子电路本地逻辑比特索引” (0, 1, 2...)
        # 示例: {0: None, 1: None} (因为 partition 里只有 2 个逻辑比特)
        local_lqid_to_pqid = {i: None for i in range(len(partition_qubits))}

        if hasattr(transpiled_circuit, "layout") and transpiled_circuit.layout is not None:
            layout = transpiled_circuit.layout.initial_layout
            # [示例] layout: Layout({ 物理: 逻辑
            # 0: Qubit(QuantumRegister(2, 'q'), 0),
            # 1: Qubit(QuantumRegister(2, 'q'), 1),
            # 2: Qubit(QuantumRegister(1, 'ancilla'), 0)
            # })

            # print(f"[DEBUG] layout (phy->log): {layout}")

            phy_to_logic_dict = layout.get_physical_bits()
            # print(f"[phy_to_log] {phy_to_logic_dict}")

            # 遍历 Layout 字典
            # phy_qid: 物理比特编号 (int, 例如 0, 1, 2)
            # logic_qubit: 逻辑比特对象 (Qubit)
            for phy_qid, logic_qubit in phy_to_logic_dict.items():
                # # 获取逻辑比特所在的寄存器名字
                reg = getattr(logic_qubit, 'register', None)
                if reg is None:
                    reg = getattr(logic_qubit, '_register', None)
                
                # # 获取逻辑比特在原始寄存器里的索引
                # 尝试获取 .index，如果没有则获取 ._index
                logic_idx = getattr(logic_qubit, 'index', None)
                if logic_idx is None:
                    logic_idx = getattr(logic_qubit, '_index', None)
                
                # 安全检查：如果都获取失败，跳过
                if reg is None or logic_idx is None:
                    raise RuntimeError(f"[ERROR] Could not extract register info from qubit: {logic_qubit}")
                
                reg_name = reg.name
                
                # print(f"[DEBUG] Checking: Phy {phy_qid} <-> Logic ({reg_name}, {logic_idx})")

                # [关键筛选]
                # 只要寄存器名字是 'q' 的，我们就要；ancilla 直接忽略
                if reg_name == 'q':
                    # 这里的 logic_idx (例如 0 或 1) 正好对应子电路里的“本地索引”
                    # 因为我们的子电路原来就是 2 个比特，名字叫 'q'
                    
                    # 安全检查：确保这个索引在我们预期的范围内
                    if logic_idx in local_lqid_to_pqid:
                        local_lqid_to_pqid[logic_idx] = phy_qid
                #         print(f"  [OK] Mapped Local Logic {logic_idx} -> Physical {phy_qid}")
                #     else:
                #         print(f"  [WARN] Logic index {logic_idx} out of range for current partition")
                # else:
                #     print(f"  [SKIP] Ignoring ancilla/other register: {reg_name}")
        else: # Layout为None，返回1:1平凡映射
            num_qubits = transpiled_circuit.num_qubits
            local_lqid_to_pqid = {i: i for i in range(num_qubits)}

        for global_q in partition_qubits:
            # 获取每个global逻辑比特对应的子线路本地逻辑比特索引
            local_idx = global_to_local_lqid[global_q]
            # 更新logical_phy_map，把物理比特索引填上
            logical_phy_map[global_q] = (logical_phy_map[global_q][0], local_lqid_to_pqid[local_idx])

        return logical_phy_map

    @staticmethod
    def get_initial_layout(
        circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]]
    ) -> dict:
        """
        构建QuantumCircuit Qubit Register到物理比特的初始布局
        """
        initial_layout = {}
        unassigned_global_qs = []
        unused_phy_ids = set(range(circuit.num_qubits))

        for q in partition_qubits:
            if logical_phy_map[q][1] is not None:
                local_lqid = global_to_local_lqid[q]
                initial_layout[circuit.qubits[local_lqid]] = logical_phy_map[q][1]
                unused_phy_ids.discard(logical_phy_map[q][1]) # type: ignore # 这个物理位已经被占用了
            else:
                unassigned_global_qs.append(q)

        # 如果initial_layout全空或全满，那么返回initial_layout
        if len(initial_layout) == 0 or len(initial_layout) == len(partition_qubits):
            return initial_layout

        # 将可用物理比特分配给未分配的逻辑比特，更新initial_layout
        for global_q, phy_id in zip(unassigned_global_qs, unused_phy_ids):
            local_lqid = global_to_local_lqid[global_q]
            initial_layout[circuit.qubits[local_lqid]] = phy_id
    
        return initial_layout
