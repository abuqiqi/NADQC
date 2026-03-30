from abc import ABC, abstractmethod
from typing import Any, Optional
import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import numpy as np
import copy
from collections import defaultdict

from qiskit import QuantumCircuit, transpile

from ..utils import Network

@dataclass
class ExecCosts:
    remote_hops: int = 0
    remote_swaps: int = 0
    epairs: int = 0
    remote_fidelity_loss: float = 0.0
    remote_fidelity_log_sum: float = 0.0
    remote_fidelity: float = 1.0
    local_fidelity_loss: float = 0.0
    local_fidelity_log_sum: float = 0.0
    local_fidelity: float = 1.0
    execution_time: float = 0.0

    local_gate_num: int = 0    # 本地量子门总个数
    
    # 只读属性（实时计算）
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
    def geometric_mean_fidelity(self) -> float:
        """
        计算多QPU系统几何平均保真度（顶会论文标准用法）
        满足：越大越好，保留连乘物理意义，无数值下溢
        """
        # 总门数量
        total_gate_num = self.local_gate_num + self.num_comms
        
        # 边界保护：没有门时返回1.0
        if total_gate_num == 0:
            return 1.0

        # 几何平均保真度（核心公式，论文标准）
        total_log = self.local_fidelity_log_sum + self.remote_fidelity_log_sum
        geo_mean_fidelity = np.exp(total_log / total_gate_num)

        return geo_mean_fidelity

    def update(self, **kwargs):
        """批量更新字段"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"ExecCosts 没有属性 {key}")
    
    # 字符串表示
    def __str__(self) -> str:
        return (
            f"ExecCosts("
            f"comms={self.num_comms}, "
            f"epairs={self.epairs}, "
            f"geo_mean_fidelity={self.geometric_mean_fidelity}, "
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
    
    # 累加方法
    def __iadd__(self, other: "ExecCosts") -> "ExecCosts":
        if not isinstance(other, ExecCosts):
            raise TypeError(f"不能与 {type(other)} 累加")
        
        self.remote_hops += other.remote_hops
        self.remote_swaps += other.remote_swaps
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

    # def to_dict(self) -> dict:
    #     """将 ExecCosts 转为字典，包含所有字段 + property 计算属性"""
    #     # 先获取默认的字段字典（真实字段）
    #     base_dict = dataclasses.asdict(self)
    #     # 手动添加 property 字段
    #     base_dict.update({
    #         "num_comms": self.num_comms,
    #         "total_fidelity_loss": self.total_fidelity_loss,
    #         "total_fidelity": self.total_fidelity
    #     })
    #     return base_dict

    def to_dict(self) -> dict:
        """将 ExecCosts 转为字典，包含所有字段 + 排序Key（学术规范版）"""
        base_dict = dataclasses.asdict(self)
        # 添加计算属性
        base_dict.update({
            "num_comms": self.num_comms,
            "total_fidelity_log_sum": self.total_fidelity_log_sum,
            "total_fidelity_loss": self.total_fidelity_loss,
            "geo_mean_fidelity": self.geometric_mean_fidelity,
            "total_fidelity": self.total_fidelity,
        })

        # 自定义固定排序顺序（按逻辑分组，整洁美观）
        sorted_keys = [
            "num_comms",
            "epairs",
            "total_fidelity_log_sum",
            "total_fidelity_loss",
            "geo_mean_fidelity",
            "total_fidelity",
            "execution_time",
            # 通信开销
            "remote_hops",
            "remote_swaps",
            # 保真度
            "local_fidelity",
            "remote_fidelity",
            "local_fidelity_loss",
            "remote_fidelity_loss",
            "local_fidelity_log_sum",
            "remote_fidelity_log_sum",
            # 门数量
            "local_gate_num"
        ]

        # 生成有序字典
        return {key: base_dict[key] for key in sorted_keys if key in base_dict}

@dataclass
class MappingRecord:
    """
    映射记录类：记录线路层级范围、映射类型、开销及时间
    """
    # 必选字段：线路层级范围
    layer_start: int          # 起始层级（第几层）
    layer_end: int            # 结束层级（第几层）
    # 必选字段：量子比特划分
    partition: list[list[int]]
    # 必选字段：映射信息
    mapping_type: str         # 映射类型（如 "teledata"、"telegate"）
    costs: ExecCosts = ExecCosts() # 执行成本，包含保真度损失、通信开销、执行时间等指标
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 冻结模式下修改字段需用 object.__setattr__
        object.__setattr__(self, "partition", copy.deepcopy(self.partition))
        object.__setattr__(self, "costs", copy.deepcopy(self.costs))
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
            "records": records_dict
        }
        # data_dict = dataclasses.asdict(self)
        data_dict = self._convert_numpy_types(data_dict)

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


class Compiler(ABC):
    """
    编译器接口
    """
    compiler_id: str

    def __init_subclass__(cls, **kwargs):
        """子类初始化时校验：必须设置 compiler_id"""
        super().__init_subclass__(** kwargs)
        if cls.compiler_id is None:
            raise NotImplementedError(f"子类 {cls.__name__} 必须定义 compiler_id 属性")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取编译器名称
        """
        pass

    @abstractmethod
    def compile(self, circuit: Any, 
                network: Any, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        编译电路
        :param circuit: 电路对象
        :param network: 网络对象
        :return: 编译后的电路对象
        """
        pass


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
            for node in layer: # node 是 DAGOpNode 对象
                if node.op.name == "barrier":
                    continue
                gate_instruction = node.op  # 获取门指令（Instruction对象）
                # qubit_indices = [q._index for q in node.qargs]  # 提取量子比特索引
                # if qubit_indices[0] == None:
                qubit_indices = [circuit.qubits.index(q) for q in node.qargs]
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
        optimization_level: int = 3
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

        # 建立每个分区对应的子线路
        subcircuits = [QuantumCircuit(len(group)) for group in partition]

        # 建立一个反向索引，用于快速查询每个量子比特属于哪个分区
        qubit_to_partition = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        # 对每个分区内，要将量子比特编号映射到0,1,...,len(group)-1，以便构建子线路
        # 例如，如果分区是 [0,2,5]，则子线路中的量子比特0对应原线路的0，量子比特1对应原线路的2，量子比特2对应原线路的5
        qubit_to_subcircuit = {}
        for group in partition:
            mapping = {original_qubit: idx for idx, original_qubit in enumerate(group)}
            qubit_to_subcircuit.update(mapping)

        # 初始化噪声
        costs = ExecCosts()

        # 遍历circuit上的每个操作，如果操作完全属于某个group，则添加到对应的subcircuit；如果操作跨越多个group，则记录为telegate操作
        for instruction in circuit:
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # 检查操作涉及的量子比特属于哪个分区
            involved_partitions = set()
            for qubit in qubits:
                involved_partitions.add(qubit_to_partition[qubit])

            if len(involved_partitions) == 1:
                # 操作完全属于一个分区，添加到对应的subcircuit
                partition_idx = involved_partitions.pop()
                # 更新操作中的量子比特编号为子线路中的编号
                mapped_qubits = [qubit_to_subcircuit[qubit] for qubit in qubits]
                subcircuits[partition_idx].append(instruction.operation, mapped_qubits)
                # print(f"[DEBUG] Added instruction {instruction.operation.name} on qubits {mapped_qubits} to subcircuit {partition_idx}")
            else: # 操作跨越多个分区，为telegate操作
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

        # 遍历每个子线路
        assert len(subcircuits) == len(network.backends)
        for idx in range(len(subcircuits)):
            # 获取每个分区单独的保真度损失
            subcircuit = subcircuits[idx]
            backend = network.backends[idx]

            transpiled_circuit = transpile(
                subcircuit,
                coupling_map=backend.coupling_map,
                basis_gates=backend.basis_gates,
                optimization_level=optimization_level
            )

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
                    print(f"[DEBUG] {gate_key}: {gate_error}")
                    exit(1)
                assert gate_error is not None, f"Gate error not found for gate_key: {gate_key} in backend.gate_dict"
                costs.local_gate_num += 1
                costs.local_fidelity_loss += gate_error
                costs.local_fidelity *= (1 - gate_error)
                costs.local_fidelity_log_sum += np.log(1 - gate_error)

        if isinstance(arg, MappingRecord):
            # 更新record的costs
            arg.costs += costs

        return costs

    @staticmethod
    def evaluate_telegate(
        arg: MappingRecord | list[list[int]],  # 兼容两种类型：record / partition
        circuit: QuantumCircuit, 
        network: Network,
        optimization_level: int = 3
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

        # 建立每个分区对应的子线路
        subcircuits = [QuantumCircuit(len(group)) for group in partition]

        # 建立一个反向索引，用于快速查询每个量子比特属于哪个分区
        qubit_to_partition = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        # 对每个分区内，要将量子比特编号映射到0,1,...,len(group)-1，以便构建子线路
        # 例如，如果分区是 [0,2,5]，则子线路中的量子比特0对应原线路的0，量子比特1对应原线路的2，量子比特2对应原线路的5
        qubit_to_subcircuit = {}
        for group in partition:
            mapping = {original_qubit: idx for idx, original_qubit in enumerate(group)}
            qubit_to_subcircuit.update(mapping)

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
    def evaluate_teledata(
        arg1: MappingRecord | list[list[int]],  # 兼容两种类型：prev_record / prev_partition
        arg2: MappingRecord | list[list[int]],  # 兼容两种类型：curr_record / curr_partition
        network: Network
    ) -> ExecCosts:
        """
        计算切换划分的通信开销，支持两种输入格式：
        格式1：arg1=prev_record(MappingRecord), arg2=curr_record(MappingRecord), network
        格式2：arg1=prev_partition(list[list[int]]), arg2=curr_partition(list[list[int]]), network
        """
        prev_record, curr_record = None, None
        prev_partition, curr_partition = None, None
        # ========== 第一步：类型判断 + 参数校验 ==========
        # 场景1：输入是 MappingRecord
        if isinstance(arg1, MappingRecord) and isinstance(arg2, MappingRecord):
            prev_record, curr_record = arg1, arg2
            # 提取 partition
            prev_partition = prev_record.partition
            curr_partition = curr_record.partition

        # 场景2：输入是 list[list[int]]
        elif isinstance(arg1, list) and isinstance(arg2, list):
            prev_partition, curr_partition = arg1, arg2

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

        for prev_part, curr_part in qubit_mapping.values():
            assert(prev_part != -1 and curr_part != -1)
            if prev_part != curr_part: # prev_part -> curr_part
                # 检查是否存在curr_part -> prev_part的边
                # 如果存在，则说明形成了环
                # 因为每次只加一条边，所以抵消掉一条就行
                if G.has_edge(curr_part, prev_part):
                    # hops = network.Hops[curr_part][prev_part]
                    # num_rswaps = 2 * hops - 1
                    
                    # costs.remote_swaps += num_rswaps
                    # costs.epairs += 2 * num_rswaps
                    # costs.remote_fidelity_loss += network.swap_fidelity_loss[prev_part][curr_part] * hops
                    # costs.remote_fidelity *= network.swap_fidelity[prev_part][curr_part] ** hops
                    # costs.remote_fidelity_log_sum += hops * np.log(network.swap_fidelity[prev_part][curr_part])
                    
                    costs = CompilerUtils.update_remote_swap_costs(
                        costs, prev_part, curr_part, 1, network
                    )

                    # 更新边权重
                    if G[curr_part][prev_part]['weight'] > 1:
                        G[curr_part][prev_part]['weight'] -= 1
                    else:
                        G.remove_edge(curr_part, prev_part)
                # 否则添加一条边prev_part -> curr_part
                else:
                    if G.has_edge(prev_part, curr_part):
                        G[prev_part][curr_part]['weight'] += 1
                    else:
                        G.add_edge(prev_part, curr_part, weight=1)
        
        all_cycles = nx.simple_cycles(G)
        cycles_by_length = defaultdict(list)
        # 收集长度大于2的环
        for cycle in all_cycles:
            length = len(cycle)
            assert(3 <= length <= network.num_backends)
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            assert(3 <= length <= network.num_backends)
            for cycle in cycles_by_length[length]:
                exist = True # 先检查是不是所有边都在
                weight = 999999
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    if not G.has_edge(u, v):
                        exist = False
                        break
                    weight = min(weight, G[u][v]['weight']) # 记录环的个数
                if not exist: # 当前环不存在了
                    continue
                for i in range(length): # 从G中移除这些环
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    if G[u][v]['weight'] > weight:
                        G[u][v]['weight'] -= weight
                    else:
                        G.remove_edge(u, v)
                    # 对环中的每一条边，计算通信开销
                    # hops = network.Hops[u][v]
                    # num_rswaps = (2 * hops - 1) * weight
                    
                    # costs.remote_swaps += num_rswaps
                    # costs.epairs += 2 * num_rswaps
                    # costs.remote_fidelity_loss += network.swap_fidelity_loss[u][v] * hops
                    # costs.remote_fidelity *= network.swap_fidelity[u][v] ** hops
                    # costs.remote_fidelity_log_sum += hops * np.log(network.swap_fidelity[u][v])

                    costs = CompilerUtils.update_remote_swap_costs(
                        costs, u, v, weight, network
                    )
        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            # path_len = network.Hops[u][v]
            # num_rmoves = path_len * data['weight']
            # costs.remote_swaps += num_rmoves
            # costs.epairs += num_rmoves
            # costs.remote_fidelity_loss += network.move_fidelity_loss[u][v] * num_rmoves
            # costs.remote_fidelity *= network.move_fidelity[u][v] ** num_rmoves
            # costs.remote_fidelity_log_sum += num_rmoves * np.log(network.move_fidelity[u][v])
            costs = CompilerUtils.update_remote_move_costs(
                costs, u, v, data['weight'], network
            )

        if isinstance(arg2, MappingRecord):
            # 更新costs
            arg2.costs += costs
            return arg2.costs

        return costs

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
