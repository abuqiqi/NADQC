from qiskit import QuantumCircuit
from abc import ABC, abstractmethod
from typing import Any, Optional
import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import numpy as np
import copy
from collections import defaultdict

from ..utils import Network

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
    costs: dict[str, Any]
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 冻结模式下修改字段需用 object.__setattr__
        object.__setattr__(self, "partition", copy.deepcopy(self.partition))
        object.__setattr__(self, "costs", copy.deepcopy(self.costs))
        if self.extra_info is not None:
            object.__setattr__(self, "extra_info", copy.deepcopy(self.extra_info))


# 辅助类：管理多条记录
@dataclass
class MappingRecordList:
    """
    映射记录管理器：批量存储、查询记录
    """
    total_costs: dict[str, Any]  = dataclasses.field(default_factory=dict)
    records: list[MappingRecord] = dataclasses.field(default_factory=list)

    def add_record(self, record: MappingRecord):
        """添加一条记录"""
        self.records.append(record)

    def add_cost(self, key: str, value: Any):
        """添加项目到total_costs"""
        self.total_costs[key] = value

    def add_cost_sum(self, key: str):
        """添加求和项到total_costs"""
        sum = 0
        for record in self.records:
            sum += record.costs[key]
        self.total_costs[key] = sum

    def add_cost_mul(self, key: str):
        mul = 1
        for record in self.records:
            mul *= record.costs[key]
        self.total_costs[key] = mul

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
        data_dict = dataclasses.asdict(self)
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

    # @abstractmethod
    # def evaluate_total_costs(self, mapping_record_list) -> MappingRecordList:
    #     """
    #     获取映射结果
    #     :return: 包含关键性能指标的字典
    #     """
    #     pass

    def allocate_qubits(self, num_qubits: int, network: Network) -> list[list[int]]:
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

    def build_qubit_interaction_graph(self, circuit: QuantumCircuit) -> nx.Graph:
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

    def evaluate_partition(self, qig: nx.Graph, 
                           partition: list[list[int]], 
                           network: Any) -> dict[str, float]:
        """
        计算qubit interaction graph在partitions下的割
        """
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, part in enumerate(partition):
            for node in part:
                node_to_partition[node] = i
        remote_hops, fidelity_loss, fidelity = 0, 0, 1
        for u, v in qig.edges(): # 遍历图中的每一条边，也就是双量子门
            qpu_u = node_to_partition[u]
            qpu_v = node_to_partition[v]
            if qpu_u != qpu_v:
                remote_hops += network.Hops[qpu_u][qpu_v] * qig[u][v]['weight']
                fidelity_loss += (1 - network.W_eff[qpu_u][qpu_v]) * qig[u][v]['weight']
                fidelity *= network.W_eff[qpu_u][qpu_v] ** qig[u][v]['weight']
        return {
            "num_comms": remote_hops,
            "remote_hops": remote_hops,
            "remote_swaps": 0,
            "fidelity_loss": fidelity_loss,
            "fidelity": fidelity
        }

    def evaluate_partition_switch(self, prev_record: MappingRecord, 
                                  curr_record: MappingRecord, 
                                  network: Network) -> dict[str, float]:
        """
        计算切换划分的通信开销
        """
        # 检查remote_hops
        assert curr_record.costs["remote_swaps"] == 0, f"curr_remote_swaps: {curr_record.costs['remote_swaps']}"
        assert prev_record.costs["num_comms"] == prev_record.costs["remote_hops"] + prev_record.costs["remote_swaps"]
        assert curr_record.costs["num_comms"] == curr_record.costs["remote_hops"]

        prev_partition = prev_record.partition
        curr_partition = curr_record.partition

        G = nx.DiGraph() # 初始化有向图
        G.add_nodes_from(range(len(prev_partition))) # 每个partition对应一个节点

        remote_swaps = 0
        fidelity_loss = 0
        fidelity = 1

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
                    num_rswaps = 2 * network.Hops[curr_part][prev_part] - 1
                    
                    remote_swaps += num_rswaps
                    fidelity_loss += (1 - network.W_eff[prev_part][curr_part]) * num_rswaps
                    fidelity *= network.W_eff[prev_part][curr_part] ** num_rswaps
                    
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
                    num_rswaps = (2 * network.Hops[u][v] - 1) * weight
                    
                    remote_swaps += num_rswaps
                    fidelity_loss += (1 - network.W_eff[u][v]) * num_rswaps
                    fidelity *= network.W_eff[u][v] ** num_rswaps

        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            path_len = 2 * network.Hops[u][v] - 1
            num_rswaps = path_len * data['weight']
            remote_swaps += num_rswaps
            fidelity_loss += (1 - network.W_eff[u][v]) * num_rswaps
            fidelity *= network.W_eff[u][v] ** num_rswaps

        # 更新num_comms
        curr_record.costs["remote_swaps"] = remote_swaps
        curr_record.costs["num_comms"] += remote_swaps
        curr_record.costs["fidelity_loss"] += fidelity_loss
        curr_record.costs["fidelity"] *= fidelity
        return curr_record.costs

    def evaluate_total_costs(self, mapping_record_list: MappingRecordList) -> MappingRecordList:
        mapping_record_list.add_cost_sum("num_comms")
        mapping_record_list.add_cost_sum("remote_hops")
        mapping_record_list.add_cost_sum("remote_swaps")
        mapping_record_list.add_cost_sum("fidelity_loss")
        mapping_record_list.add_cost_mul("fidelity")
        return mapping_record_list
