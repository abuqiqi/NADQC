from abc import ABC, abstractmethod
from typing import Any, Optional
import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import numpy as np

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
        print(f"✅ 成功保存 {len(data_dict)} 条映射记录到 JSON 文件：{filename}")
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

    @abstractmethod
    def evaluate_total_costs(self, mapping_record_list) -> MappingRecordList:
        """
        获取映射结果
        :return: 包含关键性能指标的字典
        """
        pass

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

    def evaluate_partitions(self, qig: nx.Graph, partition: list[list[int]], network: Any) -> dict[str, float]:
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
            "fidelity_loss": fidelity_loss,
            "fidelity": fidelity
        }

    def evaluate_partition_switch(self, prev_partition, curr_partition):
        """
        计算切换划分的通信开销
        """
