from abc import ABC, abstractmethod
from typing import Any, Optional
import time
from qiskit import QuantumCircuit
from qiskit import transpile

from ..utils import Network
from ..compiler import MappingRecord, CompilerUtils, MappingRecord


class TelegatePartitioner(ABC):
    """
    基于telegate的划分器接口
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """获取划分器名称"""
        pass

    @abstractmethod
    def partition(self,
                  circuit: QuantumCircuit,
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:
        """
        执行图划分
        :param circuit: 待划分的电路
        :param network: 网络拓扑信息
        :param config: 划分配置参数
        :return: 包含划分结果和代价的 MappingRecord
        """
        pass


class DirectTelegatePartitioner(TelegatePartitioner):
    """
    直接使用telegate进行划分的实现类
    """

    @property
    def name(self) -> str:
        return "DirectTelegatePartitioner"

    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:
        
        # 直接沿用config里面的partition
        partition = config.get("partition", []) if config else []
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1
        
        # 如果config里没有partition，就按顺序分配
        if not partition:
            partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        # 计算telegate代价
        qig = CompilerUtils.build_qubit_interaction_graph(circuit)
        costs = CompilerUtils.evaluate_partition(qig, partition, network)

        return MappingRecord(
            layer_start  = layer_start,
            layer_end    = layer_end,
            partition    = partition,
            mapping_type = "telegate",
            costs        = costs
        )


class PytketDQCPartitioner(TelegatePartitioner):
    @property
    def name(self) -> str:
        return "PytketDQCPartitioner"

    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:

        from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
        from pytket_dqc.utils import DQCPass
        from pytket_dqc.networks import NISQNetwork
        from pytket_dqc.distributors import CoverEmbeddingSteinerDetached, PartitioningAnnealing

        # 预处理电路
        circuit = transpile(circuit, basis_gates=["cu1", "rz", "h"], optimization_level=0)
        tk_circ = qiskit_to_tk(circuit)
        DQCPass().apply(tk_circ)

        # 构建NISQ网络
        server_coupling, server_qubits = network.get_network_coupling_and_qubits()
        nisq_network = NISQNetwork(server_coupling, server_qubits)
        
        # 获取算法
        workflow = config.get("workflow", "CE") if config else "CE"
        seed = config.get("seed", 26) if config else 26
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1

        distribution = None
        if workflow == "CE":
            print("[Pytket-DQC: CoverEmbeddingSteinerDetached]")
            distribution = CoverEmbeddingSteinerDetached().distribute(tk_circ, nisq_network, seed=seed)
        elif workflow == "PA":
            print("[Pytket-DQC: PartitioningAnnealing]")
            distribution = PartitioningAnnealing().distribute(tk_circ, nisq_network, seed=seed)
        else:
            raise ValueError("Unsupported workflow. Use 'CE' for CoverEmbeddingSteinerDetached or 'PA' for PartitioningAnnealing.")

        # 获取划分
        partition = [[] for _ in range(network.num_backends)]
        for i in range(circuit.num_qubits):
            target = distribution.placement.placement[i]
            partition[target].append(i)
        
        return MappingRecord(
            layer_start=layer_start,
            layer_end=layer_end,
            partition=partition,
            mapping_type="telegate",
            costs={
                "num_comms": distribution.cost(),
                "remote_hops": distribution.cost(),
                "remote_swaps": 0,
                "fidelity_loss": 0,
                "fidelity": 1
            }
        )


class TelegatePartitionerFactory:
    """划分分配器工厂类"""
    _registry = {
        "direct": "DirectTelegatePartitioner",
        "pytket_dqc": "PytketDQCPartitioner"
    }

    @classmethod
    def create_telegate_partitioner(cls, partitioner_type: str) -> TelegatePartitioner:
        """
        创建指定类型的划分器

        :param partitioner_type: 划分配器类型字符串
        :return: 对应的划分配器实例
        """
        partitioner_type = partitioner_type.lower()
        if partitioner_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown partitioner: '{partitioner_type}', available: {available}")
        
        # 从注册表获取类名，然后创建实例
        partitioner_class_name = cls._registry[partitioner_type]
        partitioner_class = globals()[partitioner_class_name]
        return partitioner_class()

    @classmethod
    def register_telegate_partitioner(cls, name: str, class_name: str):
        """
        动态注册新的划分分配器类型
        
        :param name: 分配器类型名称
        :param class_name: 对应的类名字符串
        """
        cls._registry[name] = class_name
    
    @classmethod
    def unregister_telegate_partitioner(cls, name: str):
        """
        移除注册的划分分配器类型
        
        :param name: 要移除的分配器类型名称
        """
        if name in cls._registry:
            del cls._registry[name]
    
    @classmethod
    def get_available_telegate_partitioners(cls):
        """
        获取所有可用的划分分配器类型
        
        :return: 可用分配器类型列表
        """
        return list(cls._registry.keys())

