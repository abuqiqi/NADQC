from abc import ABC, abstractmethod
from typing import Any, Optional
import time
import copy
from qiskit import QuantumCircuit
from qiskit import transpile

from ..utils import Network
from ..compiler import MappingRecord, CompilerUtils, MappingRecord, MappingRecordList
from ..baselines import OEE

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
                  config: Optional[dict[str, Any]]) -> MappingRecord | MappingRecordList:
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

        enable_cat_telegate = config.get("enable_cat_telegate", True) if config else True
        cat_controls = config.get("cat_controls", []) if config else []

        # 计算telegate代价
        if enable_cat_telegate:
            costs = CompilerUtils.evaluate_telegate_with_my_cat(
                partition, circuit, network, cat_controls=cat_controls
            )
        else:
            costs = CompilerUtils.evaluate_telegate(partition, circuit, network)

        return MappingRecord(
            layer_start  = layer_start,
            layer_end    = layer_end,
            partition    = partition,
            mapping_type = "telegate",
            costs        = costs
        )


class OEEPartitioner(TelegatePartitioner):
    @property
    def name(self) -> str:
        return "OEETelegatePartitioner"
    
    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:
        # 将config里面的prev_partition作为OEE的起点
        prev_partition = config.get("partition", []) if config else []
        partition = copy.deepcopy(prev_partition) if prev_partition else []
        iteration_count = config.get("iteration", 50) if config else 50
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1

        add_teledata_costs = config.get("add_teledata_costs", True) if config else True
        enable_cat_telegate = config.get("enable_cat_telegate", True) if config else True
        cat_controls = config.get("cat_controls", []) if config else []
        debug_cat_telegate = config.get("debug_cat_telegate", False) if config else False

        # 如果config里没有partition，就按顺序分配
        if not partition:
            partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        qig = CompilerUtils.build_qubit_interaction_graph(circuit)
        
        # oee_start_time = time.time()
        partition = OEE.partition(partition, qig, network, iteration_count)
        # print(f"[DEBUG] OEE Time: {time.time() - oee_start_time}")

        record = MappingRecord(
            layer_start = layer_start,
            layer_end = layer_end,
            partition = partition,
            mapping_type = "telegate",
            extra_info = {"ops": circuit},
        )

        # 评估当前线路的telegate代价
        if enable_cat_telegate:
            costs = CompilerUtils.evaluate_telegate_with_my_cat(
                partition, circuit, network, cat_controls=cat_controls
            )
        else:
            costs = CompilerUtils.evaluate_telegate(partition, circuit, network)

        if debug_cat_telegate:
            plain_costs = CompilerUtils.evaluate_telegate(partition, circuit, network)
            cat_costs = CompilerUtils.evaluate_telegate_with_my_cat(
                partition, circuit, network, cat_controls=cat_controls
            )
            print(
                f"[cat_debug][telegate_partitioner] layer=[{layer_start},{layer_end}] "
                f"controls={len(cat_controls)} plain_epairs={plain_costs.epairs} cat_epairs={cat_costs.epairs}"
            )

        # 评估和前一段线路的切分代价（如果prev_partition存在）
        if prev_partition and add_teledata_costs:
            teledata_costs, _ = CompilerUtils.evaluate_teledata(
                prev_partition,
                partition,
                network)

            costs += teledata_costs

        record.costs = costs
        return record


class CatEntPartitioner(TelegatePartitioner):
    """
    返回一个用CatEnt方式切分的量子线路
    """
    @property
    def name(self) -> str:
        return "CatEntPartitioner"
    
    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecordList:
        cfg = config or {}
        layer_start = cfg.get("layer_start", 0)
        layer_end = cfg.get("layer_end", circuit.depth() - 1)
        use_oee_init = bool(cfg.get("use_oee_init", True))

        # 先准备初始划分；若未指定则按容量顺序分配。
        base_partition = copy.deepcopy(cfg.get("partition", []))
        if not base_partition:
            base_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        if use_oee_init:
            qig = CompilerUtils.build_qubit_interaction_graph(circuit)
            oee_iteration = int(cfg.get("iteration", 50))
            init_partition = OEE.partition(base_partition, qig, network, oee_iteration)
        else:
            init_partition = base_partition

        # 在初始划分基础上调用QAutoComm，拿到cat-ent优化记录。
        from ..baselines.autocomm import QAutoComm

        autocomm_cfg = dict(cfg)
        autocomm_cfg["partition"] = init_partition
        autocomm_cfg["save_records"] = False
        autocomm_cfg["comm_only_costs"] = True

        cat_result = QAutoComm().compile(
            circuit=circuit,
            network=network,
            config=autocomm_cfg,
        )

        for record in cat_result.records:
            record.layer_start = layer_start
            record.layer_end = layer_end

        return cat_result


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

        # TODO: 计算cat-ent的telegate代价
        costs, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(partition, circuit, network)
        
        return MappingRecord(
            layer_start=layer_start,
            layer_end=layer_end,
            partition=partition,
            mapping_type="telegate",
            costs=costs
        )


class TelegatePartitionerFactory:
    """划分分配器工厂类"""
    _registry = {
        "direct": "DirectTelegatePartitioner",
        "oee": "OEEPartitioner",
        "cat": "CatEntPartitioner",
        "catent": "CatEntPartitioner",
        "cat_ent": "CatEntPartitioner",
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

