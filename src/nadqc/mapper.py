from abc import ABC, abstractmethod
import time
import numpy as np
import math
from typing import Dict, Any, List, Tuple

class Mapper(ABC):
    """
    量子线路映射器接口，定义所有映射算法必须实现的方法
    """

    @abstractmethod
    def map_circuit(self, circuit: Any, network: Any) -> Dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param circuit: 原始量子线路
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        获取映射器名称，用于标识和比较
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        获取映射性能指标
        :return: 包含关键性能指标的字典
        """
        pass

class BaselineMapper(Mapper):
    """
    基线映射器：直接使用输入的逻辑QPU顺序（逻辑QPU i -> 物理QPU i）
    作为比较基准，不进行任何优化
    """
    
    def __init__(self):
        """
        初始化BaselineMapper
        """
        super().__init__()
        self.metrics = {}
        self.mapping_sequence = None  # 保存映射序列用于后续使用

    def get_name(self) -> str:
        """获取映射器名称"""
        return "Baseline Mapper"

    def _compute_switch_demand(self, current_partition: List[List[int]], 
                             next_partition: List[List[int]]) -> Tuple[np.ndarray, Dict]:
        """
        计算单次切换的通信需求（与LinkOrientedMapper相同）
        :param current_partition: 当前时间片的分区
        :param next_partition: 下一时间片的分区
        :return: (D_switch, qubit_movements)
        """
        # 创建量子比特到逻辑QPU的映射
        current_qubit_to_logical = {}
        for logical_idx, group in enumerate(current_partition):
            for qubit in group:
                current_qubit_to_logical[qubit] = logical_idx
        
        next_qubit_to_logical = {}
        for logical_idx, group in enumerate(next_partition):
            for qubit in group:
                next_qubit_to_logical[qubit] = logical_idx
        
        # 初始化需求矩阵
        m_logical_current = len(current_partition)
        m_logical_next = len(next_partition)
        D_switch = np.zeros((m_logical_current, m_logical_next))
        
        # 记录每个量子比特的移动
        qubit_movements = {}
        
        # 计算需求
        num_qubits = len(current_qubit_to_logical)
        for qubit in range(num_qubits):
            curr_logical = current_qubit_to_logical[qubit]
            next_logical = next_qubit_to_logical[qubit]
            
            if curr_logical != next_logical:
                D_switch[curr_logical][next_logical] += 1
                qubit_movements[qubit] = (curr_logical, next_logical)
        
        return D_switch, qubit_movements

    def _evaluate_switch_cost(self, D_switch: np.ndarray, 
                            mapping_current: List[int], 
                            mapping_next: List[int], 
                            qubit_movements: Dict) -> float:
        """
        计算切换成本（与LinkOrientedMapper相同）
        :param D_switch: 通信需求矩阵
        :param mapping_current: 当前映射
        :param mapping_next: 下一映射
        :param qubit_movements: 量子比特移动详情
        :return: 切换成本
        """
        W_eff = self.network.W_eff
        
        total_cost = 0.0
        total_fidelity = 1.0
        
        # 基于量子比特移动计算
        for qubit, (from_logical, to_logical) in qubit_movements.items():
            from_physical = mapping_current[from_logical]
            to_physical = mapping_next[to_logical]
            # 移动成本 = 1 - 保真度
            cost = 1 - W_eff[from_physical][to_physical]
            total_cost += cost
            total_fidelity *= W_eff[from_physical][to_physical]
        
        return total_cost, total_fidelity

    def calculate_comm_cost(self, partition_plan: List[List[List[int]]]) -> List[List[int]]:
        """
        基线映射：使用identity映射（逻辑QPU i -> 物理QPU i）
        :param partition_plan: list of partitions for each time slice
        :return: (total_comm_cost, mapping_sequence)
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        
        # 结果存储
        total_comm_cost = 0.0
        total_fidelity = 1.0
        mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
        # 1. 为每个时间片创建identity映射
        for t in range(k):
            num_logical = len(partition_plan[t])
            # Identity mapping: logical i -> physical i
            mapping = list(range(num_logical))
            mapping_sequence.append(mapping)
        
        # 2. 计算相邻时间片之间的通信成本
        for t in range(k - 1):
            # 获取当前和下一时间片的映射
            current_mapping = mapping_sequence[t]
            next_mapping = mapping_sequence[t + 1]
            
            # 计算切换需求
            D_switch, qubit_movements = self._compute_switch_demand(
                partition_plan[t], 
                partition_plan[t + 1]
            )
            
            # 计算本次切换的成本
            switch_cost, fidelity = self._evaluate_switch_cost(
                D_switch,
                current_mapping,
                next_mapping,
                qubit_movements
            )
            
            total_comm_cost += switch_cost
            total_fidelity *= fidelity
        
        end_time = time.time()
        self.metrics = {
            "total_comm_cost": total_comm_cost,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }
        
        return mapping_sequence

    def map_circuit(self, partition_plan: Any, network: Any) -> Dict[str, Any]:
        """
        将量子线路映射到特定量子硬件（基线实现）
        :param partition_plan: 量子比特划分
        :param backend: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 确保network有必要的属性
        if not hasattr(network, 'num_backends') or not hasattr(network, 'W_eff'):
            raise AttributeError("Network must have 'num_backends' and 'W_eff' attributes")
        
        # 计算基线通信成本和映射序列
        mapping_sequence = self.calculate_comm_cost(partition_plan)
        
        # 保存映射序列
        self.mapping_sequence = mapping_sequence
        
        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "metrics": self.metrics
        }

    def get_metrics(self) -> Dict[str, float]:
        """获取映射性能指标"""
        return self.metrics

class LinkOrientedMapper(Mapper):
    """
    实现基于连接的映射算法，通过动态计算通信成本优化量子比特映射
    """
    
    def __init__(self):
        """
        初始化LinkOrientedMapper
        """
        super().__init__()
        self.metrics = {}
        self.mapping_sequence = None  # 保存映射序列用于后续使用

    def get_name(self) -> str:
        """获取映射器名称"""
        return "Link-Oriented Mapper"

    def _compute_switch_demand(self, current_partition: List[List[int]], 
                             next_partition: List[List[int]]) -> Tuple[np.ndarray, Dict]:
        """
        计算单次切换的通信需求
        :param current_partition: 当前时间片的分区
        :param next_partition: 下一时间片的分区
        :return: (D_switch, qubit_movements)
        """
        # 创建量子比特到逻辑QPU的映射
        current_qubit_to_logical = {}
        for logical_idx, group in enumerate(current_partition):
            for qubit in group:
                current_qubit_to_logical[qubit] = logical_idx
        
        next_qubit_to_logical = {}
        for logical_idx, group in enumerate(next_partition):
            for qubit in group:
                next_qubit_to_logical[qubit] = logical_idx
        
        # 初始化需求矩阵
        m_logical_current = len(current_partition)
        m_logical_next = len(next_partition)
        D_switch = np.zeros((m_logical_current, m_logical_next))
        
        # 记录每个量子比特的移动
        qubit_movements = {}
        
        # 计算需求
        num_qubits = len(current_qubit_to_logical)
        for qubit in range(num_qubits):
            curr_logical = current_qubit_to_logical[qubit]
            next_logical = next_qubit_to_logical[qubit]
            
            if curr_logical != next_logical:
                D_switch[curr_logical][next_logical] += 1
                qubit_movements[qubit] = (curr_logical, next_logical)
        
        return D_switch, qubit_movements

    def _find_optimal_mapping_for_switch(self, D_switch: np.ndarray, 
                                        current_mapping: List[int], 
                                        num_logical_next: int, 
                                        qubit_movements: Dict) -> List[int]:
        """
        为切换找到最优的下一映射
        :param D_switch: 通信需求矩阵
        :param current_mapping: 当前时间片的物理映射
        :param num_logical_next: 下一时间片的逻辑QPU数量
        :param qubit_movements: 量子比特移动详情
        :return: 下一时间片的最优物理映射
        """
        m_physical = self.network.num_backends
        
        # 获取有效保真度矩阵
        W_eff = self.network.W_eff
        
        # 1. 计算下一时间片中每个逻辑QPU的重要性
        logical_importance = np.zeros(num_logical_next)
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                logical_importance[j] += D_switch[i][j]  # 流入逻辑QPU j 的需求
        
        # 2. 计算当前时间片中每个物理QPU的"热度"
        physical_hotness = np.zeros(m_physical)
        for qubit, (from_logical, to_logical) in qubit_movements.items():
            physical_pos = current_mapping[from_logical]
            physical_hotness[physical_pos] += 1
        
        # 3. 优先分配高重要性逻辑QPU到高热度物理QPU
        importance_order = np.argsort(-logical_importance)  # 降序排序
        hotness_order = np.argsort(-physical_hotness)       # 降序排序
        
        next_mapping = [-1] * num_logical_next  # 初始化为-1（未分配）
        used_physical = [False] * m_physical
        
        # 4. 分配逻辑QPU
        for logical_next in importance_order:
            if logical_importance[logical_next] == 0:
                continue  # 无需求的逻辑QPU可以最后分配
            
            best_physical = -1
            best_score = -1
            
            # 尝试所有物理QPU
            for physical in range(m_physical):
                if used_physical[physical]:
                    continue
                
                # 评分：热度 + 与当前高需求位置的保真度
                score = physical_hotness[physical] * 0.6  # 热度权重
                
                # 考虑与当前移动源位置的保真度
                fidelity_sum = 0
                count = 0
                for qubit, (from_logical, to_logical) in qubit_movements.items():
                    if to_logical == logical_next:
                        current_physical = current_mapping[from_logical]
                        fidelity_sum += W_eff[current_physical][physical]
                        count += 1
                
                if count > 0:
                    avg_fidelity = fidelity_sum / count
                    score += avg_fidelity * 0.4  # 保真度权重
                
                if score > best_score:
                    best_score = score
                    best_physical = physical
            
            if best_physical == -1:  # 所有物理QPU已用，选第一个可用
                best_physical = used_physical.index(False)
            
            next_mapping[logical_next] = best_physical
            used_physical[best_physical] = True
        
        # 5. 分配剩余的逻辑QPU
        remaining_logicals = [i for i in range(num_logical_next) if next_mapping[i] == -1]
        remaining_physicals = [i for i in range(m_physical) if not used_physical[i]]
        
        # 按逻辑索引顺序分配
        for idx, logical in enumerate(remaining_logicals):
            next_mapping[logical] = remaining_physicals[idx]
        
        return next_mapping

    def _evaluate_switch_cost(self, D_switch: np.ndarray, 
                            mapping_current: List[int], 
                            mapping_next: List[int], 
                            qubit_movements: Dict) -> float:
        """
        计算切换成本
        :param D_switch: 通信需求矩阵
        :param mapping_current: 当前映射
        :param mapping_next: 下一映射
        :param qubit_movements: 量子比特移动详情
        :return: 切换成本
        """
        W_eff = self.network.W_eff
        
        total_cost = 0.0
        total_fidelity = 1.0
        
        # 基于量子比特移动计算
        for qubit, (from_logical, to_logical) in qubit_movements.items():
            from_physical = mapping_current[from_logical]
            to_physical = mapping_next[to_logical]
            # 移动成本 = 1 - 保真度
            cost = 1 - W_eff[from_physical][to_physical]
            total_cost += cost
            total_fidelity *= W_eff[from_physical][to_physical]
        
        return total_cost, total_fidelity

    def calculate_comm_cost_dynamic(self, partition_plan: List[List[List[int]]]) -> Tuple[float, List[List[int]]]:
        """
        动态映射：为每个partition切换计算通信成本和最优映射序列
        :param partition_plan: list of partitions for each time slice
        :return: (total_comm_cost, mapping_sequence)
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        m_physical = self.network.num_backends  # 物理QPU数量
        # n = self.circ.num_qubits  # 量子比特数
        
        # 结果存储
        total_comm_cost = 0.0
        total_fidelity = 1.0
        mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
        # 1. 设置初始映射：时间片0的逻辑QPU到物理QPU的映射
        num_logical_qpus_t0 = len(partition_plan[0])
        assert num_logical_qpus_t0 <= m_physical, f"Time slice 0 has {num_logical_qpus_t0} logical QPUs but only {m_physical} physical QPUs available"
        
        # 简单策略：identity mapping (逻辑QPU i -> 物理QPU i)
        initial_mapping = list(range(num_logical_qpus_t0))
        mapping_sequence.append(initial_mapping)
        
        # 2. 为每个时间片边界计算切换成本
        for t in range(k-1):
            # 2.1 计算当前边界(t -> t+1)的通信需求
            D_switch, qubit_movements = self._compute_switch_demand(
                partition_plan[t], 
                partition_plan[t+1]
            )
            
            # 2.2 获取下一时间片的逻辑QPU数量
            num_logical_next = len(partition_plan[t+1])
            assert num_logical_next <= m_physical, f"Time slice {t+1} has {num_logical_next} logical QPUs but only {m_physical} physical QPUs available"
            
            # 2.3 为下一时间片找到最优映射
            next_mapping = self._find_optimal_mapping_for_switch(
                D_switch,
                mapping_sequence[t],  # 当前时间片的映射
                num_logical_next,    # 下一时间片的逻辑QPU数量
                qubit_movements      # 量子比特移动详情
            )
            
            # 2.4 计算本次切换的实际成本
            switch_cost, switch_fidelity = self._evaluate_switch_cost(
                D_switch,
                mapping_sequence[t],
                next_mapping,
                qubit_movements
            )
            
            total_comm_cost += switch_cost
            total_fidelity *= switch_fidelity
            mapping_sequence.append(next_mapping)
        
        end_time = time.time()
        self.metrics = {
            "total_comm_cost": total_comm_cost,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }
        
        return mapping_sequence

    def map_circuit(self, partition_plan: Any, network: Any) -> Dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 确保network有必要的属性
        if not hasattr(network, 'num_backends') or not hasattr(network, 'W_eff'):
            raise AttributeError("Network must have 'num_backends' and 'W_eff' attributes")

        # 计算通信成本和映射序列
        mapping_sequence = self.calculate_comm_cost_dynamic(partition_plan)
        
        # 保存映射序列用于后续使用
        self.mapping_sequence = mapping_sequence
        
        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "metrics": self.metrics
        }

    def get_metrics(self) -> Dict[str, float]:
        """获取映射性能指标"""
        return self.metrics
