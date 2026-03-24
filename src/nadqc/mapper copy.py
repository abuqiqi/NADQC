from abc import ABC, abstractmethod
import dataclasses
import time
import numpy as np
import itertools
from typing import Any, Optional

from ..compiler import MappingRecordList
from ..utils import Network

class Mapper(ABC):
    """
    量子线路映射器接口，定义所有映射算法必须实现的方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取映射器名称，用于标识和比较
        """
        pass

    @abstractmethod
    # def map(self, partition_plan: Any, network: Network) -> dict[str, Any]:
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit_layers: list[Any],
            network: Network) -> dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        pass

    def _compute_switch_demand(self, 
                               current_partition: list[list[int]], 
                               next_partition: list[list[int]]) -> tuple[np.ndarray, dict]:
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
            qubit_movements[qubit] = (curr_logical, next_logical)
            D_switch[curr_logical][next_logical] += 1

        return D_switch, qubit_movements

    def _evaluate_switch_cost(self, 
                              network: Network,
                              D_switch: np.ndarray, 
                              mapping_current: list[int], 
                              mapping_next: list[int]) -> dict[str, float]:
        """
        计算切换成本
        :param D_switch: 通信需求矩阵
        :param mapping_current: 当前映射
        :param mapping_next: 下一映射
        :return: (switch_cost, switch_fidelity)
        """
        W_eff = network.W_eff
        
        mapping_score = 0.0
        fidelity_loss = 0.0
        fidelity = 1.0

        # 基于需求矩阵计算（可选）
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                demand = D_switch[i][j]
                if demand > 0:
                    from_physical = mapping_current[i]
                    to_physical = mapping_next[j]
                    mapping_score += W_eff[from_physical][to_physical] * demand
                    fidelity_loss += (1 - W_eff[from_physical][to_physical]) * demand
                    fidelity *= W_eff[from_physical][to_physical] ** demand

        return {
            "mapping_score": mapping_score,
            "fidelity_loss": fidelity_loss,
            "fidelity": fidelity
        }
    
    def _evaluate_initial_mapping(self, network: Network, mapping: list[int], D_total: np.ndarray) -> float:
        """
        评估初始映射的质量
        :param mapping: 映射
        :param D_total: 总通信需求矩阵
        :return: 映射得分
        """
        W_eff = network.W_eff
        # pprint(W_eff)
        mapping_score = 0.0
        for i in range(D_total.shape[0]):
            for j in range(D_total.shape[1]):
                from_physical = mapping[i]
                to_physical = mapping[j]
                mapping_score += W_eff[from_physical][to_physical] * D_total[i][j]
        return mapping_score

    def _validate_network_attributes(self, network):
        """验证网络对象是否具有必要属性"""
        if not hasattr(network, 'num_backends') or not hasattr(network, 'W_eff'):
            raise AttributeError("Network must have 'num_backends' and 'W_eff' attributes")

    def _build_partition_plan(self, partition_plan, mapping_sequence):
        
        # 根据映射序列更新partition_plan
        adjusted_partition_plan = []

        for t, mapping in enumerate(mapping_sequence):
            current_partition = partition_plan[t]
            
            # 构建新的分区
            new_partition = [[] for _ in range(len(current_partition))]

            for logical_idx, group in enumerate(current_partition):
                physical_idx = mapping[logical_idx]
                new_partition[physical_idx].extend(group)
            
            adjusted_partition_plan.append(new_partition)

        return adjusted_partition_plan


class SimpleMapper(Mapper):
    """
    基线映射器：直接使用输入的逻辑QPU顺序（逻辑QPU i -> 物理QPU i）
    作为比较基准，不进行任何优化
    """

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Simple Mapper"

    def _calculate_comm_cost(self, partition_plan: list[list[list[int]]]) -> list[list[int]]:
        """
        基线映射：使用identity映射（逻辑QPU i -> 物理QPU i）
        :param partition_plan: list of partitions for each time slice
        :return: mapping_sequence
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        
        # 结果存储
        total_fidelity_loss = 0.0
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
            result = self._evaluate_switch_cost(
                self.network,
                D_switch,
                current_mapping,
                next_mapping
            )
            
            total_fidelity_loss += result["fidelity_loss"]
            total_fidelity *= result["fidelity"]
        
        end_time = time.time()
        self.metrics = {
            "total_fidelity_loss": total_fidelity_loss,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }
        
        return mapping_sequence

    def map(self, 
            partition_plan: Any,
            network: Any) -> dict[str, Any]:
        """
        将量子线路映射到特定量子硬件（基线实现）
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 验证网络属性
        self._validate_network_attributes(network)
        
        # 计算基线通信成本和映射序列
        mapping_sequence = self._calculate_comm_cost(partition_plan)

        # 保存映射序列
        self.mapping_sequence = mapping_sequence

        # self.partition_plan = self._build_partition_plan(partition_plan, mapping_sequence)

        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "partition_plan": partition_plan, # 映射不变
            "metrics": self.metrics
        }


class LinkOrientedMapper(Mapper):
    """
    实现基于连接的映射算法，通过动态计算通信成本优化量子比特映射
    """
    
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Link-Oriented Mapper"

    def _find_optimal_mapping_for_switch(self, 
                                         D_switch: np.ndarray, 
                                         current_mapping: list[int], 
                                         num_logical_next: int, 
                                         qubit_movements: dict) -> list[int]:
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

    def _calculate_comm_cost_dynamic(self, partition_plan: list[list[list[int]]]) -> list[list[int]]:
        """
        动态映射：为每个partition切换计算通信成本和最优映射序列
        :param partition_plan: list of partitions for each time slice
        :return: (total_fidelity_loss, mapping_sequence)
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        m_physical = self.network.num_backends  # 物理QPU数量
        
        # 结果存储
        total_fidelity_loss = 0.0
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
            # print("D_switch:")
            # pprint(D_switch)
            # print("Qubit Movements:")
            # pprint(qubit_movements)
            
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

            # pprint(f"Mapping at time slice {t+1}: {next_mapping}")
            
            # 2.4 计算本次切换的实际成本
            result = self._evaluate_switch_cost(
                self.network,
                D_switch,
                mapping_sequence[t],
                next_mapping
            )

            # print(f"Switch Cost from time slice {t} to {t+1}: {switch_cost}, Fidelity: {switch_fidelity}")
            
            total_fidelity_loss += result["fidelity_loss"]
            total_fidelity *= result["fidelity"]
            mapping_sequence.append(next_mapping)
        
        end_time = time.time()
        self.metrics = {
            "total_fidelity_loss": total_fidelity_loss,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }
        
        return mapping_sequence

    def map(self, partition_plan: Any, network: Any) -> dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 验证网络属性
        self._validate_network_attributes(network)

        # 计算通信成本和映射序列
        mapping_sequence = self._calculate_comm_cost_dynamic(partition_plan)
        
        # 保存映射序列用于后续使用
        self.mapping_sequence = mapping_sequence
        self.partition_plan = self._build_partition_plan(partition_plan, mapping_sequence)

        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "partition_plan": self.partition_plan,
            "metrics": self.metrics
        }


class GreedyMapper(Mapper):
    """
    每一步使用贪心算法求得与上一时间步相比保真度最高的映射
    """
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Greedy Mapper"

    def map(self, partition_plan: Any, network: Any) -> dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 验证网络属性
        self._validate_network_attributes(network)

        # 计算通信成本和映射序列
        mapping_sequence = self._calculate_comm_cost_greedy(partition_plan)
        
        # 保存映射序列用于后续使用
        self.mapping_sequence = mapping_sequence
        self.partition_plan = self._build_partition_plan(partition_plan, mapping_sequence)

        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "partition_plan": self.partition_plan,
            "metrics": self.metrics
        }
    
    def _calculate_comm_cost_greedy(self, partition_plan: list[list[list[int]]]) -> list[list[int]]:
        """
        使用贪心算法计算最优映射序列
        :param partition_plan: list of partitions for each time slice
        :return: mapping_sequence
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        m_physical = self.network.num_backends  # 物理QPU数量
        
        # 结果存储
        total_fidelity = 1.0
        total_fidelity_loss = 0.0
        mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
        # 1. 设置初始映射：时间片0的逻辑QPU到物理QPU的映射
        num_logical_qpus_t0 = len(partition_plan[0])
        assert num_logical_qpus_t0 <= m_physical, f"Time slice 0 has {num_logical_qpus_t0} logical QPUs but only {m_physical} physical QPUs available"
        
        # 简单策略：identity mapping (逻辑QPU i -> 物理QPU i)
        # initial_mapping = list(range(num_logical_qpus_t0))
        # mapping_sequence.append(initial_mapping)
        initial_mapping = self._compute_initial_mapping(partition_plan)
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
            
            # 2.3 使用贪心算法为下一时间片找到映射
            next_mapping, fidelity, fidelity_loss = self._find_optimal_mapping_for_switch(
                D_switch,
                mapping_sequence[t],  # 当前时间片的映射
                num_logical_next      # 下一时间片的逻辑QPU数量
            )

            total_fidelity *= fidelity
            total_fidelity_loss += fidelity_loss
            mapping_sequence.append(next_mapping)
        
        end_time = time.time()

        self.metrics = {
            "total_fidelity_loss": total_fidelity_loss,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }

        return mapping_sequence

    def _compute_initial_mapping(self, partition_plan: list[list[list[int]]]):
        best_mapping = None
        best_score = -float('inf')
        
        D_total = self._compute_total_communication_demand(partition_plan)
        
        for perm in itertools.permutations(range(self.network.num_backends), len(partition_plan[0])):
            candidate_mapping = list(perm)
            score = self._evaluate_initial_mapping(self.network, candidate_mapping, D_total)
            if score > best_score:
                best_score = score
                best_mapping = candidate_mapping[:]

        return best_mapping

    def _compute_total_communication_demand(self, partition_plan: list[list[list[int]]]) -> np.ndarray:
        """
        计算整个partition plan的总通信需求矩阵
        :param partition_plan: list of partitions for each time slice
        :return: D_total: 总通信需求矩阵
        """
        k = len(partition_plan)

        # 初始化总需求矩阵
        num_logical_initial = len(partition_plan[0])
        D_total = np.zeros((num_logical_initial, num_logical_initial))
        
        # 遍历每个时间片边界
        for t in range(k - 1):
            current_partition = partition_plan[t]
            next_partition = partition_plan[t + 1]
            
            # 计算当前边界的需求矩阵
            D_switch, _ = self._compute_switch_demand(current_partition, next_partition)
            
            # 累加到总需求矩阵
            for i in range(D_switch.shape[0]):
                for j in range(D_switch.shape[1]):
                    D_total[i][j] += D_switch[i][j]
        
        return D_total

    def _find_optimal_mapping_for_switch(self, D_switch: np.ndarray, 
                                   current_mapping: list[int], 
                                   num_logical_next: int) -> Any:
        """
        使用贪心算法为切换找到下一映射
        :param D_switch: 通信需求矩阵
        :param current_mapping: 当前时间片的物理映射
        :param num_logical_next: 下一时间片的逻辑QPU数量
        :param qubit_movements: 量子比特移动详情
        :return: 下一时间片的物理映射
        """
        m_physical = self.network.num_backends

        # 枚举所有排列
        best_mapping = None
        best_mapping_score = 0.0
        best_fidelity_loss = float('inf')
        best_fidelity = 0.0
        for perm in itertools.permutations(range(m_physical), num_logical_next):
            candidate_mapping = list(perm)

            result = self._evaluate_switch_cost(
                self.network,
                D_switch,
                current_mapping,
                candidate_mapping
            )
            if result["mapping_score"] > best_mapping_score:
                best_mapping_score = result["mapping_score"]
                best_fidelity_loss = result["fidelity_loss"]
                best_fidelity = result["fidelity"]
                best_mapping = candidate_mapping[:]
        
        return best_mapping, best_fidelity, best_fidelity_loss


class ExactOptimizationMapper(Mapper):
    """
    精确优化映射器：使用QAP算法精确求解最优映射问题
    """
    
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Exact Optimization Mapper"

    def _exact_algorithm(self, m: int, D: np.ndarray, W: np.ndarray) -> list[int]:
        """
        精确算法：枚举所有排列，选择 R(pi) 最大的
        """
        best_pi = []
        best_score = -float('inf')
        
        # 生成所有排列 (0-based 索引: 逻辑 0..m-1 映射到物理 0..m-1)
        for perm in itertools.permutations(range(m)):
            # perm 是元组，索引 0..m-1 对应逻辑 0..m-1
            pi = list(perm)  # 0-based: pi[i] for i in 0..m-1
            
            score = self._evaluate_mapping(pi, D, W, m) # ["mapping_score"]
            if score > best_score:
                best_score = score
                best_pi = pi[:]
        
        return best_pi  # 列表，索引 0 对应逻辑 0, 索引 m-1 对应逻辑 m-1

    def _evaluate_mapping(self, pi: list[int], D: np.ndarray, W: np.ndarray, m: int) -> float:
        """
        计算目标函数 R(pi) = sum_{i<j} D_{i,j} * W_{pi(i), pi(j)}
        """
        mapping_score = 0.0
        for i in range(m):
            for j in range(m):
                p = pi[i]  # pi(i) 是物理索引
                q = pi[j]  # pi(j) 是物理索引
                mapping_score += D[i][j] * W[p][q]
        return mapping_score

    def _compute_total_communication_demand(self, partition_plan: list[list[list[int]]]) -> np.ndarray:
        """
        计算整个partition plan的总通信需求矩阵
        :param partition_plan: list of partitions for each time slice
        :return: D_total: 总通信需求矩阵
        """
        k = len(partition_plan)
        
        # 找出最大逻辑QPU数量
        max_logical = 0
        for partition in partition_plan:
            max_logical = max(max_logical, len(partition))
        
        # 初始化总需求矩阵
        D_total = np.zeros((max_logical, max_logical))
        
        # 遍历每个时间片边界
        for t in range(k - 1):
            current_partition = partition_plan[t]
            next_partition = partition_plan[t + 1]
            
            # 计算当前边界的需求矩阵
            D_switch, _ = self._compute_switch_demand(current_partition, next_partition)
            
            # 累加到总需求矩阵
            for i in range(D_switch.shape[0]):
                for j in range(D_switch.shape[1]):
                    D_total[i][j] += D_switch[i][j]
        
        return D_total

    def _calculate_comm_cost_exact(self, partition_plan: list[list[list[int]]]) -> list[list[int]]:
        """
        使用精确算法计算最优映射序列
        :param partition_plan: list of partitions for each time slice
        :return: mapping_sequence
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        m_physical = self.network.num_backends  # 物理QPU数量
        
        # 结果存储
        total_fidelity_loss = 0.0
        total_fidelity = 1.0
        mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
        # 1. 计算总通信需求矩阵
        D_total = self._compute_total_communication_demand(partition_plan)
        
        # 2. 为每个时间片边界计算切换成本
        for t in range(k):
            if t == 0:
                # 为第一个时间片找到最优映射
                num_logical_current = len(partition_plan[t])
                if num_logical_current <= 10:  # 使用精确算法
                    # 提取当前时间片对应的子矩阵
                    D_sub = D_total[:num_logical_current, :num_logical_current]
                    W_sub = self.network.W_eff[:m_physical, :m_physical]
                    
                    # 如果逻辑QPU数量小于物理QPU数量，需要扩展W矩阵
                    if num_logical_current <= m_physical:
                        # 创建映射：逻辑QPU i 映射到物理QPU 0..num_logical_current-1
                        optimal_mapping = self._exact_algorithm(num_logical_current, D_sub, W_sub)
                    else:
                        # 如果逻辑QPU数量大于物理QPU数量，报错
                        raise ValueError(f"Time slice {t} has {num_logical_current} logical QPUs but only {m_physical} physical QPUs available")
                else:
                    # 如果逻辑QPU数量太多，使用简单的映射
                    optimal_mapping = list(range(num_logical_current))
                
                mapping_sequence.append(optimal_mapping)
            else:
                # 为后续时间片找到最优映射
                num_logical_current = len(partition_plan[t])
                num_logical_prev = len(partition_plan[t-1])
                
                if num_logical_current <= 10 and num_logical_prev <= 10:  # 使用精确算法
                    # 计算当前边界的需求矩阵
                    D_switch, qubit_movements = self._compute_switch_demand(
                        partition_plan[t-1], 
                        partition_plan[t]
                    )
                    
                    # 使用前一个映射作为参考，寻找最优的当前映射
                    prev_mapping = mapping_sequence[t-1]
                    
                    # 创建当前时间片的子矩阵
                    D_sub = np.zeros((num_logical_current, num_logical_current))
                    W_sub = self.network.W_eff[:m_physical, :m_physical]
                    
                    # 通过枚举所有可能的映射来找到最优映射
                    best_mapping = None
                    best_score = -float('inf')
                    
                    # 生成所有可能的物理QPU分配
                    available_physical = list(range(m_physical))
                    for perm in itertools.permutations(available_physical, num_logical_current):
                        current_mapping = list(perm)
                        
                        # 计算当前映射的得分
                        score = self._evaluate_mapping_for_switch(
                            D_switch, prev_mapping, current_mapping, qubit_movements
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_mapping = current_mapping[:]
                    
                    if best_mapping is None:
                        # 如果找不到合适的映射，使用简单映射
                        best_mapping = list(range(min(num_logical_current, m_physical)))
                    
                    mapping_sequence.append(best_mapping)
                else:
                    # 如果逻辑QPU数量太多，使用简单的映射
                    simple_mapping = list(range(min(num_logical_current, m_physical)))
                    mapping_sequence.append(simple_mapping)
        
        # 3. 计算总通信成本和保真度
        for t in range(k - 1):
            current_mapping = mapping_sequence[t]
            next_mapping = mapping_sequence[t + 1]
            
            D_switch, qubit_movements = self._compute_switch_demand(
                partition_plan[t], 
                partition_plan[t + 1]
            )
            
            result = self._evaluate_switch_cost(
                self.network,
                D_switch,
                current_mapping,
                next_mapping
            )
            
            total_fidelity_loss += result["fidelity_loss"]
            total_fidelity *= result["fidelity"]
        
        end_time = time.time()
        self.metrics = {
            "total_fidelity_loss": total_fidelity_loss,
            "total_fidelity": total_fidelity,
            "mapping_sequence_length": len(mapping_sequence),
            "time": end_time - start_time
        }
        
        return mapping_sequence

    def _evaluate_mapping_for_switch(self, D_switch: np.ndarray, 
                                   prev_mapping: list[int], 
                                   current_mapping: list[int], 
                                   qubit_movements: dict) -> float:
        """
        评估特定映射的切换得分
        """
        W_eff = self.network.W_eff
        total_score = 0.0
        
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                demand = D_switch[i][j]
                if demand > 0:
                    from_physical = prev_mapping[i]
                    to_physical = current_mapping[j]
                    score = W_eff[from_physical][to_physical] * demand
                    total_score += score
        
        return total_score

    def map(self, partition_plan: Any, network: Any) -> dict[str, Any]:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果，包含线路、深度、门数、错误率等指标
        """
        # 保存关键对象用于内部计算
        self.network = network
        
        # 验证网络属性
        self._validate_network_attributes(network)

        # 计算通信成本和映射序列
        mapping_sequence = self._calculate_comm_cost_exact(partition_plan)
        
        # 保存映射序列用于后续使用
        self.mapping_sequence = mapping_sequence
        self.partition_plan = self._build_partition_plan(partition_plan, mapping_sequence)

        return {
            "mapping_sequence": self.mapping_sequence,  # 实际映射操作在编译器后续步骤中完成
            "partition_plan": self.partition_plan,
            "metrics": self.metrics
        }


from abc import ABC, abstractmethod
import time
import numpy as np
import itertools
from typing import Any, Optional
import copy
from math import inf

from qiskit import QuantumCircuit

from ..compiler import MappingRecordList, CompilerUtils, ExecCosts
from ..utils import Network

class Mapper(ABC):
    """
    量子线路映射器接口，定义所有映射算法必须实现的方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取映射器名称，用于标识和比较
        """
        pass

    @abstractmethod
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果
        """
        pass

    def _compute_switch_demand(self, 
                               current_partition: list[list[int]], 
                               next_partition: list[list[int]]
        ) -> tuple[np.ndarray, dict]:
        """
        计算单次切换的通信需求
        :param current_partition: 当前时间片的分区
        :param next_partition: 下一时间片的分区
        :return: (D_switch, qubit_movements)
        """
        # 创建量子比特到逻辑 QPU 的映射
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
            qubit_movements[qubit] = (curr_logical, next_logical)
            D_switch[curr_logical][next_logical] += 1

        return D_switch, qubit_movements

    def _evaluate_switch_cost(self, 
                              network: Network,
                              D_switch: np.ndarray, 
                              mapping_current: list[int], 
                              mapping_next: list[int]) -> ExecCosts:
        """
        计算切换成本（teledata 成本）
        :param network: 量子硬件网络
        :param D_switch: 通信需求矩阵
        :param mapping_current: 当前逻辑 QPU 到物理 QPU 的映射
        :param mapping_next: 下一逻辑 QPU 到物理 QPU 的映射
        :return: ExecCosts 对象
        """
        W_eff = network.W_eff
        
        fidelity_loss = 0.0
        num_comms = 0
        
        # 基于需求矩阵计算切换成本
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                demand = D_switch[i][j]
                if demand > 0:
                    from_physical = mapping_current[i]
                    to_physical = mapping_next[j]
                    # 累加通信次数
                    num_comms += int(demand)
                    # 累加保真度损失 (1 - W_eff) * demand
                    fidelity_loss += (1 - W_eff[from_physical][to_physical]) * demand

        return ExecCosts(
            total_fidelity_loss=fidelity_loss,
            num_comms=num_comms
        )


class LinkOrientedDPMapper(Mapper):
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Link-Ori DP Mapper"
    
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network
        ) -> MappingRecordList:
        """
        使用动态规划为每个时间片选择最优的物理 QPU 映射（排列），
        使得所有时间片的通信相关成本最小。
        """
        start_time = time.time()
        k = len(mapping_record_list.records)          # 时间片数量
        n_physical = network.num_backends             # 物理 QPU 数量

        # 所有可能的物理 QPU 排列（即映射状态）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # ---------- 预计算每个时间片各状态下的 telegate 成本 ----------
        telegate_costs: list[list[ExecCosts]] = []  # telegate_costs[t][idx] -> Costs 对象
        
        for t in range(k):
            record = mapping_record_list.records[t]
            original_partition = record.partition
            original_costs = record.costs
            
            # 为该时间片的所有排列状态计算 telegate 成本
            time_slice_costs = []
            for idx, perm in enumerate(all_perms):
                # 应用排列后的分区
                permuted_partition = [original_partition[i] for i in perm]
                
                # 计算当前排列下的 telegate 成本
                # 方法：基于原始成本，根据排列调整通信成本
                # 如果原始 costs 已经包含该时间片的门操作成本，可以直接使用
                # 否则需要重新计算
                
                # 这里假设原始 costs 是该时间片的基础成本
                # 排列不会改变门操作的数量，但可能改变通信路径
                # 简化处理：直接使用原始成本（因为 telegate 主要与门操作相关）
                # 更精确的做法：根据 permuted_partition 重新评估门操作的通信成本
                
                # 使用 CompilerUtils 评估当前分区下的门操作成本
                telegate_cost = CompilerUtils.evaluate_telegate(
                    circuit_layers[t],
                    permuted_partition,
                    network
                )
                time_slice_costs.append(telegate_cost)
            
            telegate_costs.append(time_slice_costs)

        # ---------- 预计算相邻时间片之间的 teledata 成本 ----------
        # 使用 D_switch 来估算，不要调用 CompilerUtils.evaluate_teledata
        teledata_costs: list[list[list[ExecCosts]]] = []  # teledata_costs[t][prev_idx][curr_idx]
        
        for t in range(k - 1):  # 从 t 到 t+1 的转移，共 k-1 个转移
            record_current = mapping_record_list.records[t]
            record_next = mapping_record_list.records[t + 1]
            
            original_partition_current = record_current.partition
            original_partition_next = record_next.partition
            
            # 为该转移的所有状态对计算 teledata 成本
            transition_costs = []
            for prev_idx in range(num_states):
                prev_perm = all_perms[prev_idx]
                # 应用排列后的当前分区
                permuted_partition_current = [original_partition_current[i] for i in prev_perm]
                
                row_costs = []
                for curr_idx in range(num_states):
                    curr_perm = all_perms[curr_idx]
                    # 应用排列后的下一分区
                    permuted_partition_next = [original_partition_next[i] for i in curr_perm]
                    
                    # 计算 D_switch 矩阵
                    D_switch, _ = self._compute_switch_demand(
                        permuted_partition_current,
                        permuted_partition_next
                    )
                    
                    # 计算切换成本
                    # mapping_current 和 mapping_next 是逻辑 QPU 到物理 QPU 的映射
                    # prev_perm 和 curr_perm 本身就是映射（逻辑索引 -> 物理索引）
                    switch_cost = self._evaluate_switch_cost(
                        network,
                        D_switch,
                        list(prev_perm),
                        list(curr_perm)
                    )
                    row_costs.append(switch_cost)
                
                transition_costs.append(row_costs)
            
            teledata_costs.append(transition_costs)

        # ---------- 动态规划 ----------
        # dp[t][idx] 存储到时间片 t 状态 idx 的最小累计成本（元组：(fidelity_loss, num_comms)）
        dp: list[list[tuple[float, int]]] = [
            [(inf, -1) for _ in range(num_states)] for _ in range(k)
        ]
        # back[t][idx] 存储达到该状态的最优前一状态索引
        back: list[list[int]] = [[-1 for _ in range(num_states)] for _ in range(k)]

        # 初始化第一个时间片
        for idx in range(num_states):
            cost = telegate_costs[0][idx]
            dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)
            back[0][idx] = -1

        # 递推后续时间片
        for t in range(1, k):
            for curr_idx in range(num_states):
                curr_telegate = telegate_costs[t][curr_idx]
                curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                best_cost = (inf, 0)
                best_prev = -1
                for prev_idx in range(num_states):
                    prev_cost = dp[t-1][prev_idx]
                    teledata_cost = teledata_costs[t-1][prev_idx][curr_idx]   # 从 t-1 到 t 的转移成本
                    tel_tuple = (teledata_cost.total_fidelity_loss, teledata_cost.num_comms)
                    total = (prev_cost[0] + curr_telegate_tuple[0] + tel_tuple[0],
                             prev_cost[1] + curr_telegate_tuple[1] + tel_tuple[1])
                    if total < best_cost:
                        best_cost = total
                        best_prev = prev_idx
                dp[t][curr_idx] = best_cost
                back[t][curr_idx] = best_prev

        # ---------- 回溯找到最优路径 ----------
        best_last = min(range(num_states), key=lambda i: dp[k-1][i])
        perm_indices = [0] * k
        perm_indices[k-1] = best_last
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]

        # ---------- 更新映射记录 ----------
        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]
            original_partition = record.partition
            best_partition = [original_partition[idx] for idx in perm]
            record.partition = copy.deepcopy(best_partition)

            # 计算该时间片对应的成本对象
            record.costs = copy.deepcopy(telegate_costs[t][perm_indices[t]])
            if t != 0:
                record.costs += teledata_costs[t-1][perm_indices[t-1]][perm_indices[t]]

        end_time = time.time()
        print(f"[INFO] DP Mapper completed in {end_time - start_time:.2f} seconds.")
        return mapping_record_list


class HybridMapper(Mapper):
    """
    混合方法映射器：根据问题规模动态选择
    """
    
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "HybridMapper"

    def map(self,
            circuit,
            partition_plan: Any, 
            network: Any) -> dict[str, Any]:
        start_time = time.time()
        
        # 验证网络属性
        self._validate_network_attributes(network)

        k = len(partition_plan)  # 时间片数量
        m = len(partition_plan[0]) # 第一个时间步的逻辑QPU数量

        # TODO: 如果m较大，num_perms会非常大，需要限制或使用启发式方法

        
        # 生成所有可能的排列 (m!)
        all_perms = list(itertools.permutations(range(m)))
        num_perms = len(all_perms)

        # dp[t][perm_idx] = 从时间步0到时间步t-1的最大保真度和
        dp = [[-float('inf')] * num_perms for _ in range(k)]
        # 路径记录: path[t][perm_idx] = 前一个时间步的排列索引
        path = [[-1] * num_perms for _ in range(k)]
        
        # 初始化dp[0]：第一个时间步的fidelity loss
        # TODO: 假设
        for perm_idx, perm in enumerate(all_perms):
            dp[0][perm_idx] = evaluate_mapping(perm, partition_plan[0], network)

        end_time = time.time()
        return {
            "partition_plan": partition_plan,
            "execution_time (sec)": end_time - start_time
        }

    # 这里的evaluate mapping和真实的量子线路有关。
    # 实际上是要评估一段量子线路，在特定映射下的保真度损失。
    # 所以map函数的输入不能只是partition_plan，还需要量子线路的描述。


class MapperFactory:
    """映射器工厂类"""
    _registry = {
        "simple": "SimpleMapper",
        "link_oriented": "LinkOrientedMapper",
        "exact": "ExactOptimizationMapper",
        "greedy": "GreedyMapper"
    }
    
    @classmethod
    def create_mapper(cls, mapper_type: str) -> Mapper:
        """
        创建指定类型的映射器
        
        :param mapper_type: 映射器类型字符串
        :return: 对应的映射器实例
        """
        mapper_type = mapper_type.lower()
        if mapper_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown mapper: '{mapper_type}'. Available mappers: {available}")
        
        # 从注册表获取类名，然后创建实例
        mapper_class_name = cls._registry[mapper_type]
        mapper_class = globals()[mapper_class_name]
        return mapper_class()
    
    @classmethod
    def register_mapper(cls, name: str, class_name: str):
        """
        动态注册新的映射器类型
        
        :param name: 映射器类型名称
        :param class_name: 对应的类名字符串
        """
        cls._registry[name.lower()] = class_name
    
    @classmethod
    def unregister_mapper(cls, name: str):
        """
        移除注册的映射器类型
        
        :param name: 要移除的映射器类型名称
        """
        if name.lower() in cls._registry:
            del cls._registry[name.lower()]
    
    @classmethod
    def get_available_mappers(cls):
        """
        获取所有可用的映射器类型
        
        :return: 可用映射器类型列表
        """
        return list(cls._registry.keys())

from abc import ABC, abstractmethod
import time
import numpy as np
import itertools
from typing import Any, Optional, Tuple, List, Dict
import copy
from math import inf

from qiskit import QuantumCircuit

# 假设这些导入的模块/类已存在
from ..compiler import MappingRecordList, CompilerUtils, ExecCosts
from ..utils import Network

class Mapper(ABC):
    """
    量子线路映射器接口，定义所有映射算法必须实现的方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """获取映射器名称，用于标识和比较"""
        pass

    @abstractmethod
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件
        :param mapping_record_list: 映射记录列表（含各时间片分区）
        :param circuit: 量子电路
        :param circuit_layers: 分层后的电路门列表
        :param network: 目标量子硬件网络
        :return: 更新后的映射记录列表
        """
        pass

    def _compute_hop_demand(self,
                            partition: list[list[int]],
                            circuit: QuantumCircuit,
                            circuit_layers: list,
                            layer_start: int,
                            layer_end: int
        ) -> np.ndarray:
        """
        计算在原始排列下，每两个QPU之间的remote hops数量（支持多比特门相邻对）
        """
        m_logical = len(partition)
        D_hop = np.zeros((m_logical, m_logical), dtype=int)

        # 创建量子比特到逻辑 QPU 索引的快速查找表
        qubit_to_logical = {}
        for logical_idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_logical[qubit] = logical_idx

        # 遍历指定范围内的所有电路层
        for layer_idx in range(layer_start, layer_end):
            if layer_idx >= len(circuit_layers):  # 修复：添加边界检查，防止索引越界
                break
            layer = circuit_layers[layer_idx]

            # 遍历该层的所有门操作
            for node in layer:
                qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                assert all(q is not None for q in qubits), "量子比特索引获取失败"
                
                # 跳过单量子比特门（无跨QPU通信需求）
                if len(qubits) < 2:
                    continue

                # 生成相邻的量子比特对（按qargs顺序）
                for i in range(len(qubits) - 1):
                    q1 = qubits[i]
                    q2 = qubits[i + 1]
                    
                    # 防护：确保量子比特在分区中
                    if q1 not in qubit_to_logical or q2 not in qubit_to_logical:
                        raise ValueError(f"量子比特 {q1}/{q2} 未在分区中找到")
                    
                    # 统计跨QPU hops
                    qpu_a = qubit_to_logical[q1]
                    qpu_b = qubit_to_logical[q2]
                    if qpu_a != qpu_b:
                        D_hop[qpu_a][qpu_b] += 1
                        D_hop[qpu_b][qpu_a] += 1

        return D_hop

    def _compute_switch_demand(self, 
                               current_partition: list[list[int]], 
                               next_partition: list[list[int]]
        ) -> tuple[np.ndarray, dict]:
        """
        计算单次切换的通信需求
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
        D_switch = np.zeros((m_logical_current, m_logical_next), dtype=int)
        
        # 记录每个量子比特的移动
        qubit_movements = {}
        
        # 计算需求（确保遍历所有量子比特，避免遗漏）
        all_qubits = set(current_qubit_to_logical.keys()).union(set(next_qubit_to_logical.keys()))
        for qubit in all_qubits:
            if qubit not in current_qubit_to_logical or qubit not in next_qubit_to_logical:
                raise ValueError(f"量子比特 {qubit} 未在当前/下一分区中找到")
            
            curr_logical = current_qubit_to_logical[qubit]
            next_logical = next_qubit_to_logical[qubit]
            qubit_movements[qubit] = (curr_logical, next_logical)
            D_switch[curr_logical][next_logical] += 1

        return D_switch, qubit_movements

    def _evaluate_hop_cost(self, 
                           network: Network,
                           D_hops: np.ndarray,
                           perm: Tuple[int, ...]) -> ExecCosts:
        """
        计算单个时间片内 remote hops 导致的通信成本（telegate成本）
        :param D_hops: 逻辑QPU间的hop需求矩阵
        :param perm: 逻辑QPU到物理QPU的排列（映射）
        :return: 封装后的成本对象
        """
        self._validate_network_attributes(network)
        W_eff = network.W_eff  # 物理QPU间的有效保真度/权重矩阵
        
        total_fidelity_loss = 0.0
        num_comms = 0
        mapping_score = 0.0
        total_fidelity = 1.0

        # 遍历所有逻辑QPU对
        m_logical = D_hops.shape[0]
        for i in range(m_logical):
            for j in range(m_logical):
                hop_count = D_hops[i][j]
                if hop_count == 0:
                    continue
                
                # 映射到物理QPU
                physical_i = perm[i]
                physical_j = perm[j]
                
                # 防护：确保物理索引合法
                if physical_i >= len(W_eff) or physical_j >= len(W_eff):
                    raise IndexError(f"物理QPU索引 {physical_i}/{physical_j} 超出范围")
                
                # 计算成本
                w = W_eff[physical_i][physical_j]
                mapping_score += w * hop_count
                fidelity_loss = (1 - w) * hop_count
                total_fidelity_loss += fidelity_loss
                total_fidelity *= (w ** hop_count)
                num_comms += hop_count  # 通信次数 = hop总数

        return ExecCosts(
            total_fidelity_loss=total_fidelity_loss,
            num_comms=num_comms,
            mapping_score=mapping_score,
            fidelity=total_fidelity
        )

    def _evaluate_switch_cost(self, 
                              network: Network,
                              D_switch: np.ndarray, 
                              mapping_current: list[int], 
                              mapping_next: list[int]) -> ExecCosts:
        """
        计算相邻时间片切换的通信成本（teledata成本）
        :return: 封装后的成本对象
        """
        self._validate_network_attributes(network)
        W_eff = network.W_eff
        
        mapping_score = 0.0
        total_fidelity_loss = 0.0
        total_fidelity = 1.0
        num_comms = 0

        # 基于切换需求矩阵计算成本
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                switch_count = D_switch[i][j]
                if switch_count == 0:
                    continue
                
                # 映射到物理QPU
                physical_i = mapping_current[i]
                physical_j = mapping_next[j]
                
                # 防护：确保物理索引合法
                if physical_i >= len(W_eff) or physical_j >= len(W_eff):
                    raise IndexError(f"物理QPU索引 {physical_i}/{physical_j} 超出范围")
                
                # 计算切换成本
                w = W_eff[physical_i][physical_j]
                mapping_score += w * switch_count
                fidelity_loss = (1 - w) * switch_count
                total_fidelity_loss += fidelity_loss
                total_fidelity *= (w ** switch_count)
                num_comms += switch_count  # 切换通信次数 = 切换量子比特数

        return ExecCosts(
            total_fidelity_loss=total_fidelity_loss,
            num_comms=num_comms,
            mapping_score=mapping_score,
            fidelity=total_fidelity
        )

    def _validate_network_attributes(self, network: Network) -> None:
        """验证网络对象是否具有必要属性，防止运行时错误"""
        required_attrs = ['num_backends', 'W_eff']
        for attr in required_attrs:
            if not hasattr(network, attr):
                raise AttributeError(f"Network 对象缺少必要属性: {attr}")
        # 验证 W_eff 是二维矩阵
        if not isinstance(network.W_eff, (np.ndarray, list)) or len(np.shape(network.W_eff)) != 2:
            raise ValueError("Network.W_eff 必须是二维矩阵")


class LinkOrientedDPMapper(Mapper):
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Link-Ori DP Mapper"
    
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network
        ) -> MappingRecordList:
        """
        使用动态规划为每个时间片选择最优的物理QPU映射，最小化通信成本
        """
        start_time = time.time()
        
        # 核心参数初始化
        k = len(mapping_record_list.records)          # 时间片数量
        if k == 0:
            raise ValueError("映射记录列表为空，无时间片可处理")
        
        n_physical = network.num_backends             # 物理QPU数量
        if n_physical == 0:
            raise ValueError("网络中无可用的物理QPU")

        # 生成所有可能的物理QPU排列（逻辑QPU -> 物理QPU的映射）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)
        if num_states == 0:
            raise ValueError("无有效的物理QPU排列（n_physical可能为0）")

        # ---------- 预计算：各时间片的 telegate 成本 ----------
        telegate_costs: List[List[ExecCosts]] = []
        for t in range(k):
            record = mapping_record_list.records[t]
            original_partition = record.partition
            
            # 1. 计算当前时间片的 hop 需求矩阵
            D_hops = self._compute_hop_demand(
                original_partition,
                circuit,
                circuit_layers,
                record.layer_start,
                record.layer_end
            )
            
            # 2. 为每个排列计算 telegate 成本
            time_slice_costs = []
            for perm in all_perms:
                telegate_cost = self._evaluate_hop_cost(network, D_hops, perm)
                time_slice_costs.append(telegate_cost)
            
            telegate_costs.append(time_slice_costs)

        # ---------- 预计算：相邻时间片的 teledata 成本 ----------
        teledata_costs: List[List[List[ExecCosts]]] = []
        for t in range(k - 1):  # 遍历所有相邻时间片对
            record_current = mapping_record_list.records[t]
            record_next = mapping_record_list.records[t + 1]
            
            original_partition_current = record_current.partition
            original_partition_next = record_next.partition
            
            # 为当前转移的所有状态对计算成本
            transition_costs = []
            for prev_idx, prev_perm in enumerate(all_perms):
                # 应用排列后的当前分区
                permuted_partition_current = [original_partition_current[i] for i in prev_perm]
                
                row_costs = []
                for curr_idx, curr_perm in enumerate(all_perms):
                    # 应用排列后的下一分区
                    permuted_partition_next = [original_partition_next[i] for i in curr_perm]
                    
                    # 计算切换需求矩阵
                    D_switch, _ = self._compute_switch_demand(
                        permuted_partition_current,
                        permuted_partition_next
                    )
                    
                    # 计算切换成本
                    switch_cost = self._evaluate_switch_cost(
                        network,
                        D_switch,
                        list(prev_perm),
                        list(curr_perm)
                    )
                    row_costs.append(switch_cost)
                
                transition_costs.append(row_costs)
            
            teledata_costs.append(transition_costs)

        # ---------- 动态规划：寻找最优映射路径 ----------
        # dp[t][idx]：时间片t处于状态idx的最小累计成本（fidelity_loss, num_comms）
        dp = [[(inf, inf)] * num_states for _ in range(k)]
        # back[t][idx]：记录最优路径的前驱状态索引
        back = [[-1 for _ in range(num_states)] for _ in range(k)]

        # 初始化：第一个时间片无前驱成本
        for idx in range(num_states):
            cost = telegate_costs[0][idx]
            dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)

        # 递推：计算后续时间片的最优成本
        for t in range(1, k):
            for curr_idx in range(num_states):
                curr_telegate = telegate_costs[t][curr_idx]
                curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                
                # 寻找最优前驱状态
                best_cost = (inf, inf)
                best_prev_idx = -1
                
                for prev_idx in range(num_states):
                    # 前驱累计成本 + 当前telegate成本 + 转移成本
                    prev_total = dp[t-1][prev_idx]
                    transfer_cost = teledata_costs[t-1][prev_idx][curr_idx]
                    transfer_tuple = (transfer_cost.total_fidelity_loss, transfer_cost.num_comms)
                    
                    # 计算总累计成本
                    total_loss = prev_total[0] + curr_telegate_tuple[0] + transfer_tuple[0]
                    total_comms = prev_total[1] + curr_telegate_tuple[1] + transfer_tuple[1]
                    total_cost = (total_loss, total_comms)
                    
                    # 更新最优解（优先按保真度损失排序，其次通信次数）
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev_idx = prev_idx
                
                # 记录最优状态
                dp[t][curr_idx] = best_cost
                back[t][curr_idx] = best_prev_idx

        # ---------- 回溯：找到最优映射路径 ----------
        # 找到最后一个时间片的最优状态
        best_last_idx = min(range(num_states), key=lambda i: dp[k-1][i])
        perm_indices = [0] * k
        perm_indices[k-1] = best_last_idx
        
        # 反向推导最优路径
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]
            if perm_indices[t] == -1:
                raise RuntimeError(f"时间片 {t} 无有效前驱状态，路径回溯失败")

        # ---------- 更新映射记录 ----------
        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]
            original_partition = record.partition
            
            # 更新为最优排列后的分区
            best_partition = [original_partition[idx] for idx in perm]
            record.partition = copy.deepcopy(best_partition)
            
            # 更新成本对象
            record.costs = copy.deepcopy(telegate_costs[t][perm_indices[t]])
            # 累加前一时间片的切换成本（非第一个时间片）
            if t > 0:
                transfer_cost = teledata_costs[t-1][perm_indices[t-1]][perm_indices[t]]
                record.costs += transfer_cost

        # 输出耗时信息
        end_time = time.time()
        print(f"[INFO] Link-Oriented DP Mapper 完成，耗时 {end_time - start_time:.2f} 秒")
        
        return mapping_record_list