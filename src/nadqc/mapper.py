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
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit_layers: list[Any],
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果
        """
        pass

    # def _compute_switch_demand(self, 
    #                            current_partition: list[list[int]], 
    #                            next_partition: list[list[int]]) -> tuple[np.ndarray, dict]:
    #     """
    #     计算单次切换的通信需求
    #     :param current_partition: 当前时间片的分区
    #     :param next_partition: 下一时间片的分区
    #     :return: (D_switch, qubit_movements)
    #     """
    #     # 创建量子比特到逻辑QPU的映射
    #     current_qubit_to_logical = {}
    #     for logical_idx, group in enumerate(current_partition):
    #         for qubit in group:
    #             current_qubit_to_logical[qubit] = logical_idx
        
    #     next_qubit_to_logical = {}
    #     for logical_idx, group in enumerate(next_partition):
    #         for qubit in group:
    #             next_qubit_to_logical[qubit] = logical_idx
        
    #     # 初始化需求矩阵
    #     m_logical_current = len(current_partition)
    #     m_logical_next = len(next_partition)
    #     D_switch = np.zeros((m_logical_current, m_logical_next))
        
    #     # 记录每个量子比特的移动
    #     qubit_movements = {}
        
    #     # 计算需求
    #     num_qubits = len(current_qubit_to_logical)
    #     for qubit in range(num_qubits):
    #         curr_logical = current_qubit_to_logical[qubit]
    #         next_logical = next_qubit_to_logical[qubit]
    #         qubit_movements[qubit] = (curr_logical, next_logical)
    #         D_switch[curr_logical][next_logical] += 1

    #     return D_switch, qubit_movements

    # def _evaluate_switch_cost(self, 
    #                           network: Network,
    #                           D_switch: np.ndarray, 
    #                           mapping_current: list[int], 
    #                           mapping_next: list[int]) -> dict[str, float]:
    #     """
    #     计算切换成本
    #     :param D_switch: 通信需求矩阵
    #     :param mapping_current: 当前映射
    #     :param mapping_next: 下一映射
    #     :return: (switch_cost, switch_fidelity)
    #     """
    #     W_eff = network.W_eff
        
    #     mapping_score = 0.0
    #     fidelity_loss = 0.0
    #     fidelity = 1.0

    #     # 基于需求矩阵计算（可选）
    #     for i in range(D_switch.shape[0]):
    #         for j in range(D_switch.shape[1]):
    #             demand = D_switch[i][j]
    #             if demand > 0:
    #                 from_physical = mapping_current[i]
    #                 to_physical = mapping_next[j]
    #                 mapping_score += W_eff[from_physical][to_physical] * demand
    #                 fidelity_loss += (1 - W_eff[from_physical][to_physical]) * demand
    #                 fidelity *= W_eff[from_physical][to_physical] ** demand

    #     return {
    #         "mapping_score": mapping_score,
    #         "fidelity_loss": fidelity_loss,
    #         "fidelity": fidelity
    #     }
    
    # def _evaluate_initial_mapping(self, network: Network, mapping: list[int], D_total: np.ndarray) -> float:
    #     """
    #     评估初始映射的质量
    #     :param mapping: 映射
    #     :param D_total: 总通信需求矩阵
    #     :return: 映射得分
    #     """
    #     W_eff = network.W_eff
    #     # pprint(W_eff)
    #     mapping_score = 0.0
    #     for i in range(D_total.shape[0]):
    #         for j in range(D_total.shape[1]):
    #             from_physical = mapping[i]
    #             to_physical = mapping[j]
    #             mapping_score += W_eff[from_physical][to_physical] * D_total[i][j]
    #     return mapping_score

    # def _validate_network_attributes(self, network):
    #     """验证网络对象是否具有必要属性"""
    #     if not hasattr(network, 'num_backends') or not hasattr(network, 'W_eff'):
    #         raise AttributeError("Network must have 'num_backends' and 'W_eff' attributes")

    # def _build_partition_plan(self, partition_plan, mapping_sequence):
        
    #     # 根据映射序列更新partition_plan
    #     adjusted_partition_plan = []

    #     for t, mapping in enumerate(mapping_sequence):
    #         current_partition = partition_plan[t]
            
    #         # 构建新的分区
    #         new_partition = [[] for _ in range(len(current_partition))]

    #         for logical_idx, group in enumerate(current_partition):
    #             physical_idx = mapping[logical_idx]
    #             new_partition[physical_idx].extend(group)
            
    #         adjusted_partition_plan.append(new_partition)

    #     return adjusted_partition_plan


class DirectMapper(Mapper):
    """
    基线映射器：直接使用输入的逻辑QPU顺序（逻辑QPU i -> 物理QPU i）
    作为比较基准，不进行任何优化
    """

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Direct Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit_layers: list[Any],
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件（基线实现）
        """
        return mapping_record_list


class GreedyMapper(Mapper):

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Greedy Mapper"
    
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit_layers: list[Any],
            network: Network) -> MappingRecordList:
        
        # 对于每一段子线路，我们尝试找到一个局部最优的映射

        # 初始线路，评估它到目标网络的保真度损失

        # 后续的，评估每个子线路在当前映射下的保真度损失，并尝试通过局部调整映射来减少损失

        return mapping_record_list

# class HybridMapper(Mapper):
#     """
#     混合方法映射器：根据问题规模动态选择
#     """
    
#     @property
#     def name(self) -> str:
#         """获取映射器名称"""
#         return "HybridMapper"

#     def map(self,
#             mapping_record_list: MappingRecordList,
#             circuit_layers: list[Any],
#             network: Network) -> MappingRecordList:
#         start_time = time.time()
        
#         # 验证网络属性
#         self._validate_network_attributes(network)

#         k = len(partition_plan)  # 时间片数量
#         m = len(partition_plan[0]) # 第一个时间步的逻辑QPU数量

#         # TODO: 如果m较大，num_perms会非常大，需要限制或使用启发式方法

        
#         # 生成所有可能的排列 (m!)
#         all_perms = list(itertools.permutations(range(m)))
#         num_perms = len(all_perms)

#         # dp[t][perm_idx] = 从时间步0到时间步t-1的最大保真度和
#         dp = [[-float('inf')] * num_perms for _ in range(k)]
#         # 路径记录: path[t][perm_idx] = 前一个时间步的排列索引
#         path = [[-1] * num_perms for _ in range(k)]
        
#         # 初始化dp[0]：第一个时间步的fidelity loss
#         # TODO: 假设
#         for perm_idx, perm in enumerate(all_perms):
#             dp[0][perm_idx] = evaluate_mapping(perm, partition_plan[0], network)

#         end_time = time.time()
#         return {
#             "partition_plan": partition_plan,
#             "execution_time (sec)": end_time - start_time
#         }

#     # 这里的evaluate mapping和真实的量子线路有关。
#     # 实际上是要评估一段量子线路，在特定映射下的保真度损失。
#     # 所以map函数的输入不能只是partition_plan，还需要量子线路的描述。


class MapperFactory:
    """映射器工厂类"""
    _registry = {
        "direct": "DirectMapper",
        # "link_oriented": "LinkOrientedMapper",
        # "exact": "ExactOptimizationMapper",
        # "greedy": "GreedyMapper"
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

