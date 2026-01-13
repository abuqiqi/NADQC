from abc import ABC, abstractmethod
import copy
from typing import List, Any, Dict


class Partitioner(ABC):
    """
    量子电路分区器接口，定义所有分区算法必须实现的方法
    """
    
    @abstractmethod
    def partition(self, components: List[List[int]]) -> List[List[List[int]]]:
        """
        将量子电路组件分配到不同QPU上
        :param components: 量子电路组件列表，每个组件是一个量子比特列表
        :return: 分区结果列表，每个元素代表一个可能的分区方案
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        获取分区器名称，用于标识和比较
        """
        pass


class PartitionerFactory:
    """分区器工厂类"""
    _registry = {
        "greedy": "GreedyPartitioner",
        "dynamic_programming": "DynamicProgrammingPartitioner", 
        "recursive_dp": "RecursiveDynamicProgrammingPartitioner"
    }
    
    @classmethod
    def create_partitioner(cls, partitioner_type: str, network, max_options: int = 1) -> Partitioner:
        partitioner_type = partitioner_type.lower()
        if partitioner_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown partitioner: {partitioner_type}, available: {available}")
        
        # 从注册表获取类名，然后创建实例
        partitioner_class_name = cls._registry[partitioner_type]
        partitioner_class = globals()[partitioner_class_name]
        return partitioner_class(network, max_options)
    
    @classmethod
    def register_partitioner(cls, name: str, class_name: str):
        """动态注册新的分区器类型"""
        cls._registry[name] = class_name
    
    @classmethod
    def unregister_partitioner(cls, name: str):
        """移除注册的分区器类型"""
        if name in cls._registry:
            del cls._registry[name]
    
    @classmethod
    def get_available_partitioners(cls):
        """获取所有可用的分区器类型"""
        return list(cls._registry.keys())


class BasePartitioner(Partitioner):
    """
    基础K路分区器，提取公共方法和属性
    """
    
    def __init__(self, network: Any, max_options: int = 1):
        self.qpus = network.get_backend_qubit_counts()
        self.max_options = max_options
        self.metrics = {}

    def get_metrics(self) -> Dict[str, float]:
        """获取分区性能指标"""
        return self.metrics

    def get_name(self) -> str:
        """获取分区器名称"""
        return "Base K-Way Partitioner"

    def _group_components_to(self, target_size: int, component_sizes: List[int]):
        """
        找出component_sizes中的元素的组合，使得其和等于target_size
        component_sizes: list[int]，每个连通分量的大小
        target_size: int，划分的大小
        """
        # QPU的容量是self.qpus[qpu_idx]
        # dp[i][j] := 前i个components，组合成含j个qubit的合法划分数
        dp = [[0]*(target_size + 1) for _ in range(len(component_sizes))]
        dp[0][0] = 1
        if component_sizes[0] <= target_size:
            dp[0][component_sizes[0]] = 1
        for i in range(1, len(component_sizes)):
            dp[i][0] = 1
            for j in range(1, target_size + 1):
                if component_sizes[i] <= j:
                    dp[i][j] = dp[i-1][j-component_sizes[i]]
                dp[i][j] = dp[i][j] or dp[i-1][j]
        return dp

    def _find_target(self, dp: List[List[int]], num_qubits: int, QPU0: int, QPU1: int) -> int:
        """
        针对QPU0的容量，找到可行的target值，尽可能多放
        """
        lowerbound = max(0, num_qubits - QPU1)  # TODO：检查，可以不分配给QPU0
        j = QPU0
        while j >= lowerbound:
            if dp[-1][j] == 1:
                return j
            j -= 1
        return -1

    def _trace(self, dp: List[List[int]], components: List[List[int]], i: int, j: int, legal_partitions: List[List[List[int]]]):
        def add(component, legal_partitions):
            for partition in legal_partitions:
                partition.extend(component)
            return legal_partitions

        legal_partitions = copy.deepcopy(legal_partitions)
        assert dp[i][j] != 0, "[ERROR] Invalid trace."
        if j == 0:
            return legal_partitions
        if i == 0:
            legal_partitions = add(components[0], legal_partitions)
            return legal_partitions
        L0, L1 = [[]], [[]]
        # 考虑是否加入第i个component
        if dp[i-1][j] == 1:  # 不加第i个component
            L0 = self._trace(dp, components, i-1, j, legal_partitions)
        # TODO: 如果legal_partitions的长度到了self.max_options，不再添加新的分支
        if len(L0) >= self.max_options and len(L0[0]) > 0:
            return L0
        # [NOTE] to speedup the trace, we only backtrack one branch
        elif len(components[i]) <= j and dp[i-1][j-len(components[i])] == 1:  # 加第i个component
            L1 = self._trace(dp, components, i-1, j-len(components[i]), legal_partitions)
            L1 = add(components[i], L1)
        if L0 == [[]]:
            return L1
        if L1 == [[]]:
            return L0
        return L0 + L1

    def _sort_partitions(self, legal_partitions: List[List[List[int]]]) -> List[List[List[int]]]:
        # legal_partitions是一个二维数组，代表多种可以放在QPU0上的合法划分
        for part in legal_partitions:
            part.sort()
        sorted_partitions = sorted(legal_partitions, key=lambda x: ''.join(map(str, x)))
        return sorted_partitions

    def _get_remaining_components(self, components: List[List[int]], partition: List[int]) -> List[List[int]]:
        """
        从components中删除出现在partition里的qubit，返回新的components
        """
        remaining_components = []
        partition_flat = set(partition)
        for comp in components:
            remaining_comp = [q for q in comp if q not in partition_flat]
            if remaining_comp:  # 
                remaining_components.append(remaining_comp)
        return remaining_components


class GreedyPartitioner(BasePartitioner):
    """
    贪心分区器：按照贪心策略分配量子电路组件到QPU
    """
    
    def get_name(self) -> str:
        return "Greedy Partitioner"

    def partition(self, components: List[List[int]]) -> List[List[List[int]]]:
        """
        使用贪心策略进行分区
        """
        partition = [[] for _ in range(len(self.qpus))]
        capacities = self.qpus.copy()  # 每个QPU的剩余容量
        
        for com in components:  # 放置第i个components
            placed = False
            for idx, qpu in enumerate(partition):  # 尝试放到第idx个QPU上
                if len(com) <= capacities[idx]:
                    qpu.extend(com)
                    capacities[idx] -= len(com)
                    placed = True
                    break
            if not placed:  # 无法完整容纳com，将com拆分开放进不同的分区
                temp_com = com.copy()  # 避免修改原始数据
                while temp_com:  # 循环拆分直到组件分配完毕
                    max_idx = capacities.index(max(capacities))  # 剩余容量最大的QPU
                    put_num = min(capacities[max_idx], len(temp_com))  # 计算本次可放入的数量
                    partition[max_idx].extend(temp_com[:put_num])
                    capacities[max_idx] -= put_num
                    temp_com = temp_com[put_num:]  # 剩余未分配的组件部分
        
        return [partition]


class DynamicProgrammingPartitioner(BasePartitioner):
    """
    动态规划分区器：使用DP算法寻找较优的分区方案
    """
    
    def get_name(self) -> str:
        return "Dynamic Programming Partitioner"

    def partition(self, components: List[List[int]]) -> List[List[List[int]]]:
        """
        使用动态规划进行分区
        """
        legal_partition = []  # 一组合法划分
        temp_components = copy.deepcopy(components)  # 避免修改原始数据
        
        for qpu_idx in range(len(self.qpus)):  # 处理第qpu_idx个QPU
            # 更新component_sizes
            component_sizes = [len(comp) for comp in temp_components]
            # 计算components是否可以组成大小不超过当前QPU容量的分区
            dp = self._group_components_to(self.qpus[qpu_idx], component_sizes)
            # 计算剩余QPU的容量
            remaining_qpu_capacity = sum(self.qpus[qpu_idx+1:])
            # 计算合适的分区大小
            target = self._find_target(dp, sum(component_sizes), self.qpus[qpu_idx], remaining_qpu_capacity)
            # 如果没有成功分离出适合第qpu_idx个QPU的划分，返回空
            if target < 0:
                return []
            # 如果成功分离了，通过trace获取可能的分区
            curr_legal_partitions = self._trace(dp, temp_components, len(temp_components)-1, target, [[]])
            # 选取字典序最小的划分
            # TODO：选取与QPU[qpu_idx]的上一个划分差距最小的划分
            sorted_curr_legal_partitions = self._sort_partitions(curr_legal_partitions)
            curr_partition = sorted_curr_legal_partitions[0]
            legal_partition.append(curr_partition)  # 加入当前QPU上的划分情况
            # 后处理，从temp_components中删除partition里的qubit
            temp_components = self._get_remaining_components(temp_components, curr_partition)
        
        return [legal_partition]


class RecursiveDynamicProgrammingPartitioner(BasePartitioner):
    """
    递归动态规划分区器：使用递归和DP生成多种分区方案
    """
    
    def get_name(self) -> str:
        return "Recursive Dynamic Programming Partitioner"

    def partition(self, components: List[List[int]]) -> List[List[List[int]]]:
        """
        利用递归和动态规划，返回多种可行的划分
        """
        self.legal_partitions = []
        temp_components = copy.deepcopy(components)  # 避免修改原始数据
        self._recursive_partition_helper([], temp_components, 0)
        return self.legal_partitions[:self.max_options]

    def _recursive_partition_helper(self, current_partition: List[List[int]], components: List[List[int]], qpu_idx: int):
        """
        递归辅助函数：
        - current_partition: 当前已经分配的划分（每个QPU对应一个划分）
        - components: 可用的组件
        - qpu_idx: 当前处理的QPU索引
        """
        assert len(current_partition) == qpu_idx, f"[ERROR] current_partition {current_partition} length mismatch qpu[{qpu_idx}]"
        if self.max_options > 1:
            current_partition = copy.deepcopy(current_partition)

        # 终止条件：所有QPU都已分配
        if qpu_idx >= len(self.qpus):
            self.legal_partitions.append(current_partition.copy())
            return
        
        # 终止条件：所有components都已分配完
        if len(components) == 0:
            for _ in range(qpu_idx, len(self.qpus)):
                current_partition.append([])
            assert len(current_partition) == len(self.qpus), f"[ERROR] Partition {current_partition} length mismatch. qpu[{qpu_idx}]"
            self.legal_partitions.append(current_partition.copy())
            return

        # 计算当前QPU的容量
        qpu_capacity = self.qpus[qpu_idx]
        component_sizes = [len(comp) for comp in components]
        total_remaining = sum(component_sizes)
        # 计算剩余QPU的总容量（用于剪枝）
        remaining_qpu_capacity = sum(self.qpus[qpu_idx+1:])

        # 如果剩余组件无法放入剩余QPU，剪枝
        if total_remaining > remaining_qpu_capacity + qpu_capacity:
            return

        # 动态规划计算当前QPU的可能划分
        dp = self._group_components_to(qpu_capacity, component_sizes)
        target = self._find_target(dp, total_remaining, qpu_capacity, remaining_qpu_capacity)
        # 如果没有合法划分，终止
        if target < 0:
            return
        # 获取对QPU[qpu_idx]的所有可能的划分
        possible_partitions = self._trace(dp, components, len(components)-1, target, [[]])
        possible_partitions = self._sort_partitions(possible_partitions)

        # 对每一种可能的划分，递归处理剩余QPU
        for cnt, partition in enumerate(possible_partitions):
            # 限制回溯的个数
            if cnt == self.max_options:
                break
            # 计算剩余组件
            remaining_components = self._get_remaining_components(components, partition)
            # 更新当前划分
            current_partition.append(partition)
            assert len(current_partition) == qpu_idx + 1, f"[ERROR] current partition {current_partition} length mismatch qpu[{qpu_idx}]"
            # 递归处理下一个QPU
            self._recursive_partition_helper(current_partition, remaining_components, qpu_idx + 1)
            # 回溯，尝试其他划分
            current_partition.pop()


# class NoiseAwarePartitioner(BasePartitioner):
#     """
#     噪声感知分区器：考虑量子硬件噪声特性的分区算法
#     """
    
#     def __init__(self, network, max_options: int = 1, noise_model=None):
#         super().__init__(network, max_options)
#         self.noise_model = noise_model  # 假设noise_model包含QPU的噪声信息

#     def get_name(self) -> str:
#         return "Noise-Aware Partitioner"

#     def _calculate_component_weights(self, components: List[List[int]], interaction_graph: Any = None):
#         """
#         计算组件权重，考虑噪声和交互强度
#         """
#         if interaction_graph is not None:
#             # 如果提供了交互图，则根据边权重计算组件重要性
#             self.component_weights = []
#             for component in components:
#                 weight = 0
#                 component_set = set(component)

#                 # 计算分量内部边的权重和
#                 for i, qubit1 in enumerate(component):
#                     for qubit2 in component[i+1:]:
#                         if interaction_graph.has_edge(qubit1, qubit2):
#                             weight += interaction_graph[qubit1][qubit2].get('weight', 1)
                
#                 total_weight = weight
#                 self.component_weights.append(total_weight)
#         else:
#             # 默认按组件大小分配权重
#             self.component_weights = [len(comp) for comp in components]

#     def partition(self, components: List[List[int]], method: str = "greedy", interaction_graph: Any = None) -> List[List[List[int]]]:
#         """
#         使用噪声感知策略进行分区
#         """
#         self._calculate_component_weights(components, interaction_graph)

#         # 按权重降序排序连通分量（重要分量优先分配）
#         weighted_components = sorted(enumerate(zip(self.component_weights, components)), 
#                                      key=lambda x: x[1][0], reverse=True)
        
#         # 按质量降序排序QPU（质量高的QPU放在前面）
#         # 这里假设我们有一个简单的质量评估方式，实际应用中可以根据噪声模型调整
#         if self.noise_model:
#             qpu_qualities = [(self.noise_model.get_quality_score(qpu_idx), qpu_idx)
#                              for qpu_idx in range(len(self.qpus))]
#             qpu_qualities = sorted(qpu_qualities, key=lambda x: x[0], reverse=True)
#         else:
#             # 如果没有噪声模型，按容量排序
#             qpu_qualities = [(capacity, idx) for idx, capacity in enumerate(self.qpus)]
#             qpu_qualities = sorted(qpu_qualities, key=lambda x: x[0], reverse=True)

#         partition = [[] for _ in range(len(self.qpus))]
#         capacities = self.qpus.copy()

#         for idx, (weight, com) in weighted_components:
#             best_qpu = -1
            
#             # 为当前分量寻找最佳QPU
#             for _, qpu_idx in qpu_qualities:
#                 if len(com) <= capacities[qpu_idx]:
#                     best_qpu = qpu_idx
#                     break

#             if best_qpu != -1:
#                 partition[best_qpu].extend(com)
#                 capacities[best_qpu] -= len(com)
#             else:
#                 # 无法完整容纳，拆分处理（尽量保持重要部分在一起）
#                 remaining_com = com.copy()
#                 # 从质量高的QPU开始，逐一放满
#                 for quality, qpu_idx in qpu_qualities:
#                     if len(remaining_com) == 0:
#                         break
#                     if capacities[qpu_idx] > 0:  # 还有余地
#                         put_num = min(capacities[qpu_idx], len(remaining_com))
#                         partition[qpu_idx].extend(remaining_com[:put_num])
#                         capacities[qpu_idx] -= put_num
#                         remaining_com = remaining_com[put_num:]
#                         break
#                 assert len(remaining_com) == 0

#         return [partition]

