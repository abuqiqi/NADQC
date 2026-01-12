from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.optimize import linear_sum_assignment
import itertools

class PartitionAssigner(ABC):
    """
    量子线路划分分配器接口，定义所有划分分配算法必须实现的方法
    """

    @abstractmethod
    def assign_partitions(self, partition_candidates: Any) -> Dict[str, Any]:
        """
        处理量子线路划分计划
        :param partition_candidates: 子电路划分候选数据结构
        :return: 处理结果，包含调整后的划分计划和性能指标
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        获取分配器名称，用于标识和比较
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        获取处理性能指标
        :return: 包含关键性能指标的字典
        """
        pass


class BasePartitionAssigner(PartitionAssigner):
    """
    基础划分分配器类，提取公共方法和属性
    """
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.partition_candidates = None
    
    def get_metrics(self) -> Dict[str, float]:
        """获取处理性能指标"""
        return self.metrics

    def _validate_inputs(self):
        """验证输入参数的有效性"""
        if self.partition_candidates is None:
            raise ValueError("partition_candidates (partition plan data structure) cannot be None")

    def _extract_initial_partition_plan(self) -> List[List[List[int]]]:
        """
        从partition_candidates中提取初始划分计划
        :return: 划分计划，每个时间片包含逻辑QPU分配
        """
        partition_plan = []
        for i in range(len(self.partition_candidates)):
            partition_plan.append(self.partition_candidates[i][0])  # 仅选择第一个划分方案
        return partition_plan


class DirectPartitionAssigner(BasePartitionAssigner):
    """
    直接划分分配器：不进行任何调整，直接返回原始划分计划
    作为比较基准
    """

    def get_name(self) -> str:
        """获取分配器名称"""
        return "Direct Partition Assigner"

    def assign_partitions(self, partition_candidates: Any) -> Dict[str, Any]:
        """
        直接处理划分计划，不进行任何调整
        :param partition_candidates: 子电路划分候选数据结构
        :return: 处理结果
        """
        start_time = time.time()
        
        # 保存输入参数
        self.partition_candidates = partition_candidates
        
        # 验证输入
        self._validate_inputs()
        
        # 提取初始划分计划
        partition_plan = self._extract_initial_partition_plan()

        # TODO: 计算通信开销
        
        end_time = time.time()
        self.metrics = {
            "processing_time": end_time - start_time,
        }
        
        return {
            "partition_plan": partition_plan,
            "metrics": self.metrics
        }


class MaxMatchPartitionAssigner(BasePartitionAssigner):
    """
    最大匹配划分分配器：使用匈牙利算法在相邻时间片之间最大化逻辑QPU交集
    """

    def get_name(self) -> str:
        """获取分配器名称"""
        return "Max-Match Partition Assigner"

    def assign_partitions(self, partition_candidates: Any) -> Dict[str, Any]:
        """
        使用匈牙利算法优化相邻时间片之间的逻辑QPU匹配
        :param partition_candidates: 子电路划分候选数据结构
        :return: 处理结果
        """
        start_time = time.time()
        
        # 保存输入参数
        self.partition_candidates = partition_candidates
        
        # 验证输入
        self._validate_inputs()
        
        # 提取初始划分计划
        partition_plan = self._extract_initial_partition_plan()
        
        # 从第二个时间步开始，为每个时间步调整逻辑QPU的标签顺序
        for t in range(1, len(partition_plan)):
            # 获取当前时间步和前一个时间步的分区方案
            prev_assign = partition_plan[t-1]
            curr_assign = partition_plan[t]
            
            # 确保逻辑QPU数量匹配
            num_qpus = len(prev_assign)
            assert num_qpus == len(curr_assign), f"QPU count mismatch: {num_qpus} != {len(curr_assign)}"
            
            # 将分区方案转换为集合列表（每个集合代表一个逻辑QPU包含的量子比特）
            prev_scheme = [set(p) for p in prev_assign]
            curr_scheme = [set(p) for p in curr_assign]
            
            # 构建权重矩阵：权重 = 两个逻辑QPU之间的交集大小
            weights = [[len(prev_scheme[i] & curr_scheme[j]) for j in range(num_qpus)] for i in range(num_qpus)]
            
            # 使用匈牙利算法找到最大权匹配（最大化交集大小）
            prev_idx, curr_idx = linear_sum_assignment(weights, maximize=True)
            
            # 创建新的当前时间步分区方案，按匹配结果重新排序
            new_curr_assign = [None] * num_qpus
            for i in range(num_qpus):
                # 将当前时间步的逻辑QPU j 映射到新的逻辑QPU位置 curr_idx[i]
                new_curr_assign[curr_idx[i]] = curr_assign[prev_idx[i]]
            
            # 更新当前时间步的分区方案
            partition_plan[t] = new_curr_assign
        
        end_time = time.time()
        self.metrics = {
            "processing_time": end_time - start_time,
            "partition_plan_length": len(partition_plan),
            "num_qpus_per_time_slice": [len(time_slice) for time_slice in partition_plan] if partition_plan else [],
            "algorithm_used": "Hungarian Algorithm for Maximum Weight Matching"
        }
        
        return {
            "partition_plan": partition_plan,
            "metrics": self.metrics
        }


class GlobalMaxMatchPartitionAssigner(BasePartitionAssigner):
    """
    全局最大匹配划分分配器：使用动态规划计算全局最优的逻辑QPU顺序
    """

    def get_name(self) -> str:
        """获取分配器名称"""
        return "Global Max-Match Partition Assigner"

    def assign_partitions(self, partition_candidates: Any) -> Dict[str, Any]:
        """
        使用动态规划计算全局最优的逻辑QPU顺序
        :param partition_candidates: 子电路划分候选数据结构
        :return: 处理结果
        """
        start_time = time.time()
        
        # 保存输入参数
        self.partition_candidates = partition_candidates
        
        # 验证输入
        self._validate_inputs()
        
        # 提取初始划分计划
        partition_plan = self._extract_initial_partition_plan()
        
        k = len(partition_plan)
        if k < 2:
            return {
                "partition_plan": partition_plan,
                "metrics": {
                    "processing_time": time.time() - start_time,
                    "partition_plan_length": len(partition_plan),
                    "num_qpus_per_time_slice": [len(time_slice) for time_slice in partition_plan] if partition_plan else [],
                    "algorithm_used": "No optimization needed (only one time slice)"
                }
            }
        
        m = len(partition_plan[0])
        
        # 生成所有可能的排列 (m!)
        all_perms = list(itertools.permutations(range(m)))
        num_perms = len(all_perms)
        
        # 将排列映射为索引
        perm_to_index = {perm: idx for idx, perm in enumerate(all_perms)}
        
        # dp[t][perm_idx] = 从时间步0到时间步t-1的最大总交集和
        dp = [[-float('inf')] * num_perms for _ in range(k)]
        # 路径记录: path[t][perm_idx] = 前一个时间步的排列索引
        path = [[-1] * num_perms for _ in range(k)]
        
        # 初始化第一个时间步（没有前驱，交集和为0）
        for idx in range(num_perms):
            dp[0][idx] = 0
        
        # 动态规划：从时间步1开始
        for t in range(1, k):
            for curr_idx in range(num_perms):
                curr_perm = all_perms[curr_idx]
                for prev_idx in range(num_perms):
                    prev_perm = all_perms[prev_idx]
                    
                    # 计算从时间步t-1（顺序prev_perm）到时间步t（顺序curr_perm）的交集和
                    total_intersection = 0
                    for i in range(m):
                        set1 = set(partition_plan[t-1][prev_perm[i]])
                        set2 = set(partition_plan[t][curr_perm[i]])
                        total_intersection += len(set1 & set2)
                    
                    # 更新dp[t][curr_idx]
                    if dp[t-1][prev_idx] + total_intersection > dp[t][curr_idx]:
                        dp[t][curr_idx] = dp[t-1][prev_idx] + total_intersection
                        path[t][curr_idx] = prev_idx
        
        # 找到最后一个时间步的最优排列
        best_last_idx = np.argmax(dp[k-1])
        
        # 回溯路径
        orders = [None] * k
        orders[k-1] = all_perms[best_last_idx]
        
        curr_idx = best_last_idx
        for t in range(k-1, 0, -1):
            prev_idx = path[t][curr_idx]
            orders[t-1] = all_perms[prev_idx]
            curr_idx = prev_idx
        
        # 确保第一个时间步的顺序为[0,1,...,m-1]（因为第一个时间步没有前驱，任意顺序都行）
        orders[0] = list(range(m))
        
        # 创建调整后的分区计划
        adjusted_plan = []
        for t in range(len(partition_plan)):
            curr_assign = partition_plan[t]
            order = orders[t]
            # 按照最优顺序重新排列逻辑QPU
            adjusted_time_step = [curr_assign[i] for i in order]
            adjusted_plan.append(adjusted_time_step)
        
        end_time = time.time()
        self.metrics = {
            "processing_time": end_time - start_time,
            "partition_plan_length": len(adjusted_plan),
            "num_qpus_per_time_slice": [len(time_slice) for time_slice in adjusted_plan] if adjusted_plan else [],
            "algorithm_used": "Dynamic Programming for Global Maximum Weight Matching",
            "num_permutations_considered": num_perms,
            "number_of_time_slices": k
        }
        
        return {
            "partition_plan": adjusted_plan,
            "metrics": self.metrics
        }


# 示例使用
if __name__ == "__main__":
    # 创建示例数据（模拟P和subc_ranges）
    class MockDataStructure:
        def __init__(self):
            # 模拟P数据结构
            # P[i][j][0]表示时间片i到j的划分方案
            self.partition_candidates = [
                [[[0, 1], [2, 3]]],  # 时间片0-0的划分方案
                [[[0, 2], [1, 3]]],  # 时间片1-1的划分方案
                [[[0, 3], [1, 2]]]   # 时间片2-2的划分方案
            ] * 3  # 扩展为3个时间段
    
    mock_data = MockDataStructure()
    
    # 创建不同的分配器实例
    assigners = [
        DirectPartitionAssigner(),
        MaxMatchPartitionAssigner(),
        GlobalMaxMatchPartitionAssigner()
    ]
    
    print("Testing different partition assigners:\n")
    
    for assigner in assigners:
        print(f"Processing with {assigner.get_name()}:")
        result = assigner.assign_partitions(mock_data.partition_candidates)
        print(f"Partition Plan: {result['partition_plan']}")
        print(f"Metrics: {result['metrics']}")
        print("-" * 50)