import networkx as nx

class BaseKWayPartitioner:
    """基础分区器（不处理噪声）"""
    def __init__(self, qpus, max_options):
        self.qpus = qpus
        self.max_options = max_options
    
    def partition(self, interaction_graph, method):
        """基础分区方法"""
        self.components = [list(comp) for comp in nx.connected_components(interaction_graph)]
        if method == "greedy":
            return self.partition_greedy()
        elif method == "recursive":
            return self.partition_recursive_dp()
        return self.partition_dp()
    
    def partition_greedy(self):
        """
        贪心划分方法，只会返回一种partition
        """
        partition = [[] for _ in range(len(self.qpus))]
        capacities = self.qpus.copy() # 每个QPU的剩余容量
        for com in self.components: # 放置第i个components
            placed = False
            for idx, qpu in enumerate(partition): # 尝试放到第idx个QPU上
                if len(com) <= capacities[idx]:
                    qpu.extend(com)
                    capacities[idx] -= len(com)
                    placed = True
                    break
            if not placed: # 无法完整容纳com，将com拆分开放进不同的分区
                while com:  # 循环拆分直到组件分配完毕
                    max_idx = capacities.index(max(capacities))  # 剩余容量最大的QPU
                    put_num = min(capacities[max_idx], len(com)) # 计算本次可放入的数量
                    partition[max_idx].extend(com[:put_num])
                    capacities[max_idx] -= put_num
                    com = com[put_num:]  # 剩余未分配的组件部分
        return [partition]

    def partition_dp(self):
        """
        动态规划划分方法，返回一种可行的划分
        比贪心方法多一个排序的优势
        """
        legal_partition = [] # 一组合法划分
        for qpu_idx in range(len(self.qpus)): # 处理第qpu_idx个QPU
            # 更新component_sizes
            self.component_sizes = [len(comp) for comp in self.components]
            # 计算components是否可以组成大小不超过当前QPU容量的分区
            dp = self.group_components_to(self.qpus[qpu_idx], self.component_sizes)
            # 计算剩余QPU的容量
            remaining_qpu_capacity = sum(self.qpus[qpu_idx+1:])
            # 计算合适的分区大小
            target = self.find_target(dp, sum(self.component_sizes), self.qpus[qpu_idx], remaining_qpu_capacity)
            # 如果没有成功分离出适合第qpu_idx个QPU的划分，返回空
            if target < 0:
                return []
            # 如果成功分离了，通过trace获取可能的分区
            curr_legal_partitions = self.trace(dp, self.components, len(self.components)-1, target, [[]])
            # print(f"curr_legal_partitions: {curr_legal_partitions}")
            # 选取字典序最小的划分
            # TODO：选取与QPU[qpu_idx]的上一个划分差距最小的划分
            sorted_curr_legal_partitions = self.sort_partitions(curr_legal_partitions)
            curr_partition = sorted_curr_legal_partitions[0]
            legal_partition.append(curr_partition) # 加入当前QPU上的划分情况
            # 后处理，从self.components中删除partition里的qubit
            self.components = self.get_remaining_components(self.components, curr_partition)
        return [legal_partition]

    def partition_recursive_dp(self):
        """
        利用递归和动态规划，返回多种可行的划分
        """
        self.legal_partitions = []
        self.recursive_partition_helper([], self.components, 0)
        # print(f"========== [DEBUG] FINAL PARTITIONS: ")
        # print(self.legal_partitions)
        # num_options = len(self.legal_partitions)
        # if num_options > 1:
        #     print(num_options)
        # print(f"[DEBUG] len(legal_partitions): {len(self.legal_partitions)}")
        return self.legal_partitions[:self.max_options]

    def recursive_partition_helper(self, current_partition, components, qpu_idx):
        """
        递归辅助函数：
        - current_partition: 当前已经分配的划分（每个QPU对应一个划分）
        - components: 可用的组件
        - qpu_idx: 当前处理的QPU索引
        """
        assert len(current_partition) == qpu_idx, f"[ERROR] current_partition {current_partition} length mismatch qpu[{qpu_idx}]"
        if self.max_options > 1:
            current_partition = copy.deepcopy(current_partition)

        # print(f"[DEBUG] call recursive_helper: curr_p: {current_partition}, re_comp: {components}, qpu_idx: {qpu_idx}")
        # 终止条件：所有QPU都已分配
        if qpu_idx >= len(self.qpus):
            # print(f"[recursive_helper] find partition: ")
            # print(current_partition)
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
        dp = self.group_components_to(qpu_capacity, component_sizes)
        target = self.find_target(dp, total_remaining, qpu_capacity, remaining_qpu_capacity)
        # 如果没有合法划分，终止
        if target < 0:
            return
        # 获取对QPU[qpu_idx]的所有可能的划分
        possible_partitions = self.trace(dp, components, len(components)-1, target, [[]])
        possible_partitions = self.sort_partitions(possible_partitions)
        # print(f"[DEBUG] [recursive_helper] possible_partitions: {possible_partitions}")

        # 对每一种可能的划分，递归处理剩余QPU
        for cnt, partition in enumerate(possible_partitions):
            # 限制回溯的个数
            if cnt == self.max_options:
                break
            # 计算剩余组件
            remaining_components = self.get_remaining_components(components, partition)
            # 更新当前划分
            current_partition.append(partition)
            assert len(current_partition) == qpu_idx + 1, f"[ERROR] current partition {current_partition} length mismatch qpu[{qpu_idx}]"
            # 递归处理下一个QPU
            self.recursive_partition_helper(current_partition, remaining_components, qpu_idx + 1)
            # 回溯，尝试其他划分
            current_partition.pop()
        return
    
    def group_components_to(self, target_size, component_sizes):
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

    def find_target(self, dp, num_qubits, QPU0, QPU1):
        """
        针对QPU0的容量，找到可行的target值，尽可能多放
        """
        # print("[DEBUG] find_target")
        lowerbound = max(0, num_qubits - QPU1) # TODO：检查，可以不分配给QPU0
        j = QPU0
        while j >= lowerbound:
            if dp[-1][j] == 1:
                return j
            j -= 1
        return -1

    def trace(self, dp, components, i, j, legal_partitions):
        def add(component, legal_partitions):
            for partition in legal_partitions:
                partition.extend(component)
            return legal_partitions

        legal_partitions = copy.deepcopy(legal_partitions)
        assert dp[i][j] != 0, "[ERROR] Invalid trace."
        # if dp[i][j] == 0:
        #     return legal_partitions
        if j == 0:
            return legal_partitions
        if i == 0:
            legal_partitions = add(components[0], legal_partitions)
            return legal_partitions
        L0, L1=[[]], [[]]
        # 考虑是否加入第i个component
        if dp[i-1][j] == 1: # 不加第i个component
            L0 = self.trace(dp, components, i-1, j, legal_partitions)
        # TODO: 如果legal_partitions的长度到了self.max_options，不再添加新的分支
        if len(L0) >= self.max_options and len(L0[0]) > 0:
            return L0
        # [NOTE] to speedup the trace, we only backtrack one branch
        elif len(components[i]) <= j and dp[i-1][j-len(components[i])] == 1: # 加第i个component
            L1 = self.trace(dp, components, i-1, j-len(components[i]), legal_partitions)
            L1 = add(components[i], L1)
        if L0 == [[]]:
            return L1
        if L1 == [[]]:
            return L0
        return L0 + L1

    def sort_partitions(self, legal_partitions):
        # legal_partitions是一个二维数组，代表多种可以放在QPU0上的合法划分
        for part in legal_partitions:
            part.sort()
        sorted_partitions = sorted(legal_partitions, key=lambda x: ''.join(map(str, x)))
        return sorted_partitions

    def get_remaining_components(self, components, partition):
        """
        从components中删除出现在partition里的qubit，返回新的components
        """
        remaining_components = []
        for comp in components:
            remaining_comp = [q for q in comp if q not in partition]
            if remaining_comp: # 
                remaining_components.append(remaining_comp)
        return remaining_components

class NoiseAwareKWayPartitioner(BaseKWayPartitioner):
    """噪声感知分区器（继承基础版本）"""
    def __init__(self, qpus, max_options, noise_model=None):
        super().__init__(qpus, max_options)
        self.noise_model = noise_model

    def partition(self, interaction_graph, method="greedy"):
        """重写分区方法，支持噪声感知"""
        self.components = [list(comp) for comp in nx.connected_components(interaction_graph)]

        # 如果没有噪声模型，回退到基础版本
        if not self.noise_model:
            return super().partition(interaction_graph, method)

        # 计算组件权重
        self.interaction_graph = interaction_graph
        self._calculate_component_weights()

        print(f"[DEBUG] Components: {self.components}")
        print(f"[DEBUG] Component Weights: {self.component_weights}")

        if method == "greedy":
            return self.partition_greedy_noise_aware()
        elif method == "recursive":
            return self.partition_recursive_dp_noise_aware()
        return self.partition_dp_noise_aware()

    def _calculate_component_weights(self):
        """计算组件权重"""
        self.component_weights = []
        for component in self.components:
            weight = 0
            component_set = set(component)

            # 计算分量内部边的权重和
            for i, qubit1 in enumerate(component):
                for qubit2 in component[i+1:]:
                    if self.interaction_graph.has_edge(qubit1, qubit2):
                        weight += self.interaction_graph[qubit1][qubit2].get('weight', 1)
            
            # # 考虑分量的重要性（大小和连接密度）
            # size_weight = len(component)  # 分量大小
            # density = weight / max(1, len(component))  # 连接密度
            
            # # 综合权重 = 边权和 × 密度因子 × 大小因子
            # total_weight = weight * (1 + density) * (1 + np.log1p(size_weight))
            total_weight = weight

            self.component_weights.append(total_weight)

    def partition_greedy_noise_aware(self):
        """噪声感知贪心分区"""
        partition = [[] for _ in range(len(self.qpus))]
        capacities = self.qpus.copy()

        # 按权重降序排序连通分量（重要分量优先分配）
        weighted_components = sorted(zip(self.component_weights, self.components), 
                                     key=lambda x: x[0], reverse=True)
        
        # 按得分降序排序QPU（质量高的QPU放在前面）
        qpu_qualities = [(self.noise_model.get_quality_score(qpu_idx), qpu_idx)
                         for qpu_idx in range(len(self.qpus))]
        qpu_qualities = sorted(qpu_qualities, key=lambda x: x[0], reverse=True)

        for weight, com in weighted_components:
            best_qpu = -1
            
            # 为当前分量寻找最佳QPU
            for _, qpu_idx in qpu_qualities:
                if len(com) <= capacities[qpu_idx]:
                    best_qpu = qpu_idx
                    break

            if best_qpu != -1:
                partition[best_qpu].extend(com)
                capacities[best_qpu] -= len(com)
            else:
                # 无法完整容纳，拆分处理（尽量保持重要部分在一起）
                remaining_com = com.copy()
                # 从质量高的QPU开始，逐一放满
                for quality, qpu_idx in qpu_qualities:
                    if len(remaining_com) == 0:
                        break
                    if capacities[qpu_idx] > 0: # 还有余地
                        put_num = min(capacities[qpu_idx], len(remaining_com))
                        partition[qpu_idx].extend(remaining_com[:put_num])
                        capacities[qpu_idx] -= put_num
                        remaining_com = remaining_com[put_num:]
                        break
                assert len(remaining_com) == 0

        return [partition]

    def partition_recursive_dp_noise_aware(self):
        """噪声感知递归动态规划分区"""
        # 实现噪声感知版本
        pass

    def partition_dp_noise_aware(self):
        """噪声感知动态规划分区"""
        # 实现噪声感知版本
        pass