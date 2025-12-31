import networkx as nx
import numpy as np
import copy
import time
import sys
from collections import defaultdict
from qiskit.converters import circuit_to_dag, dag_to_circuit
from scipy.optimize import linear_sum_assignment

from methods.pytketdqc import *
from utils.noise import *
from methods.kwayp import *

class DQTetris:
    def __init__(self, circ, qpus,
                 max_options=1,
                 min_depths=None,
                 hyper_partitioner="PA",
                 partition_method="recursive",
                 network="fc",
                 noise_model=QPUNoiseModel()):
        print("[DQTetris]")
        print(f"[INFO] max_options: {max_options}, min_depths: {min_depths}")
        print(f"[INFO] Circuit: {circ.name}, depths: {circ.depth()}")
        print(f"[INFO] #QPUs: {len(qpus)}, QPUs: {qpus}")
        self.circ = circ
        self.dag  = circuit_to_dag(circ)
        self.qpus = qpus
        self.qpus.sort(reverse=True) # QPU容量从大到小排序
        self.max_options = max_options
        self.min_depths  = min_depths
        self.hyper_partitioner = hyper_partitioner
        self.partition_method  = partition_method
        if network == "fc":
            self.network, self.swap_cost_matrix = build_fc_network(self.qpus)
        elif network == "mesh":
            self.network, self.swap_cost_matrix = build_mesh_grid_network(self.qpus, int(len(self.qpus)/2), 2)
        self.noise_model = noise_model
        self.partitioner = NoiseAwareKWayPartitioner(self.qpus, 
                                                     self.max_options,
                                                     self.noise_model)
        return

    @property
    def name(self):
        return "DQTetris"

    def distribute(self):
        start_time = time.time()
        # remove single qubit gates
        print(f"[DEBUG] remove_single_qubit_gates", file=sys.stderr)
        self.remove_single_qubit_gates()
        self.set_min_depth()
        # return

        # partition table
        print(f"[DEBUG] build_partition_table", file=sys.stderr)
        self.build_partition_table()

        # slicing table and slicing results
        self.build_slicing_table()
        self.subc_ranges = []
        self.get_sliced_subc(0, len(self.P)-1) # -> self.subc_ranges
        # print(f"[DEBUG] subc_ranges: {self.subc_ranges}")

        # find partitions for each subcircuits
        self.legal_paths = []
        for ran in self.subc_ranges:
            self.legal_paths.append(self.P[ran[0]][ran[1]])
        if self.max_options == 1:
            self.num_comms = self.find_min_comms_path_greedy()
        else:
            self.num_comms = self.find_min_comms_path()
        print(f"[INFO] SWAP-ONLY #comms: {self.num_comms}, #subcs: {len(self.legal_paths)}")
        end_time = time.time()
        print(f"[INFO] Stage 1 Runtime: {end_time - start_time} seconds")

        # 对浅层的子线路使用gate teleportation进行优化
        self.num_gates = 0
        self.final_path = []
        self.group_shallow_subcircuits()
        self.num_swaps = self.num_comms - self.num_gates
        end_time = time.time()

        print(f"=====")
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}")
        print(f"#swap_comms: {self.num_swaps}")
        self.exec_time = end_time - start_time
        print(f"Time: {self.exec_time} seconds")
        self.save_entry_to_file(f"./outputs/paths/{self.circ.name[:11]}_{self.circ.num_qubits}_ours")
        print(f"=====\n\n")

        # self.check_num_swaps()
        return

    def set_min_depth(self):
        def sigmoid_decay(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 - 1 / (1 + np.exp(-k * (gate_density - c))))
        def sigmoid_increase(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 / (1 + np.exp(-k * (gate_density - c))))
        if self.min_depths == None:
            self.min_depths = int(sigmoid_increase(self.cu1_density, self.circ.depth()))
        print(f"[INFO] gate_density: {self.gate_density}")
        print(f"[INFO] cu1_density: {self.cu1_density}")
        print(f"[INFO] min_depth: {self.min_depths}")
        return

    def remove_single_qubit_gates(self):
        """
        Remove single qubit gates to construct self.dag_multiq
        Calculate gate density
        """
        # 根据原始线路的dag，逐层保留双量子比特门，并记录双量子比特门的层号
        start_time = time.time()
        self.dag_multiq = []
        self.map_to_init_layer = {}
        self.map_to_dag_multi_layer = {}
        pos_count = 0
        cu1_count = 0
        gate_count = 0
        # dag_debug = self.dag.copy_empty_like()
        # 遍历self.dag的每一层，如果是双量子比特门
        # 则添加到self.dag_multiq
        # TODO：每层最多len(self.qpus)个双量子比特门，防止单层出现不可分的情况
        layers = list(self.dag.layers())
        for lev, layer in enumerate(layers):
            curr_layer = []
            for node in layer["graph"].op_nodes():
                # print(f"{lev} {node.op.name} {node.qargs}")
                pos_count += len(node.qargs)
                gate_count += 1
                if len(node.qargs) > 1:
                    if node.op.name == "barrier":
                        continue
                    if len(node.qargs) != 2:
                        print(node.op.name)
                    assert(len(node.qargs) == 2)
                    if node.op.name == "cu1":
                        cu1_count += 1
                    curr_layer.append(node)
                    # dag_debug.apply_operation_back(node.op, node.qargs, node.cargs)
            if len(curr_layer) > 0:
                self.dag_multiq.append(curr_layer)
                # 记录双量子比特门在原始线路的层号
                self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
        self.gate_density = pos_count / (self.circ.num_qubits * self.circ.depth())
        self.cu1_density = cu1_count / gate_count
        # print(self.map_to_init_layer)

        # subc = dag_to_circuit(dag_debug)
        # print("[remove_single_qubit_gates]")
        # print(subc)
        # for lev, nodes in enumerate(self.dag_multiq):
        #     print(f"lev {lev}")
        #     for node in nodes:
        #         print(node.op.name, node.qargs)
        end_time = time.time()
        print(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        return

    def build_partition_table(self):
        """
        An efficient way of building the partition table
        """
        start_time = time.time()
        num_depths = len(self.dag_multiq)
        self.P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        print(f"[DEBUG] num_depths: {num_depths}", file=sys.stderr)
        # build the qubit interaction nxGraph for the entire circuit
        qig = self.build_qubit_interaction_graph((0, num_depths-1))
        is_changed = True

        for i in range(num_depths):
            # print(f"depth [{i}]", file=sys.stderr)
            # ===== P[i][numDepths-1] =====
            # rebuild qig
            if i != 0: # remove the (i-1)-th level of the remaining qig
                is_changed = self.remove_qig_edge(qig, i-1)

            if len(self.P[i][num_depths-1]) > 0: # inherit from the upper grid
                assert i != 0, f"[ERROR] P[{i}][{num_depths-1}] should be empty."
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    self.P[i+1][num_depths-1] = self.P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    self.P[i][num_depths-1] = self.get_qig_partitions(qig)
                    cnt += 1
                    if len(self.P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            self.P[i+1][num_depths-1] = self.P[i][num_depths-1] # downward propagation
        
            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self.remove_qig_edge(qig_tmp, j+1)
                # print(f"depth [{i}][{j}]", file=sys.stderr)
                if len(self.P[i][j]) > 0: # inherit from the upper grid
                    success = True # leftward propagation
                    if i + 1 <= j: # i + 1 < numDepths
                        self.P[i+1][j] = self.P[i][j] # downward propagation
                elif success: # inherit from the right grid
                    self.P[i][j] = self.P[i][j+1]
                else:
                    # print(f"is_changed: {is_changed}", file=sys.stderr)
                    if is_changed:
                        self.P[i][j] = self.get_qig_partitions(qig_tmp)
                        cnt += 1
                        if len(self.P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                self.P[i+1][j] = self.P[i][j]
        end_time = time.time()
        print(f"[build_partition_table] Partition calculation times: {cnt}.")
        print(f"[build_partition_table] Time: {end_time - start_time} seconds")
        return

    def build_qubit_interaction_graph(self, level_range):
        G = nx.Graph()
        for qubit in range(self.circ.num_qubits):
            G.add_node(qubit)
        for lev in range(level_range[0], level_range[1]+1):
            for node in self.dag_multiq[lev]:
                qubits = [qubit._index for qubit in node.qargs]
                if qubits[0] == None:
                    qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                if G.has_edge(qubits[0], qubits[1]):
                    G[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    G.add_edge(qubits[0], qubits[1], weight=1)
        return G

    def remove_qig_edge(self, qig, lev):
        """
        从qig中移除self.dag_multiq第lev列的量子门
        """
        is_changed = False # whether an edge is removed from qig
        for node in self.dag_multiq[lev]:
            qubits = [qubit._index for qubit in node.qargs]
            if qig.has_edge(qubits[0], qubits[1]):
                qig[qubits[0]][qubits[1]]['weight'] -= 1
                if qig[qubits[0]][qubits[1]]['weight'] == 0:
                    qig.remove_edge(qubits[0], qubits[1])
                    is_changed = True
        return is_changed

    def get_qig_partitions(self, qig):
        # start_time = time.time()
        # components = [list(comp) for comp in nx.connected_components(qig)]

        # 计算每个连通分量的权重
        # component_weights = []
        # for comp in components:
            # weight = self.weight_calculator.calculate_component_weight(comp, qig)
            # component_weights.append(weight)
        
        # 将组件和权重打包
        # weighted_components = list(zip(components, component_weights))

        # legal_partitions = self.partitioner.partition(components, self.partition_method)
        legal_partitions = self.partitioner.partition(qig, self.partition_method)
        # end_time = time.time()
        # print(f"[get_qig_partitions] Time: {end_time - start_time} seconds")
        return legal_partitions

    # 
    # S, T table
    # 
    def build_slicing_table(self):
        start_time = time.time()
        num_depths = len(self.P)
        self.T = [[0]  * num_depths for _ in range(num_depths)]
        self.S = [[-1] * num_depths for _ in range(num_depths)]

        for i in range(num_depths):
            if len(self.P[i][i]) == 0:
                print(f"[ERROR] P[{i}][{i}] is empty.")
                exit(1)
        # print("[build_t_table] ", end="")
        for depth in range(2, num_depths + 1): # depth: 2, 3, ..., num_depths
            # print(depth, end="")
            for i in range(0, num_depths - depth + 1): # 左边界
                j = i + depth - 1 # 右边界
                if len(self.P[i][j]) == 0:
                    self.T[i][j] = float('inf')
                    # 利用四边形优化缩小枚举范围
                    lower_k = self.S[i][j-1] if self.S[i][j-1] != -1 else i
                    upper_k = self.S[i+1][j] if self.S[i+1][j] != -1 else j-1
                    for k in range(lower_k, upper_k + 1):
                    # for k in range(i, j):
                        comms = self.T[i][k] + self.T[k+1][j] + 1
                        if comms < self.T[i][j]:
                            self.T[i][j] = comms
                            self.S[i][j] = k
                    # check if S[i][j-1] <= S[i][j] <= S[i+1][j]
                    # print(i, self.S[i][j-1], self.S[i][j], self.S[i+1][j], j)
                    # if self.S[i][j-1] != -1:
                    #     assert(self.S[i][j-1] <= self.S[i][j])
                    # if self.S[i+1][j] != -1:
                    #     assert(self.S[i][j] <= self.S[i+1][j])
        # print()
        end_time = time.time()
        print(f"[build_slicing_table] Time: {end_time - start_time} seconds")
        return
    
    def get_sliced_subc(self, i, j):
        if self.S[i][j] == -1:
            self.subc_ranges.append((i, j))
            return
        self.get_sliced_subc(i, self.S[i][j])
        self.get_sliced_subc(self.S[i][j] + 1, j)
        return

    def find_min_comms_path_greedy(self):
        """
        基于self.legal_paths，即每个子线路合法的划分
        找到最少swap次数的路径
        """
        start_time = time.time()

        # dp[i][A]：表示第 i 个子线路选择分配方案 A 时的最小累计SWAP次数。
        # dp = {}
        self.num_comms = 0
        self.swap_only_path = [] # 记录最优路径
        self.swap_prefix_sums = [0 for _ in range(len(self.legal_paths))] # 记录最优路径上的交换次数

        for i in range(len(self.legal_paths)):
            assert(len(self.legal_paths[i]) == 1)
            self.swap_only_path.append(self.legal_paths[i][0])
            if i > 0:
                comms = self.calculate_nonlocal_communications(self.swap_only_path[-2], self.swap_only_path[-1])
                self.num_comms += comms
                self.swap_prefix_sums[i] = self.swap_prefix_sums[i-1] + comms

        end_time = time.time()
        print(f"[find_min_comms_path] Time: {end_time - start_time} seconds")
        return self.num_comms

    def find_min_comms_path(self):
        """
        基于self.legal_paths，即每个子线路合法的划分
        找到最少swap次数的路径
        """
        start_time = time.time()

        # dp[i][j]：表示第 i 个子线路选择分配方案 j 时的最小累计SWAP次数。
        num_subcs = len(self.legal_paths)
        dp = [[float('inf')] * len(self.legal_paths[i]) for i in range(num_subcs)]
        # 初始化 prev 数组，用于记录路径信息
        prev = [[-1] * len(self.legal_paths[i]) for i in range(num_subcs)]

        # 初始化第一层的 dp 值
        for j in range(len(self.legal_paths[0])):
            dp[0][j] = 0

        # 动态规划计算 dp 数组
        for i in range(1, num_subcs):
            for j in range(len(self.legal_paths[i])):
                # if self.max_options == 1:
                #     assert j == 0, "[ERROR] max_options is 0 but more than 1 clean partition detected."
                for k in range(len(self.legal_paths[i - 1])):
                    # if self.max_options == 1:
                    #     assert k == 0, "[ERROR] max_options is 0 but more than 1 clean partition detected."
                    comms = self.calculate_nonlocal_communications(self.legal_paths[i - 1][k], self.legal_paths[i][j])
                    if dp[i - 1][k] + comms < dp[i][j]:
                        dp[i][j] = dp[i - 1][k] + comms
                        prev[i][j] = k

        # 找到min_comms最后一层的最小 comms 元素
        min_index = dp[-1].index(min(dp[-1]))
        min_comms = dp[-1][min_index]

        # 回溯路径
        self.swap_only_path = [] # 记录最优路径
        self.swap_prefix_sums = [0 for _ in range(len(self.legal_paths))] # 记录最优路径上的交换次数
        index = min_index
        for i in range(num_subcs - 1, -1, -1):
            self.swap_only_path.append(self.legal_paths[i][index])
            k = prev[i][index]
            if k != -1:
                # 可以计算dp[i][index]和dp[i-1][index']之间的通信次数
                self.swap_prefix_sums[i] = dp[i][index]
            index = k
        self.swap_only_path.reverse()

        end_time = time.time()
        print(f"[find_min_comms_path] Time: {end_time - start_time} seconds")
        return min_comms

    def calculate_nonlocal_communications(self, prev_assign, curr_assign):
        # TODO：把划分从数组改成set
        prev_assign = [set(p) for p in prev_assign]
        curr_assign = [set(p) for p in curr_assign]
        num_qpus = len(prev_assign)
        assert num_qpus == len(self.qpus), f"[ERROR] prev_assign: {prev_assign}'s size {num_qpus} != len(self.qpus): {len(self.qpus)}"
        # 构建权重矩阵（partition A和B集合的交集大小）
        weights = [[len(prev_assign[i] & curr_assign[j]) for j in range(num_qpus)] for i in range(num_qpus)]
        # 使用匈牙利算法找到最大权匹配
        prev_assign_idx, curr_assign_idx = linear_sum_assignment(weights, maximize=True)

        num_qubits = self.circ.num_qubits
        G = nx.DiGraph() # 初始化有向图
        G.add_nodes_from(range(len(prev_assign))) # 每个partition对应一个节点

        communication_cost = 0

        # 记录每个qubit在prev和curr的分区号
        qubit_mapping = [[-1, -1] for _ in range(num_qubits)]
        for pno, (i, j) in enumerate(zip(prev_assign_idx, curr_assign_idx)):
            prev_partition, curr_partition = prev_assign[i], curr_assign[j]
            for qubit in prev_partition:
                qubit_mapping[qubit][0] = pno
            for qubit in curr_partition:
                qubit_mapping[qubit][1] = pno

        # 遍历映射，若前后分配不同，添加边到图中
        for prev_part, curr_part in qubit_mapping:
            assert(prev_part != -1 and curr_part != -1)
            if prev_part != curr_part: # prev_part -> curr_part
                # 检查是否存在curr_part -> prev_part的边
                # 如果存在，则说明形成了环
                # 因为每次只加一条边，所以抵消掉一条就行
                if G.has_edge(curr_part, prev_part):
                    communication_cost += \
                        self.swap_cost_matrix[curr_part][prev_part] # one RSWAP
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
            assert(3 <= length <= len(self.qpus))
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            assert(3 <= length <= len(self.qpus))
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
                    swap_cost = self.swap_cost_matrix[u][v]
                    communication_cost += swap_cost * weight

        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            path_len = (self.swap_cost_matrix[u][v] + 1) / 2
            communication_cost += path_len * data['weight']

        return communication_cost

    # 
    # Group shallow subcircuits
    # 
    def group_shallow_subcircuits(self):
        # 
        # 收集单层的量子线路
        # 
        start_time = time.time()
        left = right = 0
        left_subc_idx = right_subc_idx = -1
        for i in range(len(self.swap_only_path)):
            # 获取每个子线路的层数
            depth = self.subc_ranges[i][1] - self.subc_ranges[i][0] + 1
            # print(f"subc {self.subc_ranges[i]}: {self.swap_only_path[i]} {depth}")

            # 如果连续的量子线路是浅层的，就加入list
            if depth < self.min_depths:
                right = self.subc_ranges[i][1]
                right_subc_idx = i + 1
            else:
                # 当前的子线路层数比较多，不需要做gate teleportation
                # 1. 先把之前收集的子线路（如果有）用gate teleportation试一下
                if left_subc_idx < right_subc_idx:
                    self.try_replace_with_gate_tele(left_subc_idx, right_subc_idx, left, right)
                # 2. 当前子线路还是采用swap，记录
                self.add_final_entry(self.subc_ranges[i], "swap", self.swap_only_path[i])
                # 3. 再把left和right更新成下一个子线路的左起点
                left_subc_idx = right_subc_idx = i
                if i != len(self.swap_only_path) - 1: # subc[i]不是最后一个子线路
                    left = right = self.subc_ranges[i+1][0]
                # 如果subc[i]是最后一个子线路，处理完毕

        if left_subc_idx < right_subc_idx: # 处理最后一个
            self.try_replace_with_gate_tele(left_subc_idx, right_subc_idx, left, right)
        end_time = time.time()
        print(f"[group_shallow_subcircuits] Time: {end_time - start_time} seconds")
        return

    def try_replace_with_gate_tele(self, left_subc_idx, right_subc_idx, left, right):
        """
        Try to replace dag_multiq[left, right] with gate teleportation
        """
        # print(f"[debug][try replace] {left_subc_idx} {right_subc_idx} {left} {right}")
        # 1. 尝试替换[left, right]这部分子线路
        # 1.1. 获取替换后的分区以及cat-ent costs
        partition, cat_ent_costs = self.hyper_partition(left, right)
        assert len(partition) == len(self.qpus), f"[ERROR] partition: {partition} != {len(self.qpus)}"
        # 1.2. 计算新分区和左分区的swap costs
        swap_costs = 0
        # 获取左子线路的partition
        assert(left_subc_idx < len(self.swap_only_path) - 1)
        if left_subc_idx != -1:
            left_partition = self.swap_only_path[left_subc_idx]
            # print(left_partition)
            # left_swap = calculate_nonlocal_communication(self.circ.num_qubits, left_partition, partition)
            # print(left_partition, partition[0])
            left_swap = self.calculate_nonlocal_communications(left_partition, partition)
            # print(f"left_swap: {left_swap}")
            swap_costs += left_swap
        # 1.3. 计算新分区和右分区的swap costs
        assert(right_subc_idx > 0)
        if right_subc_idx != len(self.swap_only_path):
            right_partition = self.swap_only_path[right_subc_idx]
            # print(right_partition)
            # right_swap = calculate_nonlocal_communication(self.circ.num_qubits, partition, right_partition)
            right_swap = self.calculate_nonlocal_communications(partition, right_partition)
            # print(f"right_swap: {right_swap}")
            swap_costs += right_swap
        new_costs = cat_ent_costs + swap_costs

        # 2. 和全部基于swap的通信代价进行比较
        # 2.1. 获取[left, right]这部分子线路的swap costs
        old_costs = 0
        if right_subc_idx >= len(self.swap_only_path):
            old_costs = self.swap_prefix_sums[right_subc_idx - 1]
        else:
            old_costs = self.swap_prefix_sums[right_subc_idx]
        if left_subc_idx >= 0:
            old_costs -= self.swap_prefix_sums[left_subc_idx]
        # print(f"[try_replace_with_gate_tele] old_costs: {old_costs}, new_costs: {new_costs}")

        # 2. 如果新的ecosts更低，那么这部分线路用gate teleportation
        if new_costs < old_costs:
            self.num_comms += new_costs - old_costs
            self.add_final_entry((left, right), "gate", partition)
            self.num_gates += cat_ent_costs
            return

        for j in range(left_subc_idx + 1, right_subc_idx):
            self.add_final_entry(self.subc_ranges[j], "swap", self.swap_only_path[j])
        return

    def get_ori_subc(self, ori_left, ori_right):
        """
        获取原线路中[ori_left, ori_right]的子线路
        """
        layers = list(self.dag.layers())
        sub_dag = self.dag.copy_empty_like()
        for lev in range(ori_left, ori_right + 1):
            for node in layers[lev]["graph"].op_nodes():
                sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        sub_qc = dag_to_circuit(sub_dag)
        return sub_qc

    def hyper_partition(self, left, right):
        """
        Call Pytket-DQC's partitioner
        """
        # 
        # 1. 获取原线路中的[map[left], map[right]]这部分子线路
        # 
        # print(f"[hyper_partition] {left}-{right}")
        ori_left = self.map_to_init_layer[left]
        ori_right = self.map_to_init_layer[right]
        # layers = list(self.dag.layers())
        # sub_dag = self.dag.copy_empty_like()
        # for lev in range(ori_left, ori_right + 1):
        #     for node in layers[lev]["graph"].op_nodes():
        #         sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        sub_qc = self.get_ori_subc(ori_left, ori_right)
        # 
        # 2. 转译成Pytket-DQC可以处理的线路
        # 
        sub_qc = transpile(sub_qc, basis_gates=["cu1", "rz", "h"])
        sub_qc = qiskit_to_tk(sub_qc)
        DQCPass().apply(sub_qc)
        # print(f"[hyper_partition] sub_qc: {ori_left}-{ori_right}")
        # output_circuit_as_html(sub_qc, "hyper_partition")
        # 
        # 3. 调用Pytket-DQC的库，计算划分和ecosts
        #
        if self.hyper_partitioner == "CE":
            distribution = CoverEmbeddingSteinerDetached().distribute(sub_qc, self.network, seed=26)
        else:
            distribution = PartitioningAnnealing().distribute(sub_qc, self.network, seed=26)
            # distributed_circ = distribution.to_pytket_circuit()
            # output_circuit_as_html(distributed_circ, "distributed_circ")
        # 
        # 4. 返回前后的划分，以及通信次数
        # 
        partition = [[] for _ in range(len(self.qpus))] # 每个qpu上一个划分
        # print(f"[get_qubit_mapping]")
        # qubit_mapping = distribution.get_qubit_mapping()
        # print(qubit_mapping)
        # for e in qubit_mapping:
        #     print(e.index[0], qubit_mapping[e])
        for i in range(self.circ.num_qubits): # q[i]->server[j]
            # print(i, distribution.placement.placement[i])
            partition[distribution.placement.placement[i]].append(i)
        # print(partition)
        # print(distribution.cost())
        return partition, distribution.cost()

    def add_final_entry(self, subc_range, comm_type, partition):
        # map to original range
        ori_left = self.map_to_init_layer[subc_range[0]]
        ori_right = self.map_to_init_layer[subc_range[1]]
        entry = {"range": (ori_left, ori_right), "comm_type": comm_type, "partition": partition}
        # print(f"[DEBUG] range: {subc_range}")
        # print(entry)
        self.final_path.append(entry)
        return

    def save_entry_to_file(self, filename):
        """
        将二维数组的每一行输出到一个文件中。

        参数:
        - array: 二维数组（列表的列表）
        - filename: 文件名前缀，生成的文件名将是 filename_0.txt, filename_1.txt, ...
        """
        filename = f"{filename}.txt"
        with open(filename, 'w') as file:
            for entry in self.final_path:
                file.write(f"{entry}")
                file.write('\n')

    def count_cut_edges(self, graph, partitions):
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, partition in enumerate(partitions):
            for node in partition:
                node_to_partition[node] = i
        cut_edges = 0
        for u, v in graph.edges(): # 遍历图中的每一条边
            if node_to_partition[u] != node_to_partition[v]:
                cut_edges += graph[u][v]['weight']
        return cut_edges

    def check_num_swaps(self):
        """
        检查num_swaps的计算对不对
        """
        print("[DEBUG] check_num_swaps", file=sys.stderr)
        num_swaps = 0 # remote swaps
        # 对于每个子线路，先判断是基于swap还是cat-ent
        for idx, entry in enumerate(self.final_path):
            ran, comm_type, partition = entry["range"], entry["comm_type"], entry["partition"]
            if entry["comm_type"] == "swap":
                # 获取ran在dag_multiq中的位置
                left = self.map_to_dag_multi_layer[ran[0]]
                right = self.map_to_dag_multi_layer[ran[1]]
                # 获取qig
                qig = self.build_qubit_interaction_graph((left, right))
                # 调用count_cut_edge检查partition是否是clean partition
                cut_edges = self.count_cut_edges(qig, partition)
                assert cut_edges == 0, f"[ERROR] partition {partition} is not clean partition."
            # 计算两个partitions的switching cost
            if idx > 0:
                prev_partition = self.final_path[idx - 1]["partition"]
                # print(f"prev: {prev_partition}, curr: {partition}")
                # 计算当前的partition和prev_partition的swap costs
                swap_costs = self.calculate_nonlocal_communications(prev_partition, partition)
                num_swaps += swap_costs
        assert num_swaps == self.num_swaps, f"[ERROR] num_swaps: {num_swaps} != {self.num_swaps}"
        return
