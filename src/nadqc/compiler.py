from qiskit import QuantumCircuit
import time
import sys
import networkx as nx
import numpy as np
from collections import defaultdict
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pprint import pprint
from scipy.optimize import linear_sum_assignment
import itertools
import networkx as nx
import copy

from . import Network, PartitionerFactory

class NADQC:
    def __init__(self, 
                 circ: QuantumCircuit, 
                 network: Network,
                 partition_method: str = "recursive_dp"):
        self.circ = circ
        self.network = network
        self.partitioner = PartitionerFactory.create_partitioner(partition_method, network, max_options=1)
        # self.mapper = Mapper()
        return
    
    @property
    def name(self):
        return "NADQC"

    def distribute(self):
        self._remove_single_qubit_gates()
        self._build_partition_table()
        self._build_slicing_table()
        # self.partition_plan = self.get_partition_plan()
        # self.map(self.partition_plan)
        return

    def get_partition_candidates(self):
        self.subc_ranges = []
        self._get_sliced_subc(0, len(self.P) - 1)
        partition_candidates = []
        for (i, j) in self.subc_ranges:
            # 通过P[i][j][0]，获取每个qubit被映射到哪个逻辑QPU
            partition_candidates.append(self.P[i][j]) # 返回所有划分方案
        return partition_candidates

    def _remove_single_qubit_gates(self):
        """
        Remove single qubit gates to construct self.dag_multiq
        Calculate gate density
        """
        # 根据原始线路的dag，逐层保留双量子比特门，并记录双量子比特门的层号
        start_time = time.time()
        self.dag  = circuit_to_dag(self.circ)
        self.dag_multiq = []
        self.map_to_init_layer = {}
        self.map_to_dag_multi_layer = {}
        pos_count, cu1_count, gate_count = 0, 0, 0
        # dag_debug = self.dag.copy_empty_like()
        # 遍历self.dag的每一层，如果是双量子比特门
        # 则添加到self.dag_multiq
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
                    assert len(node.qargs) == 2, f"[ERROR] Found gate with more than 2 qubits: {node.op.name} on {node.qargs}"
                    if node.op.name == "cu1":
                        cu1_count += 1
                    curr_layer.append(node)
                    # dag_debug.apply_operation_back(node.op, node.qargs, node.cargs)
            if len(curr_layer) > 0:
                if len(curr_layer) == self.circ.num_qubits // 2 and len(curr_layer) % self.network.num_backends != 0:
                    # split the layer into two layers
                    split_point = len(curr_layer) // 2
                    first_half = curr_layer[:split_point]
                    second_half = curr_layer[split_point:]
                    self.dag_multiq.append(first_half)
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
                    self.dag_multiq.append(second_half)
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
                else:
                    self.dag_multiq.append(curr_layer)
                    # 记录双量子比特门在原始线路的层号
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
        self.gate_density = pos_count / (self.circ.num_qubits * self.circ.depth())
        self.cu1_density = cu1_count / gate_count

        end_time = time.time()
        print(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        # self._reconstruct_and_visualize_circuit()
        return

    def _reconstruct_and_visualize_circuit(self):
        """
        Reconstructs the multi-qubit gate circuit from self.dag_multiq and visualizes it.
        
        Creates a new QuantumCircuit containing only the multi-qubit gates (with barrier separators),
        then prints a text-based visualization using Qiskit's circuit_drawer.
        """
        # 如果没有双量子门，直接返回
        if not self.dag_multiq:
            print("No multi-qubit gates found. Cannot visualize circuit.")
            return

        pprint(self.dag_multiq)
        # 创建新的量子线路（量子比特数与原始线路一致）
        n_qubits = self.circ.num_qubits
        recon_circ = QuantumCircuit(n_qubits)
        
        # 按层添加双量子门
        for i, layer in enumerate(self.dag_multiq):
            for node in layer:
                # 添加门操作（包含参数和量子比特）
                recon_circ.append(node.op, node.qargs, node.cargs)
            
            # 在每层后添加 barrier（最后一层不加）
            if i < len(self.dag_multiq) - 1:
                recon_circ.barrier()
        
        # 打印可视化结果
        print("\n" + "="*50)
        print("Reconstructed Multi-Qubit Gate Circuit:")
        print("="*50)
        print(recon_circ)
        print("\n" + "="*50)
        print(f"Total layers: {len(self.dag_multiq)} | Total gates: {sum(len(layer) for layer in self.dag_multiq)}")
        print("="*50)
        
        return recon_circ  # 返回重建的电路对象供进一步使用

    def _build_partition_table(self):
        """
        An efficient way of building the partition table
        """
        start_time = time.time()
        num_depths = len(self.dag_multiq)
        self.P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        print(f"[DEBUG] num_depths: {num_depths}", file=sys.stderr)
        # build the qubit interaction nxGraph for the entire circuit
        qig = self._build_qubit_interaction_graph((0, num_depths-1))
        is_changed = True

        for i in range(num_depths):
            # print(f"depth [{i}]", file=sys.stderr)
            # ===== P[i][numDepths-1] =====
            # rebuild qig
            if i != 0: # remove the (i-1)-th level of the remaining qig
                is_changed = self._remove_qig_edge(qig, i-1)

            if len(self.P[i][num_depths-1]) > 0: # inherit from the upper grid
                assert i != 0, f"[ERROR] P[{i}][{num_depths-1}] should be empty."
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    self.P[i+1][num_depths-1] = self.P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    self.P[i][num_depths-1] = self._get_qig_partitions(qig)
                    cnt += 1
                    if len(self.P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            self.P[i+1][num_depths-1] = self.P[i][num_depths-1] # downward propagation
        
            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self._remove_qig_edge(qig_tmp, j+1)
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
                        self.P[i][j] = self._get_qig_partitions(qig_tmp)
                        cnt += 1
                        if len(self.P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                self.P[i+1][j] = self.P[i][j]
        end_time = time.time()
        print(f"[build_partition_table] Partition calculation times: {cnt}.")
        print(f"[build_partition_table] Time: {end_time - start_time} seconds")
        return

    def _build_qubit_interaction_graph(self, level_range):
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

    def _remove_qig_edge(self, qig, lev):
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

    def _get_qig_partitions(self, qig):
        components = [list(comp) for comp in nx.connected_components(qig)]
        legal_partitions = self.partitioner.partition(components)
        return legal_partitions



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
    # S, T table
    # 
    def _build_slicing_table(self):
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
    
    def _get_sliced_subc(self, i, j):
        if self.S[i][j] == -1:
            self.subc_ranges.append((i, j))
            return
        self._get_sliced_subc(i, self.S[i][j])
        self._get_sliced_subc(self.S[i][j] + 1, j)
        return

    # def get_partition_plan(self):
    #     self.subc_ranges = []
    #     self._get_sliced_subc(0, len(self.P) - 1)
    #     partition_plan = []
    #     for (i, j) in self.subc_ranges:
    #         # 通过P[i][j][0]，获取每个qubit被映射到哪个逻辑QPU
    #         partition_plan.append(self.P[i][j][0]) # 仅选择第一个划分方案
    #     return partition_plan

    # def calculate_comm_cost_dynamic(self, partition_plan):
    #     """
    #     动态映射：为每个partition切换计算通信成本和最优映射序列
    #     :param partition_plan: list of partitions for each time slice
    #         Example: [[[0,1], [2]], [[0,2], [1]]] 
    #         - Time slice 0: logical QPU 0 has qubits [0,1], logical QPU 1 has qubit [2]
    #         - Time slice 1: logical QPU 0 has qubits [0,2], logical QPU 1 has qubit [1]
    #     :return: (total_comm_cost, mapping_sequence)
    #     """
    #     start_time = time.time()
    #     k = len(partition_plan)  # 时间片数量
    #     m_physical = self.network.num_backends  # 物理QPU数量
    #     n = self.circ.num_qubits  # 量子比特数
        
    #     # 结果存储
    #     total_comm_cost = 0.0
    #     mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
    #     # 1. 设置初始映射：时间片0的逻辑QPU到物理QPU的映射
    #     num_logical_qpus_t0 = len(partition_plan[0])
    #     assert num_logical_qpus_t0 <= m_physical, f"Time slice 0 has {num_logical_qpus_t0} logical QPUs but only {m_physical} physical QPUs available"
        
    #     # 简单策略：identity mapping (逻辑QPU i -> 物理QPU i)
    #     initial_mapping = list(range(num_logical_qpus_t0))
    #     mapping_sequence.append(initial_mapping)
        
    #     print(f"[Initial Mapping] Time slice 0: {initial_mapping} (logical -> physical)", file=sys.stderr)
        
    #     # 2. 为每个时间片边界计算切换成本
    #     for t in range(k-1):
    #         # 2.1 计算当前边界(t -> t+1)的通信需求
    #         D_switch, qubit_movements = self._compute_switch_demand(
    #             partition_plan[t], 
    #             partition_plan[t+1]
    #         )
            
    #         # 2.2 获取下一时间片的逻辑QPU数量
    #         num_logical_next = len(partition_plan[t+1])
    #         assert num_logical_next <= m_physical, f"Time slice {t+1} has {num_logical_next} logical QPUs but only {m_physical} physical QPUs available"
            
    #         # 2.3 为下一时间片找到最优映射
    #         next_mapping = self._find_optimal_mapping_for_switch(
    #             D_switch,
    #             mapping_sequence[t],  # 当前时间片的映射
    #             num_logical_next,    # 下一时间片的逻辑QPU数量
    #             qubit_movements      # 量子比特移动详情
    #         )
            
    #         # 2.4 计算本次切换的实际成本
    #         switch_cost = self._evaluate_switch_cost(
    #             D_switch,
    #             mapping_sequence[t],
    #             next_mapping,
    #             qubit_movements
    #         )
            
    #         total_comm_cost += switch_cost
    #         mapping_sequence.append(next_mapping)
            
    #         print(f"[Switch {t}->{t+1}] Cost: {switch_cost:.4f}, Mapping: {next_mapping}", file=sys.stderr)
    #         # 调试：打印量子比特移动
    #         print(f"  Qubit movements: {qubit_movements}", file=sys.stderr)
        
    #     end_time = time.time()
    #     print(f"[calculate_comm_cost_dynamic] Total comm cost: {total_comm_cost:.4f}, Time: {end_time - start_time:.4f}s", file=sys.stderr)
        
    #     return total_comm_cost, mapping_sequence


    # def _find_optimal_mapping_for_switch(self, D_switch, current_mapping, num_logical_next, qubit_movements):
    #     """
    #     为切换找到最优的下一映射
    #     :param D_switch: 通信需求矩阵 (m_logical_current x m_logical_next)
    #     :param current_mapping: 当前时间片的物理映射 [logical_current -> physical]
    #     :param num_logical_next: 下一时间片的逻辑QPU数量
    #     :param qubit_movements: 量子比特移动详情 {qubit: (from_logical, to_logical)}
    #     :return: next_mapping, 下一时间片的最优物理映射 [logical_next -> physical]
    #     """
    #     m_physical = self.network.num_backends
        
    #     # 获取有效保真度矩阵
    #     # if not hasattr(self.network, 'W_eff'):
    #     #     self.network.W_eff, _ = self.network.compute_effective_fidelity()
    #     W_eff = self.network.W_eff
        
    #     # 策略：贪心分配 - 高需求逻辑QPU优先分配到高保真度位置
        
    #     # 1. 计算下一时间片中每个逻辑QPU的重要性
    #     logical_importance = np.zeros(num_logical_next)
    #     for i in range(D_switch.shape[0]):
    #         for j in range(D_switch.shape[1]):
    #             logical_importance[j] += D_switch[i][j]  # 流入逻辑QPU j 的需求
        
    #     # 2. 计算当前时间片中每个物理QPU的"热度"（高需求量子比特所在位置）
    #     physical_hotness = np.zeros(m_physical)
    #     for qubit, (from_logical, to_logical) in qubit_movements.items():
    #         physical_pos = current_mapping[from_logical]
    #         physical_hotness[physical_pos] += 1
        
    #     # 3. 优先分配高重要性逻辑QPU到高热度物理QPU
    #     importance_order = np.argsort(-logical_importance)  # 降序排序
    #     hotness_order = np.argsort(-physical_hotness)       # 降序排序
        
    #     next_mapping = [-1] * num_logical_next  # 初始化为-1（未分配）
    #     used_physical = [False] * m_physical
        
    #     # 4. 分配逻辑QPU
    #     for logical_next in importance_order:
    #         if logical_importance[logical_next] == 0:
    #             continue  # 无需求的逻辑QPU可以最后分配
            
    #         best_physical = -1
    #         best_score = -1
            
    #         # 尝试所有物理QPU
    #         for physical in range(m_physical):
    #             if used_physical[physical]:
    #                 continue
                
    #             # 评分：热度 + 与当前高需求位置的保真度
    #             score = physical_hotness[physical] * 0.6  # 热度权重
                
    #             # 考虑与当前移动源位置的保真度
    #             fidelity_sum = 0
    #             count = 0
    #             for qubit, (from_logical, to_logical) in qubit_movements.items():
    #                 if to_logical == logical_next:
    #                     current_physical = current_mapping[from_logical]
    #                     fidelity_sum += W_eff[current_physical][physical]
    #                     count += 1
                
    #             if count > 0:
    #                 avg_fidelity = fidelity_sum / count
    #                 score += avg_fidelity * 0.4  # 保真度权重
                
    #             if score > best_score:
    #                 best_score = score
    #                 best_physical = physical
            
    #         if best_physical == -1:  # 所有物理QPU已用，选第一个可用
    #             best_physical = used_physical.index(False)
            
    #         next_mapping[logical_next] = best_physical
    #         used_physical[best_physical] = True
        
    #     # 5. 分配剩余的逻辑QPU
    #     remaining_logicals = [i for i in range(num_logical_next) if next_mapping[i] == -1]
    #     remaining_physicals = [i for i in range(m_physical) if not used_physical[i]]
        
    #     # 按逻辑索引顺序分配
    #     for logical, physical in zip(remaining_logicals, remaining_physicals):
    #         next_mapping[logical] = physical
        
    #     return next_mapping


class OEE:
    def __init__(self, 
                 circ: QuantumCircuit, 
                 network: Network,
                 iteration: int = 10):
        self.circ = circ
        self.network = network
        self.iteration = iteration
        return

    @property
    def name(self):
        return "S-OEE"
    
    def distribute(self):
        print("[Static OEE]")
        start_time = time.time()
        self.build_qubit_interaction_graph(self.circ)
        self.k_way_OEE(self.qig)
        self.path = []
        self.path.append(self.partitions)
        self.num_comms = self.num_gates = self.count_cut_edges(self.qig, self.partitions)
        self.num_swaps = 0
        end_time = time.time()
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}")
        print(f"#swap_comms: {self.num_swaps}")
        print(f"fidelity_loss: {self.fidelity_loss}")
        self.exec_time = end_time - start_time
        print(f"Time: {self.exec_time} seconds\n\n")
        # self.save_path_to(f"./outputs/paths/{self.circ.name[:11]}_{self.circ.num_qubits}_oee")
        return
    
    def get_partition_candidates(self):
        partition_candidates = []
        partition_candidates.append([self.partitions])
        return partition_candidates

    def build_qubit_interaction_graph(self, circuit):
        self.qig = nx.Graph()
        for qubit in range(circuit.num_qubits):
            self.qig.add_node(qubit)
        for instruction in circuit:
            # gate = instruction.operation
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) > 1:
                if instruction.name == "barrier":
                    continue
                if len(qubits) != 2:
                    print(instruction)
                assert(len(qubits) == 2)
                if self.qig.has_edge(qubits[0], qubits[1]):
                    self.qig[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    self.qig.add_edge(qubits[0], qubits[1], weight=1)
        return

    def k_way_OEE(self, graph):
        nodes = list(graph.nodes())
        n = len(nodes)
        k = self.network.num_backends
        self.allocate_qubits() # initialize self.partitions
        for _itr in range(self.iteration):
            C = nodes.copy()
            D = np.zeros((n, k))
            # Step 1: Calculate the D(i, l) value corresponding to each node i and each subset l
            for node in nodes:
                current_col = next(j for j, subset in enumerate(self.partitions) if node in subset)
                for l in range(k):
                    D[node, l] = self.calculate_d(graph, node, self.partitions[l], self.partitions[current_col])
            g_values = []
            exchange_pairs = []
            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None
                # Step 2: Find the two nodes a and b that maximize the reduction in exchange cost g(a, b)
                for a in C:
                    for b in C:
                        if a < b:
                            col_a = next(j for j, subset in enumerate(self.partitions) if a in subset)
                            col_b = next(j for j, subset in enumerate(self.partitions) if b in subset)
                            if graph.has_edge(a, b):
                                g = D[a, col_b] + D[b, col_a] - 2 * graph[a][b].get('weight', 1)
                            else:
                                g = D[a, col_b] + D[b, col_a]
                            if g > max_g:
                                max_g = g
                                best_a, best_b = a, b
                # print(f"remove: {best_a}, {best_b}, max_g: {max_g}")
                C.remove(best_a)
                C.remove(best_b)
                # print(C)
                g_values.append(max_g)
                exchange_pairs.append((best_a, best_b))

                # Step 3: Update D-values
                col_a = next(j for j, subset in enumerate(self.partitions) if best_a in subset)
                col_b = next(j for j, subset in enumerate(self.partitions) if best_b in subset)
                for node in C:
                    col_i = next(j for j, subset in enumerate(self.partitions) if node in subset)
                    w_ia = graph[best_a][node].get('weight', 1) if graph.has_edge(best_a, node) else 0
                    w_ib = graph[best_b][node].get('weight', 1) if graph.has_edge(best_b, node) else 0
                    # print(f"w_ia: {w_ia}, w_ib: {w_ib}")
                    for l in range(k):
                        if l == col_a:
                            if col_i != col_a and col_i != col_b:
                                D[node, l] += w_ib - w_ia
                            elif col_i == col_b:
                                D[node, l] += 2 * w_ib - 2 * w_ia
                        elif l == col_b:
                            if col_i != col_a and col_i != col_b:
                                D[node, l] += w_ia - w_ib
                            elif col_i == col_a:
                                D[node, l] += 2 * w_ia - 2 * w_ib
                        elif col_i == col_a and l != col_a and l != col_b:
                            D[node, l] += w_ia - w_ib
                        elif col_i == col_b and l != col_a and l != col_b:
                            D[node, l] += w_ib - w_ia

            # Step 4: Find the optimal time m
            max_g_sum = float('-inf')
            best_m = 0
            g_sum = 0
            for m in range(len(g_values)):
                g_sum += g_values[m]
                if g_sum > max_g_sum:
                    max_g_sum = g_sum
                    best_m = m

            # Step 5: Record the maximum total reduction cost
            g_max = max_g_sum

            # Step 6: Determine whether to continue iterating
            if g_max <= 0:
                break
            # Exchange the m pairs of nodes before
            for i in range(best_m + 1):
                a, b = exchange_pairs[i]
                col_a = next(j for j, subset in enumerate(self.partitions) if a in subset)
                col_b = next(j for j, subset in enumerate(self.partitions) if b in subset)
                self.partitions[col_a].remove(a)
                self.partitions[col_b].append(a)
                self.partitions[col_b].remove(b)
                self.partitions[col_a].append(b)
        return

    def allocate_qubits(self):
        """
        Initialize the partitions
        """
        self.partitions = []
        cnt_qubits = 0
        for qpu_size in self.network.backend_sizes:
            remain = self.circ.num_qubits - cnt_qubits
            if remain == 0:
                break
            end_index = min(cnt_qubits + qpu_size, self.circ.num_qubits)
            partition = list(range(cnt_qubits, end_index))
            self.partitions.append(partition)
            cnt_qubits = end_index
        assert(cnt_qubits == self.circ.num_qubits)
        for _ in range(len(self.partitions), self.network.num_backends):
            self.partitions.append([])
        return
    
    def calculate_w(self, graph, node, subset):
        """
        计算节点到子集的边权重之和
        """
        weight_sum = 0
        for neighbor in subset:
            if graph.has_edge(node, neighbor):
                weight_sum += graph[node][neighbor].get('weight', 1)
        return weight_sum

    def calculate_d(self, graph, node, target_subset, current_subset):
        """
        计算 D 值
        """
        w_target = self.calculate_w(graph, node, target_subset)
        w_current = self.calculate_w(graph, node, current_subset)
        return w_target - w_current

    def count_cut_edges(self, graph, partitions):
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, partition in enumerate(partitions):
            for node in partition:
                node_to_partition[node] = i
        cut_edges = 0
        W_eff = self.network.W_eff
        self.fidelity_loss = 0
        for u, v in graph.edges(): # 遍历图中的每一条边
            qpu_u = node_to_partition[u]
            qpu_v = node_to_partition[v]
            if qpu_u != qpu_v:
                path_len = self.network.Hops[qpu_u][qpu_v]
                cut_edges += path_len * graph[u][v]['weight']
                self.fidelity_loss += (1 - W_eff[qpu_u][qpu_v]) * graph[u][v]['weight']
        return cut_edges

    def save_path_to(self, filename="./outputs/paths/oee"):
        filename = f"{filename}.txt"
        with open(filename, 'w') as file:
            for partition in self.path:
                # 将行中的每个元素转换为字符串并写入文件
                file.write(' '.join(map(str, partition)) + '\n')
        return

class FGP_rOEE(OEE):
    @property
    def name(self):
        return "FGP-rOEE"

    def distribute(self):
        print(f"[FGP_rOEE]")
        start_time = time.time()
        self.k_way_FGP_rOEE()
        end_time = time.time()
        print(f"#comms: {self.num_comms}")
        print(f"#gate_comms: {self.num_gates}")
        print(f"#swap_comms: {self.num_swaps}")
        self.exec_time = end_time - start_time
        print(f"Time: {self.exec_time} seconds\n\n")
        self.save_path_to(f"./outputs/paths/{self.circ.name[:11]}_{self.circ.num_qubits}_FGP_rOEE")
        return

    def k_way_FGP_rOEE(self):
        self.dag = circuit_to_dag(self.circ)
        self.layers = list(self.dag.layers())
        self.num_depths = self.circ.depth()
        print(f"num_depths: {self.num_depths}")
        self.allocate_qubits()
        self.path = []
        self.num_gates = 0
        self.num_swaps = 0
        for lev in range(self.num_depths):
            lookahead_graph, time_slice_graph = self.build_lookahead_graphs(lev)
            num_gate_cut = self.k_way_rOEE(lookahead_graph, time_slice_graph)
            self.path.append(copy.deepcopy(self.partitions))
            self.num_gates += num_gate_cut
            if lev > 0:
                min_num_comms = self.calculate_nonlocal_communications(self.path[-2], self.path[-1])
                self.num_swaps += min_num_comms
        self.num_comms = self.num_gates + self.num_swaps
        return

    def build_lookahead_graphs(self, level):
        def lookahead_weight(n, sigma=1.0):
            return 2 ** (-n / sigma)
        G = nx.Graph()
        G.add_nodes_from(range(self.circ.num_qubits))
        for current_level in range(level, len(self.layers)):
            weight = lookahead_weight(current_level - level) # the lookahead weight of the current level
            if current_level == level:
                weight = 999 # float('inf')
            for node in self.layers[current_level]["graph"].op_nodes():
                # print(f"node.op: {node.op}, node.qargs: {node.qargs}, node.cargs: {node.cargs}")
                if len(node.qargs) == 2:
                    qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                    if qubits[0] == None:
                        qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                        # print(f"none: qubits: {qubits}")
                        # exit(0)
                    if G.has_edge(qubits[0], qubits[1]):
                        G[qubits[0]][qubits[1]]['weight'] += weight
                    else:
                        G.add_edge(qubits[0], qubits[1], weight=weight)
        # 返回当前层的图
        G_current = nx.Graph()
        G_current.add_nodes_from(range(self.circ.num_qubits))
        for node in self.layers[level]["graph"].op_nodes():
            if len(node.qargs) == 2:
                qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                if qubits[0] == None:
                    qubits = [self.circ.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                if G_current.has_edge(qubits[0], qubits[1]):
                    G_current[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    G_current.add_edge(qubits[0], qubits[1], weight=1)
        return G, G_current

    def k_way_rOEE(self, lagraph, tsgraph):
        nodes = list(lagraph.nodes())
        n = len(nodes)
        cnt = 0
        k = len(self.partitions)

        while self.count_cut_edges(tsgraph, self.partitions) != 0:
            cnt += 1
            if cnt > self.iteration:
                break
            # print(f"=== iteration {cnt} ===")
            C = nodes.copy()
            D = np.zeros((n, k))
            # 步骤 1: 计算每个节点 i 和每个子集 l 对应的 D(i, l) 值
            for node in nodes:
                current_col = next(j for j, subset in enumerate(self.partitions) if node in subset)
                for l in range(k):
                    D[node, l] = self.calculate_d(lagraph, node, self.partitions[l], self.partitions[current_col])
            g_values = []
            exchange_pairs = []
            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None
                # 步骤 2: 寻找使减少交换成本 g(a, b) 最大的两个节点 a 和 b
                for a in C:
                    for b in C:
                        if a < b:
                            col_a = next(j for j, subset in enumerate(self.partitions) if a in subset)
                            col_b = next(j for j, subset in enumerate(self.partitions) if b in subset)
                            if lagraph.has_edge(a, b):
                                g = D[a, col_b] + D[b, col_a] - 2 * lagraph[a][b].get('weight', 1)
                            else:
                                g = D[a, col_b] + D[b, col_a]
                            if g > max_g:
                                max_g = g
                                best_a, best_b = a, b
                # print(f"remove: {best_a}, {best_b}, max_g: {max_g}")
                C.remove(best_a)
                C.remove(best_b)
                g_values.append(max_g)
                exchange_pairs.append((best_a, best_b))

                # 步骤 3: 更新 D 值
                col_a = next(j for j, subset in enumerate(self.partitions) if best_a in subset)
                col_b = next(j for j, subset in enumerate(self.partitions) if best_b in subset)
                # print(f"col_a: {col_a}, col_b: {col_b}")
                for node in C:
                    col_i = next(j for j, subset in enumerate(self.partitions) if node in subset)
                    w_ia = lagraph[best_a][node].get('weight', 1) if lagraph.has_edge(best_a, node) else 0
                    w_ib = lagraph[best_b][node].get('weight', 1) if lagraph.has_edge(best_b, node) else 0
                    # print(f"w_ia: {w_ia}, w_ib: {w_ib}")
                    for l in range(k):
                        if l == col_a:
                            if col_i != col_a and col_i != col_b:
                                D[node, l] += w_ib - w_ia
                            elif col_i == col_b:
                                D[node, l] += 2 * w_ib - 2 * w_ia
                        elif l == col_b:
                            if col_i != col_a and col_i != col_b:
                                D[node, l] += w_ia - w_ib
                            elif col_i == col_a:
                                D[node, l] += 2 * w_ia - 2 * w_ib
                        elif col_i == col_a and l != col_a and l != col_b:
                            D[node, l] += w_ia - w_ib
                        elif col_i == col_b and l != col_a and l != col_b:
                            D[node, l] += w_ib - w_ia
            # 步骤 5: 寻找最优时间 m
            max_g_sum = float('-inf')
            best_m = 0
            g_sum = 0
            for m in range(len(g_values)):
                g_sum += g_values[m]
                if g_sum > max_g_sum:
                    max_g_sum = g_sum
                    best_m = m
            # 步骤 6: 记录最大总减少成本
            g_max = max_g_sum
            # 步骤 7: 判断是否继续迭代
            if g_max <= 0:
                break
            # 交换前 m 对节点
            for i in range(best_m + 1):
                a, b = exchange_pairs[i]
                col_a = next(j for j, subset in enumerate(self.partitions) if a in subset)
                col_b = next(j for j, subset in enumerate(self.partitions) if b in subset)
                self.partitions[col_a].remove(a)
                self.partitions[col_b].append(a)
                self.partitions[col_b].remove(b)
                self.partitions[col_a].append(b)
        num_gate_cut = self.count_cut_edges(tsgraph, self.partitions)
        return num_gate_cut

    def calculate_nonlocal_communications(self, prev_assign, curr_assign):
        num_qubits = self.circ.num_qubits
        G = nx.DiGraph() # 初始化有向图
        G.add_nodes_from(range(len(prev_assign))) # 每个partition对应一个节点

        communication_cost = 0

        # 记录每个qubit在prev和curr的分区号
        qubit_mapping = [[-1, -1] for _ in range(num_qubits)]
        for pno, partition in enumerate(prev_assign):
            # print(f"{pno}: {partition}")
            for qubit in partition:
                qubit_mapping[qubit][0] = pno
        for pno, partition in enumerate(curr_assign):
            # print(f"{pno}: {partition}")
            for qubit in partition:
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
                        2 * self.network.Hops[curr_part][prev_part] - 1 # RSWAP
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
            assert(3 <= length <= self.network.num_backends)
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            assert(3 <= length <= self.network.num_backends)
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
                    swap_cost = 2 * self.network.Hops[u][v] - 1
                    communication_cost += swap_cost * weight

        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            path_len = 2 * self.network.Hops[u][v] - 1
            communication_cost += path_len * data['weight']

        return communication_cost
