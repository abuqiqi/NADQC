from qiskit import QuantumCircuit
import time
import sys
import networkx as nx
import numpy as np
from collections import defaultdict
from qiskit.converters import circuit_to_dag, dag_to_circuit

from .backends import Network
from .partitioner import KWayPartitioner

class NADQC:
    def __init__(self, 
                 circ: QuantumCircuit, 
                 network: Network,
                 partition_method: str = "recursive"):
        self.circ = circ
        self.network = network
        self.partitioner = KWayPartitioner(network, max_options=1)
        self.partition_method = partition_method
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

        end_time = time.time()
        print(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        return

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
        legal_partitions = self.partitioner.partition(components, self.partition_method)
        return legal_partitions

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

    def get_partition_plan(self):
        self.subc_ranges = []
        self._get_sliced_subc(0, len(self.P) - 1)
        partition_plan = []
        # mapping_sequence = []
        for (i, j) in self.subc_ranges:
            # 通过P[i][j][0]，获取每个qubit被映射到哪个逻辑QPU
            partition_plan.append(self.P[i][j][0]) # 仅选择第一个划分方案
        return partition_plan

    def calculate_comm_cost_dynamic(self, partition_plan):
        """
        动态映射：为每个partition切换计算通信成本和最优映射序列
        :param partition_plan: list of partitions for each time slice
            Example: [[[0,1], [2]], [[0,2], [1]]] 
            - Time slice 0: logical QPU 0 has qubits [0,1], logical QPU 1 has qubit [2]
            - Time slice 1: logical QPU 0 has qubits [0,2], logical QPU 1 has qubit [1]
        :return: (total_comm_cost, mapping_sequence)
        """
        start_time = time.time()
        k = len(partition_plan)  # 时间片数量
        m_physical = self.network.num_backends  # 物理QPU数量
        n = self.circ.num_qubits  # 量子比特数
        
        # 结果存储
        total_comm_cost = 0.0
        mapping_sequence = []  # mapping_sequence[t][i] = time slice t 中逻辑QPU i 映射到的物理QPU
        
        # 1. 设置初始映射：时间片0的逻辑QPU到物理QPU的映射
        num_logical_qpus_t0 = len(partition_plan[0])
        assert num_logical_qpus_t0 <= m_physical, f"Time slice 0 has {num_logical_qpus_t0} logical QPUs but only {m_physical} physical QPUs available"
        
        # 简单策略：identity mapping (逻辑QPU i -> 物理QPU i)
        initial_mapping = list(range(num_logical_qpus_t0))
        mapping_sequence.append(initial_mapping)
        
        print(f"[Initial Mapping] Time slice 0: {initial_mapping} (logical -> physical)", file=sys.stderr)
        
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
            switch_cost = self._evaluate_switch_cost(
                D_switch,
                mapping_sequence[t],
                next_mapping,
                qubit_movements
            )
            
            total_comm_cost += switch_cost
            mapping_sequence.append(next_mapping)
            
            print(f"[Switch {t}->{t+1}] Cost: {switch_cost:.4f}, Mapping: {next_mapping}", file=sys.stderr)
            # 调试：打印量子比特移动
            print(f"  Qubit movements: {qubit_movements}", file=sys.stderr)
        
        end_time = time.time()
        print(f"[calculate_comm_cost_dynamic] Total comm cost: {total_comm_cost:.4f}, Time: {end_time - start_time:.4f}s", file=sys.stderr)
        
        return total_comm_cost, mapping_sequence

    def _compute_switch_demand(self, current_partition, next_partition):
        """
        计算单次切换的通信需求
        :param current_partition: 当前时间片的分区 [[qubits_in_logical_0], [qubits_in_logical_1], ...]
        :param next_partition: 下一时间片的分区 [[qubits_in_logical_0], [qubits_in_logical_1], ...]
        :return: (D_switch, qubit_movements)
            - D_switch: m_logical x m_logical 矩阵，D_switch[i][j] = 从逻辑QPU i 移动到 j 的量子比特数
            - qubit_movements: dict {qubit: (from_logical, to_logical)}
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
        for qubit in range(self.circ.num_qubits):
            curr_logical = current_qubit_to_logical[qubit]
            next_logical = next_qubit_to_logical[qubit]
            
            if curr_logical != next_logical:
                D_switch[curr_logical][next_logical] += 1
                qubit_movements[qubit] = (curr_logical, next_logical)
        
        return D_switch, qubit_movements

    def _find_optimal_mapping_for_switch(self, D_switch, current_mapping, num_logical_next, qubit_movements):
        """
        为切换找到最优的下一映射
        :param D_switch: 通信需求矩阵 (m_logical_current x m_logical_next)
        :param current_mapping: 当前时间片的物理映射 [logical_current -> physical]
        :param num_logical_next: 下一时间片的逻辑QPU数量
        :param qubit_movements: 量子比特移动详情 {qubit: (from_logical, to_logical)}
        :return: next_mapping, 下一时间片的最优物理映射 [logical_next -> physical]
        """
        m_physical = self.network.num_backends
        
        # 获取有效保真度矩阵
        # if not hasattr(self.network, 'W_eff'):
        #     self.network.W_eff, _ = self.network.compute_effective_fidelity()
        W_eff = self.network.W_eff
        
        # 策略：贪心分配 - 高需求逻辑QPU优先分配到高保真度位置
        
        # 1. 计算下一时间片中每个逻辑QPU的重要性
        logical_importance = np.zeros(num_logical_next)
        for i in range(D_switch.shape[0]):
            for j in range(D_switch.shape[1]):
                logical_importance[j] += D_switch[i][j]  # 流入逻辑QPU j 的需求
        
        # 2. 计算当前时间片中每个物理QPU的"热度"（高需求量子比特所在位置）
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
        for logical, physical in zip(remaining_logicals, remaining_physicals):
            next_mapping[logical] = physical
        
        return next_mapping

    def _evaluate_switch_cost(self, D_switch, mapping_current, mapping_next, qubit_movements):
        """
        计算切换成本
        :param D_switch: 通信需求矩阵
        :param mapping_current: 当前映射 [logical_current -> physical]
        :param mapping_next: 下一映射 [logical_next -> physical]
        :param qubit_movements: 量子比特移动详情 {qubit: (from_logical, to_logical)}
        :return: 切换成本
        """
        if not hasattr(self.network, 'W_eff'):
            self.network.W_eff, _ = self.network.compute_effective_fidelity()
        W_eff = self.network.W_eff
        
        total_cost = 0.0
        
        # 方法1: 基于量子比特移动计算
        for qubit, (from_logical, to_logical) in qubit_movements.items():
            from_physical = mapping_current[from_logical]
            to_physical = mapping_next[to_logical]
            # 移动成本 = 1 - 保真度
            cost = 1 - W_eff[from_physical][to_physical]
            total_cost += cost
        
        # 方法2: 基于需求矩阵计算（作为验证）
        # validation_cost = 0.0
        # for i in range(D_switch.shape[0]):
        #     for j in range(D_switch.shape[1]):
        #         if D_switch[i][j] > 0:
        #             from_physical = mapping_current[i]
        #             to_physical = mapping_next[j]
        #             cost = (1 - W_eff[from_physical][to_physical]) * D_switch[i][j]
        #             validation_cost += cost
        
        return total_cost