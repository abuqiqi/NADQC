from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from typing import Any, Optional
import networkx as nx
import numpy as np
import copy
import time
from collections import defaultdict

from ..compiler import Compiler, MappingRecord, MappingRecordList
from ..utils import Network

class FGP_rOEE(Compiler):
    """
    FGP_rOEE
    """
    compiler_id = "fgproee"

    def __init__(self):
        super().__init__()
        return

    @property
    def name(self) -> str:
        return "FGP-rOEE"

    def compile(self, circuit: QuantumCircuit, 
                    network: Network, 
                    config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit
        """
        print(f"Compiling with [{self.name}]...")

        start_time = time.time()
        circuit_name = config.get("circuit_name", "circ") if config else "circ"
        
        mapping_record_list = self._k_way_FGP_rOEE(circuit, network)
        
        end_time = time.time()

        mapping_record_list.add_cost("exec_time (sec)", end_time - start_time)
        mapping_record_list = self.evaluate_total_costs(mapping_record_list)
        # TODO: 计算num_comms
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        return mapping_record_list
   
    def _k_way_FGP_rOEE(self, circuit: QuantumCircuit, network: Network) -> MappingRecordList:
        circuit_dag = circuit_to_dag(circuit)
        self.layers = list(circuit_dag.layers())
        circuit_depth = circuit.depth()
        print(f"[DEBUG] num_depths: {circuit_depth}")
        
        partition = self.allocate_qubits(circuit.num_qubits, network)
        mapping_record_list = MappingRecordList()

        for lev in range(circuit_depth):
            lookahead_graph, time_slice_graph = self._build_lookahead_graphs(circuit, lev)
        
        # self.path = []
        # self.num_gates = 0
        # self.num_swaps = 0
        # for lev in range(circuit_depth):
            # lookahead_graph, time_slice_graph = self._build_lookahead_graphs(lev)
            mapping_record = self.k_way_rOEE(lookahead_graph, time_slice_graph)
            mapping_record_list.add_record(mapping_record)


            # num_gate_cut = self.k_way_rOEE(lookahead_graph, time_slice_graph)
            # self.path.append(copy.deepcopy(self.partitions))
            # self.num_gates += num_gate_cut
            if lev > 0:
                min_num_comms = self.calculate_nonlocal_communications(self.path[-2], self.path[-1])
                self.num_swaps += min_num_comms
        self.num_comms = self.num_gates + self.num_swaps

        return mapping_record_list

    def _build_lookahead_graphs(self, circuit: QuantumCircuit, level: int):
        def lookahead_weight(n, sigma=1.0):
            return 2 ** (-n / sigma)
        

    # def build_lookahead_graphs(self, level):
    #     def lookahead_weight(n, sigma=1.0):
    #         return 2 ** (-n / sigma)
        G = nx.Graph()
        G.add_nodes_from(range(circuit.num_qubits))
        for current_level in range(level, len(self.layers)):
            weight = lookahead_weight(current_level - level) # the lookahead weight of the current level
            if current_level == level:
                weight = 999 # float('inf')
            for node in self.layers[current_level]["graph"].op_nodes():
                # print(f"node.op: {node.op}, node.qargs: {node.qargs}, node.cargs: {node.cargs}")
                if len(node.qargs) == 2:
                    qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                    if qubits[0] == None:
                        qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                        # print(f"none: qubits: {qubits}")
                        # exit(0)
                    if G.has_edge(qubits[0], qubits[1]):
                        G[qubits[0]][qubits[1]]['weight'] += weight
                    else:
                        G.add_edge(qubits[0], qubits[1], weight=weight)
        # 返回当前层的图
        G_current = nx.Graph()
        G_current.add_nodes_from(range(circuit.num_qubits))
        for node in self.layers[level]["graph"].op_nodes():
            if len(node.qargs) == 2:
                qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                if qubits[0] == None:
                    qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
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
