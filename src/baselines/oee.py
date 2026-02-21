import networkx as nx
import numpy as np
import copy
import time
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import defaultdict

class OEE:
    def __init__(self, 
                 circ, 
                 network,
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
