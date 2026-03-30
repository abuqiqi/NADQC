from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from typing import Any, Optional
import networkx as nx
import numpy as np
import time

from .oee import OEE
from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network

class FGPrOEE(Compiler):
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

    def compile(self, 
                circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit
        """
        print(f"Compiling with [{self.name}]...")
        # self.circ = circuit
        # self.qpus = network.backend_sizes
        # self.swap_cost_matrix = build_fc_network(network.backend_sizes)
        
        # print(f"[DEBUG] swap_cost_matrix:\n{self.swap_cost_matrix}\n")
        # print(f"[DEBUG] network.Hops: {network.Hops}\n")

        start_time = time.time()
        iteration_count = config.get("iteration", 10) if config else 10
        circuit_name = config.get("circuit_name", "circ") if config else "circ"
        
        mapping_record_list = self._k_way_FGP_rOEE(circuit, network, iteration_count)
        
        end_time = time.time()

        mapping_record_list.summarize_total_costs()
        mapping_record_list.update_total_costs(execution_time = end_time - start_time)
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        
        # print(f"[DEBUG] num_swaps: {self.num_swaps}\nnum_gates: {self.num_gates}")
        print(f"[DEBUG] Total swaps: {sum(self.num_swaps)}, Total gates: {sum(self.num_gates)}")
        print(f"[DEBUG] Total comms: {sum(self.num_swaps) + sum(self.num_gates)}\n")
        return mapping_record_list

    def _k_way_FGP_rOEE(self, circuit: QuantumCircuit, 
                        network: Network, 
                        iteration_count: int) -> MappingRecordList:
        circuit_dag = circuit_to_dag(circuit)
        circuit_layers = list(circuit_dag.layers())
        circuit_depth = circuit.depth()
        print(f"[DEBUG] num_depths: {circuit_depth}")
        
        partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
        mapping_record_list = MappingRecordList()

        self.num_swaps = []
        self.num_gates = []

        for lev in range(circuit_depth):
            lookahead_graph, time_slice_graph = self._build_lookahead_graphs(circuit, circuit_layers, lev)
            partition = self._k_way_rOEE(partition,
                                         lookahead_graph, 
                                         time_slice_graph, 
                                         network, 
                                         iteration_count)
            # 获取子线路
            sub_qc = self._extract_subcircuit(circuit.num_qubits, circuit_layers, lev)

            # 评估划分
            record = MappingRecord(
                layer_start = lev, 
                layer_end = lev,
                partition = partition,
                mapping_type = "telegate"
            )
            # TODO: 检查为什么本地量子门数量这么少
            _ = CompilerUtils.evaluate_local_and_telegate(record, sub_qc, network)

            # 更新mapping_record_list
            mapping_record_list.add_record(record)

            if lev > 0:
                # TODO: CHECK
                # self.num_swaps.append(self.calculate_nonlocal_communications(mapping_record_list.records[-2].partition, 
                #                                                              mapping_record_list.records[-1].partition))
                CompilerUtils.evaluate_teledata(mapping_record_list.records[-2],
                                                mapping_record_list.records[-1],
                                                network)
        return mapping_record_list

    # def _build_lookahead_graphs(self, circuit: QuantumCircuit, 
    #                             circuit_layers: list, 
    #                             level: int):
    #     def lookahead_weight(n, sigma=1.0):
    #         return 2 ** (-n / sigma)
    #     lookahead_graph = nx.Graph()
    #     lookahead_graph.add_nodes_from(range(circuit.num_qubits))
    #     for current_level in range(level, len(circuit_layers)):
    #         weight = lookahead_weight(current_level - level) # the lookahead weight of the current level
    #         if current_level == level:
    #             weight = 999 # float('inf')
    #         for node in circuit_layers[current_level]["graph"].op_nodes():
    #             # print(f"node.op: {node.op}, node.qargs: {node.qargs}, node.cargs: {node.cargs}")
    #             if len(node.qargs) == 2:
    #                 qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
    #                 if qubits[0] == None:
    #                     qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
    #                     # print(f"none: qubits: {qubits}")
    #                     # exit(0)
    #                 if lookahead_graph.has_edge(qubits[0], qubits[1]):
    #                     lookahead_graph[qubits[0]][qubits[1]]['weight'] += weight
    #                 else:
    #                     lookahead_graph.add_edge(qubits[0], qubits[1], weight=weight)
    #     # 返回当前层的图
    #     time_slice_graph = nx.Graph()
    #     time_slice_graph.add_nodes_from(range(circuit.num_qubits))
    #     for node in circuit_layers[level]["graph"].op_nodes():
    #         if len(node.qargs) == 2:
    #             qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
    #             if qubits[0] == None:
    #                 qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
    #             if time_slice_graph.has_edge(qubits[0], qubits[1]):
    #                 time_slice_graph[qubits[0]][qubits[1]]['weight'] += 1
    #             else:
    #                 time_slice_graph.add_edge(qubits[0], qubits[1], weight=1)
    #     return lookahead_graph, time_slice_graph

    def _k_way_rOEE(self, partition: list[list[int]], 
                   lookahead_graph: nx.Graph, 
                   time_slice_graph: nx.Graph, 
                   network: Network, 
                   iteration_count: int) -> list[list[int]]:
        nodes = list(lookahead_graph.nodes())
        n = len(nodes)
        cnt = 0
        k = len(partition)

        # --- 优化1: 预构建节点到分区的映射 (O(1)查找) ---
        node_to_part = {}
        for part_idx, part in enumerate(partition):
            for node in part:
                node_to_part[node] = part_idx

        num_remote_hops = CompilerUtils.evaluate_remote_hops(time_slice_graph, partition, network)

        while num_remote_hops != 0:
            cnt += 1
            if cnt > iteration_count:
                break
            # print(f"=== iteration {cnt} ===")
            C = nodes.copy()
            D = np.zeros((n, k))
            # 步骤 1: 计算每个节点 i 和每个子集 l 对应的 D(i, l) 值
            for node in nodes:
                current_col = node_to_part[node]  # 优化查找
                for l in range(k):
                    D[node, l] = OEE._calculate_d(lookahead_graph, node, partition[l], partition[current_col])
            
            g_values = []
            exchange_pairs = []
            
            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None
                
                # --- 优化2: 更紧凑的循环结构 ---
                # 只遍历 a < b 的组合，避免重复检查
                for i in range(len(C)):
                    a = C[i]
                    col_a = node_to_part[a]
                    for j in range(i + 1, len(C)):
                        b = C[j]
                        col_b = node_to_part[b]
                        
                        if lookahead_graph.has_edge(a, b):
                            g = D[a, col_b] + D[b, col_a] - 2 * lookahead_graph[a][b].get('weight', 1)
                        else:
                            g = D[a, col_b] + D[b, col_a]
                        
                        if g > max_g:
                            max_g = g
                            best_a, best_b = a, b

                C.remove(best_a)
                C.remove(best_b)
                g_values.append(max_g)
                exchange_pairs.append((best_a, best_b))

                # 步骤 3: 更新 D 值
                col_a = node_to_part[best_a]
                col_b = node_to_part[best_b]
                
                for node in C:
                    col_i = node_to_part[node]
                    w_ia = lookahead_graph[best_a][node].get('weight', 1) if lookahead_graph.has_edge(best_a, node) else 0
                    w_ib = lookahead_graph[best_b][node].get('weight', 1) if lookahead_graph.has_edge(best_b, node) else 0
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
            
            if max_g_sum <= 0:
                break

            # 执行交换并更新映射
            for i in range(best_m + 1):
                a, b = exchange_pairs[i]
                col_a = node_to_part[a]
                col_b = node_to_part[b]

                # 更新 partition 数据结构
                partition[col_a].remove(a)
                partition[col_b].append(a)
                partition[col_b].remove(b)
                partition[col_a].append(b)

                # --- 优化3: 同步更新映射字典 ---
                node_to_part[a] = col_b
                node_to_part[b] = col_a

            # 更新下一轮迭代的条件
            num_remote_hops = CompilerUtils.evaluate_remote_hops(time_slice_graph, partition, network)

        return partition

    def _build_lookahead_graphs(self, circuit: QuantumCircuit, 
                                circuit_layers: list, 
                                level: int):
        def lookahead_weight(n, sigma=1.0):
            return 2 ** (-n / sigma)
            
        lookahead_graph = nx.Graph()
        lookahead_graph.add_nodes_from(range(circuit.num_qubits))
        
        # 构建 Lookahead 图
        for current_level in range(level, len(circuit_layers)):
            weight = lookahead_weight(current_level - level) # the lookahead weight of the current level
            if current_level == level:
                weight = 999  # 当前层权重极大化
                
            layer_graph = circuit_layers[current_level]["graph"]
            for node in layer_graph.op_nodes():
                qargs = node.qargs
                if len(qargs) != 2:
                    continue
                
                # 简化索引获取逻辑
                # q0_idx = qargs[0]._index
                # q1_idx = qargs[1]._index
                # if q0_idx is None:
                q0_idx = circuit.qubits.index(qargs[0])
                q1_idx = circuit.qubits.index(qargs[1])

                if lookahead_graph.has_edge(q0_idx, q1_idx):
                    lookahead_graph[q0_idx][q1_idx]['weight'] += weight
                else:
                    lookahead_graph.add_edge(q0_idx, q1_idx, weight=weight)

        # 构建当前层 Time Slice 图
        time_slice_graph = nx.Graph()
        time_slice_graph.add_nodes_from(range(circuit.num_qubits))
        layer_graph = circuit_layers[level]["graph"]
        
        for node in layer_graph.op_nodes():
            qargs = node.qargs
            if len(qargs) != 2:
                continue
                
            # q0_idx = qargs[0]._index
            # q1_idx = qargs[1]._index
            # if q0_idx is None:
            q0_idx = circuit.qubits.index(qargs[0])
            q1_idx = circuit.qubits.index(qargs[1])

            if time_slice_graph.has_edge(q0_idx, q1_idx):
                time_slice_graph[q0_idx][q1_idx]['weight'] += 1
            else:
                time_slice_graph.add_edge(q0_idx, q1_idx, weight=1)

        return lookahead_graph, time_slice_graph

    def _extract_subcircuit(self, num_qubits: int, circuit_layers: list, level: int) -> QuantumCircuit:
        sub_qc = QuantumCircuit(num_qubits)
        for node in circuit_layers[level]["graph"].op_nodes():
            sub_qc.append(node.op, qargs=node.qargs, cargs=node.cargs)
        return sub_qc

    # def _calculate_d(self, graph: nx.Graph, node: int, target_subset: list[int], current_subset: list[int]) -> float:
    #     """
    #     Calculate the D(i, l) value for a node and a target subset
    #     """
    #     w_target = self._calculate_w(graph, node, target_subset)
    #     w_current = self._calculate_w(graph, node, current_subset)
    #     return w_target - w_current
    
    # def _calculate_w(self, graph: nx.Graph, node: int, subset: list[int]) -> float:
    #     """
    #     Calculate the sum of edge weights from a node to a subset of nodes
    #     """
    #     weight_sum = 0
    #     for neighbor in subset:
    #         if graph.has_edge(node, neighbor):
    #             weight_sum += graph[node][neighbor].get('weight', 1)
    #     return weight_sum

    # def count_cut_edges(self, graph, partitions):
    #     node_to_partition = {} # 构建节点到划分编号的映射
    #     for i, partition in enumerate(partitions):
    #         for node in partition:
    #             node_to_partition[node] = i
    #     cut_edges = 0
    #     for u, v in graph.edges(): # 遍历图中的每一条边
    #         qpu_u = node_to_partition[u]
    #         qpu_v = node_to_partition[v]
    #         if qpu_u != qpu_v:
    #             path_len = (self.swap_cost_matrix[qpu_u][qpu_v] + 1) / 2
    #             cut_edges += path_len * graph[u][v]['weight']
    #     return cut_edges

    # def calculate_nonlocal_communications(self, prev_assign, curr_assign):
    #     num_qubits = self.circ.num_qubits
    #     G = nx.DiGraph() # 初始化有向图
    #     G.add_nodes_from(range(len(prev_assign))) # 每个partition对应一个节点

    #     communication_cost = 0

    #     # 记录每个qubit在prev和curr的分区号
    #     qubit_mapping = [[-1, -1] for _ in range(num_qubits)]
    #     for pno, partition in enumerate(prev_assign):
    #         # print(f"{pno}: {partition}")
    #         for qubit in partition:
    #             qubit_mapping[qubit][0] = pno
    #     for pno, partition in enumerate(curr_assign):
    #         # print(f"{pno}: {partition}")
    #         for qubit in partition:
    #             qubit_mapping[qubit][1] = pno

    #     # 遍历映射，若前后分配不同，添加边到图中
    #     for prev_part, curr_part in qubit_mapping:
    #         assert(prev_part != -1 and curr_part != -1)
    #         if prev_part != curr_part: # prev_part -> curr_part
    #             # 检查是否存在curr_part -> prev_part的边
    #             # 如果存在，则说明形成了环
    #             # 因为每次只加一条边，所以抵消掉一条就行
    #             if G.has_edge(curr_part, prev_part):
    #                 communication_cost += \
    #                     self.swap_cost_matrix[curr_part][prev_part] # one RSWAP
    #                 # 更新边权重
    #                 if G[curr_part][prev_part]['weight'] > 1:
    #                     G[curr_part][prev_part]['weight'] -= 1
    #                 else:
    #                     G.remove_edge(curr_part, prev_part)
    #             # 否则添加一条边prev_part -> curr_part
    #             else:
    #                 if G.has_edge(prev_part, curr_part):
    #                     G[prev_part][curr_part]['weight'] += 1
    #                 else:
    #                     G.add_edge(prev_part, curr_part, weight=1)

    #     all_cycles = nx.simple_cycles(G)
    #     cycles_by_length = defaultdict(list)
    #     # 收集长度大于2的环
    #     for cycle in all_cycles:
    #         length = len(cycle)
    #         assert(3 <= length <= len(self.qpus))
    #         cycles_by_length[length].append(cycle)

    #     for length in sorted(cycles_by_length.keys()):
    #         assert(3 <= length <= len(self.qpus))
    #         for cycle in cycles_by_length[length]:
    #             exist = True # 先检查是不是所有边都在
    #             weight = 999999
    #             for i in range(length):
    #                 u = cycle[i]
    #                 v = cycle[(i + 1) % length]
    #                 if not G.has_edge(u, v):
    #                     exist = False
    #                     break
    #                 weight = min(weight, G[u][v]['weight']) # 记录环的个数
    #             if not exist: # 当前环不存在了
    #                 continue
    #             for i in range(length): # 从G中移除这些环
    #                 u = cycle[i]
    #                 v = cycle[(i + 1) % length]
    #                 if G[u][v]['weight'] > weight:
    #                     G[u][v]['weight'] -= weight
    #                 else:
    #                     G.remove_edge(u, v)
    #                 # 对环中的每一条边，计算通信开销
    #                 swap_cost = self.swap_cost_matrix[u][v]
    #                 communication_cost += swap_cost * weight

    #     # 获取剩余的边
    #     remaining_edges = G.edges(data=True)
    #     for u, v, data in remaining_edges:
    #         path_len = (self.swap_cost_matrix[u][v] + 1) / 2
    #         communication_cost += path_len * data['weight']

    #     return communication_cost
