from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from typing import Any, Optional
import networkx as nx
import numpy as np
import time

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
        iteration_count = config.get("iteration", 10) if config else 10
        circuit_name = config.get("circuit_name", "circ") if config else "circ"
        
        mapping_record_list = self._k_way_FGP_rOEE(circuit, network, iteration_count)
        
        end_time = time.time()

        mapping_record_list.add_cost("exec_time (sec)", end_time - start_time)
        mapping_record_list = self.evaluate_total_costs(mapping_record_list)
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        return mapping_record_list

    def _k_way_FGP_rOEE(self, circuit: QuantumCircuit, 
                        network: Network, 
                        iteration_count: int) -> MappingRecordList:
        circuit_dag = circuit_to_dag(circuit)
        circuit_layers = list(circuit_dag.layers())
        circuit_depth = circuit.depth()
        print(f"[DEBUG] num_depths: {circuit_depth}")
        
        partition = self.allocate_qubits(circuit.num_qubits, network)
        mapping_record_list = MappingRecordList()

        for lev in range(circuit_depth):
            lookahead_graph, time_slice_graph = self._build_lookahead_graphs(circuit, circuit_layers, lev)
            mapping_record = self.k_way_rOEE(partition, 
                                             lookahead_graph, 
                                             time_slice_graph, 
                                             network, 
                                             iteration_count)
            mapping_record.layer_start = lev
            mapping_record.layer_end = lev + 1
            mapping_record_list.add_record(mapping_record)
            if lev > 0:
                self.evaluate_partition_switch(mapping_record_list.records[-2], 
                                               mapping_record_list.records[-1],
                                               network)
        return mapping_record_list

    def _build_lookahead_graphs(self, circuit: QuantumCircuit, 
                                circuit_layers: list, 
                                level: int):
        def lookahead_weight(n, sigma=1.0):
            return 2 ** (-n / sigma)
        lookahead_graph = nx.Graph()
        lookahead_graph.add_nodes_from(range(circuit.num_qubits))
        for current_level in range(level, len(circuit_layers)):
            weight = lookahead_weight(current_level - level) # the lookahead weight of the current level
            if current_level == level:
                weight = 999 # float('inf')
            for node in circuit_layers[current_level]["graph"].op_nodes():
                # print(f"node.op: {node.op}, node.qargs: {node.qargs}, node.cargs: {node.cargs}")
                if len(node.qargs) == 2:
                    qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                    if qubits[0] == None:
                        qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                        # print(f"none: qubits: {qubits}")
                        # exit(0)
                    if lookahead_graph.has_edge(qubits[0], qubits[1]):
                        lookahead_graph[qubits[0]][qubits[1]]['weight'] += weight
                    else:
                        lookahead_graph.add_edge(qubits[0], qubits[1], weight=weight)
        # 返回当前层的图
        time_slice_graph = nx.Graph()
        time_slice_graph.add_nodes_from(range(circuit.num_qubits))
        for node in circuit_layers[level]["graph"].op_nodes():
            if len(node.qargs) == 2:
                qubits = [node.qargs[i]._index for i in range(len(node.qargs))]
                if qubits[0] == None:
                    qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                if time_slice_graph.has_edge(qubits[0], qubits[1]):
                    time_slice_graph[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    time_slice_graph.add_edge(qubits[0], qubits[1], weight=1)
        return lookahead_graph, time_slice_graph

    def k_way_rOEE(self, partition: list[list[int]], 
                   lookahead_graph: nx.Graph, 
                   time_slice_graph: nx.Graph, 
                   network: Network, 
                   iteration_count: int) -> MappingRecord:
        nodes = list(lookahead_graph.nodes())
        n = len(nodes)
        cnt = 0
        k = len(partition)

        costs = self.evaluate_partition(time_slice_graph, partition, network)

        while costs["remote_hops"] != 0:
            cnt += 1
            if cnt > iteration_count:
                break
            # print(f"=== iteration {cnt} ===")
            C = nodes.copy()
            D = np.zeros((n, k))
            # 步骤 1: 计算每个节点 i 和每个子集 l 对应的 D(i, l) 值
            for node in nodes:
                current_col = next(j for j, subset in enumerate(partition) if node in subset)
                for l in range(k):
                    D[node, l] = self._calculate_d(lookahead_graph, node, partition[l], partition[current_col])
            g_values = []
            exchange_pairs = []
            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None
                # 步骤 2: 寻找使减少交换成本 g(a, b) 最大的两个节点 a 和 b
                for a in C:
                    for b in C:
                        if a < b:
                            col_a = next(j for j, subset in enumerate(partition) if a in subset)
                            col_b = next(j for j, subset in enumerate(partition) if b in subset)
                            if lookahead_graph.has_edge(a, b):
                                g = D[a, col_b] + D[b, col_a] - 2 * lookahead_graph[a][b].get('weight', 1)
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
                col_a = next(j for j, subset in enumerate(partition) if best_a in subset)
                col_b = next(j for j, subset in enumerate(partition) if best_b in subset)
                # print(f"col_a: {col_a}, col_b: {col_b}")
                for node in C:
                    col_i = next(j for j, subset in enumerate(partition) if node in subset)
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
            # 步骤 6: 记录最大总减少成本
            g_max = max_g_sum
            # 步骤 7: 判断是否继续迭代
            if g_max <= 0:
                break
            # 交换前 m 对节点
            for i in range(best_m + 1):
                a, b = exchange_pairs[i]
                col_a = next(j for j, subset in enumerate(partition) if a in subset)
                col_b = next(j for j, subset in enumerate(partition) if b in subset)
                partition[col_a].remove(a)
                partition[col_b].append(a)
                partition[col_b].remove(b)
                partition[col_a].append(b)
        
        costs = self.evaluate_partition(time_slice_graph, partition, network)

        record = MappingRecord(
            layer_start = 0,
            layer_end = 0,
            partition = partition,
            mapping_type = "telegate",
            costs = costs
        )

        return record

    def _calculate_d(self, graph: nx.Graph, node: int, target_subset: list[int], current_subset: list[int]) -> float:
        """
        Calculate the D(i, l) value for a node and a target subset
        """
        w_target = self._calculate_w(graph, node, target_subset)
        w_current = self._calculate_w(graph, node, current_subset)
        return w_target - w_current
    
    def _calculate_w(self, graph: nx.Graph, node: int, subset: list[int]) -> float:
        """
        Calculate the sum of edge weights from a node to a subset of nodes
        """
        weight_sum = 0
        for neighbor in subset:
            if graph.has_edge(node, neighbor):
                weight_sum += graph[node][neighbor].get('weight', 1)
        return weight_sum
