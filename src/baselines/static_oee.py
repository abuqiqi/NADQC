from qiskit import QuantumCircuit
from typing import Any, Optional
import networkx as nx
import numpy as np
import time

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network

class StaticOEE(Compiler):
    """
    Static OEE
    """
    compiler_id = "staticoee"

    def __init__(self):
        super().__init__()
        return

    @property
    def name(self) -> str:
        return "Static OEE"

    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit using Static OEE algorithm
        """
        print(f"Compiling with [{self.name}]...")
        
        start_time = time.time()
        iteration_count = config.get("iteration", 10) if config else 10
        circuit_name = config.get("circuit_name", "circ") if config else "circ"

        qig = CompilerUtils.build_qubit_interaction_graph(circuit)
        
        # TODO: use OEE.partition
        partition = self._k_way_OEE(qig, network, iteration_count)

        # TODO: fine-grained mapping

        end_time = time.time()

        costs = CompilerUtils.evaluate_partition(qig, partition, network)
        
        record = MappingRecord(
            layer_start = 0, 
            layer_end = circuit.depth() - 1,
            partition = partition,
            mapping_type = "telegate",
            costs = costs
        )
        mapping_record_list = MappingRecordList()
        mapping_record_list.add_record(record)
        
        mapping_record_list.add_cost("exec_time (sec)", end_time - start_time)
        mapping_record_list = CompilerUtils.evaluate_total_costs(mapping_record_list)
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        return mapping_record_list
    
    def _k_way_OEE(self, 
                   qig: nx.Graph, 
                   network: Network, 
                   iteration_count: int) -> list[list[int]]:
        """
        Partition the qubits into k subsets using the OEE algorithm
        """
        nodes = list(qig.nodes())
        num_qubits = len(nodes)
        k = network.num_backends
        partition = CompilerUtils.allocate_qubits(num_qubits, network) # initialize partition
        for _ in range(iteration_count):
            C = nodes.copy()
            D = np.zeros((num_qubits, k))
            # Step 1: Calculate the D(i, l) value corresponding to each node i and each subset l
            for node in nodes:
                current_col = next(j for j, subset in enumerate(partition) if node in subset)
                for l in range(k):
                    D[node, l] = self._calculate_d(qig, node, partition[l], partition[current_col])
            g_values = []
            exchange_pairs = []
            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None
                # Step 2: Find the two nodes a and b that maximize the reduction in exchange cost g(a, b)
                for a in C:
                    for b in C:
                        if a < b:
                            col_a = next(j for j, subset in enumerate(partition) if a in subset)
                            col_b = next(j for j, subset in enumerate(partition) if b in subset)
                            if qig.has_edge(a, b):
                                g = D[a, col_b] + D[b, col_a] - 2 * qig[a][b].get('weight', 1)
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
                col_a = next(j for j, subset in enumerate(partition) if best_a in subset)
                col_b = next(j for j, subset in enumerate(partition) if best_b in subset)
                for node in C:
                    col_i = next(j for j, subset in enumerate(partition) if node in subset)
                    w_ia = qig[best_a][node].get('weight', 1) if qig.has_edge(best_a, node) else 0
                    w_ib = qig[best_b][node].get('weight', 1) if qig.has_edge(best_b, node) else 0
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
                col_a = next(j for j, subset in enumerate(partition) if a in subset)
                col_b = next(j for j, subset in enumerate(partition) if b in subset)
                partition[col_a].remove(a)
                partition[col_b].append(a)
                partition[col_b].remove(b)
                partition[col_a].append(b)
        return partition

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
