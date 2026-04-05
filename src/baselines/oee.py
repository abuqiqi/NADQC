import networkx as nx
import numpy as np

from ..utils import Network

class OEE:
    def __init__(self):
        pass

    @classmethod
    def partition(cls, 
                  initial_partition: list[list[int]], 
                  qig: nx.Graph, 
                  network: Network, 
                  iteration_count: int) -> list[list[int]]:
        """
        Partition the qubits into k subsets using the OEE algorithm
        @param partition: Initial partition of qubits into subsets
        @param qig: Qubit interaction graph
        @param network: Network information for partitioning
        @param iteration_count: Number of iterations for the OEE algorithm
        @return: Final partition of qubits into subsets
        """
        nodes = list(qig.nodes())
        partition = [subset.copy() for subset in initial_partition]  # 深拷贝，避免修改原输入
        num_qubits = len(nodes)
        k = network.num_backends

        # 建立节点索引映射 (Node -> Index)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # 建立节点 -> 分区索引的字典 (Node -> PartitionID)
        node_to_part = {}
        partition_set: list[set[int]] = []

        for part_idx, subset in enumerate(partition):
            part_set = set(subset)
            partition_set.append(part_set)
            for node in subset:
                node_to_part[node] = part_idx

        for itr in range(iteration_count):
            # cut_edge = 0
            # # 遍历原图所有边，只需找到一条跨分区边即可判定需要继续
            # for u, v in qig.edges():
            #     if node_to_part[u] != node_to_part[v]:
            #         cut_edge += 1
            #         print(f"[DEBUG] node[{u}] in part[{node_to_part[u]}], node[{v}] in part[{node_to_part[v]}]")
            # print(f"[DEBUG] itr[{itr}] cut_edge: {cut_edge}")

            # C = nodes.copy()
            # ============================================================
            # 将候选集 C 从 list 改为 set
            # 后续的 C.remove(best_a) 操作，list 是 O(n)，set 是 O(1)
            # ============================================================
            C = set(nodes)
            D = np.zeros((num_qubits, k), dtype=np.float64)
            
            # Step 1: Calculate the D(i, l) value corresponding to each node i and each subset l
            for node in nodes:
                current_col = node_to_part[node]
                node_idx = node_to_idx[node] # 【优化 1 应用】
                for l in range(k):
                    D[node_idx, l] = cls._calculate_d(qig, node, partition_set[l], partition_set[current_col])

            g_values = []
            exchange_pairs = []

            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None

                # 【性能优化】将候选集按分区分组，仅遍历跨分区Pair
                part_to_nodes = cls._group_candidate_by_partition(C, node_to_part)
                parts_in_C = list(part_to_nodes.keys())

                # 遍历所有跨分区的节点对
                for i in range(len(parts_in_C)):
                    p1 = parts_in_C[i]
                    nodes_p1 = part_to_nodes[p1]
                    for j in range(i + 1, len(parts_in_C)):
                        p2 = parts_in_C[j]
                        nodes_p2 = part_to_nodes[p2]
                        
                        # 在p1和p2的节点中找最大g(a,b)
                        current_max_pair = cls._find_max_g_in_two_parts(
                            nodes_p1, nodes_p2, p1, p2, qig, node_to_idx, D
                        )

                        if current_max_pair[2] > max_g:
                            max_g = current_max_pair[2]
                            best_a, best_b = current_max_pair[0], current_max_pair[1]

                if best_a is None or best_b is None:
                    break  # 无有效可交换pair，退出当前pass的while循环

                # # Step 2: Find the two nodes a and b that maximize the reduction in exchange cost g(a, b)
                # for a, b in itertools.combinations(C, 2):  # 直接生成所有 (a,b) 组合，且 a≠b、不重复

                # # for a in C:
                #     col_a = node_to_part[a] # 【优化 2 应用】
                #     idx_a = node_to_idx[a]  # 【优化 1 应用】
                    
                #     # for b in C:
                #     col_b = node_to_part[b] # 【优化 2 应用】
                #     idx_b = node_to_idx[b]  # 【优化 1 应用】

                #     if col_a == col_b: # CHECK: 这里是否要跳过同分区内的节点？
                #         continue

                #     if qig.has_edge(a, b):
                #         g = D[idx_a, col_b] + D[idx_b, col_a] - 2 * qig[a][b].get('weight', 1)
                #     else:
                #         g = D[idx_a, col_b] + D[idx_b, col_a]
                    
                #     if g > max_g:
                #         max_g = g
                #         best_a, best_b = a, b
                # # print(f"remove: {best_a}, {best_b}, max_g: {max_g}")
                # # 【优化 3 应用】Set 的 remove 极快
                
                # if best_a is None or best_b is None:
                #     break  # 无有效可交换pair，退出当前pass的while循环
                
                C.remove(best_a)
                C.remove(best_b)
                # print(C)
                g_values.append(max_g)
                exchange_pairs.append((best_a, best_b))

                # Step 3: Update D-values
                # col_a = next(j for j, subset in enumerate(partition) if best_a in subset)
                # col_b = next(j for j, subset in enumerate(partition) if best_b in subset)
                col_a = node_to_part[best_a] # 【优化 2 应用】
                col_b = node_to_part[best_b] # 【优化 2 应用】
                # idx_a = node_to_idx[best_a]  # 【优化 1 应用】
                # idx_b = node_to_idx[best_b]  # 【优化 1 应用】

                for node in C:
                    # col_i = next(j for j, subset in enumerate(partition) if node in subset)
                    col_i = node_to_part[node] # 【优化 2 应用】
                    node_idx = node_to_idx[node] # 【优化 1 应用】
                    
                    w_ia = qig[best_a][node].get('weight', 1) if qig.has_edge(best_a, node) else 0
                    w_ib = qig[best_b][node].get('weight', 1) if qig.has_edge(best_b, node) else 0
                    # print(f"w_ia: {w_ia}, w_ib: {w_ib}")
                    
                    for l in range(k):
                        if l == col_a:
                            if col_i != col_a and col_i != col_b:
                                D[node_idx, l] += w_ib - w_ia
                            elif col_i == col_b:
                                D[node_idx, l] += 2 * w_ib - 2 * w_ia
                        elif l == col_b:
                            if col_i != col_a and col_i != col_b:
                                D[node_idx, l] += w_ia - w_ib
                            elif col_i == col_a:
                                D[node_idx, l] += 2 * w_ia - 2 * w_ib
                        elif col_i == col_a and l != col_a and l != col_b:
                            D[node_idx, l] += w_ia - w_ib
                        elif col_i == col_b and l != col_a and l != col_b:
                            D[node_idx, l] += w_ib - w_ia

            # Step 4: Find the optimal time m
            max_g_sum = float('-inf')
            best_m = -1
            g_sum = 0
            for m in range(len(g_values)):
                g_sum += g_values[m]
                if g_sum > max_g_sum:
                    max_g_sum = g_sum
                    best_m = m

            if max_g_sum <= 0:
                best_m = -1  # 如果所有增益都是负的，设置 m=-1 表示不执行交换

            # print(f"[DEBUG] max_g_sum: {max_g_sum}, best_m: {best_m}")
            # print(f"[DEBUG] exchange_pairs: {exchange_pairs}")

            # Step 5: Determine whether to continue iterating
            if max_g_sum <= 0 or best_m == -1:
                # print(f"[DEBUG] iteration: {itr}")
                break

            # Exchange the m pairs of nodes before
            for i in range(best_m + 1):
                a, b = exchange_pairs[i]
                col_a = node_to_part[a] # 【优化 2 应用】
                col_b = node_to_part[b] # 【优化 2 应用】
                
                partition[col_a].remove(a)
                partition[col_b].append(a)
                partition[col_b].remove(b)
                partition[col_a].append(b)

                partition_set[col_a].remove(a)
                partition_set[col_b].add(a)
                partition_set[col_b].remove(b)
                partition_set[col_a].add(b)

                # ============================================================
                # 必须同步更新字典！
                # ============================================================
                node_to_part[a], node_to_part[b] = node_to_part[b], node_to_part[a]

        return partition

    @classmethod
    def _calculate_d(cls, graph: nx.Graph, node: int, target_subset: set[int], current_subset: set[int]) -> float:
        """
        Calculate the D(i, l) value for a node and a target subset
        """
        w_target = cls._calculate_w(graph, node, target_subset)
        w_current = cls._calculate_w(graph, node, current_subset)
        return w_target - w_current

    @classmethod
    def _calculate_w(cls, graph: nx.Graph, node: int, subset: set[int]) -> float:
        """
        Calculate the sum of edge weights from a node to a subset of nodes
        """
        weight_sum = 0
        # 只遍历 node 的实际邻居，而不是遍历整个 subset 去 check has_edge
        for neighbor in graph.neighbors(node):
            if neighbor in subset:
                weight_sum += graph[node][neighbor].get('weight', 1)
        return weight_sum

    @classmethod
    def _group_candidate_by_partition(cls, C: set[int], node_to_part: dict[int, int]) -> dict[int, list[int]]:
        """将候选集C中的节点按分区ID分组"""
        part_to_nodes = {}
        for node in C:
            p = node_to_part[node]
            if p not in part_to_nodes:
                part_to_nodes[p] = []
            part_to_nodes[p].append(node)
        return part_to_nodes
    
    @classmethod
    def _find_max_g_in_two_parts(cls, 
                                 nodes_p1: list[int], nodes_p2: list[int],
                                 p1: int, p2: int, qig: nx.Graph,
                                 node_to_idx: dict[int, int], D: np.ndarray) -> tuple[int | None, int | None, float]:
        """在两个分区的节点中找到使g(a,b)最大的节点对"""
        max_g = float('-inf')
        best_a, best_b = None, None
        for a in nodes_p1:
            idx_a = node_to_idx[a]
            for b in nodes_p2:
                idx_b = node_to_idx[b]
                # 论文Proposition 2公式：g(i,j) = D(i,col(j)) + D(j,col(i)) - 2*w_ij
                if qig.has_edge(a, b):
                    g = D[idx_a, p2] + D[idx_b, p1] - 2 * qig[a][b].get('weight', 1.0)
                else:
                    g = D[idx_a, p2] + D[idx_b, p1]
                if g > max_g:
                    max_g = g
                    best_a, best_b = a, b
        return best_a, best_b, max_g
