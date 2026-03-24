import networkx as nx
import numpy as np
import itertools

from ..utils import Network

class OEE:
    def __init__(self):
        pass

    @classmethod
    def partition(cls, 
                  partition: list[list[int]], 
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
        num_qubits = len(nodes)
        k = network.num_backends

        # ============================================================
        # 【优化 1】建立节点索引映射 (Node -> Index)
        # 原因：防止节点 ID 不是从 0 开始的连续整数，导致 D[node, l] 报错 IndexError
        # ============================================================
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # ============================================================
        # 【优化 2】建立节点 -> 分区索引的字典 (Node -> PartitionID)
        # 原因：将原来 O(n) 的 next(...) 查找降为 O(1)，这是最核心的性能瓶颈修复
        # ============================================================
        node_to_part = {}
        for part_idx, subset in enumerate(partition):
            for node in subset:
                node_to_part[node] = part_idx

        for _ in range(iteration_count):
            # C = nodes.copy()
            # ============================================================
            # 【优化 3】将候选集 C 从 list 改为 set
            # 原因：后续的 C.remove(best_a) 操作，list 是 O(n)，set 是 O(1)
            # ============================================================
            C = set(nodes)
            D = np.zeros((num_qubits, k))
            # Step 1: Calculate the D(i, l) value corresponding to each node i and each subset l
            for node in nodes:
                # current_col = next(j for j, subset in enumerate(partition) if node in subset)
                # 【优化 2 应用】直接查字典，不再遍历
                current_col = node_to_part[node]
                node_idx = node_to_idx[node] # 【优化 1 应用】
                
                for l in range(k):
                    # D[node, l] = cls._calculate_d(qig, node, partition[l], partition[current_col])
                    # 注意：这里传入 partition[l] 是为了计算权重，逻辑保持不变
                    D[node_idx, l] = cls._calculate_d(qig, node, partition[l], partition[current_col])

            g_values = []
            exchange_pairs = []

            while len(C) > 1:
                max_g = float('-inf')
                best_a, best_b = None, None

                # Step 2: Find the two nodes a and b that maximize the reduction in exchange cost g(a, b)
                for a, b in itertools.combinations(C, 2):  # 直接生成所有 (a,b) 组合，且 a≠b、不重复

                # for a in C:
                    col_a = node_to_part[a] # 【优化 2 应用】
                    idx_a = node_to_idx[a]  # 【优化 1 应用】
                    
                    # for b in C:
                    col_b = node_to_part[b] # 【优化 2 应用】
                    idx_b = node_to_idx[b]  # 【优化 1 应用】

                        # if a < b:
                            # col_a = next(j for j, subset in enumerate(partition) if a in subset)
                            # col_b = next(j for j, subset in enumerate(partition) if b in subset)
                            
                            # if qig.has_edge(a, b):
                            #     g = D[a, col_b] + D[b, col_a] - 2 * qig[a][b].get('weight', 1)
                            # else:
                            #     g = D[a, col_b] + D[b, col_a]

                    if qig.has_edge(a, b):
                        g = D[idx_a, col_b] + D[idx_b, col_a] - 2 * qig[a][b].get('weight', 1)
                    else:
                        g = D[idx_a, col_b] + D[idx_b, col_a]
                    
                    if g > max_g:
                        max_g = g
                        best_a, best_b = a, b
                # print(f"remove: {best_a}, {best_b}, max_g: {max_g}")
                # 【优化 3 应用】Set 的 remove 极快
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
                idx_a = node_to_idx[best_a]  # 【优化 1 应用】
                idx_b = node_to_idx[best_b]  # 【优化 1 应用】

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
                # col_a = next(j for j, subset in enumerate(partition) if a in subset)
                # col_b = next(j for j, subset in enumerate(partition) if b in subset)
                col_a = node_to_part[a] # 【优化 2 应用】
                col_b = node_to_part[b] # 【优化 2 应用】
                
                partition[col_a].remove(a)
                partition[col_b].append(a)
                partition[col_b].remove(b)
                partition[col_a].append(b)

                # ============================================================
                # 【优化 2 附加】必须同步更新字典！
                # 原因：否则下一次迭代查字典就是错的
                # ============================================================
                node_to_part[a], node_to_part[b] = node_to_part[b], node_to_part[a]

        return partition

    @classmethod
    def _calculate_d(cls, graph: nx.Graph, node: int, target_subset: list[int], current_subset: list[int]) -> float:
        """
        Calculate the D(i, l) value for a node and a target subset
        """
        w_target = cls._calculate_w(graph, node, target_subset)
        w_current = cls._calculate_w(graph, node, current_subset)
        return w_target - w_current

    @classmethod
    def _calculate_w(cls, graph: nx.Graph, node: int, subset: list[int]) -> float:
        """
        Calculate the sum of edge weights from a node to a subset of nodes
        """
        # weight_sum = 0
        # for neighbor in subset:
        #     if graph.has_edge(node, neighbor):
        #         weight_sum += graph[node][neighbor].get('weight', 1)
        # return weight_sum
        """
        【优化 4】微优化权重计算
        原因：虽然这不是主要瓶颈，但利用 NetworkX 的邻居查询比遍历整个 subset 稍快
        """
        weight_sum = 0
        # 只遍历 node 的实际邻居，而不是遍历整个 subset 去 check has_edge
        for neighbor in graph.neighbors(node):
            if neighbor in subset:
                weight_sum += graph[node][neighbor].get('weight', 1)
        return weight_sum