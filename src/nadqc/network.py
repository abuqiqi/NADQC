import random
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

class Network:
    def __init__(self, network_config: dict, backend_config: list):
        self.num_backends = len(backend_config)
        self.network_coupling = self._build_network_coupling(network_config)
        self.network_graph = self._build_weighted_network_graph(self.network_coupling)
        self.W_eff, self.Hops, self.optimal_paths = self._compute_effective_fidelity()
        self.backends = backend_config
        return

    def _build_network_coupling(self, network_config: dict) -> dict:
        self.net_type = network_config.get('type', 'all_to_all')
        self.size = network_config.get('size', (self.num_backends, 1))
        self.fidelity_range = network_config.get('fidelity_range', (0.96, 0.99))

        # 验证保真度范围
        if not (0 < self.fidelity_range[0] <= self.fidelity_range[1] < 1):
            raise ValueError(f"Invalid fidelity range: {self.fidelity_range}. Must be in (0, 1).")

        network_coupling = {}

        if self.net_type == 'all_to_all':
            network_coupling = {
                (i, j): random.uniform(self.fidelity_range[0], self.fidelity_range[1])
                for i in range(self.num_backends)
                for j in range(i+1, self.num_backends)
            }
        elif self.net_type == 'mesh_grid':
            n_rows, n_cols = self.size
            assert self.num_backends == n_rows * n_cols, "Size does not match number of backends"
            
            for row in range(n_rows):
                for col in range(n_cols - 1):
                    network_coupling[(row * n_cols + col, row * n_cols + col + 1)] = random.uniform(self.fidelity_range[0], self.fidelity_range[1])
            for row in range(n_rows - 1):
                for col in range(n_cols):
                    network_coupling[(row * n_cols + col, (row + 1) * n_cols + col)] = random.uniform(self.fidelity_range[0], self.fidelity_range[1])
        elif self.net_type == 'self_defined':
            network_coupling = network_config.get('network_coupling', {})
            self.fidelity_range = (
                min(network_coupling.values()),
                max(network_coupling.values())
            )
        else:
            raise ValueError(f"Unsupported network type: {self.net_type}")
        return network_coupling

    def _build_network_graph(self, network_coupling: dict) -> nx.Graph:
        G = nx.Graph()
        for (u, v), _ in network_coupling.items():
            G.add_edge(u, v)
        return G
    
    def _build_weighted_network_graph(self, network_coupling: dict) -> nx.Graph:
        G = nx.Graph()
        # hop_weight 必须 > (max possible -log(fidelity)) * (max hops)
        self.hop_weight = -math.log(self.fidelity_range[0]) * self.num_backends
        for (u, v), link_fidelity in network_coupling.items():
            # G.add_edge(u, v, weight=(1, -math.log(link_fidelity)))
            # 边权重 = 1 * LARGE + (-log(fidelity))
            G.add_edge(u, v, weight=self.hop_weight + (-math.log(link_fidelity)), fidelity=link_fidelity)
        return G

    def _count_shortest_communication_paths(self):
        """
        高效计算无权图中所有节点对之间的最短路径数量。
        返回: (dist, count) 两个 n x n numpy 数组
        """
        # 邻接表
        n = self.num_backends
        adj = [[] for _ in range(n)]
        for (u, v), _ in self.network_coupling.items():
            adj[u].append(v)
            adj[v].append(u)  # 假设为无向图；若为有向图，移除此行

        dist = np.full((n, n), np.inf)
        count = np.zeros((n, n), dtype=int)

        for s in range(n):
            dist[s, s] = 0
            count[s, s] = 1
            q = deque([s])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if np.isinf(dist[s, v]):          # 首次访问
                        dist[s, v] = dist[s, u] + 1
                        count[s, v] = count[s, u]
                        q.append(v)
                    elif dist[s, v] == dist[s, u] + 1: # 发现另一条最短路径
                        count[s, v] += count[s, u]
        return dist, count

    def _compute_effective_fidelity(self) -> tuple[np.ndarray, np.ndarray, list[list[list[int]]]]:
        """
        计算有效保真度、跳数和最优路径
        :return:
            W_eff: m x m 矩阵，有效保真度
            Hops: m x m 矩阵，最优跳数
            optimal_paths: m x m 列表，optimal_paths[i][j] = 从i到j的最优路径节点列表
        """
        m = self.num_backends
        
        # 初始化矩阵
        W_eff = np.zeros((m, m))
        Hops = np.zeros((m, m), dtype=int)
        optimal_paths = [[[] for _ in range(m)] for __ in range(m)]  # m x m 空列表
        
        # 自环设置
        for i in range(m):
            W_eff[i][i] = 1.0
            Hops[i][i] = 0
            optimal_paths[i][i] = [i]  # 自环路径

        # 计算所有点对的最短路径 (带路径记录)
        for source in range(m):
            try:
                # 获取长度和路径
                lengths, paths = nx.single_source_dijkstra(
                    self.network_graph,
                    source,
                    weight='weight'
                )
            except nx.NetworkXNoPath:
                lengths, paths = {}, {}

            for target in range(m):
                if source == target:
                    continue

                if target in lengths:
                    # 获取路径
                    path = paths[target]  # 节点列表 [source, ..., target]
                    optimal_paths[source][target] = path
                    
                    # 计算跳数（边数 = 节点数 - 1）
                    Hops[source][target] = len(path) - 1

                    # 计算总保真度成本
                    total_fidelity_cost = 1.0
                    # 遍历路径上的每条边
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k+1]
                        edge_data = self.network_graph.get_edge_data(u, v)
                        if edge_data:
                            # 元组权重 (hop, fidelity_cost)
                            fidelity_cost = edge_data['fidelity']
                            total_fidelity_cost *= fidelity_cost
                    W_eff[source][target] = total_fidelity_cost
                else:
                    # 不可达
                    W_eff[source][target] = 0.0
                    Hops[source][target] = -1
                    optimal_paths[source][target] = []  # 空路径
        
        return W_eff, Hops, optimal_paths

    def get_effective_fidelity(self, src: int, dst: int) -> float:
        """获取两点间有效保真度 (安全访问)"""
        if not (0 <= src < self.num_backends and 0 <= dst < self.num_backends):
            raise IndexError(f"Backend index out of range: ({src}, {dst})")
        return self.W_eff[src][dst]
    
    def get_hop_count(self, src: int, dst: int) -> int:
        """获取两点间最优跳数 (安全访问)"""
        if not (0 <= src < self.num_backends and 0 <= dst < self.num_backends):
            raise IndexError(f"Backend index out of range: ({src}, {dst})")
        return self.Hops[src][dst]
    
    def get_optimal_path(self, src: int, dst: int) -> list[int]:
        """
        获取两点间最优路径
        :param src: 源节点索引
        :param dst: 目标节点索引
        :return: 节点列表，例如 [0, 2, 3] 表示 0->2->3
                 若不可达，返回空列表 []
        """
        if not (0 <= src < self.num_backends and 0 <= dst < self.num_backends):
            raise IndexError(f"Backend index out of range: ({src}, {dst})")
        
        path = self.optimal_paths[src][dst]
        # 验证路径有效性
        if not path and src != dst:
            return []  # 明确返回空列表表示不可达
        return path.copy()  # 返回副本防止外部修改

    def get_backend_qubit_counts(self) -> list[int]:
        """获取每个后端的qubit容量，从大到小排序"""
        return sorted([backend.num_qubits for backend in self.backends], reverse=True)

    def draw_network_graph(self, filename="network_graph", seed=42, highlight_path=None):
        """
        可视化带权重的 NetworkX 图，在边上显示权重（保留四位小数）。
        
        Parameters
        ----------
        filename : str
            保存图片的文件名（不含扩展名）
        seed : int
            布局随机种子
        highlight_path : list of int, optional
            要高亮显示的节点路径，例如 [0, 2, 5]
            函数会将路径中连续节点之间的边设为红色
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.network_graph, seed=seed, weight=None)

        # 绘制节点（默认样式）
        nx.draw_networkx_nodes(self.network_graph, pos, node_color='lightblue', node_size=500)

        # 默认边（非高亮）
        all_edges = list(self.network_graph.edges())
        
        if highlight_path is not None and len(highlight_path) > 1:
            # 构建高亮边集合（使用 frozenset 保证无向图兼容）
            highlight_edges = set()
            for i in range(len(highlight_path) - 1):
                u, v = highlight_path[i], highlight_path[i + 1]
                # 无向图：(u,v) 和 (v,u) 是同一条边
                if self.network_graph.has_edge(u, v):
                    highlight_edges.add((u, v))
                elif self.network_graph.has_edge(v, u):
                    highlight_edges.add((v, u))
                else:
                    print(f"Warning: Edge ({u}, {v}) not in graph! Skipping in highlight.")

            # 非高亮边
            normal_edges = [e for e in all_edges if e not in highlight_edges and (e[1], e[0]) not in highlight_edges]
            
            # 先画普通边（灰色）
            nx.draw_networkx_edges(self.network_graph, pos, edgelist=normal_edges, edge_color='gray', width=1.0)
            # 再画高亮边（红色，更粗）
            nx.draw_networkx_edges(self.network_graph, pos, edgelist=list(highlight_edges), edge_color='red', width=2.5)
        else:
            # 无高亮路径：画所有边为默认颜色
            nx.draw_networkx_edges(self.network_graph, pos, edge_color='gray', width=1.0)

        # 边标签
        edge_labels = {}
        for u, v, d in self.network_graph.edges(data=True):
            weight_val = d.get('weight', 0.0)
            fidelity_val = d.get('fidelity', 0.0)
            edge_labels[(u, v)] = f"w:{weight_val:.4f}\nfid:{fidelity_val:.4f}"

        nx.draw_networkx_edge_labels(self.network_graph, pos, edge_labels=edge_labels)

        # 节点标签
        nx.draw_networkx_labels(self.network_graph, pos, font_size=10)

        plt.axis('off')
        plt.tight_layout()

        # 确保输出目录存在
        os.makedirs("./outputs", exist_ok=True)
        
        plt.savefig(f"./outputs/{filename}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return

    def print_info(self):
        print(f"Network with {self.num_backends} backends")
        print("Network Coupling (edges with fidelities):")
        for (u, v), fidelity in self.network_coupling.items():
            print(f"  Backend {u} <-> Backend {v}: Fidelity = {fidelity:.4f}")
        print(f"Hop weight: {self.hop_weight}")
        print(f"Fidelity range: {self.fidelity_range}")
        return
