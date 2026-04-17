import random
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

from .backend import Backend

class Network:
    def __init__(self, network_config: dict, backend_list: list[Backend], build_all_to_all_copy: bool = True):
        self.num_backends = len(backend_list)
        self.network_coupling = self._build_network_coupling(network_config)
        self.network_graph = self._build_weighted_network_graph(self.network_coupling)
        # 计算telegate的保真度和跳数
        self.move_fidelity, self.move_fidelity_loss, self.Hops, self.optimal_paths = self._compute_effective_fidelity()
        # 计算量子比特交换的保真度和损失
        self.swap_fidelity, self.swap_fidelity_loss = self._compute_swap_fidelity()
        self.backends = backend_list
        self.comm_slot_reserve = int(network_config.get('comm_slot_reserve', 0) or 0)
        # 后端实际容量（用于transpile/通信临时槽）
        self.backend_sizes_full = [backend.num_qubits for backend in self.backends]
        # 核心计算容量（用于partition阶段），由 full - reserve 得到
        self.backend_sizes = [max(0, size - self.comm_slot_reserve) for size in self.backend_sizes_full]
        self.basis_gates, self.two_qubit_gates = self._get_basis_gates(backend_list)

        # 预构建all-to-all副本（重算move/swap等所有依赖连通性的矩阵）。
        self.all_to_all_copy = None
        if build_all_to_all_copy:
            self.all_to_all_copy = self._build_all_to_all_copy(network_config)
        return

    def _build_all_to_all_copy(self, network_config: dict) -> "Network":
        """
        基于network_config构建一个all-to-all副本：
        - 直接将网络类型切换为all_to_all；
        - 由副本自行重算move/swap矩阵，确保与连通性变化一致。
        """
        all_to_all_config = dict(network_config)
        all_to_all_config["type"] = "all_to_all"
        return Network(all_to_all_config, self.backends, build_all_to_all_copy=False)

    @property
    def name(self):
        return f"net{self.backend_sizes}"

    def info(self):
        return {
            "net_type": self.net_type,
            "backends": [backend.name for backend in self.backends],
            "fidelity": self.fidelity_range
        }

    def _build_network_coupling(self, network_config: dict) -> dict:
        self.net_type = network_config.get('type', 'all_to_all')
        self.size = network_config.get('size', (self.num_backends, 1))
        self.fidelity_range = network_config.get('fidelity_range', [0.95, 0.98])

        # 验证保真度范围
        if not (0 < self.fidelity_range[0] <= self.fidelity_range[1] < 1):
            raise ValueError(f"Invalid fidelity range: {self.fidelity_range}. Must be in [0, 1].")

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
        # TODO: chain
        elif self.net_type == 'self_defined':
            network_coupling = network_config.get('network_coupling', {})
            self.fidelity_range = [
                min(network_coupling.values()),
                max(network_coupling.values())
            ]
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

    def _compute_effective_fidelity(self) -> tuple:
        """
        计算有效保真度、跳数和最优路径
        :return:
            move_fidelity: m x m 矩阵，有效保真度
            Hops: m x m 矩阵，最优跳数
            optimal_paths: m x m 列表，optimal_paths[i][j] = 从i到j的最优路径节点列表
        """
        m = self.num_backends
        
        # 初始化矩阵
        # move_fidelity = np.zeros((m, m))
        # Hops = np.zeros((m, m), dtype=int)
        move_fidelity = [[0.0 for _ in range(m)] for __ in range(m)]
        move_fidelity_loss = [[0.0 for _ in range(m)] for __ in range(m)]
        Hops = [[0 for _ in range(m)] for __ in range(m)]
        
        optimal_paths = [[[] for _ in range(m)] for __ in range(m)]  # m x m 空列表
        
        # 自环设置
        for i in range(m):
            move_fidelity[i][i] = 1.0
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

                assert type(lengths) == dict and type(paths) == dict, "Expected lengths and paths to be dictionaries"

                if target in lengths:
                    # 获取路径
                    path = paths[target]  # 节点列表 [source, ..., target]
                    optimal_paths[source][target] = path.copy()
                    
                    # 计算跳数（边数 = 节点数 - 1）
                    Hops[source][target] = len(path) - 1

                    # 计算总保真度成本
                    total_fidelity_cost = 1.0
                    total_fidelity_loss = 0.0
                    # 遍历路径上的每条边
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k+1]
                        edge_data = self.network_graph.get_edge_data(u, v)
                        # if edge_data: # 必须有data，没有报错
                        # 元组权重 (hop, fidelity_cost)
                        fidelity_cost = edge_data['fidelity']
                        total_fidelity_cost *= fidelity_cost
                        total_fidelity_loss += (1 - fidelity_cost)
                    move_fidelity[source][target] = total_fidelity_cost
                    move_fidelity_loss[source][target] = total_fidelity_loss
                else:
                    # 不可达
                    # move_fidelity[source][target] = 0.0
                    # Hops[source][target] = -1
                    # optimal_paths[source][target] = []  # 空路径
                    # 报错
                    raise RuntimeError(f"[ERROR] Cannot reach from {source} to {target}.")
        
        return move_fidelity, move_fidelity_loss, Hops, optimal_paths

    # === 新增：计算量子比特交换的保真度和损失 ===
    def _compute_swap_fidelity(self) -> tuple:
        """
        计算任意两个节点之间交换量子比特的总保真度和保真度损失
        原理：对于最短路径 [v0, v1, ..., vk]，需执行 (2k-1) 次 SWAP 操作
             操作序列：(v0,v1) → (v1,v2) → ... → (vk-1,vk) → (vk-2,vk-1) → ... → (v0,v1)
        :return:
            swap_fidelity: m x m 矩阵，swap_fidelity[i][j] 表示交换i和j上量子比特的总保真度
            swap_fidelity_loss: m x m 矩阵，swap_fidelity_loss[i][j] = 1 - swap_fidelity[i][j]
        """
        m = self.num_backends
        swap_fidelity = [[0.0 for _ in range(m)] for __ in range(m)]
        swap_fidelity_loss = [[0.0 for _ in range(m)] for __ in range(m)]

        for i in range(m):
            swap_fidelity[i][i] = 1.0
            swap_fidelity_loss[i][i] = 0.0
            for j in range(m):
                if i == j:
                    continue
                path = self.optimal_paths[i][j]
                if not path:
                    # 不可达
                    swap_fidelity[i][j] = 0.0
                    swap_fidelity_loss[i][j] = 1.0
                    continue
                k = len(path) - 1  # 跳数（边数）
                # 生成 SWAP 操作的边序列
                swap_edges = []
                # 第一阶段：从 path[0] 到 path[k]
                for idx in range(k):
                    u = path[idx]
                    v = path[idx + 1]
                    swap_edges.append((u, v))
                # 第二阶段：从 path[k-2] 回到 path[0]
                for idx in range(k-2, -1, -1):
                    u = path[idx]
                    v = path[idx + 1]
                    swap_edges.append((u, v))
                # 计算总保真度（所有 SWAP 边保真度的乘积）和总保真度损失
                total_fid = 1.0
                total_fid_loss = 0.0
                for (u, v) in swap_edges:
                    edge_data = self.network_graph.get_edge_data(u, v)
                    fid = edge_data['fidelity']
                    total_fid *= (fid ** 2)
                    total_fid_loss += 2 * (1 - fid)

                swap_fidelity[i][j] = total_fid
                swap_fidelity_loss[i][j] = total_fid_loss
        return swap_fidelity, swap_fidelity_loss

    def get_effective_fidelity(self, src: int, dst: int) -> float:
        """获取两点间有效保真度 (安全访问)"""
        if not (0 <= src < self.num_backends and 0 <= dst < self.num_backends):
            raise IndexError(f"Backend index out of range: ({src}, {dst})")
        return self.move_fidelity[src][dst]
    
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

    def get_backend_qubit_counts(self, include_comm_slot: bool = False) -> list[int]:
        """获取每个后端的qubit容量。"""
        # 默认返回基础容量（用于partition阶段）；include_comm_slot=True返回完整容量。
        if include_comm_slot:
            return self.backend_sizes_full
        return self.backend_sizes

    def get_backend_qubit_counts_full(self) -> list[int]:
        """获取考虑通信预留槽位后的完整后端容量。"""
        return self.backend_sizes_full

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
        print("========== Network Info ==========")
        print(f"Network with {self.num_backends} backends ({self.net_type})")
        print("Network Coupling (edges with fidelities):")
        for (u, v), fidelity in self.network_coupling.items():
            print(f"  Backend {u} <-> Backend {v}: Fidelity = {fidelity:.4f}")
        print(f"Hop weight: {self.hop_weight}")
        print(f"Fidelity range: {self.fidelity_range}")
        print(f"Backend sizes: {self.backend_sizes}")
        print(f"Backend sizes (full): {self.backend_sizes_full}")
        # 输出每个backend的coupling map
        print("Backend Coupling Maps:")
        for i, backend in enumerate(self.backends):
            print(f"  Backend {i} ({backend.name}): Coupling Map = {backend.coupling_map}")
        return

    def get_network_coupling_and_qubits(self):
        """
        根据网络关系构建点对点耦合结构，类似于示例中的 server_coupling、server_qubits 和 swap_cost_matrix。

        返回:
            server_coupling (list[list[int]]): 每条边的两端节点列表，例如 [[0,1], [0,2], ...]
            server_qubits (dict[int, list[int]]): 每个后端所拥有的全局量子比特索引列表，键为后端索引，值为该后端包含的量子比特列表。
            swap_cost_matrix (np.ndarray): 形状 (n_backends, n_backends) 的矩阵，表示从一个后端到另一个后端的 SWAP 成本，
                                            通常基于最短路径跳数（或自定义因子）。对角线为 0，不可达标记为 -1（或视情况处理）。
        """
        # 1. 构建 server_coupling：将内部存储的边字典转换为列表的列表
        #    network_coupling 是一个字典，键为 (u, v) 元组，值为 fidelity
        #    由于是无向图，每条边只出现一次（通常 u < v）
        server_coupling = []
        for (u, v) in self.network_coupling.keys():
            server_coupling.append([u, v])
            # server_coupling.append([v, u])

        # 2. 构建 server_qubits：为每个后端分配全局量子比特索引
        #    按后端顺序切片，得到每个后端对应的量子比特索引段
        server_qubits_list = []
        current = 0
        for backend in self.backends:
            backend_qubits = list(range(current, current + backend.num_qubits))
            server_qubits_list.append(backend_qubits)
            current += backend.num_qubits

        #    转换为字典形式，方便通过后端索引快速获取其量子比特列表
        server_qubits = {i: qlist for i, qlist in enumerate(server_qubits_list)}

        return server_coupling, server_qubits

    def _get_basis_gates(self, backend_list: list[Backend]):
        # 理论上所有后端用同样的basis gates
        basis_gates = backend_list[0].basis_gates  # 从第一个后端开始，初始化basis_gates集合
        two_qubit_gates = backend_list[0].two_qubit_gates  # 从第一个后端开始，初始化two_qubit_gates集合
        # 从每一个backend中获取basis_gates，假设和basis_gates不同，发出警告
        for backend in backend_list:
            if set(backend.basis_gates) != set(basis_gates):
                print(f"[Warning] Backend {backend.name} has different basis gates: {backend.basis_gates} vs {basis_gates}")

        return basis_gates, two_qubit_gates

"""
# 构建一个全连接网络
def build_fc_network(qpus):
    server_coupling = [
        list(combination)
        for combination in combinations([i for i in range(len(qpus))], 2)
    ]
    qubits = [i for i in range(sum(qpus))]
    server_qubits_list = [
        qubits[sum(qpus[:i]) : sum(qpus[:i+1])]
        for i in range(len(qpus))
    ]
    server_qubits = {
        i: qubits_list
        for i, qubits_list in enumerate(server_qubits_list)
    }
    # 计算每个节点到其他节点的qubit swap cost
    swap_cost_matrix = np.zeros((len(qpus), len(qpus)), dtype=int)
    for i in range(len(qpus)):
        for j in range(len(qpus)):
            swap_cost_matrix[i][j] = 1
        swap_cost_matrix[i][i] = 0
    return NISQNetwork(server_coupling, server_qubits), swap_cost_matrix

# 构建一个mesh-grid网络
def build_mesh_grid_network(qpus, n_rows=4, n_cols=2):
    assert len(qpus) == n_rows * n_cols
    server_coupling = []
    for row in range(n_rows): # 生成水平连接（左右）
        for col in range(n_cols - 1):
            server_coupling.append([row * n_cols + col, row * n_cols + col + 1])
    for row in range(n_rows - 1): # 生成垂直连接（上下）
        for col in range(n_cols):
            server_coupling.append([row * n_cols + col, (row + 1) * n_cols + col])
    qubits = [i for i in range(sum(qpus))]
    server_qubits_list = [
        qubits[sum(qpus[:i]) : sum(qpus[:i+1])]
        for i in range(len(qpus))
    ]
    server_qubits = {
        i: qubits_list
        for i, qubits_list in enumerate(server_qubits_list)
    }
    # 计算每个节点到其他节点的qubit swap cost
    G = nx.Graph()
    G.add_edges_from(server_coupling)
    swap_cost_matrix = np.zeros((len(G.nodes()), len(G.nodes())), dtype=int)
    for i in G.nodes():
        for j in G.nodes():
            swap_cost_matrix[i][j] = 2 * nx.shortest_path_length(G, source=i, target=j) - 1
        swap_cost_matrix[i][i] = 0
    return NISQNetwork(server_coupling, server_qubits), swap_cost_matrix

"""