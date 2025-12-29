from qiskit_ibm_runtime import QiskitRuntimeService
from pprint import pprint
import pickle as pkl
import datetime
import os

import datetime
import pandas as pd
import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class QiskitBackendImporter:
    def __init__(self, token, instance, proxies):
        QiskitRuntimeService.save_account(token, instance=instance, overwrite=True, proxies=proxies)
        self.service = QiskitRuntimeService()
        return

    def download_backend_info(self, backend_name, start_date, end_date, folder):
        backend = self.service.backend(backend_name)
        self._print_backend(backend)
        config = backend.configuration()

        folder = f'{folder}{backend_name}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        date = start_date
        while date <= end_date:
            properties = backend.properties(datetime=date)
            pkl.dump(
                (properties, config),
                open(
                    os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.data'),
                    'wb'
                )
            )
            print('Device properties saved for date:', date.strftime("%Y-%m-%d"))

            # 将properties写入xlsx文件
            self._to_xlsx(properties.to_dict(), os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.xlsx'))

            date += datetime.timedelta(days=1)
        return

    def download_all_backend_info(self, start_date, end_date, folder):
        for backend in self.service.backends():
            self.download_backend_info(backend.name, start_date, end_date, folder)
        return

    def _print_backend(self, backend):
        print(
            f"Name: {backend.name}\n"
            f"Version: {backend.version}\n"
            f"#Qubits: {backend.num_qubits}\n"
        )
        return

    def _to_iso(self, dt):
        if isinstance(dt, datetime.datetime):
            return dt.isoformat()
        return dt

    def _build_basic_info(self, data: dict):
        row = {
            'backend_name': data.get('backend_name'),
            'backend_version': data.get('backend_version'),
            'general_qlists': data.get('general_qlists'),
            'last_update_date': self._to_iso(data.get('last_update_date')),
        }
        return pd.DataFrame([row])

    def _build_gate_info(self, data: dict):
        gates = data.get('gates', [])
        rows = []
        for g in gates:
            row = {
                'name': g.get('name'),
                'gate': g.get('gate'),
            }
            # qubits 列：保存为逗号分隔字符串（也可改为列表）
            qubits = g.get('qubits', [])
            row['qubits'] = ','.join(map(str, qubits)) if isinstance(qubits, list) else qubits

            # 展开 parameters 列表 -> {param_name}_value/unit/date
            params = g.get('parameters', [])
            for p in params or []:
                p_name = p.get('name')
                if not p_name:
                    continue
                row[f'{p_name}_value'] = p.get('value')
                row[f'{p_name}_unit'] = p.get('unit')
                row[f'{p_name}_date'] = self._to_iso(p.get('date'))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_qubit_info(self, data: dict):
        qubits = data.get('qubits', [])
        rows = []
        for qid, qlist in enumerate(qubits):
            row = {'qubit_id': qid}
            if isinstance(qlist, list):
                for item in qlist:
                    name = item.get('name')
                    if not name:
                        continue
                    row[f'{name}_value'] = item.get('value')
                    row[f'{name}_unit'] = item.get('unit')
                    row[f'{name}_date'] = self._to_iso(item.get('date'))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_general_info(self, data: dict):
        gens = data.get('general', [])
        rows = []
        for item in gens:
            name = item.get('name')
            row = {
                'name': name,
                'value': item.get('value'),
                'unit': item.get('unit'),
                'date': self._to_iso(item.get('date')),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _save_to_excel(self, basic_df, gate_df, qubit_df, general_df, out_path: str):
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            basic_df.to_excel(writer, index=False, sheet_name='basic_info')
            gate_df.to_excel(writer, index=False, sheet_name='gate_info')
            qubit_df.to_excel(writer, index=False, sheet_name='qubit_info')
            general_df.to_excel(writer, index=False, sheet_name='general_info')

    def _to_xlsx(self, data: dict, output_path: str = 'noise_parsed.xlsx'):
        basic_df = self._build_basic_info(data)
        gate_df = self._build_gate_info(data)
        qubit_df = self._build_qubit_info(data)
        general_df = self._build_general_info(data)

        # 可选：按列名排序，让 *_value, *_unit, *_date 更整齐
        def sort_cols(df):
            cols = list(df.columns)
            # 将核心列放前面
            front = [c for c in ['backend_name','backend_version','general_qlists','last_update_date',
                                'name','gate','qubits','qubit_id','value','unit','date'] if c in cols]
            rest = [c for c in cols if c not in front]
            return df[front + sorted(rest)]
        try:
            basic_df = sort_cols(basic_df)
            gate_df = sort_cols(gate_df)
            qubit_df = sort_cols(qubit_df)
            general_df = sort_cols(general_df)
        except Exception:
            pass

        self._save_to_excel(basic_df, gate_df, qubit_df, general_df, output_path)
        print(f'已生成 Excel：{output_path}')

class Backend:
    def __init__(self):
        # 初始化QPU上的噪声信息
        return

    def load_properties(self, config: dict, backend_name: str, date: datetime.datetime):

        filepath = f"{config['device_properties_folder']}{backend_name}/{backend_name}_{date.strftime('%Y-%m-%d')}.xlsx"

        if not os.path.exists(filepath):
            print(f"Downloading the device properties for {backend_name} on {date.strftime('%Y-%m-%d')} ...")
            backend_importer = QiskitBackendImporter(token=config["ibm_quantum_token"],
                                                    instance=config["ibm_quantum_instance"],
                                                    proxies=config.get("proxies", None)
                                                    )
            
            backend_importer.download_backend_info(backend_name=backend_name,
                                                start_date=date,
                                                end_date=date,
                                                folder=config["device_properties_folder"])

        # 从文件加载噪声数据
        # 读取basic_info, gate_info, qubit_info, general_info表
        basic_df = pd.read_excel(filepath, sheet_name='basic_info')
        gate_df = pd.read_excel(filepath, sheet_name='gate_info')
        qubit_df = pd.read_excel(filepath, sheet_name='qubit_info')
        general_df = pd.read_excel(filepath, sheet_name='general_info')

        # 初始化basic_info
        self.basic_info = basic_df.to_dict(orient='records')[0]
        self.name = self.basic_info.get('backend_name', 'unknown')
        # 初始化gate_info
        self.gate_info = gate_df.to_dict(orient='records')
        # 初始化qubit_info
        self.qubit_info = qubit_df.to_dict(orient='records')
        self.num_qubits = len(self.qubit_info)
        # 初始化general_info
        self.general_info = general_df.to_dict(orient='records')
        return
    
    def print(self):
        print(f"Backend Name: {self.name}, #Qubits: {self.num_qubits}")
        # pprint(self.basic_info)
        # pprint(self.gate_info)
        # pprint(self.qubit_info)
        # pprint(self.general_info)
        return

class Network:
    def __init__(self, network_config: dict, backend_config: list):
        self.num_backends = len(backend_config)
        self.network_coupling = self._build_network_coupling(network_config)
        self.network_graph = self._build_weighted_network_graph(self.network_coupling)
        self.Weff, self.Hops, self.optimal_paths = self._compute_effective_fidelity()
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

    def draw_network_graph(self, filename="network_graph", seed=42):
        """
        可视化带权重的 NetworkX 图，在边上显示权重（保留两位小数）。
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.network_graph, seed=seed, weight=None)

        # 绘制节点
        nx.draw_networkx_nodes(self.network_graph, pos)

        # 绘制边
        nx.draw_networkx_edges(self.network_graph, pos)

        # 提取fidelity值用于显示，不修改原图
        edge_labels = {}
        for u, v, d in self.network_graph.edges(data=True):
            weight_val = d.get('weight', 0.0)
            fidelity_val = d.get('fidelity', 0.0)  # 默认值防止无权重的情况
            # 保留小数
            edge_labels[(u, v)] = f"{weight_val:.4f} ({fidelity_val:.4f})"

        # 绘制边标签（显示第二个值）
        nx.draw_networkx_edge_labels(self.network_graph, pos, edge_labels=edge_labels)

        # 绘制节点标签
        nx.draw_networkx_labels(self.network_graph, pos)

        plt.savefig(f"./outputs/{filename}.png")
        plt.show()
        return

    def print_info(self):
        print(f"Network with {self.num_backends} backends")
        print("Network Coupling (edges with fidelities):")
        for (u, v), fidelity in self.network_coupling.items():
            print(f"  Backend {u} <-> Backend {v}: Fidelity = {fidelity:.4f}")
        print(f"Hop weight: {self.hop_weight}")
        print(f"Fidelity range: {self.fidelity_range}")
        return
