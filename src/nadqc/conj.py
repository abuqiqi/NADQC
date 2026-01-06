import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional, Callable

class Network:
    def __init__(self, network_config: dict, backend_config: list):
        self.num_backends = len(backend_config)
        self.backends = backend_config
        
        # 新增：链路容量配置
        self.link_capacity = network_config.get('link_capacity', 10.0)  # 默认容量
        self.capacity_matrix = self._build_capacity_matrix(network_config)
        
        # 原有初始化
        self.network_coupling = self._build_network_coupling(network_config)
        self.network_graph = self._build_weighted_graph()
        
        # 预计算基础指标
        self.W_eff_base, self.Hops_base, self.optimal_paths_base = self.compute_effective_fidelity()
        
        # 初始化负载感知指标
        self.reset_load_aware_metrics()
    
    def _build_capacity_matrix(self, network_config: dict) -> np.ndarray:
        """构建链路容量矩阵 (m x m)"""
        m = self.num_backends
        capacity = np.zeros((m, m))
        
        # 全局容量设置
        if 'link_capacity' in network_config:
            cap = network_config['link_capacity']
            for i in range(m):
                for j in range(m):
                    if i != j:
                        capacity[i][j] = cap
        
        # 覆盖特定链路容量
        if 'link_capacity_map' in network_config:
            for (i, j), cap in network_config['link_capacity_map'].items():
                if 0 <= i < m and 0 <= j < m:
                    capacity[i][j] = cap
                    capacity[j][i] = cap  # 无向图对称
        
        return capacity
    
    def reset_load_aware_metrics(self):
        """重置负载相关指标（用于新映射）"""
        self.current_load = np.zeros((self.num_backends, self.num_backends))
        self.W_eff_loaded = self.W_eff_base.copy()  # 负载感知保真度
        self.congestion_map = {}  # 记录拥塞路径
    
    def compute_link_load(self, D: np.ndarray, pi: List[int]) -> np.ndarray:
        """
        计算给定映射pi下的链路负载
        :param D: m x m 通信需求矩阵 (0-based)
        :param pi: 逻辑->物理映射 [物理索引 for 逻辑0..m-1]
        :return: m x m 负载矩阵
        """
        m = self.num_backends
        load = np.zeros((m, m))
        
        # 遍历所有逻辑对
        for i in range(m):
            for j in range(i + 1, m):
                demand = D[i][j]
                if demand <= 0:
                    continue
                
                # 获取物理节点
                p_phys = pi[i]
                q_phys = pi[j]
                
                # 获取最优路径
                path = self.optimal_paths_base[p_phys][q_phys]
                if not path or len(path) < 2:
                    continue
                
                # 累加路径上的每条边
                for k in range(len(path) - 1):
                    u = path[k]
                    v = path[k+1]
                    load[u][v] += demand
                    load[v][u] += demand  # 无向图
        
        return load
    
    def update_load_aware_fidelity(self, load_matrix: np.ndarray, 
                                  congestion_model: str = 'exponential') -> np.ndarray:
        """
        更新负载感知的有效保真度
        :param load_matrix: 当前链路负载
        :param congestion_model: 拥塞模型 ('exponential', 'linear', 'threshold')
        :return: 更新后的 W_eff_loaded
        """
        m = self.num_backends
        W_eff_new = np.zeros((m, m))
        
        # 选择拥塞函数
        if congestion_model == 'exponential':
            def congestion_func(load, capacity):
                alpha = 0.1  # 拥塞敏感度参数
                return math.exp(-alpha * (load / max(capacity, 1e-9)))
        
        elif congestion_model == 'linear':
            def congestion_func(load, capacity):
                beta = 0.5
                return max(0.0, 1.0 - beta * (load / max(capacity, 1e-9)))
        
        elif congestion_model == 'threshold':
            def congestion_func(load, capacity):
                threshold = 0.8  # 80% 容量阈值
                if load / max(capacity, 1e-9) > threshold:
                    return 0.5  # 严重拥塞时保真度减半
                return 1.0
        
        else:
            raise ValueError(f"Unknown congestion model: {congestion_model}")
        
        # 更新每对物理节点的保真度
        for i in range(m):
            W_eff_new[i][i] = 1.0  # 自环保真度
            for j in range(m):
                if i == j:
                    continue
                
                # 基础保真度
                base_fid = self.W_eff_base[i][j]
                if base_fid <= 0:
                    W_eff_new[i][j] = 0.0
                    continue
                
                # 获取路径上的最大拥塞
                path = self.optimal_paths_base[i][j]
                if not path or len(path) < 2:
                    W_eff_new[i][j] = 0.0
                    continue
                
                min_congestion = 1.0
                congested_edges = []
                
                # 遍历路径上的每条边
                for k in range(len(path) - 1):
                    u = path[k]
                    v = path[k+1]
                    current_load = load_matrix[u][v]
                    capacity = self.capacity_matrix[u][v]
                    
                    # 计算该边的拥塞因子
                    cong_factor = congestion_func(current_load, capacity)
                    if cong_factor < min_congestion:
                        min_congestion = cong_factor
                        congested_edges = [(u, v)]
                    elif abs(cong_factor - min_congestion) < 1e-6:
                        congested_edges.append((u, v))
                
                # 应用最坏边的拥塞（木桶效应）
                W_eff_new[i][j] = base_fid * min_congestion
                
                # 记录拥塞信息（用于调试）
                if min_congestion < 0.9:  # 显著拥塞
                    self.congestion_map[(i, j)] = {
                        'path': path,
                        'congested_edges': congested_edges,
                        'congestion_factor': min_congestion
                    }
        
        self.W_eff_loaded = W_eff_new
        return W_eff_new
    
    def compute_load_aware_mapping(self, D: np.ndarray, 
                                  max_iterations: int = 10,
                                  convergence_threshold: float = 0.01) -> Tuple[List[int], Dict]:
        """
        计算负载感知的映射
        :param D: 通信需求矩阵 (m x m, 0-based)
        :return: (最优映射, 优化历史)
        """
        m = self.num_backends
        history = {
            'iterations': [],
            'total_fidelity': [],
            'max_utilization': []
        }
        
        # 步骤1: 初始映射（无负载）
        pi_current = self._initial_mapping(D, self.W_eff_base)
        best_pi = pi_current.copy()
        best_score = -1
        
        # 步骤2: 迭代优化
        for iter in range(max_iterations):
            # 2.1 计算当前负载
            load_matrix = self.compute_link_load(D, pi_current)
            max_util = np.max(load_matrix / (self.capacity_matrix + 1e-9))
            
            # 2.2 更新负载感知保真度
            W_eff_current = self.update_load_aware_fidelity(load_matrix)
            
            # 2.3 评估当前映射
            current_score = self._evaluate_mapping(D, W_eff_current, pi_current)
            
            # 记录历史
            history['iterations'].append(iter)
            history['total_fidelity'].append(current_score)
            history['max_utilization'].append(max_util)
            
            # 2.4 检查收敛
            if current_score > best_score:
                best_score = current_score
                best_pi = pi_current.copy()
            
            if iter > 0 and abs(current_score - history['total_fidelity'][-2]) < convergence_threshold:
                break
            
            # 2.5 生成新映射（基于当前负载感知保真度）
            pi_new = self._optimize_mapping_iterative(D, W_eff_current, pi_current.copy())
            
            # 2.6 接受更好映射
            new_score = self._evaluate_mapping(D, W_eff_current, pi_new)
            if new_score > current_score:
                pi_current = pi_new
        
        # 最终更新
        final_load = self.compute_link_load(D, best_pi)
        self.update_load_aware_fidelity(final_load)
        
        return best_pi, history
    
    def _initial_mapping(self, D: np.ndarray, W: np.ndarray) -> List[int]:
        """生成初始映射（忽略负载）"""
        # 使用贪心算法：高需求对映射到高保真度链路
        m = self.num_backends
        available = list(range(m))
        mapping = [-1] * m
        
        # 按通信需求排序逻辑对
        pairs = [(i, j, D[i][j]) for i in range(m) for j in range(i+1, m) if D[i][j] > 0]
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心分配
        for i, j, demand in pairs:
            if mapping[i] == -1 or mapping[j] == -1:
                # 选择最佳物理对
                best_pair = None
                best_val = -1
                for p in available:
                    for q in available:
                        if p == q:
                            continue
                        val = demand * W[p][q]
                        if val > best_val:
                            best_val = val
                            best_pair = (p, q)
                
                if best_pair and mapping[i] == -1 and mapping[j] == -1:
                    mapping[i], mapping[j] = best_pair
                    available.remove(best_pair[0])
                    available.remove(best_pair[1])
                elif best_pair and mapping[i] == -1:
                    mapping[i] = best_pair[0]
                    available.remove(best_pair[0])
                elif best_pair and mapping[j] == -1:
                    mapping[j] = best_pair[1]
                    available.remove(best_pair[1])
        
        # 填充剩余
        for i in range(m):
            if mapping[i] == -1:
                mapping[i] = available.pop()
        
        return mapping
    
    def _optimize_mapping_iterative(self, D: np.ndarray, W: np.ndarray, pi_start: List[int]) -> List[int]:
        """基于当前保真度矩阵优化映射"""
        m = self.num_backends
        pi = pi_start.copy()
        improved = True
        
        while improved:
            improved = False
            best_improvement = 0
            best_swap = None
            
            # 尝试所有逻辑QPU对的交换
            for i in range(m):
                for j in range(i+1, m):
                    # 交换逻辑i和j的物理映射
                    pi_new = pi.copy()
                    pi_new[i], pi_new[j] = pi_new[j], pi_new[i]
                    
                    # 评估改进
                    current_score = self._evaluate_mapping(D, W, pi)
                    new_score = self._evaluate_mapping(D, W, pi_new)
                    improvement = new_score - current_score
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i, j)
            
            if best_swap and best_improvement > 1e-6:
                i, j = best_swap
                pi[i], pi[j] = pi[j], pi[i]
                improved = True
        
        return pi
    
    def _evaluate_mapping(self, D: np.ndarray, W: np.ndarray, pi: List[int]) -> float:
        """评估映射质量（总保真度）"""
        total = 0.0
        m = self.num_backends
        for i in range(m):
            for j in range(i+1, m):
                p = pi[i]
                q = pi[j]
                total += D[i][j] * W[p][q]
        return total
    
    def visualize_congestion(self, figsize=(12, 10)):
        """可视化网络拥塞状态"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not hasattr(self, 'current_load'):
            print("No congestion data available. Run compute_load_aware_mapping first.")
            return
        
        plt.figure(figsize=figsize)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 链路负载热力图
        sns.heatmap(self.current_load, ax=axes[0,0], annot=True, fmt=".1f", 
                   cmap="YlOrRd", cbar_kws={'label': 'Load'})
        axes[0,0].set_title("Link Load Matrix")
        axes[0,0].set_xlabel("Physical QPU")
        axes[0,0].set_ylabel("Physical QPU")
        
        # 2. 容量利用率
        utilization = self.current_load / (self.capacity_matrix + 1e-9)
        sns.heatmap(utilization * 100, ax=axes[0,1], annot=True, fmt=".1f", 
                   cmap="coolwarm", vmin=0, vmax=100,
                   cbar_kws={'label': 'Utilization (%)'})
        axes[0,1].set_title("Link Utilization (%)")
        axes[0,1].set_xlabel("Physical QPU")
        axes[0,1].set_ylabel("Physical QPU")
        
        # 3. 拥塞影响
        congestion_impact = np.zeros((self.num_backends, self.num_backends))
        for i in range(self.num_backends):
            for j in range(self.num_backends):
                if i != j and self.W_eff_base[i][j] > 0:
                    congestion_impact[i][j] = self.W_eff_loaded[i][j] / self.W_eff_base[i][j]
        
        sns.heatmap(congestion_impact, ax=axes[1,0], annot=True, fmt=".2f", 
                   cmap="RdYlGn", vmin=0, vmax=1,
                   cbar_kws={'label': 'Fidelity Ratio'})
        axes[1,0].set_title("Congestion Impact (Fidelity Ratio)")
        axes[1,0].set_xlabel("Physical QPU")
        axes[1,0].set_ylabel("Physical QPU")
        
        # 4. 拓扑图（带拥塞边）
        pos = nx.spring_layout(self.network_graph, seed=42)
        
        # 绘制所有边
        edge_colors = []
        edge_widths = []
        edge_labels = {}
        
        for u, v, data in self.network_graph.edges(data=True):
            load = self.current_load[u][v]
            capacity = self.capacity_matrix[u][v]
            util = load / max(capacity, 1e-9)
            
            # 颜色编码
            if util > 0.8:
                color = 'red'
            elif util > 0.5:
                color = 'orange'
            else:
                color = 'green'
            
            edge_colors.append(color)
            edge_widths.append(1 + 3 * util)  # 宽度反映利用率
            
            # 标签
            fid_orig = math.exp(-data['weight'][1])
            fid_curr = self.W_eff_loaded[u][v] if hasattr(self, 'W_eff_loaded') else fid_orig
            edge_labels[(u, v)] = f"{load:.1f}/{capacity:.1f}\n{fid_curr:.3f}"
        
        nx.draw_networkx_nodes(self.network_graph, pos, node_size=700, node_color='lightblue', ax=axes[1,1])
        nx.draw_networkx_edges(self.network_graph, pos, edge_color=edge_colors, 
                              width=edge_widths, alpha=0.7, ax=axes[1,1])
        nx.draw_networkx_labels(self.network_graph, pos, font_size=10, ax=axes[1,1])
        nx.draw_networkx_edge_labels(self.network_graph, pos, edge_labels=edge_labels, 
                                    font_size=8, ax=axes[1,1])
        
        axes[1,1].set_title("Network Topology with Congestion")
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.show()