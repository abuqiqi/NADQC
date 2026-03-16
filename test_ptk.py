import networkx as nx
from pytket_dqc.circuits import Distribution, Hyperedge
from pytket_dqc.utils import steiner_tree
from typing import Set, Tuple, Dict

# -------------------------- 第一步：定义基础配置 --------------------------
# 1. 为每对服务器配置链路保真度（可替换为实测/仿真数据）
# 格式：{(server1, server2): 保真度值}，无向链路需双向配置
LINK_FIDELITY: Dict[Tuple[int, int], float] = {
    (0, 1): 0.99, (1, 0): 0.99,
    (1, 2): 0.98, (2, 1): 0.98,
    (0, 2): 0.97, (2, 0): 0.97,
    # 可扩展到更多服务器对
}

# 2. 默认链路保真度（未配置的链路使用此值）
DEFAULT_LINK_FID = 0.99

# -------------------------- 第二步：提取通信链路 --------------------------
def get_hyperedge_communication_links(
    distribution: Distribution,
    hyperedge: Hyperedge
) -> Set[Tuple[int, int]]:
    """
    提取单个超边对应的所有 Ebit 通信链路（服务器对）
    核心：复用 pytket-dqc 内置的 steiner_tree/ALAP 逻辑，与 cost() 对齐
    """
    # 步骤1：获取超边涉及的所有服务器
    servers = [distribution.placement.placement[v] for v in hyperedge.vertices]
    # 步骤2：计算 Steiner 树（与 hyperedge_cost 逻辑一致）
    server_graph = distribution.network.get_server_nx()
    tree = steiner_tree(server_graph, servers)
    
    # 步骤3：处理 H-嵌入场景（ALAP 路径）
    requires_h_embed = distribution.circuit.requires_h_embedded_cu1(hyperedge)
    if requires_h_embed:
        # H-嵌入场景需额外提取 ALAP 路径的链路（复用 hyperedge_cost 的 ALAP 逻辑）
        # 简化版：直接用 Steiner 树近似（若需精确值，需复刻 hyperedge_cost 的 ALAP 循环）
        pass
    
    # 步骤4：返回树的所有边（服务器对），去重且无序
    links = set()
    for s1, s2 in tree.edges:
        links.add((min(s1, s2), max(s2, s1)))  # 统一格式：(小ID, 大ID)
    return links

def get_all_communication_links(distribution: Distribution) -> Set[Tuple[int, int]]:
    """提取所有超边对应的 Ebit 通信链路（全局去重）"""
    all_links = set()
    for he in distribution.circuit.hyperedge_list:
        he_links = get_hyperedge_communication_links(distribution, he)
        all_links.update(he_links)
    return all_links

# -------------------------- 第三步：保真度计算 --------------------------
def calculate_communication_fidelity(distribution: Distribution) -> float:
    """计算所有通信操作的总保真度（基于 Ebit 链路）"""
    if not distribution.is_valid():
        raise ValueError("Invalid distribution, cannot compute fidelity")
    
    # 步骤1：获取所有通信链路
    all_links = get_all_communication_links(distribution)
    if not all_links:
        return 1.0  # 无通信，保真度100%
    
    # 步骤2：累积保真度（乘积模型：每条链路的损失独立）
    total_fidelity = 1.0
    for (s1, s2) in all_links:
        # 查找链路保真度（兼容双向配置）
        link_fid = LINK_FIDELITY.get((s1, s2), LINK_FIDELITY.get((s2, s1), DEFAULT_LINK_FID))
        total_fidelity *= link_fid
    
    return total_fidelity

def calculate_fidelity_loss(distribution: Distribution) -> float:
    """计算总保真度损失（1 - 总保真度）"""
    return 1.0 - calculate_communication_fidelity(distribution)

def calculate_non_local_gate_fidelity_loss(distribution: Distribution) -> Dict[int, float]:
    """计算每个非本地门对应的保真度损失（按服务器对拆分）"""
    non_local_gates = distribution.non_local_gate_list()
    gate_loss = {}
    
    for idx, gate_vertex in enumerate(non_local_gates):
        # 步骤1：找到门对应的超边
        gate_hyperedge = None
        for he in distribution.circuit.hyperedge_list:
            if gate_vertex in he.vertices:
                gate_hyperedge = he
                break
        if not gate_hyperedge:
            gate_loss[idx] = 0.0
            continue
        
        # 步骤2：计算该超边的保真度损失
        he_links = get_hyperedge_communication_links(distribution, gate_hyperedge)
        he_fidelity = 1.0
        for (s1, s2) in he_links:
            link_fid = LINK_FIDELITY.get((s1, s2), LINK_FIDELITY.get((s2, s1), DEFAULT_LINK_FID))
            he_fidelity *= link_fid
        gate_loss[idx] = 1.0 - he_fidelity
    
    return gate_loss

# -------------------------- 第四步：使用示例 --------------------------
if __name__ == "__main__":
    # 假设你已构造好 Distribution 实例（circuit + placement + network）
    # distribution = Distribution(circuit, placement, network)
    
    # 1. 验证分布有效性
    if distribution.is_valid():
        # 2. 提取所有通信链路
        all_links = get_all_communication_links(distribution)
        print(f"所有 Ebit 通信链路（服务器对）：{all_links}")
        
        # 3. 计算总保真度 & 损失
        total_fid = calculate_communication_fidelity(distribution)
        total_loss = calculate_fidelity_loss(distribution)
        print(f"总通信保真度：{total_fid:.4f}")
        print(f"总保真度损失：{total_loss:.4f}")
        
        # 4. 计算每个非本地门的保真度损失
        non_local_loss = calculate_non_local_gate_fidelity_loss(distribution)
        print("每个非本地门的保真度损失：", non_local_loss)
        
        # 5. 结合 cost() 计算加权指标（兼顾通信量和损失）
        total_ebits = distribution.cost()
        weighted_cost = total_ebits * total_loss
        print(f"带保真度损失的加权 cost：{weighted_cost:.4f}")