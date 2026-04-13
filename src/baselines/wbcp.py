import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from typing import Any, Optional
import networkx as nx
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import datetime

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network
from .oee import OEE

class WBCP(Compiler):
    """
    WBCP: Window-Based Circuit Partitioning
    """
    compiler_id = "wbcp"

    def __init__(self):
        super().__init__()
        return

    @property
    def name(self) -> str:
        return "WBCP"
    
    def compile(self, 
                circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit
        """
        print(f"Compiling with [{self.name}]...")
        
        start_time = time.time()
        iteration_count = config.get("iteration", 50) if config else 50
        circuit_name = config.get("circuit_name", "circ") if config else "circ"

        mapping_record_list = self._k_way_WBCP(circuit, network, iteration_count)

        end_time = time.time()

        mapping_record_list.summarize_total_costs()
        mapping_record_list.update_total_costs(execution_time = end_time - start_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mapping_record_list.save_records(f"./outputs/{circuit_name}/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")
        
        return mapping_record_list

    def _k_way_WBCP(self, 
                    circuit: QuantumCircuit, 
                    network: Network,
                    iteration_count: int) -> MappingRecordList:
        """
        The main function for WBCP compilation
        """
        circuit_dag = circuit_to_dag(circuit)
        circuit_layers = list(circuit_dag.layers())

        num_depths = circuit.depth()
        win_len = max(1, min(num_depths // 20, 500)) # 范围在[1, 500]，过长的线路设置成500
        if win_len == 0:
            win_len = num_depths
        num_subc = math.ceil(num_depths / win_len)

        # split qc into 'num_subc' sub-circuits
        # build the qubit interaction graph for each sub-circuit
        mapping_record_list = MappingRecordList()

        # 第一个子线路直接用OEE算法得到初始划分
        right = min(win_len-1, num_depths-1)
        sub_qc = self._get_subcircuit_by_levels(circuit_dag, circuit_layers, (0, right))
        qig = CompilerUtils.build_qubit_interaction_graph(sub_qc)

        # 初始化划分
        partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
        partition = OEE.partition(partition, qig, network, iteration_count)

        # 完成逻辑QPU到物理QPU的映射
        partition = self._map_logical_to_physical(partition, qig, network)
        logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        record = MappingRecord(
            layer_start = 0,
            layer_end = right,
            partition = partition,
            mapping_type = "telegate",
            logical_phy_map = logical_phy_map
        )
        _ = CompilerUtils.evaluate_local_and_telegate_with_cat(record, sub_qc, network)
        mapping_record_list.add_record(record)

        # 处理后续的子线路
        for i in range(1, num_subc):

            # print(f"\n\n\n[DEBUG] Processing sub-circuit {i}/{num_subc - 1}, layers {i*win_len} to {min((i+1)*win_len-1, num_depths-1)}")

            # 获取子线路段
            right = min((i+1)*win_len-1, num_depths-1)
            sub_qc = self._get_subcircuit_by_levels(circuit_dag, circuit_layers, (i*win_len, right))

            # print(f"[DEBUG] subc:\n{sub_qc}")

            # 构造子线路的qubit interaction graph
            qig = CompilerUtils.build_qubit_interaction_graph(sub_qc)

            # 获取上一个子线路的划分结果
            previous_record = mapping_record_list.records[-1]
            previous_partition = previous_record.partition

            # 构造子线路带权重的qubit interaction graph，WBCP特有
            weighted_qig = self._build_weighted_qigraph(sub_qc, previous_partition)
            
            # 继续用OEE算法得到划分
            current_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
            current_partition = OEE.partition(current_partition, weighted_qig, network, iteration_count)

            # 先确定partition，再确定logical到物理的映射
            current_partition = self._map_logical_to_physical(current_partition, qig, network)

            # 构建当前划分对应的mapping record
            current_record = MappingRecord(
                layer_start = i*win_len,
                layer_end = right,
                partition = current_partition,
                mapping_type = "telegate",
                logical_phy_map = previous_record.logical_phy_map
            )

            # 如果采用当前的record，评估当前划分的costs
            # 先评估teledata并更新logical_phy_map
            _ = CompilerUtils.evaluate_teledata(previous_record, current_record, network)
            # 再评估local和telegate
            _ = CompilerUtils.evaluate_local_and_telegate_with_cat(current_record, sub_qc, network)

            # 检查沿用上一个partition是否更好
            # 如果沿用上一个partition，只有remote telegate开销，没有teledata开销
            num_prev_remote_hops = CompilerUtils.evaluate_remote_hops(qig, previous_partition, network)

            # 比较哪个开销小
            # 如果先前的partition更好
            if num_prev_remote_hops < current_record.costs.num_comms:
                # 更新previous record的costs和layers
                record = MappingRecord(
                    layer_start = i*win_len,
                    layer_end = right,
                    partition = previous_partition,
                    mapping_type = "telegate",
                    logical_phy_map = previous_record.logical_phy_map # 沿用上一个record的logical_phy_map作为初始状态
                )
                # 完整评估划分，只有local and telegate开销
                _ = CompilerUtils.evaluate_local_and_telegate_with_cat(record, sub_qc, network)
                mapping_record_list.add_record(record)
            else:
                mapping_record_list.add_record(current_record)

        return mapping_record_list

    def _get_subcircuit_by_levels(self, circuit_dag, circuit_layers, level_range: tuple[int, int]) -> QuantumCircuit:
        """
        Get the sub-circuit by the given levels
        """
        sub_dag = circuit_dag.copy_empty_like()
        for level in range(level_range[0], level_range[1] + 1):
            for node in circuit_layers[level]["graph"].op_nodes():
                sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        # dag_drawer(sub_dag, scale=0.8, filename=f"dag_{level_range}.png")
        sub_qc = dag_to_circuit(sub_dag)
        return sub_qc

    def _build_weighted_qigraph(self, 
                                circuit: QuantumCircuit, 
                                previous_partition: list[list[int]]) -> nx.Graph:
        G = nx.Graph()
        for node in range(circuit.num_qubits):
            G.add_node(node) # 添加num_qubits个节点

        # 记录每个qubits所属的分区编号
        qubit_partition = {}
        for i, partition in enumerate(previous_partition):
            for qubit in partition:
                qubit_partition[qubit] = i
        
        for instruction in circuit:
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) > 1:
                if instruction.name == "barrier":
                    continue
                assert(len(qubits) == 2)
                edge_weight = 1
                # 检查qubits[0]和qubits[1]是否在同一个分区
                if qubit_partition[qubits[0]] == qubit_partition[qubits[1]]:
                    edge_weight = 2
                if G.has_edge(qubits[0], qubits[1]):
                    G[qubits[0]][qubits[1]]['weight'] += edge_weight
                else:
                    G.add_edge(qubits[0], qubits[1], weight = edge_weight)
        return G

    # def _map_logical_to_physical(self, 
    #                              partition: list[list[int]],
    #                              qig: nx.Graph,
    #                              network: Network) -> list[list[int]]:
    #     """
    #     Map logical QPUs in the partition to physical QPUs in the network.
    #     Uses a MILP formulation to minimize total weighted EPR cost.

    #     Args:
    #         partition: logical partition (list of lists, each list contains qubit indices)
    #         qig: current qubit interaction graph for the window (with edge weights)
    #         network: physical network object

    #     Returns:
    #         new_partition: partition after physical mapping (list of lists, length = number of physical nodes)
    #     """
    #     # 提取非空逻辑组及其索引
    #     logical_qpu_ids = [i for i, logical_qpu in enumerate(partition) if logical_qpu]
        
    #     if len(logical_qpu_ids) == 0: # 全部逻辑QPU都没有分配到任何qubit，直接返回空的物理划分
    #         return partition
        
    #     # 提取物理QPU索引
    #     physical_qpu_ids = list(range(network.num_backends))

    #     # 获取每两个逻辑QPU之间的交互权重
    #     # ---------- 建立量子比特到组的映射 ----------
    #     qubit_to_logical_qpu = {}
    #     for logical_qpu in logical_qpu_ids:
    #         for qubit in partition[logical_qpu]:
    #             qubit_to_logical_qpu[qubit] = logical_qpu

    #     inter_logical_qpu_weights = {}
    #     for (physical_qpu_id, logical_qpu_id, data) in qig.edges(data=True): # 遍历所有的2q操作
    #         assert physical_qpu_id in qubit_to_logical_qpu and logical_qpu_id in qubit_to_logical_qpu, f"Qubit {physical_qpu_id} or {logical_qpu_id} not found in any logical QPU"
    #         logical_qpu1, logical_qpu2 = qubit_to_logical_qpu[physical_qpu_id], qubit_to_logical_qpu[logical_qpu_id]
    #         if logical_qpu1 == logical_qpu2:
    #             continue
    #         if logical_qpu1 > logical_qpu2:
    #             logical_qpu1, logical_qpu2 = logical_qpu2, logical_qpu1
    #         weight = data["weight"]
    #         inter_logical_qpu_weights[(logical_qpu1, logical_qpu2)] = inter_logical_qpu_weights.get((logical_qpu1, logical_qpu2), 0) + weight

    #     # 获取两个物理QPU之间的交互权重
    #     # network.network_graph
    #     inter_physical_qpu_weights = {}
    #     for (physical_qpu_id, logical_qpu_id, data) in network.network_graph.edges(data=True):
    #         # 提取权重（原逻辑所有边都有weight，无需默认值）
    #         weight = data["weight"]
    #         # 存储正向和反向键，确保双向查询都能拿到值
    #         inter_physical_qpu_weights[(physical_qpu_id, logical_qpu_id)] = weight
    #         inter_physical_qpu_weights[(logical_qpu_id, physical_qpu_id)] = weight

    #     # ---------- 若无组间通信，直接顺序映射 ----------
    #     if not inter_logical_qpu_weights:
    #         return partition

    #     # ---------- 构建MILP模型 ----------
    #     model = gp.Model("logical_to_physical_mapping")
    #     model.setParam('OutputFlag', 1)  # Gurobi输出

    #     logical_edges = list(inter_logical_qpu_weights.keys())
    #     physical_edges = list(inter_physical_qpu_weights.keys())
    #     logical_edges_weights = list(inter_logical_qpu_weights.values())
    #     physical_edges_costs = list(inter_physical_qpu_weights.values())

    #     # ---------- 创建变量 ----------
    #     y_vars = {}  # 逻辑QPU到物理QPU的映射变量
    #     x_vars = {}  # 逻辑边到物理边的映射变量
    #     z1_vars = {} # 辅助变量 z1ij = y[u1][v1] * y[u2][v2]
    #     z2_vars = {} # 辅助变量 z2ij = y[u1][v2] * y[u2][v1]

    #     # y[physical_node][logical_node]: 逻辑QPU v 是否分配到物理QPU u
    #     for physical_qpu_id in physical_qpu_ids:
    #         for logical_qpu_id in logical_qpu_ids:
    #             y_vars[(physical_qpu_id, logical_qpu_id)] = model.addVar(vtype=GRB.BINARY, name=f"y_{physical_qpu_id}_{logical_qpu_id}")
        
    #     # x[physical_edge_idx][logical_edge_idx]: 逻辑边 j 是否映射到物理边 i
    #     # z1_vars[i][j], z2_vars[i][j]: 辅助变量用于线性化二次项
    #     for i, _ in enumerate(physical_edges):
    #         for j, _ in enumerate(logical_edges):
    #             x_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    #             z1_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z1_{i}_{j}")
    #             z2_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z2_{i}_{j}")
     
    #     # 设置目标函数：最小化总加权EPR使用量
    #     obj_expr = gp.LinExpr()
    #     for i, cost in enumerate(physical_edges_costs):
    #         for j, weight in enumerate(logical_edges_weights):
    #             obj_expr += cost * weight * x_vars[(i, j)]
    #     model.setObjective(obj_expr, GRB.MINIMIZE)

    #     # ---------- 顶点映射约束 ----------
    #     # 每个逻辑节点必须映射到一个物理节点
    #     for logical_qpu_id in logical_qpu_ids:
    #         model.addConstr(gp.quicksum(y_vars[(u, logical_qpu_id)] for u in physical_qpu_ids) == 1,
    #                         name=f"logic_to_phys_{logical_qpu_id}")
            
    #     # 每个物理节点最多承载一个逻辑节点
    #     for physical_qpu_id in physical_qpu_ids:
    #         model.addConstr(gp.quicksum(y_vars[(physical_qpu_id, v)] for v in logical_qpu_ids) <= 1,
    #                         name=f"phys_to_logic_{physical_qpu_id}")
            
    #     # 每条逻辑边必须映射到一条物理边
    #     for j, _ in enumerate(logical_edges):
    #         model.addConstr(gp.quicksum(x_vars[(i, j)] for i, _ in enumerate(physical_edges)) == 1,
    #                         name=f"logic_edge_map_{j}")

    #     # ---------- 线性化约束（定义 z1, z2）----------
    #     for i, (u1, u2) in enumerate(physical_edges):
    #         for j, (v1, v2) in enumerate(logical_edges):
    #             # z1ij = y[u1][v1] * y[u2][v2]
    #             # z1ij <= y[u1][v1]
    #             model.addConstr(z1_vars[(i, j)] <= y_vars[(u1, v1)], name=f"z1_leq_y1_{i}_{j}")
    #             # z1ij <= y[u2][v2]
    #             model.addConstr(z1_vars[(i, j)] <= y_vars[(u2, v2)], name=f"z1_leq_y2_{i}_{j}")
    #             # z1ij >= y[u1][v1] + y[u2][v2] - 1
    #             model.addConstr(z1_vars[(i, j)] >= y_vars[(u1, v1)] + y_vars[(u2, v2)] - 1,
    #                             name=f"z1_geq_sum_{i}_{j}")

    #             # z2 = y[u1][v2] * y[u2][v1]
    #             # z2ij <= y[u1][v2]
    #             model.addConstr(z2_vars[(i, j)] <= y_vars[(u1, v2)], name=f"z2_leq_y1_{i}_{j}")
    #             # z2ij <= y[u2][v1]
    #             model.addConstr(z2_vars[(i, j)] <= y_vars[(u2, v1)], name=f"z2_leq_y2_{i}_{j}")
    #             # z2ij >= y[u1][v2] + y[u2][v1] - 1
    #             model.addConstr(z2_vars[(i, j)] >= y_vars[(u1, v2)] + y_vars[(u2, v1)] - 1,
    #                             name=f"z2_geq_sum_{i}_{j}")

    #             # x = z1 + z2
    #             model.addConstr(x_vars[(i, j)] == z1_vars[(i, j)] + z2_vars[(i, j)],
    #                             name=f"consistency_{i}_{j}")
                
    #     # ---------- 求解 ----------
    #     model.optimize()

    #     # ---------- 处理求解结果 ----------
    #     if model.status == GRB.OPTIMAL:
    #         print(f"Physical mapping optimal solution found, objective = {model.objVal}")

    #         # 构建物理节点到逻辑组ID的映射
    #         physical_to_logical = {}
    #         for physical_qpu_id in physical_qpu_ids:
    #             for logical_qpu_id in logical_qpu_ids:
    #                 if y_vars[(physical_qpu_id, logical_qpu_id)].x > 0.5:
    #                     physical_to_logical[physical_qpu_id] = logical_qpu_id
    #                     break

    #         # 更新partition
    #         new_partition = [[] for _ in range(len(physical_qpu_ids))]
    #         for physical_qpu_id, logical_qpu_id in physical_to_logical.items():
    #             new_partition[physical_qpu_id] = partition[logical_qpu_id]

    #         return new_partition

    #     # 求解失败，回退到顺序分配
    #     print("Warning: Physical mapping optimization failed. Falling back to sequential assignment.")
    #     return partition

    def _map_logical_to_physical(self, 
                                 partition: list[list[int]],
                                 qig: nx.Graph,
                                 network: Network) -> list[list[int]]:
        """
        Map logical QPUs in the partition to physical QPUs in the network.
        Uses a simplified MILP formulation to minimize total weighted communication cost.
        """
        # 1. 提取非空逻辑组及其索引
        logical_qpu_ids = [i for i, logical_qpu in enumerate(partition) if logical_qpu]
        
        if len(logical_qpu_ids) == 0: 
            return partition
        
        # 2. 提取物理QPU索引
        physical_qpu_ids = list(range(network.num_backends))

        # 3. 安全检查：如果逻辑QPU数量 > 物理QPU数量，必然无解（通常不会发生）
        if len(logical_qpu_ids) > len(physical_qpu_ids):
            print("Warning: More logical QPUs than physical QPUs. Falling back.")
            return partition

        # ---------- 建立量子比特到逻辑QPU的映射 ----------
        qubit_to_logical = {}
        for l_idx in logical_qpu_ids:
            for qubit in partition[l_idx]:
                qubit_to_logical[qubit] = l_idx

        # ---------- 计算逻辑QPU之间的交互权重 (Logical Weight Matrix) ----------
        # W_logical[l1][l2]: 逻辑QPU l1 和 l2 之间的交互量
        W_logical = defaultdict(int)
        for q1, q2, data in qig.edges(data=True):
            # 确保 q1 和 q2 都在当前的 partition 里
            if q1 not in qubit_to_logical or q2 not in qubit_to_logical:
                continue
                
            l1 = qubit_to_logical[q1]
            l2 = qubit_to_logical[q2]
            
            if l1 == l2:
                continue
            
            # 确保顺序一致，避免重复计算 (l1, l2) 和 (l2, l1)
            if l1 > l2:
                l1, l2 = l2, l1
            
            W_logical[(l1, l2)] += data.get("weight", 1)

        # ---------- 若无组间通信，直接返回 ----------
        if not W_logical:
            return partition

        # ---------- TODO: 获取物理QPU之间的通信成本 (Physical Cost Matrix) ----------
        # C_physical[p1][p2]: 物理QPU p1 和 p2 之间的通信代价 (通常是跳数)
        # 使用 networkx 的 shortest_path_length 预计算所有物理节点对的距离
        # 这样就不需要物理拓扑是全连接的了！
        C_physical = {}
        for p1 in physical_qpu_ids:
            # 计算 p1 到所有点的最短路径
            lengths = nx.shortest_path_length(network.network_graph, source=p1)
            for p2 in physical_qpu_ids:
                if p1 != p2:
                    # 如果网络没有边权重，默认用跳数；如果有，用权重
                    # 这里假设 network_graph 的 'weight' 就是我们需要的代价
                    try:
                        # 尝试直接获取边权重（如果是直连）
                        if network.network_graph.has_edge(p1, p2):
                            cost = network.network_graph[p1][p2].get('weight', 1)
                        else:
                            # 非直连，成本 = 跳数 * 单位成本（这里简化为跳数）
                            cost = lengths.get(p2, float('inf'))
                    except:
                        cost = lengths.get(p2, float('inf'))
                    C_physical[(p1, p2)] = cost

        # ---------- 构建MILP模型 (简化版) ----------
        model = gp.Model("logical_to_physical_mapping")
        model.setParam('OutputFlag', 1)
        # 设定一个时间限制，防止卡死
        model.setParam('TimeLimit', 60) 

        # ---------- 创建变量 ----------
        # y[p][l]: 逻辑QPU l 是否放置在物理QPU p 上
        y = model.addVars(physical_qpu_ids, logical_qpu_ids, vtype=GRB.BINARY, name="y")

        # ---------- 设置目标函数 ----------
        # Minimize sum_{l1<l2} sum_{p1, p2} W_logical[l1][l2] * C_physical[p1][p2] * y[p1][l1] * y[p2][l2]
        # 由于 Gurobi 不能直接处理二次项，我们需要用线性化技巧，或者直接使用 gurobi 的 qp 功能
        # 这里为了简单，我们直接设置目标为 QuadExpr
        obj = gp.QuadExpr()
        
        for (l1, l2), w in W_logical.items():
            for p1 in physical_qpu_ids:
                for p2 in physical_qpu_ids:
                    if p1 == p2:
                        continue # 同一个物理QPU没有通信成本（或者已经在前面过滤了）
                    cost = C_physical.get((p1, p2), 1e6) # 如果不可达，给一个极大惩罚
                    # 添加项: weight * cost * y[p1][l1] * y[p2][l2]
                    obj += w * cost * y[p1, l1] * y[p2, l2]

        model.setObjective(obj, GRB.MINIMIZE)

        # ---------- 约束 ----------
        # 1. 每个逻辑QPU必须恰好放在一个物理QPU上
        for l in logical_qpu_ids:
            model.addConstr(gp.quicksum(y[p, l] for p in physical_qpu_ids) == 1, name=f"AssignL_{l}")
            
        # 2. 每个物理QPU最多放一个逻辑QPU
        for p in physical_qpu_ids:
            model.addConstr(gp.quicksum(y[p, l] for l in logical_qpu_ids) <= 1, name=f"AssignP_{p}")

        # ---------- 求解 ----------
        # 告诉 Gurobi 这是一个非凸二次规划 (MIQP)，需要使用特定的求解器
        model.setParam('NonConvex', 2) 
        model.optimize()

        # ---------- 处理求解结果 ----------
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            if model.status == GRB.OPTIMAL:
                print(f"Physical mapping optimal solution found, objective = {model.objVal}")
            else:
                print(f"Physical mapping suboptimal solution found (status {model.status}), objective = {model.objVal}")

            # 构建映射关系
            physical_to_logical_idx = {}
            for p in physical_qpu_ids:
                for l in logical_qpu_ids:
                    if y[p, l].x > 0.5:
                        physical_to_logical_idx[p] = l
                        break

            # 更新 partition
            # new_partition 的索引对应物理QPU
            new_partition = [[] for _ in range(network.num_backends)]
            for p, l_idx in physical_to_logical_idx.items():
                new_partition[p] = partition[l_idx]

            return new_partition

        # 求解失败，打印 IIS (Irreducible Inconsistent Subsystem) 帮助调试
        print("Warning: Physical mapping optimization failed. Computing IIS...")
        try:
            model.computeIIS()
            model.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
        except:
            pass
            
        return partition