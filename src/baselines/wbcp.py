import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from typing import Any, Optional
import networkx as nx
import numpy as np
import time
from itertools import combinations
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network
from .oee import OEE

class WBCP(Compiler):
    """
    WBCP: Window-Based Circuit Partitioning
    """
    compiler_id = "fgproee"

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
        iteration_count = config.get("iteration", 10) if config else 10
        circuit_name = config.get("circuit_name", "circ") if config else "circ"

        mapping_record_list = self._k_way_WBCP(circuit, network, iteration_count)

        end_time = time.time()

        # mapping_record_list.add_cost("exec_time (sec)", end_time - start_time)
        # mapping_record_list = CompilerUtils.evaluate_total_costs(mapping_record_list)
        mapping_record_list.summarize_total_costs()
        mapping_record_list.update_total_costs(execution_time = end_time - start_time)
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        
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
        win_len = num_depths // 20
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

        # TODO: 完成逻辑QPU到物理QPU的映射
        partition = self._map_logical_to_physical(partition, qig, network)

        record = MappingRecord(
            layer_start = 0,
            layer_end = right,
            partition = partition,
            mapping_type = "telegate"
        )
        _ = CompilerUtils.evaluate_local_and_telegate(record, sub_qc, network)
        mapping_record_list.add_record(record)

        # 处理后续的子线路
        for i in range(1, num_subc):
            # 获取子线路段
            right = min((i+1)*win_len-1, num_depths-1)
            sub_qc = self._get_subcircuit_by_levels(circuit_dag, circuit_layers, (i*win_len, right))
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

            current_record = MappingRecord(
                layer_start = i*win_len,
                layer_end = right,
                partition = current_partition,
                mapping_type = "telegate"
            )
            _ = CompilerUtils.evaluate_local_and_telegate(current_record, sub_qc, network)
            costs_for_current_partition = CompilerUtils.evaluate_teledata(previous_record, current_record, network)

            # 检查沿用上一个partition是否更好
            # 如果沿用上一个partition，只有local和remote telegate开销，没有teledata开销
            num_prev_remote_hops = CompilerUtils.evaluate_remote_hops(qig, previous_partition, network)

            # 比较哪个开销小
            if num_prev_remote_hops < costs_for_current_partition.num_comms:
                # 更新previous record的costs和layers
                record = MappingRecord(
                    layer_start = i*win_len,
                    layer_end = right,
                    partition = previous_partition,
                    mapping_type = "telegate"
                )
                _ = CompilerUtils.evaluate_local_and_telegate(record, sub_qc, network)
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

    def _map_logical_to_physical(self, 
                                 partition: list[list[int]],
                                 qig: nx.Graph,
                                 network: Network) -> list[list[int]]:
        """
        Map logical QPUs in the partition to physical QPUs in the network.
        Uses a MILP formulation to minimize total weighted EPR cost.

        Args:
            partition: logical partition (list of lists, each list contains qubit indices)
            qig: current qubit interaction graph for the window (with edge weights)
            network: physical network object

        Returns:
            new_partition: partition after physical mapping (list of lists, length = number of physical nodes)
        """
        # 提取非空逻辑组及其索引
        logical_qpu_ids = [i for i, logical_qpu in enumerate(partition) if logical_qpu]
        
        if len(logical_qpu_ids) == 0: # 全部逻辑QPU都没有分配到任何qubit，直接返回空的物理划分
            return partition
        
        # 提取物理QPU索引
        physical_qpu_ids = list(range(network.num_backends))

        # 获取每两个逻辑QPU之间的交互权重
        # ---------- 建立量子比特到组的映射 ----------
        qubit_to_logical_qpu = {}
        for logical_qpu in logical_qpu_ids:
            for qubit in partition[logical_qpu]:
                qubit_to_logical_qpu[qubit] = logical_qpu

        inter_logical_qpu_weights = {}
        for (physical_qpu_id, logical_qpu_id, data) in qig.edges(data=True): # 遍历所有的2q操作
            assert physical_qpu_id in qubit_to_logical_qpu and logical_qpu_id in qubit_to_logical_qpu, f"Qubit {physical_qpu_id} or {logical_qpu_id} not found in any logical QPU"
            logical_qpu1, logical_qpu2 = qubit_to_logical_qpu[physical_qpu_id], qubit_to_logical_qpu[logical_qpu_id]
            if logical_qpu1 == logical_qpu2:
                continue
            if logical_qpu1 > logical_qpu2:
                logical_qpu1, logical_qpu2 = logical_qpu2, logical_qpu1
            weight = data["weight"]
            inter_logical_qpu_weights[(logical_qpu1, logical_qpu2)] = inter_logical_qpu_weights.get((logical_qpu1, logical_qpu2), 0) + weight

        # 获取两个物理QPU之间的交互权重
        # network.network_graph
        inter_physical_qpu_weights = {}
        for (physical_qpu_id, logical_qpu_id, data) in network.network_graph.edges(data=True):
            # 提取权重（原逻辑所有边都有weight，无需默认值）
            weight = data["weight"]
            # 存储正向和反向键，确保双向查询都能拿到值
            inter_physical_qpu_weights[(physical_qpu_id, logical_qpu_id)] = weight
            inter_physical_qpu_weights[(logical_qpu_id, physical_qpu_id)] = weight

        # ---------- 若无组间通信，直接顺序映射 ----------
        if not inter_logical_qpu_weights:
            return partition

        # ---------- 构建MILP模型 ----------
        model = gp.Model("logical_to_physical_mapping")
        model.setParam('OutputFlag', 1)  # Gurobi输出

        logical_edges = list(inter_logical_qpu_weights.keys())
        physical_edges = list(inter_physical_qpu_weights.keys())
        logical_edges_weights = list(inter_logical_qpu_weights.values())
        physical_edges_costs = list(inter_physical_qpu_weights.values())

        # ---------- 创建变量 ----------
        y_vars = {}  # 逻辑QPU到物理QPU的映射变量
        x_vars = {}  # 逻辑边到物理边的映射变量
        z1_vars = {} # 辅助变量 z1ij = y[u1][v1] * y[u2][v2]
        z2_vars = {} # 辅助变量 z2ij = y[u1][v2] * y[u2][v1]

        # y[physical_node][logical_node]: 逻辑QPU v 是否分配到物理QPU u
        for physical_qpu_id in physical_qpu_ids:
            for logical_qpu_id in logical_qpu_ids:
                y_vars[(physical_qpu_id, logical_qpu_id)] = model.addVar(vtype=GRB.BINARY, name=f"y_{physical_qpu_id}_{logical_qpu_id}")
        
        # x[physical_edge_idx][logical_edge_idx]: 逻辑边 j 是否映射到物理边 i
        # z1_vars[i][j], z2_vars[i][j]: 辅助变量用于线性化二次项
        for i, _ in enumerate(physical_edges):
            for j, _ in enumerate(logical_edges):
                x_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                z1_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z1_{i}_{j}")
                z2_vars[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z2_{i}_{j}")
     
        # 设置目标函数：最小化总加权EPR使用量
        obj_expr = gp.LinExpr()
        for i, cost in enumerate(physical_edges_costs):
            for j, weight in enumerate(logical_edges_weights):
                obj_expr += cost * weight * x_vars[(i, j)]
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # ---------- 顶点映射约束 ----------
        # 每个逻辑节点必须映射到一个物理节点
        for logical_qpu_id in logical_qpu_ids:
            model.addConstr(gp.quicksum(y_vars[(u, logical_qpu_id)] for u in physical_qpu_ids) == 1,
                            name=f"logic_to_phys_{logical_qpu_id}")
            
        # 每个物理节点最多承载一个逻辑节点
        for physical_qpu_id in physical_qpu_ids:
            model.addConstr(gp.quicksum(y_vars[(physical_qpu_id, v)] for v in logical_qpu_ids) <= 1,
                            name=f"phys_to_logic_{physical_qpu_id}")
            
        # 每条逻辑边必须映射到一条物理边
        for j, _ in enumerate(logical_edges):
            model.addConstr(gp.quicksum(x_vars[(i, j)] for i, _ in enumerate(physical_edges)) == 1,
                            name=f"logic_edge_map_{j}")

        # ---------- 线性化约束（定义 z1, z2）----------
        for i, (u1, u2) in enumerate(physical_edges):
            for j, (v1, v2) in enumerate(logical_edges):
                # z1ij = y[u1][v1] * y[u2][v2]
                # z1ij <= y[u1][v1]
                model.addConstr(z1_vars[(i, j)] <= y_vars[(u1, v1)], name=f"z1_leq_y1_{i}_{j}")
                # z1ij <= y[u2][v2]
                model.addConstr(z1_vars[(i, j)] <= y_vars[(u2, v2)], name=f"z1_leq_y2_{i}_{j}")
                # z1ij >= y[u1][v1] + y[u2][v2] - 1
                model.addConstr(z1_vars[(i, j)] >= y_vars[(u1, v1)] + y_vars[(u2, v2)] - 1,
                                name=f"z1_geq_sum_{i}_{j}")

                # z2 = y[u1][v2] * y[u2][v1]
                # z2ij <= y[u1][v2]
                model.addConstr(z2_vars[(i, j)] <= y_vars[(u1, v2)], name=f"z2_leq_y1_{i}_{j}")
                # z2ij <= y[u2][v1]
                model.addConstr(z2_vars[(i, j)] <= y_vars[(u2, v1)], name=f"z2_leq_y2_{i}_{j}")
                # z2ij >= y[u1][v2] + y[u2][v1] - 1
                model.addConstr(z2_vars[(i, j)] >= y_vars[(u1, v2)] + y_vars[(u2, v1)] - 1,
                                name=f"z2_geq_sum_{i}_{j}")

                # x = z1 + z2
                model.addConstr(x_vars[(i, j)] == z1_vars[(i, j)] + z2_vars[(i, j)],
                                name=f"consistency_{i}_{j}")
                
        # ---------- 求解 ----------
        model.optimize()

        # ---------- 处理求解结果 ----------
        if model.status == GRB.OPTIMAL:
            print(f"Physical mapping optimal solution found, objective = {model.objVal}")

            # 构建物理节点到逻辑组ID的映射
            physical_to_logical = {}
            for physical_qpu_id in physical_qpu_ids:
                for logical_qpu_id in logical_qpu_ids:
                    if y_vars[(physical_qpu_id, logical_qpu_id)].x > 0.5:
                        physical_to_logical[physical_qpu_id] = logical_qpu_id
                        break

            # 更新partition
            new_partition = [[] for _ in range(len(physical_qpu_ids))]
            for physical_qpu_id, logical_qpu_id in physical_to_logical.items():
                new_partition[physical_qpu_id] = partition[logical_qpu_id]

            return new_partition

        # 求解失败，回退到顺序分配
        print("Warning: Physical mapping optimization failed. Falling back to sequential assignment.")
        return partition