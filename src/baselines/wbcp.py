import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from typing import Any, Optional
import networkx as nx
import numpy as np
import time
from itertools import combinations
from collections import defaultdict

from ..compiler import Compiler, MappingRecord, MappingRecordList
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

        mapping_record_list.add_cost("exec_time (sec)", end_time - start_time)
        mapping_record_list = self.evaluate_total_costs(mapping_record_list)
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
        qig = self.build_qubit_interaction_graph(sub_qc)

        # 初始化划分
        partition = self.allocate_qubits(circuit.num_qubits, network)
        partition = OEE.partition(partition, qig, network, iteration_count)
        record = MappingRecord(
            layer_start = 0,
            layer_end = right,
            partition = partition,
            mapping_type = "telegate",
            costs = self.evaluate_partition(qig, partition, network)
        )
        mapping_record_list.add_record(record)

        # 处理后续的子线路
        for i in range(1, num_subc):
            # 获取子线路段
            right = min((i+1)*win_len-1, num_depths-1)
            sub_qc = self._get_subcircuit_by_levels(circuit_dag, circuit_layers, (i*win_len, right))
            # 构造子线路的qubit interaction graph
            qig = self.build_qubit_interaction_graph(sub_qc)

            # 获取上一个子线路的划分结果
            previous_record = mapping_record_list.records[-1]
            previous_partition = previous_record.partition

            # 构造子线路带权重的qubit interaction graph，WBCP特有
            weighted_qig = self._build_weighted_qigraph(sub_qc, previous_partition)
            
            # 继续用OEE算法得到划分
            current_partition = self.allocate_qubits(circuit.num_qubits, network)
            current_partition = OEE.partition(current_partition, weighted_qig, network, iteration_count)
            current_record = MappingRecord(
                layer_start = i*win_len,
                layer_end = right,
                partition = current_partition,
                mapping_type = "telegate",
                costs = self.evaluate_partition(qig, current_partition, network)
            )
            costs_for_current_partition = self.evaluate_partition_switch(previous_record, current_record, network)

            # 检查用上一个partition是否更好
            costs_for_previous_partition = self.evaluate_partition(qig, previous_partition, network)

            # 比较哪个开销小
            if costs_for_previous_partition["num_comms"] < costs_for_current_partition["num_comms"]:
                # 更新previous record的costs和layers
                record = MappingRecord(
                    layer_start = i*win_len,
                    layer_end = right,
                    partition = previous_partition,
                    mapping_type = "telegate",
                    costs = costs_for_previous_partition
                )
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
