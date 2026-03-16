from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit

from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
from pytket_dqc.utils import DQCPass
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.distributors import CoverEmbeddingSteinerDetached, PartitioningAnnealing

from typing import Any, Optional
import time
from pprint import pprint
import networkx as nx
import sys
import numpy as np
import copy

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network
from .partitioner import PartitionerFactory
from .partition_assigner import PartitionAssignerFactory
from .mapper import MapperFactory

class NADQC(Compiler):
    """
    Noise-Aware Distributed Quantum Compiler
    """
    compiler_id = "nadqc"

    def __init__(self):
        super().__init__()
        return
    
    @property
    def name(self) -> str:
        return "NADQC"
    
    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit
        """
        print(f"Compiling with [{self.name}]...")
        
        self.circuit = circuit
        self.network = network
        circuit_name = config.get("circuit_name", "circ") if config else "circ"
        
        partitioner_type = config.get("partitioner", 
                                 "recursive_dp") if config else "recursive_dp"
        self.max_option = config.get("max_option", 1) if config else 1
        self.min_depths = config.get("min_depths", None) if config else None
        self.partitioner = PartitionerFactory.create_partitioner(partitioner_type, 
                                                                 network, 
                                                                 max_options=self.max_option)
        
        partition_assigner_type = config.get("partition_assigner", "global_max_match") if config else "global_max_match"
        self.partition_assigner = PartitionAssignerFactory.create_assigner(partition_assigner_type)

        mapper_type = config.get("mapper", "simple") if config else "simple"
        self.mapper = MapperFactory.create_mapper(mapper_type)

        hyper_partitioner_type = config.get("hyper_partitioner", "CE") if config else "CE"
        self.hyper_partitioner = hyper_partitioner_type
        server_coupling, server_qubits = self.network.get_network_coupling_and_qubits()
        self.nisq_network = NISQNetwork(server_coupling, server_qubits)

        start_time = time.time()

        # 获取仅有双量子比特门的线路
        print(f"[DEBUG] remove_single_qubit_gates", file=sys.stderr)
        self.remove_single_qubit_gates()
        self.set_min_depth()

        # partition table
        print(f"[DEBUG] build_partition_table", file=sys.stderr)
        self.build_partition_table()
        
        # slicing table and slicing results
        self.build_slicing_table()
        self.subc_ranges = []
        self.get_sliced_subc(0, len(self.P) - 1)

        # teledata-only partition candidates
        self.partition_candidates = []
        for (i, j) in self.subc_ranges:
            self.partition_candidates.append(self.P[i][j]) # 返回所有划分方案
        
        # 测试匹配一致性
        self.partition_plan = self.partition_assigner.assign(self.partition_candidates)["partition_plan"]

        # 构建mapping_record_list，此时层号是在2q gates only线路上的
        mapping_record_list = self.construct_teledata_only_mapping_record_list(self.partition_plan)

        # 尝试使用telegate并恢复在原始线路中的层号
        grouped_records = self.group_shallow_subcircuits(mapping_record_list)

        # TODO: 改成考虑噪声信息的
        # 考虑异构噪声的QPU间链路
        self.partition_plan = self.mapper.map(grouped_records, self.network)["partition_plan"]
        print(f"[DEBUG] partition_plan: {self.partition_plan}", file=sys.stderr)

        end_time = time.time()

        grouped_records.add_cost("exec_time (sec)", end_time - start_time)
        grouped_records = CompilerUtils.evaluate_total_costs(grouped_records)
        grouped_records.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        return grouped_records

    def get_partition_candidates(self):
        return self.partition_candidates

    def set_min_depth(self):
        def sigmoid_decay(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 - 1 / (1 + np.exp(-k * (gate_density - c))))
        def sigmoid_increase(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 / (1 + np.exp(-k * (gate_density - c))))
        if self.min_depths is None:
            self.min_depths = int(sigmoid_increase(self.cu1_density, self.circuit.depth()))
        print(f"[INFO] gate_density: {self.gate_density}")
        print(f"[INFO] cu1_density: {self.cu1_density}")
        print(f"[INFO] min_depth: {self.min_depths}")
        return

    def remove_single_qubit_gates(self):
        """
        Remove single qubit gates to construct self.dag_multiq
        Calculate gate density
        """
        # 根据原始线路的dag，逐层保留双量子比特门，并记录双量子比特门的层号
        start_time = time.time()
        self.dag = circuit_to_dag(self.circuit)
        self.dag_multiq = []
        self.map_to_init_layer = {}
        self.map_to_dag_multi_layer = {}
        pos_count, cu1_count, gate_count = 0, 0, 0
        # dag_debug = self.dag.copy_empty_like()
        # 遍历self.dag的每一层，如果是双量子比特门
        # 则添加到self.dag_multiq
        layers = list(self.dag.layers())
        for lev, layer in enumerate(layers):
            curr_layer = []
            for node in layer["graph"].op_nodes():
                # print(f"{lev} {node.op.name} {node.qargs}")
                pos_count += len(node.qargs)
                gate_count += 1
                if len(node.qargs) > 1:
                    if node.op.name == "barrier":
                        continue
                    assert len(node.qargs) == 2, f"[ERROR] Found gate with more than 2 qubits: {node.op.name} on {node.qargs}"
                    if node.op.name == "cu1":
                        cu1_count += 1
                    curr_layer.append(node)
                    # dag_debug.apply_operation_back(node.op, node.qargs, node.cargs)
            if len(curr_layer) > 0:
                if len(curr_layer) == self.circuit.num_qubits // 2 and len(curr_layer) % self.network.num_backends != 0:
                    # split the layer into two layers
                    split_point = len(curr_layer) // 2
                    first_half = curr_layer[:split_point]
                    second_half = curr_layer[split_point:]
                    self.dag_multiq.append(first_half)
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
                    self.dag_multiq.append(second_half)
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
                else:
                    self.dag_multiq.append(curr_layer)
                    # 记录双量子比特门在原始线路的层号
                    self.map_to_init_layer[len(self.dag_multiq)-1] = lev
                    self.map_to_dag_multi_layer[lev] = len(self.dag_multiq)-1
        self.gate_density = pos_count / (self.circuit.num_qubits * self.circuit.depth())
        self.cu1_density = cu1_count / gate_count

        end_time = time.time()
        print(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        # self.reconstruct_and_visualize_circuit()
        return
    
    def reconstruct_and_visualize_circuit(self):
        """
        Reconstructs the multi-qubit gate circuit from self.dag_multiq and visualizes it.
        
        Creates a new QuantumCircuit containing only the multi-qubit gates (with barrier separators),
        then prints a text-based visualization using Qiskit's circuit_drawer.
        """
        # 如果没有双量子门，直接返回
        if not self.dag_multiq:
            print("No multi-qubit gates found. Cannot visualize circuit.")
            return

        pprint(self.dag_multiq)
        # 创建新的量子线路（量子比特数与原始线路一致）
        n_qubits = self.circuit.num_qubits
        recon_circ = QuantumCircuit(n_qubits)
        
        # 按层添加双量子门
        for i, layer in enumerate(self.dag_multiq):
            for node in layer:
                # 添加门操作（包含参数和量子比特）
                recon_circ.append(node.op, node.qargs, node.cargs)
            
            # 在每层后添加 barrier（最后一层不加）
            if i < len(self.dag_multiq) - 1:
                recon_circ.barrier()
        
        # 打印可视化结果
        print("\n" + "="*50)
        print("Reconstructed Multi-Qubit Gate Circuit:")
        print("="*50)
        print(recon_circ)
        print("\n" + "="*50)
        print(f"Total layers: {len(self.dag_multiq)} | Total gates: {sum(len(layer) for layer in self.dag_multiq)}")
        print("="*50)
        
        return recon_circ  # 返回重建的电路对象供进一步使用

    # 
    # P table
    # 
    def build_partition_table(self):
        """
        An efficient way of building the partition table
        """
        start_time = time.time()
        num_depths = len(self.dag_multiq)
        self.P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        print(f"[DEBUG] num_depths: {num_depths}", file=sys.stderr)
        # build the qubit interaction nxGraph for the entire circuit
        qig = self.build_qubit_interaction_graph_by_level((0, num_depths-1))
        is_changed = True

        for i in range(num_depths):
            # print(f"depth [{i}]", file=sys.stderr)
            # ===== P[i][numDepths-1] =====
            # rebuild qig
            if i != 0: # remove the (i-1)-th level of the remaining qig
                is_changed = self.remove_qig_edge(qig, i-1)

            if len(self.P[i][num_depths-1]) > 0: # inherit from the upper grid
                assert i != 0, f"[ERROR] P[{i}][{num_depths-1}] should be empty."
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    self.P[i+1][num_depths-1] = self.P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    self.P[i][num_depths-1] = self.get_qig_partitions(qig)
                    cnt += 1
                    if len(self.P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            self.P[i+1][num_depths-1] = self.P[i][num_depths-1] # downward propagation
        
            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self.remove_qig_edge(qig_tmp, j+1)
                # print(f"depth [{i}][{j}]", file=sys.stderr)
                if len(self.P[i][j]) > 0: # inherit from the upper grid
                    success = True # leftward propagation
                    if i + 1 <= j: # i + 1 < numDepths
                        self.P[i+1][j] = self.P[i][j] # downward propagation
                elif success: # inherit from the right grid
                    self.P[i][j] = self.P[i][j+1]
                else:
                    # print(f"is_changed: {is_changed}", file=sys.stderr)
                    if is_changed:
                        self.P[i][j] = self.get_qig_partitions(qig_tmp)
                        cnt += 1
                        if len(self.P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                self.P[i+1][j] = self.P[i][j]
        end_time = time.time()
        print(f"[build_partition_table] Partition calculation times: {cnt}.")
        print(f"[build_partition_table] Time: {end_time - start_time} seconds")
        return
    
    def build_qubit_interaction_graph_by_level(self, level_range):
        G = nx.Graph()
        for qubit in range(self.circuit.num_qubits):
            G.add_node(qubit)
        for lev in range(level_range[0], level_range[1]+1):
            for node in self.dag_multiq[lev]:
                qubits = [qubit._index for qubit in node.qargs]
                if qubits[0] == None:
                    qubits = [self.circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                if G.has_edge(qubits[0], qubits[1]):
                    G[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    G.add_edge(qubits[0], qubits[1], weight=1)
        return G

    def remove_qig_edge(self, qig, lev):
        """
        从qig中移除self.dag_multiq第lev列的量子门
        """
        is_changed = False # whether an edge is removed from qig
        for node in self.dag_multiq[lev]:
            qubits = [qubit._index for qubit in node.qargs]
            if qubits[0] == None:
                qubits = [self.circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]

            if qig.has_edge(qubits[0], qubits[1]):
                qig[qubits[0]][qubits[1]]['weight'] -= 1
                if qig[qubits[0]][qubits[1]]['weight'] == 0:
                    qig.remove_edge(qubits[0], qubits[1])
                    is_changed = True
        return is_changed
    
    def get_qig_partitions(self, qig):
        components = [list(comp) for comp in nx.connected_components(qig)]
        legal_partitions = self.partitioner.partition(components)
        return legal_partitions
    
    # 
    # S, T table
    # 
    def build_slicing_table(self):
        start_time = time.time()
        num_depths = len(self.P)
        self.T = [[0]  * num_depths for _ in range(num_depths)]
        self.S = [[-1] * num_depths for _ in range(num_depths)]

        for i in range(num_depths):
            if len(self.P[i][i]) == 0:
                print(f"[ERROR] P[{i}][{i}] is empty.")
                exit(1)
        # print("[build_t_table] ", end="")
        for depth in range(2, num_depths + 1): # depth: 2, 3, ..., num_depths
            # print(depth, end="")
            for i in range(0, num_depths - depth + 1): # 左边界
                j = i + depth - 1 # 右边界
                if len(self.P[i][j]) == 0:
                    self.T[i][j] = 10 ** 100
                    # 利用四边形优化缩小枚举范围
                    lower_k = self.S[i][j-1] if self.S[i][j-1] != -1 else i
                    upper_k = self.S[i+1][j] if self.S[i+1][j] != -1 else j-1
                    for k in range(lower_k, upper_k + 1):
                    # for k in range(i, j):
                        comms = self.T[i][k] + self.T[k+1][j] + 1
                        if comms < self.T[i][j]:
                            self.T[i][j] = comms
                            self.S[i][j] = k
                    # check if S[i][j-1] <= S[i][j] <= S[i+1][j]
                    # print(i, self.S[i][j-1], self.S[i][j], self.S[i+1][j], j)
                    # if self.S[i][j-1] != -1:
                    #     assert(self.S[i][j-1] <= self.S[i][j])
                    # if self.S[i+1][j] != -1:
                    #     assert(self.S[i][j] <= self.S[i+1][j])
        # print()
        end_time = time.time()
        print(f"[build_slicing_table] Time: {end_time - start_time} seconds")
        return
    
    def get_sliced_subc(self, i, j):
        if self.S[i][j] == -1:
            self.subc_ranges.append((i, j))
            return
        self.get_sliced_subc(i, self.S[i][j])
        self.get_sliced_subc(self.S[i][j] + 1, j)
        return

    def construct_teledata_only_mapping_record_list(self, partition_plan):
        """
        基于每个子线路合法的划分
        找到最少swap次数的路径
        """
        mapping_record_list = MappingRecordList()

        # dp[i][A]：表示第 i 个子线路选择分配方案 A 时的最小累计SWAP次数。
        # dp = {}
        self.num_comms = 0
        self.swap_only_path = [] # 记录最优路径 TODO: remove
        self.swap_prefix_sums = [0 for _ in range(len(partition_plan))] # 记录最优路径上的交换次数

        for i in range(len(partition_plan)):
            # assert(len(partition_plan[i]) == 1)
            self.swap_only_path.append(partition_plan[i])

            (left, right) = self.subc_ranges[i]

            record = MappingRecord(
                layer_start = left, # self.map_to_init_layer[left],
                layer_end = right, # self.map_to_init_layer[right],
                partition = partition_plan[i],
                mapping_type = "teledata",
                costs = {
                    "num_comms": 0,
                    "remote_hops": 0,
                    "remote_swaps": 0,
                    "fidelity_loss": 0,
                    "fidelity": 1
                } # 没有telegate操作，所以通信成本初始化为0
            )
            mapping_record_list.add_record(record)

            if i > 0:
                CompilerUtils.evaluate_partition_switch(mapping_record_list.records[-2], 
                                               mapping_record_list.records[-1],
                                               self.network)
                comms = mapping_record_list.records[-1].costs["remote_swaps"]
                self.num_comms += comms
                self.swap_prefix_sums[i] = self.swap_prefix_sums[i-1] + comms

        return mapping_record_list

    # 
    # Group shallow subcircuits
    # 
    def group_shallow_subcircuits(self, mapping_record_list):
        """
        Group adjacent subcircuits with shallow depth into a single subcircuit to reduce communication times.
        """
        start_time = time.time()
        left = right = 0
        left_subc_idx = right_subc_idx = -1

        records = mapping_record_list.records

        grouped_records = MappingRecordList()

        for i in range(len(records)):
            record = records[i]
            depth = record.layer_end - record.layer_start + 1
            if depth < self.min_depths:
                right = record.layer_end
                right_subc_idx = i + 1
            else:
                # 当前的子线路层数比较多，不需要做gate teleportation
                # 1. 先把之前收集的子线路（如果有）用gate teleportation试一下
                if left_subc_idx < right_subc_idx:
                    self.try_replace_with_gate_tele(mapping_record_list, 
                                                    left_subc_idx, 
                                                    right_subc_idx, 
                                                    left, 
                                                    right,
                                                    grouped_records)

                # 2. 当前子线路还是采用swap，记录
                # 已经在try_replace_with_gate_tele更新过当前record的costs
                record.layer_start = self.map_to_init_layer[record.layer_start]
                record.layer_end = self.map_to_init_layer[record.layer_end]
                grouped_records.add_record(record)

                # 3. 再把left和right更新成下一个子线路的左起点
                left_subc_idx = right_subc_idx = i
                if i != len(records) - 1: # subc[i]不是最后一个子线路
                    left = right = records[i+1].layer_start
                # 如果subc[i]是最后一个子线路，处理完毕

        if left_subc_idx < right_subc_idx: # 处理最后一个
            self.try_replace_with_gate_tele(mapping_record_list, left_subc_idx, right_subc_idx, left, right, grouped_records)
        end_time = time.time()
        print(f"[group_shallow_subcircuits] Time: {end_time - start_time} seconds")
        return grouped_records

    def try_replace_with_gate_tele(self, 
                                   mapping_record_list,
                                   left_subc_idx, 
                                   right_subc_idx, 
                                   left, 
                                   right, 
                                   grouped_records):
        """
        Try to replace dag_multiq[left, right] with gate teleportation
        """
        # print(f"[debug][try replace] {left_subc_idx} {right_subc_idx} {left} {right}")
        # 1. 尝试替换[left, right]这部分子线路
        # 1.1. 获取替换后的分区以及cat-ent costs
        # partition, cat_ent_costs = self.hyper_partition(left, right)
        telegate_record = self.hyper_partition(left, right)
        cat_ent_costs = telegate_record.costs["remote_hops"]
        # assert len(partition) == len(self.network.backend_sizes), f"[ERROR] partition: {partition} != {len(self.network.backend_sizes)}"
        
        # 1.2. 计算新分区和左分区的swap costs
        swap_costs, left_swaps, right_swaps = 0, 0, 0
        right_record = None
        # 获取左子线路的partition
        assert(left_subc_idx < len(mapping_record_list) - 1)
        if left_subc_idx != -1:
            left_record = mapping_record_list.records[left_subc_idx]
            # left_partition = left_record.partition
            # print(left_partition)
            # left_swap = calculate_nonlocal_communication(self.circ.num_qubits, left_partition, partition)
            # print(left_partition, partition[0])
            left_swaps = CompilerUtils.evaluate_partition_switch(left_record, telegate_record, self.network)["remote_swaps"]
            # print(f"left_swap: {left_swap}")
            swap_costs += left_swaps
        
        # 1.3. 计算新分区和右分区的swap costs
        assert(right_subc_idx > 0)
        if right_subc_idx != len(mapping_record_list.records):
            # copy一个新的right_record
            right_record = copy.deepcopy(mapping_record_list.records[right_subc_idx])
            # right_partition = mapping_record_list.records[right_subc_idx].partition
            # print(right_partition)
            # right_swap = calculate_nonlocal_communication(self.circ.num_qubits, partition, right_partition)
            right_swaps = CompilerUtils.evaluate_partition_switch(telegate_record, right_record, self.network)["remote_swaps"]
            # print(f"right_swap: {right_swap}")
            swap_costs += right_swaps
        new_costs = cat_ent_costs + swap_costs

        # 2. 和全部基于swap的通信代价进行比较
        # 2.1. 获取[left, right]这部分子线路的swap costs
        old_costs = 0
        if right_subc_idx >= len(mapping_record_list.records):
            old_costs = self.swap_prefix_sums[right_subc_idx - 1]
        else:
            old_costs = self.swap_prefix_sums[right_subc_idx]
        if left_subc_idx >= 0:
            old_costs -= self.swap_prefix_sums[left_subc_idx]
        # print(f"[try_replace_with_gate_tele] old_costs: {old_costs}, new_costs: {new_costs}")

        # 2. 如果新的ecosts更低，那么这部分线路用gate teleportation
        if new_costs < old_costs:
            self.num_comms += new_costs - old_costs
            # self.num_gates += cat_ent_costs
            # self.add_final_entry((left, right), "gate", partition)
            grouped_records.add_record(telegate_record)

            # 更新下一个子线路的record
            if right_subc_idx < len(mapping_record_list.records):
                mapping_record_list.records[right_subc_idx] = right_record

            return True

        # 3. 如果旧的ecosts更低，那么不变
        for j in range(left_subc_idx + 1, right_subc_idx):
            # self.add_final_entry(self.subc_ranges[j], "swap", self.swap_only_path[j])
            # 更新层号
            record = mapping_record_list.records[j]
            record.layer_start = self.map_to_init_layer[record.layer_start]
            record.layer_end = self.map_to_init_layer[record.layer_end]
            grouped_records.add_record(record)
        return False

    def hyper_partition(self, left, right):
        """
        Call Pytket-DQC's partitioner
        """
        # 
        # 1. 获取原线路中的[map[left], map[right]]这部分子线路
        # 
        # print(f"[hyper_partition] {left}-{right}")
        ori_left = self.map_to_init_layer[left]
        ori_right = self.map_to_init_layer[right]
        # layers = list(self.dag.layers())
        # sub_dag = self.dag.copy_empty_like()
        # for lev in range(ori_left, ori_right + 1):
        #     for node in layers[lev]["graph"].op_nodes():
        #         sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        sub_qc = self.get_ori_subc(ori_left, ori_right)
        # 
        # 2. 转译成Pytket-DQC可以处理的线路
        # 
        sub_qc = transpile(sub_qc, basis_gates=["cu1", "rz", "h"])
        sub_qc = qiskit_to_tk(sub_qc)
        DQCPass().apply(sub_qc)
        # print(f"[hyper_partition] sub_qc: {ori_left}-{ori_right}")
        # output_circuit_as_html(sub_qc, "hyper_partition")
        # 
        # 3. 调用Pytket-DQC的库，计算划分和ecosts
        #
        distribution = None
        if self.hyper_partitioner == "CE":
            distribution = CoverEmbeddingSteinerDetached().distribute(sub_qc, self.nisq_network, seed=26)
        else:
            distribution = PartitioningAnnealing().distribute(sub_qc, self.nisq_network, seed=26)
            # distributed_circ = distribution.to_pytket_circuit()
            # output_circuit_as_html(distributed_circ, "distributed_circ")
        # 
        # 4. 返回前后的划分，以及通信次数
        # 
        partition = [[] for _ in range(self.network.num_backends)] # 每个qpu上一个划分
        # print(f"[get_qubit_mapping]")
        # qubit_mapping = distribution.get_qubit_mapping()
        # print(qubit_mapping)
        # for e in qubit_mapping:
        #     print(e.index[0], qubit_mapping[e])
        for i in range(self.circuit.num_qubits): # q[i]->server[j]
            # print(i, distribution.placement.placement[i])
            partition[distribution.placement.placement[i]].append(i)
        # print(partition)
        # print(distribution.cost())
        record = MappingRecord(
            layer_start = ori_left,
            layer_end = ori_right,
            partition = partition,
            mapping_type = "telegate",
            costs = {
                "num_comms": distribution.cost(),
                "remote_hops": distribution.cost(),
                "remote_swaps": 0,
                "fidelity_loss": 0, # TODO: calculate fidelity loss based on the distribution result
                "fidelity": 1
            }
        )
        return record

    def get_ori_subc(self, ori_left, ori_right):
        """
        获取原线路中[ori_left, ori_right]的子线路
        """
        layers = list(self.dag.layers())
        sub_dag = self.dag.copy_empty_like()
        for lev in range(ori_left, ori_right + 1):
            for node in layers[lev]["graph"].op_nodes():
                sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        sub_qc = dag_to_circuit(sub_dag)
        return sub_qc
