from dataclasses import dataclass, field
from typing import Any, Optional
import time
import sys
import copy
import numpy as np
import networkx as nx
import datetime

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network
from .partitioner import Partitioner, PartitionerFactory
from .partition_assigner import PartitionAssigner, PartitionAssignerFactory
from .telegate_partitioner import TelegatePartitioner, TelegatePartitionerFactory
from .mapper import Mapper, MapperFactory

@dataclass
class CompilationContext:
    """
    存储编译过程中所有中间状态的数据容器
    """
    # --- 输入 (Input) ---
    circuit: QuantumCircuit
    network: Network
    config: dict[str, Any]
    
    # --- 预处理状态 (Preprocessing State) ---
    dag: Any = None
    circuit_layers: list[list[Any]] = field(default_factory=list) # Any <- DAGOpNode
    multiq_layers: list[list[Any]] = field(default_factory=list)
    map_to_circuit_layer: dict[int, int] = field(default_factory=dict)
    gate_density: float = 0.0
    twoq_gate_density: float = 0.0
    min_depth: int = 0
    
    # --- 分区表 (Partition Table - P) ---
    P_table: list[list[list[Any]]] = field(default_factory=list)

    # --- 切片表 (Slicing Tables - S, T) ---
    S_table: list[list[int]] = field(default_factory=list)
    T_table: list[list[int]] = field(default_factory=list)
    subc_ranges: list[tuple[int, int]] = field(default_factory=list)

    # --- 结果与计划 (Results & Plans) ---
    partition_candidates: list[Any] = field(default_factory=list)
    partition_plan: list[Any] = field(default_factory=list)
    hybrid_records: MappingRecordList = field(default_factory=MappingRecordList)
    final_records: MappingRecordList = field(default_factory=MappingRecordList)

    # --- 统计信息 (Statistics) ---
    swap_prefix_sums: list[int] = field(default_factory=list)
    epair_prefix_sums: list[int] = field(default_factory=list)
    telegate_cache_hits: int = 0
    telegate_cache_relaxed_hits: int = 0
    telegate_cache_misses: int = 0
    telegate_cache: dict[Any, MappingRecordList] = field(default_factory=dict)
    telegate_total_calls: int = 0
    telegate_strict_seen_keys: set[Any] = field(default_factory=set)
    telegate_range_seen_keys: set[Any] = field(default_factory=set)
    ptable_telegate_hints: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    
    # --- 运行时组件 (Runtime Components) ---
    partitioner: Optional[Partitioner] = None
    partition_assigner: Optional[PartitionAssigner] = None
    telegate_partitioner: Optional[TelegatePartitioner] = None
    mapper: Optional[Mapper] = None

    # --- 编译阶段完成标志 ---
    preprocessed: bool = False
    partition_planned: bool = False
    telegate_optimized: bool = False


class NAVI(Compiler):
    """
    Noise-Aware Distributed Quantum Compiler
    """
    compiler_id = "navi"

    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "NAVI"

    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit using the NADQC algorithm.
        """
        print(f"Compiling with [{self.name}]...")
        
        if config is None:
            config = {}

        # print(f"[DEBUG] [compile] circuit: \n{circuit}")

        # 1. 解析配置
        circuit_name = config.get("circuit_name", "circ")
        min_depth_cfg = config.get("min_depth", None)
        
        # 2. 初始化组件
        partitioner_type = config.get("partitioner", "recursive_dp")
        max_option = config.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(partitioner_type, network, max_options=max_option)

        partition_assigner_type = config.get("partition_assigner", "direct")
        # partition_assigner_type = config.get("partition_assigner", "global_max_match")
        partition_assigner = PartitionAssignerFactory.create_assigner(partition_assigner_type)

        telegate_partitioner_type = config.get("telegate_partitioner", "oee")
        telegate_partitioner = TelegatePartitionerFactory.create_telegate_partitioner(telegate_partitioner_type)
        
        # mapper_type = config.get("mapper", "dp")
        mapper_type = config.get("mapper", "boundeddp_neighbor")
        mapper = MapperFactory.create_mapper(mapper_type)
        
        # 3. 初始化上下文 (Context)
        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=config,
            
            partitioner=partitioner,
            partition_assigner=partition_assigner,
            telegate_partitioner=telegate_partitioner,
            mapper=mapper
        )

        start_time = time.time()

        # --- 编译流水线 ---
        
        # Step 1: 移除单量子比特门并计算量子门密度
        self._step_remove_single_qubit_gates(ctx)
        
        # Step 2: 设置最小深度
        ctx.min_depth = self._step_set_min_depth(ctx, min_depth_cfg)
        print(f"[DEBUG] min_depth: {ctx.min_depth}", file=sys.stderr)

        # Step 3: 构建分区表 (P Table)
        ctx.P_table = self._step_build_partition_table(ctx)

        # Step 4: 构建切片表 (S, T) 并获取子线路范围
        ctx.subc_ranges = self._step_build_slicing_table(ctx)

        # Step 5: 生成分区候选
        ctx.partition_candidates = [ctx.P_table[i][j] for (i, j) in ctx.subc_ranges]
        
        # Step 6: 分配分区计划
        assert ctx.partition_assigner is not None
        assign_result = ctx.partition_assigner.assign(ctx.partition_candidates)
        ctx.partition_plan = assign_result["partition_plan"]

        # Step 7: 构建初始 Mapping Record (Teledata-only)
        mapping_record_list = self._step_construct_teledata_only_records(ctx)

        # Step 8: 分组浅层子线路并尝试 Gate Teleportation 优化
        # grouped_records = self._step_group_and_optimize(ctx, mapping_record_list)

        # Step 9: 最终映射 (考虑噪声)
        assert ctx.mapper is not None
        final_result = ctx.mapper.map(
            # grouped_records, 
            mapping_record_list,
            ctx.circuit,
            ctx.circuit_layers, 
            ctx.network
        )
        # final_result = mapping_record_list # 先不进行telegate优化，直接映射看看效果

        end_time = time.time()
        exec_time = end_time - start_time

        final_result.summarize_total_costs()
        final_result.update_total_costs(execution_time = exec_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_result.save_records(f"./outputs/{circuit_name}/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")

        return final_result

    def preprocess(self, circuit: QuantumCircuit, network: Network, config: Optional[dict[str, Any]] = None) -> CompilationContext:
        """
        预处理阶段：移除单量子比特门，计算密度，构建DAG多量子门表示
        """
        if config is None:
            config = {}

        min_depth_cfg = config.get("min_depth", None)

        # 初始化上下文
        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=config
        )

        # Step 1: 移除单量子比特门并计算量子门密度
        self._step_remove_single_qubit_gates(ctx)
        
        # Step 2: 设置最小深度
        ctx.min_depth = self._step_set_min_depth(ctx, min_depth_cfg)

        ctx.preprocessed = True
        return ctx

    def generate_partition_candidates(self, ctx: CompilationContext, partitioner_type: str, max_option: int = 1) -> CompilationContext:
        """
        生成分区候选
        """
        if not ctx.preprocessed:
            raise RuntimeError("[ERROR] Context must be preprocessed before generating partition candidates.")
        
        # 配置partitioner
        ctx.partitioner = PartitionerFactory.create_partitioner(partitioner_type, ctx.network, max_options=max_option)

        # Step 3: 构建分区表 (P Table)
        ctx.P_table = self._step_build_partition_table(ctx)

        # Step 4: 构建切片表 (S, T) 并获取子线路范围
        ctx.subc_ranges = self._step_build_slicing_table(ctx)

        # Step 5: 生成分区候选
        ctx.partition_candidates = [ctx.P_table[i][j] for (i, j) in ctx.subc_ranges]

        return ctx

    def generate_partition_plan(self, ctx: CompilationContext, partition_assigner_type: str) -> CompilationContext:
        """
        生成分区计划
        """
        if not ctx.partition_candidates:
            raise RuntimeError("[ERROR] Partition candidates must be generated before generating partition plan.")
        
        # 配置partition assigner
        ctx.partition_assigner = PartitionAssignerFactory.create_assigner(partition_assigner_type)

        # Step 6: 分配分区计划
        assign_result = ctx.partition_assigner.assign(ctx.partition_candidates)
        ctx.partition_plan = assign_result["partition_plan"]
        ctx.partition_planned = True

        return ctx

    def optimize_with_telegate(self, ctx: CompilationContext, telegate_partitioner_type: str) -> CompilationContext:
        """
        分组浅层子线路并尝试 Gate Teleportation 优化
        """
        if not ctx.partition_planned:
            raise RuntimeError("[ERROR] Partition plan must be generated before optimizing with telegate.")
        
        # Step 7: 构建初始 Mapping Record (Teledata-only)
        mapping_record_list = self._step_construct_teledata_only_records(ctx)

        # Step 8: 分组浅层子线路并尝试 Gate Teleportation 优化
        ctx.telegate_partitioner = TelegatePartitionerFactory.create_telegate_partitioner(telegate_partitioner_type)

        grouped_records = self._step_group_and_optimize(ctx, mapping_record_list)
        grouped_records.summarize_total_costs()
        
        ctx.hybrid_records = grouped_records
        ctx.telegate_optimized = True

        return ctx

    def optimize_mapping(self, ctx: CompilationContext, mapper_type: str) -> CompilationContext:
        """最终映射 (考虑噪声)"""
        if not ctx.telegate_optimized:
            raise RuntimeError("[ERROR] Telegate optimization must be done before final mapping.")
        
        # Step 9: 最终映射 (考虑噪声)
        ctx.mapper = MapperFactory.create_mapper(mapper_type)

        # 获取评估过总成本的final_records
        ctx.final_records = ctx.mapper.map(
            ctx.hybrid_records, 
            ctx.circuit,
            ctx.circuit_layers, 
            ctx.network
        )

        ctx.final_records.summarize_total_costs()

        return ctx

    # =========================================================================
    # 步骤实现 (Step Implementations)
    # =========================================================================

    def _step_remove_single_qubit_gates(self, ctx: CompilationContext):
        """
        Remove single qubit gates and calculate densities.
        Updates: ctx.multiq_layers, ctx.map_*, ctx.gate_density, ctx.twoq_gate_density
        """
        start_time = time.time()
        ctx.dag = circuit_to_dag(ctx.circuit)
        
        circuit_layers = [] # 每一层存放的是原始量子线路上所有量子门，如果双量子比特门被拆分了，它也要被拆分
        multiq_layers = []
        map_to_circuit_layer = {}
        # map_to_multiq_layer = {}
        # pos_count, cu1_count, gate_count = 0, 0, 0
        pos_count, twoq_count, gate_count = 0, 0, 0

        n_backends = ctx.network.num_backends
        
        layers = list(ctx.dag.layers())
        for lev, layer in enumerate(layers):
            twoq_gates = []
            all_gates = []

            for node in layer["graph"].op_nodes():
                if not hasattr(node, 'qargs') or node.qargs is None:
                    raise ValueError("[ERROR] node.qargs does not exist or is none")
                pos_count += len(node.qargs)
                gate_count += 1
                
                if len(node.qargs) > 1:
                    if node.op.name == "barrier":
                        continue
                    if len(node.qargs) != 2:
                        raise ValueError(f"[ERROR] Found gate with more than 2 qubits: {node.op.name} on {node.qargs}")
                    twoq_count += 1
                    twoq_gates.append(node)
                else:
                    all_gates.append(node)

            if len(twoq_gates) > 0: # 逻辑：如果层满且不能整除后端数，则拆分
                if len(twoq_gates) == ctx.circuit.num_qubits // 2 and len(twoq_gates) % n_backends != 0:
                    split_point = len(twoq_gates) // 2
                    halves = [twoq_gates[:split_point], twoq_gates[split_point:]]

                    # 前一半，直接extend到all_gates当中，加入circuit_layers
                    all_gates.extend(halves[0]) # 单量子比特门 + 前一半双量子比特门
                    circuit_layers.append(all_gates)
                    multiq_layers.append(halves[0])
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1

                    # 后一半，作为新的一层，加入circuit_layers
                    circuit_layers.append(halves[1]) # 后一半双量子比特门作为新的一层
                    multiq_layers.append(halves[1])
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1
                else:
                    all_gates.extend(twoq_gates) # 单量子比特门 + 双量子比特门
                    circuit_layers.append(all_gates)
                    multiq_layers.append(twoq_gates)
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1
            elif len(all_gates) > 0: # 不是空层
                # 没有双量子比特门的层，直接放到circuit_layers里
                circuit_layers.append(all_gates)
        
        assert 0 in map_to_circuit_layer, f"[ERROR] map_to_circuit_layer must contain the first layer mapping. {map_to_circuit_layer}"
        # map_to_circuit_layer[0] = 0
        # map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1

        # 计算密度
        depth = ctx.circuit.depth()
        num_q = ctx.circuit.num_qubits
        ctx.gate_density = pos_count / (num_q * depth) if depth > 0 and num_q > 0 else 0.0
        ctx.twoq_gate_density = twoq_count / gate_count if gate_count > 0 else 0.0
        
        # 更新 Context
        ctx.circuit_layers = circuit_layers
        ctx.multiq_layers = multiq_layers
        ctx.map_to_circuit_layer = map_to_circuit_layer

        end_time = time.time()
        print(f"[INFO] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        # print(f"[INFO] gate_density: {ctx.gate_density}")
        # print(f"[INFO] twoq_gate_density: {ctx.twoq_gate_density}")
        return

    def _step_set_min_depth(self, ctx: CompilationContext, min_depth_cfg: Optional[int]) -> int:
        def sigmoid_decay(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 - 1 / (1 + np.exp(-k * (gate_density - c))))
        def sigmoid_increase(gate_density, depth, k=15, c=0.5):
            return 0.6 * depth * (1 / (1 + np.exp(-k * (gate_density - c))))

        if min_depth_cfg is None:
            print(f"[DEBUG] twoq_gate_density: {ctx.twoq_gate_density}", file=sys.stderr)
            print(f"[DEBUG] circuit depth: {ctx.circuit.depth()}", file=sys.stderr)
            print(f"[DEBUG] sigmoid_decay: {sigmoid_increase(ctx.twoq_gate_density, ctx.circuit.depth())}", file=sys.stderr)
            return int(sigmoid_increase(ctx.twoq_gate_density, ctx.circuit.depth()))

        return min_depth_cfg

    def _step_build_partition_table(self, ctx: CompilationContext) -> list[Any]:
        """
        Build the P table.
        """
        print(f"[build_partition_table]")
        start_time = time.time()
        multiq_layers = ctx.multiq_layers
        num_depths = len(multiq_layers)
        
        if num_depths == 0:
            return []

        P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        
        qig = self._build_qubit_interaction_graph_by_level(ctx, (0, num_depths-1))
        is_changed = True

        assert ctx.partitioner is not None

        for i in range(num_depths):
            # ===== P[i][numDepths-1] =====
            if i != 0:
                is_changed = self._remove_qig_edge(ctx, qig, i-1, multiq_layers)

            if len(P[i][num_depths-1]) > 0:
                if i == 0:
                    raise RuntimeError(f"[ERROR] P[{i}][{num_depths-1}] should be empty.")
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    P[i+1][num_depths-1] = P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    P[i][num_depths-1] = self._get_qig_partitions(qig, ctx.partitioner)
                    cnt += 1
                    if len(P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            P[i+1][num_depths-1] = P[i][num_depths-1]
        
            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self._remove_qig_edge(ctx, qig_tmp, j+1, multiq_layers)
                
                if len(P[i][j]) > 0: # inherit from the upper grid
                    success = True # leftward propagation
                    if i + 1 <= j:
                        P[i+1][j] = P[i][j] # downward propagation
                elif success: # inherit from the right grid
                    P[i][j] = P[i][j+1]
                else:
                    if is_changed:
                        P[i][j] = self._get_qig_partitions(qig_tmp, ctx.partitioner)
                        cnt += 1
                        if len(P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                P[i+1][j] = P[i][j]

        print(f"[build_partition_table] Partition calculation times: {cnt}.")
        print(f"[build_partition_table] Time: {time.time() - start_time} seconds")
        return P

    def _step_build_slicing_table(self, ctx: CompilationContext) -> list[Any]:
        """
        Build S and T tables and extract subc_ranges.
        """
        print(f"[build_slicing_table]")
        start_time = time.time()
        P = ctx.P_table
        num_depths = len(P)

        if num_depths == 0:
            return []

        T = [[0] * num_depths for _ in range(num_depths)]
        S = [[-1] * num_depths for _ in range(num_depths)]

        for i in range(num_depths):
            if len(P[i][i]) == 0:
                # 输出当前层上的量子操作
                for node in ctx.multiq_layers[i]:
                    print(f"[DEBUG] node: {node}, {node.op.name}, {node.qargs}", file=sys.stderr)
                raise RuntimeError(f"[ERROR] P[{i}][{i}] is empty. Cannot build slicing table.")

        for depth in range(2, num_depths + 1):
            for i in range(0, num_depths - depth + 1): # 左边界
                j = i + depth - 1 # 右边界
                if len(P[i][j]) == 0: # 需要切分
                    lower_k = S[i][j-1] if S[i][j-1] != -1 else i
                    upper_k = S[i+1][j] if S[i+1][j] != -1 else j-1
                    
                    best_k = -1
                    min_val = 10 ** 100
                    for k in range(lower_k, upper_k + 1):
                        comms = T[i][k] + T[k+1][j] + 1
                        if comms < min_val:
                            min_val = comms
                            best_k = k
                    T[i][j] = min_val
                    S[i][j] = best_k

        # Extract ranges
        ctx.S_table = S
        ctx.T_table = T

        subc_ranges = []
        self._get_sliced_subc_recursive(S, 0, num_depths - 1, subc_ranges)
        
        print(f"[build_slicing_table] Time: {time.time() - start_time} seconds")
        return subc_ranges

    def _step_construct_teledata_only_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        Construct initial mapping records based on partition plan.
        Updates: ctx.swap_prefix_sums, ctx.epair_prefix_sums
        """
        print(f"[construct_teledata_only_records]")
        mapping_record_list = MappingRecordList()
        partition_plan = ctx.partition_plan
        subc_ranges = ctx.subc_ranges
        network = ctx.network
        
        # ctx.num_comms = 0
        swap_prefix_sums = [0] * len(partition_plan)
        epair_prefix_sums = [0] * len(partition_plan)

        logical_phy_map = {}

        for i, partition in enumerate(partition_plan):
            if i == 0:
                # 对第一个子线路生成logical到物理的映射
                logical_phy_map = CompilerUtils.init_logical_phy_map(partition)
            else:
                # 沿用上一个record的logical_phy_map作为初始状态
                logical_phy_map = mapping_record_list.records[-1].logical_phy_map

            # 构建当前划分
            left, right = subc_ranges[i]
            ori_left, ori_right = self.get_original_layer_idx(ctx, (left, right))
            record = MappingRecord(
                layer_start = ori_left, # 注意，如果要group，这里要换成原始的层号
                layer_end = ori_right,
                partition = partition,
                mapping_type = "teledata",
                logical_phy_map = logical_phy_map
            )

            if i > 0: # 计算teledata并更新logical_phy_map
                prev_rec = mapping_record_list.records[-1]
                costs, _ = CompilerUtils.evaluate_teledata(prev_rec, record, network)
                comms = costs.remote_swaps
                
                # ctx.num_comms += comms
                swap_prefix_sums[i] = swap_prefix_sums[i-1] + comms
                epair_prefix_sums[i] = epair_prefix_sums[i-1] + costs.epairs

            mapping_record_list.add_record(record)

        ctx.swap_prefix_sums = swap_prefix_sums
        ctx.epair_prefix_sums = epair_prefix_sums

        return mapping_record_list

    def _step_group_and_optimize(self, ctx: CompilationContext, mapping_record_list: MappingRecordList) -> MappingRecordList:
        """
        Group shallow subcircuits and try to replace with gate teleportation.
        """
        start_time = time.time()
        records = mapping_record_list.records
        
        grouped_records = MappingRecordList()

        left = right = 0
        left_subc_idx = right_subc_idx = -1

        for i, record in enumerate(records):
            depth = record.layer_end - record.layer_start + 1
            if depth < ctx.min_depth:
                right = record.layer_end
                right_subc_idx = i + 1
            else:
                if left_subc_idx < right_subc_idx:
                    # TODO: 改层号
                    self._try_replace_with_telegate(
                        ctx, mapping_record_list, grouped_records, left_subc_idx, right_subc_idx, left, right
                    )

                # Add current record (with updated layer indices)
                # TODO: 注意层号
                # record.layer_start = ctx.map_to_circuit_layer[record.layer_start]
                # record.layer_end = ctx.map_to_circuit_layer[record.layer_end]
                record.layer_start, record.layer_end = self.get_original_layer_idx(
                    ctx, (record.layer_start, record.layer_end))
                grouped_records.add_record(record)

                left_subc_idx = right_subc_idx = i
                if i != len(records) - 1:
                    left = right = records[i+1].layer_start

        # Handle tail
        if left_subc_idx < right_subc_idx:
            self._try_replace_with_telegate(
                ctx, mapping_record_list, grouped_records, left_subc_idx, right_subc_idx, left, right
            )

        print(f"[group_shallow_subcircuits] Time: {time.time() - start_time} seconds")
        return grouped_records

    # =========================================================================
    # 底层逻辑辅助函数 (Helper Functions)
    # =========================================================================

    # P_table
    def _build_qubit_interaction_graph_by_level(self, ctx: CompilationContext, level_range: tuple[int, int]) -> nx.Graph:
        G = nx.Graph()
        for qubit in range(ctx.circuit.num_qubits):
            G.add_node(qubit)
        
        for lev in range(level_range[0], level_range[1]+1):
            for node in ctx.multiq_layers[lev]:
                qubits = []
                for q in node.qargs:
                    # if hasattr(q, '_index'):
                    #     qubits.append(q._index)
                    # else:
                    qubits.append(ctx.circuit.qubits.index(q))

                assert len(qubits) == 2

                q0, q1 = qubits
                if G.has_edge(q0, q1):
                    G[q0][q1]['weight'] += 1
                else:
                    G.add_edge(q0, q1, weight=1)
        return G

    def _remove_qig_edge(self, 
                         ctx: CompilationContext, 
                         qig: nx.Graph, 
                         lev: int, 
                         multiq_layers: list) -> bool:
        is_changed = False
        for node in multiq_layers[lev]:
            qubits = []
            for q in node.qargs:
                # if hasattr(q, '_index'):
                #     qubits.append(q._index)
                # else:
                qubits.append(ctx.circuit.qubits.index(q))
            # print(f"[DEBUG] qubits: {qubits}", file=sys.stderr)

            assert len(qubits) == 2

            q0, q1 = qubits
            if qig.has_edge(q0, q1):
                qig[q0][q1]['weight'] -= 1
                if qig[q0][q1]['weight'] == 0:
                    qig.remove_edge(q0, q1)
                    is_changed = True
        return is_changed

    def _get_qig_partitions(self, qig: nx.Graph, partitioner: Partitioner):
        components = [list(comp) for comp in nx.connected_components(qig)]
        return partitioner.partition(components)

    # S_table
    def _get_sliced_subc_recursive(self, S: list[list[int]], i: int, j: int, result_list: list[tuple[int, int]]):
        if S[i][j] == -1:
            result_list.append((i, j))
            return
        self._get_sliced_subc_recursive(S, i, S[i][j], result_list)
        self._get_sliced_subc_recursive(S, S[i][j] + 1, j, result_list)

    def _try_replace_with_telegate(self, 
                           ctx: CompilationContext, 
                           mapping_record_list: MappingRecordList, 
                           grouped_records: MappingRecordList,
                           left_subc_idx, right_subc_idx, left, right):
        """
        Logic to try replacing a range with gate teleportation.
        """
        # 1. 获取原始子电路范围
        ori_left, ori_right = self.get_original_layer_idx(ctx, (left, right))
        sub_qc = self._get_ori_subc(ctx, ori_left, ori_right)

        # 获取上一个partition的partition方案，作为telegate partition的初始方案
        prev_partition = []
        if left_subc_idx != -1:
            prev_partition = mapping_record_list.records[left_subc_idx].partition
        
        # 2. 调用telegate_partitioner进行划分
        assert ctx.telegate_partitioner is not None
        telegate_record = ctx.telegate_partitioner.partition(
            circuit = sub_qc,
            network = ctx.network,
            config = {
                "layer_start": ori_left,
                "layer_end": ori_right,
                "partition": prev_partition
            }
        )

        new_epairs = telegate_record.costs.epairs

        # 获取右子线路的record
        right_record = None
        if right_subc_idx < len(mapping_record_list.records):
            # 拷贝一份right_record，可能会直接加入grouped_records
            right_record = copy.deepcopy(mapping_record_list.records[right_subc_idx])
            costs, _ = CompilerUtils.evaluate_teledata(telegate_record, right_record, ctx.network)
            new_epairs += costs.epairs

        # Calculate old costs
        end_idx = right_subc_idx if right_subc_idx < len(ctx.swap_prefix_sums) else len(ctx.swap_prefix_sums) - 1
        # old_costs = ctx.swap_prefix_sums[end_idx]
        old_epairs = ctx.epair_prefix_sums[end_idx]
        if left_subc_idx >= 0:
            # old_costs -= ctx.swap_prefix_sums[left_subc_idx]
            old_epairs -= ctx.epair_prefix_sums[left_subc_idx]

        if new_epairs < old_epairs:
            grouped_records.add_record(telegate_record) # 层号已更新

            # 更新下一个子线路的record
            if right_record:
                mapping_record_list.records[right_subc_idx] = right_record

            return True

        # If not replacing, we still need to add the intermediate records to grouped_records later
        for idx in range(left_subc_idx + 1, right_subc_idx):
            # 更新层号
            record = mapping_record_list.records[idx]
            # record.layer_start = ctx.map_to_circuit_layer[record.layer_start]
            # record.layer_end = ctx.map_to_circuit_layer[record.layer_end]
            record.layer_start, record.layer_end = self.get_original_layer_idx(
                ctx, (record.layer_start, record.layer_end))
            grouped_records.add_record(record)
        return False

    def _get_ori_subc(self, ctx: CompilationContext, ori_left: int, ori_right: int) -> QuantumCircuit:
        """
        获取原线路中[ori_left, ori_right]的子线路
        """
        # 1. 复制原电路的寄存器结构（包含量子和经典比特）
        # 这样可以完美保留原电路的比特命名和结构
        qregs = ctx.circuit.qregs
        cregs = ctx.circuit.cregs
        subcircuit = QuantumCircuit(*qregs, *cregs)

        # 2. 建立映射：{原电路比特对象: 子电路对应索引的比特对象}
        # 量子比特映射
        q_map = {old_q: subcircuit.qubits[i] for i, old_q in enumerate(ctx.circuit.qubits)}
        # 经典比特映射（防止有测量门等操作经典比特的情况）
        c_map = {old_c: subcircuit.clbits[i] for i, old_c in enumerate(ctx.circuit.clbits)} if subcircuit.clbits else {}

        # 3. 遍历并追加门，使用映射后的比特
        for lev in range(ori_left, ori_right + 1):
            for node in ctx.circuit_layers[lev]:
                # 替换量子比特
                new_qargs = [q_map[q] for q in node.qargs]
                # 替换经典比特（如果有的话）
                old_cargs = getattr(node, 'cargs', [])
                new_cargs = [c_map[c] for c in old_cargs] if old_cargs else []
                
                subcircuit.append(node.op, new_qargs, new_cargs)

        return subcircuit

        # subcircuit = QuantumCircuit(ctx.circuit.num_qubits)
        # for lev in range(ori_left, ori_right + 1):
        #     for node in ctx.circuit_layers[lev]:
        #         subcircuit.append(node.op, node.qargs, getattr(node, 'cargs', []))
        # return subcircuit
        # # layers = list(ctx.dag.layers())
        # sub_dag = ctx.dag.copy_empty_like()
        # for lev in range(ori_left, ori_right + 1):
        #     # for node in layers[lev]["graph"].op_nodes():
        #     for node in ctx.circuit_layers[lev]:
        #         sub_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        # return dag_to_circuit(sub_dag) # 改成circuit_layers的形式

    def reconstruct_and_visualize_circuit(self, ctx: CompilationContext, verbose: bool = True) -> QuantumCircuit:
        """
        Reconstructs the multi-qubit gate circuit from the context's multiq_layers and visualizes it.
        
        Args:
            ctx (CompilationContext): The compilation context containing multiq_layers and original circuit info.
            verbose (bool): If True, prints the visualization and statistics to stdout.
            
        Returns:
            QuantumCircuit: The reconstructed circuit containing only multi-qubit gates.
        """
        multiq_layers = ctx.multiq_layers
        
        # 1. 检查数据有效性
        if not multiq_layers:
            if verbose:
                print("[INFO] No multi-qubit gates found in context. Cannot visualize circuit.")
            # 返回一个空的量子线路，保持返回值类型一致
            return QuantumCircuit(ctx.circuit.num_qubits)

        # 2. 调试信息 (可选)
        if verbose:
            print(f"[DEBUG] Reconstructing circuit from {len(multiq_layers)} layers...")
            # 如果 multiq_layers 很大，pprint 可能会刷屏，这里仅打印概要，必要时可开启详细打印
            # pprint(multiq_layers) 

        # 3. 创建新的量子线路
        n_qubits = ctx.circuit.num_qubits
        recon_circ = QuantumCircuit(n_qubits)
        
        total_gates = 0
        # 4. 按层添加双量子门
        for i, layer in enumerate(multiq_layers):
            for node in layer:
                # 安全性检查：确保 node 有必要的属性
                if hasattr(node, 'op') and hasattr(node, 'qargs'):
                    recon_circ.append(node.op, node.qargs, getattr(node, 'cargs', []))
                    total_gates += 1
                else:
                    print(f"[WARNING] Skipping invalid node in layer {i}: {node}")

            # 在每层后添加 barrier（最后一层不加）
            if i < len(multiq_layers) - 1:
                recon_circ.barrier()

        # 5. 打印可视化结果
        if verbose:
            print("\n" + "="*60)
            print("Reconstructed Multi-Qubit Gate Circuit Visualization")
            print("="*60)
            print(recon_circ) # Qiskit 会自动调用 circuit_drawer
            print("="*60)
            print(f"Statistics:")
            print(f"  - Total Layers: {len(multiq_layers)}")
            print(f"  - Total Gates : {total_gates}")
            print(f"  - Num Qubits  : {n_qubits}")
            print("="*60 + "\n")
        
        return recon_circ

    def get_original_layer_idx(self, ctx: CompilationContext, layer_range: tuple[int, int]) -> tuple[int, int]:
        """
        获取指定层在原始电路中的对应层号。

        Args:
            ctx (CompilationContext): 编译上下文。
            layer_range (tuple[int, int]): 指定的层范围。

        Returns:
            int: 原始电路中的层号。
        """
        left, right = layer_range
        ori_left, ori_right = ctx.map_to_circuit_layer[left], ctx.map_to_circuit_layer[right]

        if left == 0 and ctx.map_to_circuit_layer[0] != 0: # 检查起点是否对应原始电路的起点
            ori_left = 0
        
        if right == len(ctx.multiq_layers) - 1 and ctx.map_to_circuit_layer[right] != len(ctx.circuit_layers) - 1: # 检查终点是否对应原始电路的终点
            ori_right = len(ctx.circuit_layers) - 1

        return (ori_left, ori_right)