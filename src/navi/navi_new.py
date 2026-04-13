from dataclasses import dataclass, field
from typing import Any, Optional
import time
import sys
import copy
import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Gate  # 自定义门必须继承的基类

from ..compiler import Compiler, CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..utils import Network
from .partitioner import Partitioner, PartitionerFactory
from .partition_assigner import PartitionAssigner, PartitionAssignerFactory
from .telegate_partitioner import TelegatePartitioner, TelegatePartitionerFactory
from .mapper import Mapper, MapperFactory
from .navi_compiler import CompilationContext


@dataclass
class TGIncrementalState:
    """
    TG增量更新状态（骨架）：
    - partition: 当前区间使用的telegate分区
    - edge_weights: 区间内二比特交互计数（逻辑边->权重）
    - cat_groups: CAT候选门计数（分区无关键值）
    - costs: 当前区间telegate/CAT代价
    """
    partition: list[list[int]] = field(default_factory=list)
    edge_weights: dict[tuple[int, int], int] = field(default_factory=dict)
    cat_groups: dict[tuple[int, int, str], int] = field(default_factory=dict)
    costs: ExecCosts = field(default_factory=ExecCosts)


class NAVINew(Compiler):
    """
    Noise-Aware Distributed Quantum Compiler
    """
    compiler_id = "navinew"

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "NAVI_NEW"
    
    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit using the NADQC algorithm.
        """
        print(f"Compiling with [{self.name}]...")

        if config is None:
            config = {}

        # 1. 解析配置
        circuit_name = config.get("circuit_name", "circ")
        partitioner_type = config.get("partitioner", "recursive_dp")
        max_option = config.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(partitioner_type, network, max_options=max_option)
        mapper_type = config.get("mapper", "boundeddp_neighbor")
        mapper = MapperFactory.create_mapper(mapper_type)
        
        # 2. 构建编译上下文
        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=config,
            
            partitioner=partitioner,
            mapper=mapper,
        )

        start_time = time.time()

        # --- 编译流水线（确定性版本骨架） ---
        # 1) 用multiq层做切分索引；
        # 2) 用原始层做telegate/CAT计费；
        # 3) 在统一DP里同时考虑teledata与telegate。
        self._step_remove_single_qubit_gates(ctx)

        # Step 2: 构建P-table（teledata无通信分区候选）
        # Step 3: 在同一遍历中同步构建TG-table（基于原始层结构的telegate/CAT区间代价）
        ctx.P_table, tg_table = self._step_build_partition_and_telegate_tables(ctx)

        # Step 4: 统一S/T-table（同时考虑teledata与telegate）
        mode_table, split_table = self._step_build_unified_s_t_table(ctx, tg_table)

        # Step 5: 根据模式与切分回溯，直接构建最终records（不再后置hybrid）
        planned_result = self._step_construct_records_from_mode_plan(ctx, mode_table, split_table, tg_table)

        # Step 6: 一致性校验（DP目标值 vs records汇总epairs）
        self._step_validate_dp_record_consistency(ctx, planned_result)

        # Step 7: 与其他编译器口径一致，进入mapper获得最终通信/保真度统计
        assert ctx.mapper is not None
        final_result = ctx.mapper.map(
            planned_result,
            ctx.circuit,
            ctx.circuit_layers,
            ctx.network,
            config=ctx.config,
        )

        exec_time = time.time() - start_time
        final_result.summarize_total_costs()
        final_result.update_total_costs(execution_time=exec_time)

        print(f"[INFO] {self.name} skeleton pipeline time: {time.time() - start_time} sec", file=sys.stderr)
        return final_result


    # =========================================================================
    # 步骤实现 (Step Implementations)
    # =========================================================================

    def _step_remove_single_qubit_gates(self, ctx: CompilationContext):
        """
        Remove single qubit gates and calculate densities.
        Updates: ctx.dag, ctx.circuit_layers, ctx.multiq_layers, ctx.map_to_circuit_layer
        """
        start_time = time.time()
        ctx.dag = circuit_to_dag(ctx.circuit)
        
        circuit_layers = [] # 每一层存放的是原始量子线路上所有量子门，如果双量子比特门被拆分了，它也要被拆分
        multiq_layers = []
        map_to_circuit_layer = {}
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


        # 确保map_to_circuit_layer[0] = 0, map_to_circuit_layer[-1] = len(circuit_layers) - 1
        assert 0 in map_to_circuit_layer, "[ERROR] map_to_circuit_layer must contain the first layer mapping."
        map_to_circuit_layer[0] = 0
        map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1

        # 计算密度
        # depth = ctx.circuit.depth()
        # num_q = ctx.circuit.num_qubits
        # ctx.gate_density = pos_count / (num_q * depth) if depth > 0 and num_q > 0 else 0.0
        # ctx.twoq_gate_density = twoq_count / gate_count if gate_count > 0 else 0.0
        
        # 更新 Context
        ctx.circuit_layers = circuit_layers
        ctx.multiq_layers = multiq_layers
        ctx.map_to_circuit_layer = map_to_circuit_layer

        end_time = time.time()
        print(f"[DEBUG] remove_single_qubit_gates: {end_time - start_time} seconds", file=sys.stderr)
        return

    def _step_build_partition_and_telegate_tables(
        self,
        ctx: CompilationContext,
    ) -> tuple[list[list[list[Any]]], list[list[Optional[ExecCosts]]]]:
        """
        在同一遍历中构建：
        - P-table: teledata无通信分区候选
        - TG-table: telegate/CAT区间代价（骨架占位，后续补真实估算）

        设计目标：
        - 共享区间遍历顺序，避免第二次完整区间扫描；
        - 索引仍然是multiq层区间 [i, j]；
        - 代价评估通过 map_to_circuit_layer 映射回原始层区间。
        """
        print(f"[build_partition_and_telegate_tables]")
        start_time = time.time()

        multiq_layers = ctx.multiq_layers
        num_depths = len(multiq_layers)
        
        if num_depths == 0:
            return [], []
        
        # partition_table记录分区结果以及分区的telegate开销
        # list[list[int]], ExecCosts
        # 相当于每个都是一个MappingRecord
        P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        tg_table: list[list[Optional[ExecCosts]]] = [[None for _ in range(num_depths)] for _ in range(num_depths)]
        partition_table = [[MappingRecord() for _ in range(num_depths)] for _ in range(num_depths)]
        clean_success = [[False for _ in range(num_depths)] for _ in range(num_depths)]

        # TG初始化会通过_select_representative_partition读取ctx.P_table。
        # 在同一构建流程中先将本地P绑定到ctx，避免越界访问。
        ctx.P_table = P

        cnt = 0

        qig = self._build_qubit_interaction_graph_by_level(ctx, (0, num_depths-1))
        is_changed = True
        assert ctx.partitioner is not None
        tg_state_table: list[list[Optional[TGIncrementalState]]] = [[None for _ in range(num_depths)] for _ in range(num_depths)]

        # 用recursive dp和OEE都算一次，确认是不是每次有cut=0的都能找到                                                         
        for i in range(num_depths):
            # ===== P[i][numDepths-1] =====
            self._update_tg_cell_incremental(ctx, tg_table, tg_state_table, i, num_depths - 1)
            if i != 0:
                is_changed = self._remove_qig_edge(ctx, qig, i-1, multiq_layers)

            # if len(P[i][num_depths-1]) > 0:
            if clean_success[i][num_depths-1]:
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
                removed_level = j + 1
                is_changed = self._remove_qig_edge(ctx, qig_tmp, removed_level, multiq_layers)
                base_state = tg_state_table[i][j + 1]

                tg_changed = self._tg_cell_needs_update(
                    ctx=ctx,
                    left=i,
                    right=j,
                    removed_level=removed_level,
                    p_changed=is_changed,
                    base_partition=base_state.partition if base_state is not None else None,
                )
                if tg_changed:
                    self._update_tg_cell_incremental(ctx, tg_table, tg_state_table, i, j)
                else:
                    self._inherit_tg_cell_from_right(tg_table, tg_state_table, i, j)
                
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

        print(f"[build_partition_and_telegate_tables] Partition calculation times: {cnt}.")
        print(f"[build_partition_and_telegate_tables] Time: {time.time() - start_time} seconds")

        tg_partition_table: list[list[list[list[int]]]] = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        for i in range(num_depths):
            for j in range(i, num_depths):
                state = tg_state_table[i][j]
                if state is not None and state.partition:
                    tg_partition_table[i][j] = self._normalize_partition(state.partition)
        setattr(ctx, "tg_partition_table", tg_partition_table)

        return P, tg_table

    def _tg_cell_needs_update(
        self,
        ctx: CompilationContext,
        left: int,
        right: int,
        removed_level: int,
        p_changed: bool,
        base_partition: Optional[list[list[int]]] = None,
    ) -> bool:
        """
        TG单元是否需要更新（变化触发式骨架）。

        规则保持确定性：
        - 若P-table对应QIG发生变化，则允许触发TG更新；
        - 若被移除层触及潜在CAT相关门，也触发TG更新；
        - 否则可继承右邻TG[i][right+1]，避免重复计算。

        TODO: 用更精确的增量统计替换当前保守触发条件。
        """
        if p_changed:
            return True

        if removed_level < 0 or removed_level >= len(ctx.multiq_layers):
            return False

        # 若没有热启动分区，保守更新一次。
        if base_partition is None:
            return True

        # 仅当该层在当前分区下确实影响telegate/cat统计时才更新。
        return self._level_impacts_tg_under_partition(ctx, removed_level, base_partition)

    def _level_impacts_tg_under_partition(
        self,
        ctx: CompilationContext,
        level: int,
        partition: list[list[int]],
    ) -> bool:
        """
        判断某层在给定分区下是否影响TG统计：
        - 存在跨分区二比特门，影响telegate边开销；
        - 存在可作为CAT候选的跨分区门，影响cat复用分组。
        """
        if level < 0 or level >= len(ctx.circuit_layers):
            return False

        qubit_to_part = self._build_qubit_to_partition_map(partition)
        cat_gate_set = {"cx", "cz", "rzz"}

        for node in ctx.circuit_layers[level]:
            if node.op.name == "barrier" or len(node.qargs) != 2:
                continue
            q0 = ctx.circuit.qubits.index(node.qargs[0])
            q1 = ctx.circuit.qubits.index(node.qargs[1])
            p0 = qubit_to_part.get(q0)
            p1 = qubit_to_part.get(q1)
            if p0 is None or p1 is None:
                continue
            if p0 != p1:
                return True
            if node.op.name in cat_gate_set and p0 != p1:
                return True

        return False

    def _inherit_tg_cell_from_right(
        self,
        tg_table: list[list[Optional[ExecCosts]]],
        tg_state_table: list[list[Optional[TGIncrementalState]]],
        i: int,
        j: int,
    ) -> None:
        """
        TG继承路径：TG[i][j] <- TG[i][j+1]。
        """
        right_neighbor = tg_table[i][j + 1]
        tg_table[i][j] = copy.deepcopy(right_neighbor) if right_neighbor is not None else ExecCosts()
        right_state = tg_state_table[i][j + 1]
        tg_state_table[i][j] = copy.deepcopy(right_state) if right_state is not None else TGIncrementalState()

    def _update_tg_cell_incremental(
        self,
        ctx: CompilationContext,
        tg_table: list[list[Optional[ExecCosts]]],
        tg_state_table: list[list[Optional[TGIncrementalState]]],
        i: int,
        j: int,
    ) -> None:
        """
        TG更新路径（增量实现占位）。

        这里先保留框架：
        - 索引仍为multiq区间[i, j]；
        - 评估在原始层区间上进行；
        - 后续替换为真实的增量CAT/telegate计费，而非全量重算。
        """
        # 热启动来源：优先复用右邻区间状态，降低更新成本。
        base_state: Optional[TGIncrementalState] = None
        if j + 1 < len(ctx.multiq_layers):
            base_state = tg_state_table[i][j + 1]

        # 1) 先初始化或复制状态
        state = self._initialize_tg_state_for_range(ctx, i, j, base_state)

        # 2) 应用区间缩减的增量变化（删除第j+1层的贡献）
        # 这里保留接口，后续填入边计数/CAT分组的增量加减逻辑。
        removed_level = j + 1
        if removed_level < len(ctx.multiq_layers):
            self._apply_removed_level_to_tg_state(ctx, state, removed_level)

        # 3) 基于增量状态做确定性局部修复，得到telegate/cat partition
        state = self._local_repair_telegate_partition(ctx, state, i, j)

        # 4) 从状态汇总代价并写回表项
        state.costs = self._summarize_tg_costs_from_state(ctx, state, i, j)
        tg_state_table[i][j] = state
        tg_table[i][j] = copy.deepcopy(state.costs)

    def _initialize_tg_state_for_range(
        self,
        ctx: CompilationContext,
        i: int,
        j: int,
        base_state: Optional[TGIncrementalState],
    ) -> TGIncrementalState:
        """
        初始化区间[i,j]的TG状态。
        - 若base_state存在，则复制热启动；
        - 否则构造确定性初始状态（例如来自P-table代表分区）。
        """
        if base_state is not None:
            return copy.deepcopy(base_state)

        state = TGIncrementalState()
        state.partition = self._select_representative_partition(ctx, i, j)

        # 从区间子线路初始化边计数/CAT组统计。
        ori_left, ori_right = self._get_original_layer_idx(ctx, (i, j))
        for lev in range(ori_left, ori_right + 1):
            self._accumulate_layer_to_tg_state(ctx, state, lev, sign=1)
        return state

    def _apply_removed_level_to_tg_state(
        self,
        ctx: CompilationContext,
        state: TGIncrementalState,
        removed_level: int,
    ) -> None:
        """
        增量删除一层门对TG状态的影响。
        目标：只更新受影响的edge_weights/cat_groups，而不是重建整个子线路。
        """
        self._accumulate_layer_to_tg_state(ctx, state, removed_level, sign=-1)

    def _local_repair_telegate_partition(
        self,
        ctx: CompilationContext,
        state: TGIncrementalState,
        i: int,
        j: int,
    ) -> TGIncrementalState:
        """
        确定性局部修复（骨架）：
        - 基于当前状态尝试局部迁移qubit；
        - 采用字典序tie-break，不引入随机与reward。
        """
        if not state.partition:
            return state

        # 分区规范化，保证后续tie-break稳定。
        state.partition = self._normalize_partition(state.partition)

        # 固定轮数上限，保证确定性和可控开销。
        max_iters = max(1, ctx.circuit.num_qubits)
        for _ in range(max_iters):
            qubit_to_part = self._build_qubit_to_partition_map(state.partition)
            best_delta: Optional[int] = None
            candidate_moves: list[tuple[int, int]] = []  # (qubit, target_part)
            backend_caps = ctx.network.get_backend_qubit_counts(include_comm_slot=True)

            for q in range(ctx.circuit.num_qubits):
                src_part = qubit_to_part.get(q)
                if src_part is None:
                    continue

                # 不把源分区搬空，避免产生空分区。
                if len(state.partition[src_part]) <= 1:
                    continue

                for tgt_part in range(len(state.partition)):
                    if tgt_part == src_part:
                        continue
                    if tgt_part < len(backend_caps) and len(state.partition[tgt_part]) >= backend_caps[tgt_part]:
                        continue
                    delta = self._delta_epairs_for_move(
                        q=q,
                        src_part=src_part,
                        tgt_part=tgt_part,
                        state=state,
                        qubit_to_part=qubit_to_part,
                        network=ctx.network,
                    )
                    if delta >= 0:
                        continue
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        candidate_moves = [(q, tgt_part)]
                    elif delta == best_delta:
                        candidate_moves.append((q, tgt_part))

            if best_delta is None or not candidate_moves:
                break

            # 二级目标：在epairs同等改进时，优先选择CAT复用分数更高的迁移。
            # 三级目标：按(qubit, target_part)字典序稳定选取。
            best_move: Optional[tuple[int, int]] = None
            best_cat_score: Optional[int] = None
            for q, tgt_part in sorted(candidate_moves):
                trial_partition = copy.deepcopy(state.partition)
                trial_src = self._build_qubit_to_partition_map(trial_partition)[q]
                trial_partition[trial_src].remove(q)
                trial_partition[tgt_part].append(q)
                trial_partition = self._normalize_partition(trial_partition)
                cat_score = self._estimate_cat_reuse_score_from_state(state, trial_partition)
                if best_move is None or best_cat_score is None or cat_score > best_cat_score:
                    best_move = (q, tgt_part)
                    best_cat_score = cat_score

            assert best_move is not None
            q, tgt_part = best_move
            src_part = self._build_qubit_to_partition_map(state.partition)[q]
            state.partition[src_part].remove(q)
            state.partition[tgt_part].append(q)
            state.partition = self._normalize_partition(state.partition)

        state.partition = self._enforce_partition_capacity(
            state.partition,
            ctx.network,
            ctx.circuit.num_qubits,
        )
        return state

    def _summarize_tg_costs_from_state(
        self,
        ctx: CompilationContext,
        state: TGIncrementalState,
        i: int,
        j: int,
    ) -> ExecCosts:
        """
        从增量状态汇总TG代价（骨架）。
        """
        # 1) 基于增量状态直接汇总telegate/CAT代价；
        # 2) 可选对拍：与evaluate_telegate_with_my_cat比较；
        # 3) 结果走缓存复用。
        if not state.partition:
            return ExecCosts()

        cache: dict[Any, ExecCosts] = getattr(ctx, "tg_cost_cache", {})
        if not isinstance(cache, dict):
            cache = {}
        setattr(ctx, "tg_cost_cache", cache)

        key = (i, j, self._partition_signature(state.partition))
        if key in cache:
            return copy.deepcopy(cache[key])

        use_incremental = bool(ctx.config.get("enable_incremental_tg_cost", False))
        if use_incremental:
            costs = self._calculate_tg_costs_incremental(ctx, state)

            # 可选：对拍严格实现，便于验证增量模型准确性。
            if bool(ctx.config.get("debug_tg_incremental_check", False)):
                ori_left, ori_right = self._get_original_layer_idx(ctx, (i, j))
                subcircuit = self._build_subcircuit_from_original_layers(ctx, ori_left, ori_right)
                ref = CompilerUtils.evaluate_telegate_with_my_cat(state.partition, subcircuit, ctx.network)
                if ref.epairs != costs.epairs:
                    print(
                        f"[WARNING] tg_incremental_check mismatch at [{i},{j}]: inc_epairs={costs.epairs}, ref_epairs={ref.epairs}",
                        file=sys.stderr,
                    )
        else:
            # 默认使用精确评估，优先保证决策质量和可比性。
            ori_left, ori_right = self._get_original_layer_idx(ctx, (i, j))
            subcircuit = self._build_subcircuit_from_original_layers(ctx, ori_left, ori_right)
            costs = CompilerUtils.evaluate_telegate_with_my_cat(state.partition, subcircuit, ctx.network)

        cache[key] = copy.deepcopy(costs)
        return costs

    def _select_representative_partition(self, ctx: CompilationContext, i: int, j: int) -> list[list[int]]:
        """
        从P-table中确定性选择代表分区：
        - 优先用P[i][j]第一个候选；
        - 若P为空，则采用顺序分配作为回退。
        """
        candidates = ctx.P_table[i][j]
        if len(candidates) == 0:
            partition = CompilerUtils.allocate_qubits(ctx.circuit.num_qubits, ctx.network)
            return self._enforce_partition_capacity(partition, ctx.network, ctx.circuit.num_qubits)

        first = candidates[0]
        # 兼容两种可能：
        # 1) first是partition: list[list[int]]
        # 2) first是单个分组: list[int]，这时candidates本身是partition
        if len(first) > 0 and isinstance(first[0], int):
            partition = candidates
        else:
            partition = first
        return self._enforce_partition_capacity(partition, ctx.network, ctx.circuit.num_qubits)

    def _select_partition_for_mode(self, ctx: CompilationContext, i: int, j: int, mode: str) -> list[list[int]]:
        """
        根据模式确定性选择输出record使用的partition。
        - td: 使用P-table代表分区
        - tg: 优先使用TG增量状态输出分区，缺失时回退到代表分区
        """
        if mode == "tg":
            tg_partition_table = getattr(ctx, "tg_partition_table", None)
            if isinstance(tg_partition_table, list):
                cell = tg_partition_table[i][j]
                if isinstance(cell, list) and len(cell) > 0:
                    return self._enforce_partition_capacity(cell, ctx.network, ctx.circuit.num_qubits)
        return self._enforce_partition_capacity(
            self._select_representative_partition(ctx, i, j),
            ctx.network,
            ctx.circuit.num_qubits,
        )

    def _normalize_partition(self, partition: list[list[int]]) -> list[list[int]]:
        normalized = [sorted(group) for group in partition]
        normalized.sort(key=lambda g: (len(g), g))
        return normalized

    def _enforce_partition_capacity(
        self,
        partition: list[list[int]],
        network: Network,
        num_qubits: int,
    ) -> list[list[int]]:
        """
        将partition校正到后端容量约束内：
        - 分区数固定为network.num_backends；
        - 每个分区大小不超过对应backend容量；
        - 覆盖全部逻辑比特且不重复。
        """
        k = network.num_backends
        caps = list(network.get_backend_qubit_counts(include_comm_slot=True))

        # 收集唯一量子比特（保持出现顺序）
        ordered: list[int] = []
        seen = set()
        for group in partition:
            for q in group:
                if q not in seen:
                    ordered.append(q)
                    seen.add(q)
        for q in range(num_qubits):
            if q not in seen:
                ordered.append(q)
                seen.add(q)

        # 按容量顺序确定性分配
        fixed: list[list[int]] = [[] for _ in range(k)]
        idx = 0
        for pidx in range(k):
            cap = caps[pidx] if pidx < len(caps) else 0
            take = min(cap, len(ordered) - idx)
            if take > 0:
                fixed[pidx] = ordered[idx: idx + take]
            idx += max(0, take)

        # 正常情况下容量总和应覆盖num_qubits；若不足则回退到原始分配接口。
        if idx < len(ordered):
            fallback = CompilerUtils.allocate_qubits(num_qubits, network)
            return self._normalize_partition(fallback)

        return self._normalize_partition(fixed)

    def _partition_signature(self, partition: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        normalized = self._normalize_partition(partition)
        return tuple(tuple(group) for group in normalized)

    def _estimate_cat_reuse_score_from_state(
        self,
        state: TGIncrementalState,
        partition: list[list[int]],
    ) -> int:
        """
        估计CAT复用分数（确定性）：
        - 使用分区无关的cat_groups计数；
        - 在给定partition下映射为(anchor, src_part, dst_part, gate)组。
        """
        qubit_to_part = self._build_qubit_to_partition_map(partition)
        grouped_targets: dict[tuple[int, int, int, str], set[int]] = {}
        grouped_count: dict[tuple[int, int, int, str], int] = {}

        for (anchor, target, gname), cnt in state.cat_groups.items():
            if cnt <= 0:
                continue
            src_part = qubit_to_part.get(anchor)
            dst_part = qubit_to_part.get(target)
            if src_part is None or dst_part is None:
                continue
            if src_part == dst_part:
                continue

            key = (anchor, src_part, dst_part, gname)
            if key not in grouped_targets:
                grouped_targets[key] = set()
                grouped_count[key] = 0
            grouped_targets[key].add(target)
            grouped_count[key] += int(cnt)

        score = 0
        for key, targets in grouped_targets.items():
            if len(targets) >= 2 and grouped_count[key] >= 2:
                score += 1
        return score

    def _calculate_tg_costs_incremental(
        self,
        ctx: CompilationContext,
        state: TGIncrementalState,
    ) -> ExecCosts:
        """
        基于增量状态直接计算telegate/CAT成本。
        """
        costs = ExecCosts()
        qubit_to_part = self._build_qubit_to_partition_map(state.partition)

        # 1) 先按普通telegate计费（所有2Q边贡献）
        for (u, v), w in state.edge_weights.items():
            if w <= 0:
                continue
            pu = qubit_to_part.get(u)
            pv = qubit_to_part.get(v)
            if pu is None or pv is None or pu == pv:
                continue
            self._apply_remote_move_weight(costs, pu, pv, int(w), ctx.network)

        # 2) 对CAT有效组做折扣修正：组内总计数n -> 2
        grouped_targets: dict[tuple[int, int, int, str], set[int]] = {}
        grouped_count: dict[tuple[int, int, int, str], int] = {}
        for (anchor, target, gname), cnt in state.cat_groups.items():
            if cnt <= 0:
                continue
            src_p = qubit_to_part.get(anchor)
            dst_p = qubit_to_part.get(target)
            if src_p is None or dst_p is None or src_p == dst_p:
                continue
            gkey = (anchor, src_p, dst_p, gname)
            if gkey not in grouped_targets:
                grouped_targets[gkey] = set()
                grouped_count[gkey] = 0
            grouped_targets[gkey].add(target)
            grouped_count[gkey] += int(cnt)

        for (anchor, src_p, dst_p, gname), targets in grouped_targets.items():
            total_cnt = grouped_count[(anchor, src_p, dst_p, gname)]
            if len(targets) >= 2 and total_cnt >= 2:
                # 基础计费已加了total_cnt次，这里回退到2次。
                delta = 2 - total_cnt
                self._apply_remote_move_weight(costs, src_p, dst_p, int(delta), ctx.network)
                costs.cat_ents += 1

        return costs

    def _apply_remote_move_weight(
        self,
        costs: ExecCosts,
        src: int,
        dst: int,
        weight: int,
        network: Network,
    ) -> None:
        """
        按权重(可正可负)更新remote move相关成本字段。
        负权重用于CAT折扣回退。
        """
        if src == dst or weight == 0:
            return
        hops = network.Hops[src][dst]
        costs.remote_hops += hops * weight
        costs.epairs += hops * weight
        costs.remote_fidelity_loss += network.move_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.move_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.move_fidelity[src][dst]) * weight

    def _accumulate_layer_to_tg_state(
        self,
        ctx: CompilationContext,
        state: TGIncrementalState,
        level: int,
        sign: int,
    ) -> None:
        """
        对单层执行增量更新：
        - sign=+1: 添加该层贡献
        - sign=-1: 删除该层贡献
        """
        if level < 0 or level >= len(ctx.circuit_layers):
            return

        cat_gate_set = {"cx", "cz", "rzz"}

        for node in ctx.circuit_layers[level]:
            if node.op.name == "barrier" or len(node.qargs) != 2:
                continue
            q0 = ctx.circuit.qubits.index(node.qargs[0])
            q1 = ctx.circuit.qubits.index(node.qargs[1])
            edge = (q0, q1) if q0 < q1 else (q1, q0)

            new_val = state.edge_weights.get(edge, 0) + sign
            if new_val <= 0:
                state.edge_weights.pop(edge, None)
            else:
                state.edge_weights[edge] = new_val

            # CAT候选门计数（分区无关）：(anchor, target, gate_name)->count。
            if node.op.name not in cat_gate_set:
                continue

            if node.op.name == "cx":
                anchor = q0
                target = q1
            else:
                # 对称门使用较小逻辑位作锚点，确保确定性。
                anchor = min(q0, q1)
                target = q1 if anchor == q0 else q0

            cat_key = (anchor, target, node.op.name)
            cat_val = state.cat_groups.get(cat_key, 0) + sign
            if cat_val <= 0:
                state.cat_groups.pop(cat_key, None)
            else:
                state.cat_groups[cat_key] = cat_val

    def _build_qubit_to_partition_map(self, partition: list[list[int]]) -> dict[int, int]:
        qubit_to_part: dict[int, int] = {}
        for pidx, group in enumerate(partition):
            for q in group:
                qubit_to_part[q] = pidx
        return qubit_to_part

    def _delta_epairs_for_move(
        self,
        q: int,
        src_part: int,
        tgt_part: int,
        state: TGIncrementalState,
        qubit_to_part: dict[int, int],
        network: Network,
    ) -> int:
        """
        计算单次迁移 q: src->tgt 的epairs增量（局部边增量）。
        """
        delta = 0
        for (u, v), w in state.edge_weights.items():
            if u != q and v != q:
                continue
            other = v if u == q else u
            other_part = qubit_to_part.get(other)
            if other_part is None:
                continue

            old_cross = src_part != other_part
            new_cross = tgt_part != other_part

            old_cost = network.Hops[src_part][other_part] * w if old_cross else 0
            new_cost = network.Hops[tgt_part][other_part] * w if new_cross else 0
            delta += (new_cost - old_cost)
        return delta

    def _step_build_unified_s_t_table(
        self,
        ctx: CompilationContext,
        tg_table: list[list[Optional[ExecCosts]]],
    ) -> tuple[list[list[str]], list[list[int]]]:
        """
        统一DP骨架：
        - mode_table[i][j] 记录区间最优模式："td" / "tg" / "split"；
        - split_table[i][j] 记录切分点；
        - 成本函数保持确定性，不使用reward/可调权重。
        """
        print("[build_unified_s_t_table]")
        start_time = time.time()

        num_depths = len(ctx.multiq_layers)
        if num_depths == 0:
            return [], []

        dp = [[float("inf") for _ in range(num_depths)] for _ in range(num_depths)]
        mode_table = [["" for _ in range(num_depths)] for _ in range(num_depths)]
        split_table = [[-1 for _ in range(num_depths)] for _ in range(num_depths)]
        # 辅助表：记录当前最优方案的起止分区，用于split边界teledata代价。
        start_partition_table: list[list[Optional[list[list[int]]]]] = [[None for _ in range(num_depths)] for _ in range(num_depths)]
        end_partition_table: list[list[Optional[list[list[int]]]]] = [[None for _ in range(num_depths)] for _ in range(num_depths)]

        for i in range(num_depths):
            # 单层：在td/tg之间做确定性比较。
            best_cost = float("inf")
            best_mode = ""
            best_partition: Optional[list[list[int]]] = None

            td_partition = self._select_partition_for_mode(ctx, i, i, "td")
            td_cost = 0.0
            if td_cost < best_cost:
                best_cost = td_cost
                best_mode = "td"
                best_partition = copy.deepcopy(td_partition)

            tg_cost_obj = tg_table[i][i]
            tg_cost = float(tg_cost_obj.epairs) if tg_cost_obj is not None else float("inf")
            if tg_cost < best_cost:
                best_cost = tg_cost
                best_mode = "tg"
                best_partition = self._select_partition_for_mode(ctx, i, i, "tg")

            dp[i][i] = best_cost
            mode_table[i][i] = best_mode
            split_table[i][i] = -1
            start_partition_table[i][i] = copy.deepcopy(best_partition) if best_partition is not None else None
            end_partition_table[i][i] = copy.deepcopy(best_partition) if best_partition is not None else None

        for span in range(2, num_depths + 1):
            for i in range(0, num_depths - span + 1):
                j = i + span - 1

                # 方案A: 直接teledata块（需要P可行）
                if len(ctx.P_table[i][j]) > 0:
                    td_cost = 0.0
                    if td_cost < dp[i][j]:
                        dp[i][j] = td_cost
                        mode_table[i][j] = "td"
                        split_table[i][j] = -1
                        td_partition = self._select_partition_for_mode(ctx, i, j, "td")
                        start_partition_table[i][j] = copy.deepcopy(td_partition)
                        end_partition_table[i][j] = copy.deepcopy(td_partition)

                # 方案B: 直接telegate/CAT块（来自TG-table）
                tg_cost_obj = tg_table[i][j]
                tg_cost = float(tg_cost_obj.epairs) if tg_cost_obj is not None else float("inf")
                if tg_cost < dp[i][j]:
                    dp[i][j] = tg_cost
                    mode_table[i][j] = "tg"
                    split_table[i][j] = -1
                    tg_partition = self._select_partition_for_mode(ctx, i, j, "tg")
                    start_partition_table[i][j] = copy.deepcopy(tg_partition)
                    end_partition_table[i][j] = copy.deepcopy(tg_partition)

                # 方案C: split
                for k in range(i, j):
                    left_end_partition = end_partition_table[i][k]
                    right_start_partition = start_partition_table[k + 1][j]
                    if left_end_partition is None or right_start_partition is None:
                        continue

                    bridge_costs, _ = CompilerUtils.evaluate_teledata(
                        left_end_partition,
                        right_start_partition,
                        ctx.network,
                    )
                    split_cost = dp[i][k] + dp[k + 1][j] + float(bridge_costs.epairs)
                    if split_cost < dp[i][j]:
                        dp[i][j] = split_cost
                        mode_table[i][j] = "split"
                        split_table[i][j] = k
                        start_partition_table[i][j] = copy.deepcopy(start_partition_table[i][k])
                        end_partition_table[i][j] = copy.deepcopy(end_partition_table[k + 1][j])

        setattr(ctx, "unified_dp_epairs", float(dp[0][num_depths - 1]))

        print(f"[build_unified_s_t_table] Time: {time.time() - start_time} seconds")
        return mode_table, split_table

    def _step_validate_dp_record_consistency(
        self,
        ctx: CompilationContext,
        result: MappingRecordList,
    ) -> None:
        """
        校验统一DP目标值与最终records汇总epairs的一致性。
        将差值挂到上下文并输出日志，便于回归验证。
        """
        dp_target = getattr(ctx, "unified_dp_epairs", None)
        if not isinstance(dp_target, (int, float)):
            return

        record_epairs = float(result.total_costs.epairs)
        diff = record_epairs - float(dp_target)
        setattr(ctx, "dp_record_epairs_diff", diff)

        print(
            f"[consistency] dp_epairs={float(dp_target)}, record_epairs={record_epairs}, diff={diff}"
        )
        print(
            f"[consistency] dp_epairs={float(dp_target)}, record_epairs={record_epairs}, diff={diff}",
            file=sys.stderr,
        )

        # 默认行为：出现差异时给出告警，便于批量日志排查。
        if diff != 0:
            print(
                f"[WARNING] consistency mismatch detected: diff={diff}"
            )
            print(
                f"[WARNING] consistency mismatch detected: diff={diff}",
                file=sys.stderr,
            )

        # 可选严格模式：用于实验回归门禁，发现差异直接失败。
        strict_check = bool(ctx.config.get("strict_consistency_check", False))
        if strict_check and diff != 0:
            raise RuntimeError(
                f"[ERROR] consistency mismatch under strict mode: dp={float(dp_target)}, record={record_epairs}, diff={diff}"
            )

    def _step_construct_records_from_mode_plan(
        self,
        ctx: CompilationContext,
        mode_table: list[list[str]],
        split_table: list[list[int]],
        tg_table: list[list[Optional[ExecCosts]]],
    ) -> MappingRecordList:
        """
        从统一DP结果直接回溯生成records（骨架）。
        这里先生成最小可用结构，后续补充分区分配和真实成本写入。
        """
        print("[construct_records_from_mode_plan]")

        result = MappingRecordList()
        num_depths = len(ctx.multiq_layers)
        if num_depths == 0:
            return result

        def _reconstruct(i: int, j: int):
            mode = mode_table[i][j]
            if mode == "split":
                k = split_table[i][j]
                if k < 0:
                    raise RuntimeError(f"[ERROR] split mode without valid split point: ({i}, {j})")
                _reconstruct(i, k)
                _reconstruct(k + 1, j)
                return

            ori_left, ori_right = self._get_original_layer_idx(ctx, (i, j))
            partition = self._select_partition_for_mode(ctx, i, j, mode)

            record = MappingRecord(
                layer_start=ori_left,
                layer_end=ori_right,
                partition=partition,
                mapping_type="teledata" if mode == "td" else "telegate",
            )
            if mode == "tg" and tg_table[i][j] is not None:
                tg_costs = tg_table[i][j]
                assert tg_costs is not None
                record.costs = copy.deepcopy(tg_costs)
                subcircuit = self._build_subcircuit_from_original_layers(ctx, ori_left, ori_right)
                cat_controls = CompilerUtils._extract_cat_controls_for_circuit(subcircuit)
                record.extra_info = {
                    "cat_controls": cat_controls,
                }

            result.add_record(record)

        _reconstruct(0, num_depths - 1)
        result.summarize_total_costs()

        # 将相邻块之间的teledata边界切换成本显式计入总成本，
        # 与 unified S/T 中的 split bridge 代价保持一致。
        bridge_total = ExecCosts()
        for ridx in range(1, len(result.records)):
            prev_rec = result.records[ridx - 1]
            curr_rec = result.records[ridx]
            bridge_costs, _ = CompilerUtils.evaluate_teledata(
                prev_rec.partition,
                curr_rec.partition,
                ctx.network,
            )
            bridge_total += bridge_costs

            if curr_rec.extra_info is None:
                curr_rec.extra_info = {}
            curr_rec.extra_info["bridge_epairs_from_prev"] = int(bridge_costs.epairs)

        result.total_costs += bridge_total
        setattr(ctx, "bridge_total_costs", bridge_total)
        return result

    def _get_original_layer_idx(self, ctx: CompilationContext, block: tuple[int, int]) -> tuple[int, int]:
        """
        multiq层区间 -> 原始层区间映射。
        """
        left, right = block
        ori_left = ctx.map_to_circuit_layer[left]
        ori_right = ctx.map_to_circuit_layer[right]
        return ori_left, ori_right

    def _build_subcircuit_from_original_layers(
        self,
        ctx: CompilationContext,
        ori_left: int,
        ori_right: int,
    ) -> QuantumCircuit:
        """
        从原始DAG层重建子线路（保留single-qubit门），供CAT相关评估使用。
        """
        subc = QuantumCircuit(ctx.circuit.num_qubits)
        original_layers = ctx.circuit_layers

        for lev in range(ori_left, ori_right + 1):
            if lev < 0 or lev >= len(original_layers):
                continue
            for node in original_layers[lev]:
                if node.op.name == "barrier":
                    continue
                qids = [ctx.circuit.qubits.index(q) for q in node.qargs]
                subc.append(node.op, qids, [])

        return subc


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
