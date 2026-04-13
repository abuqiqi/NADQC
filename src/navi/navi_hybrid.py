from dataclasses import dataclass, field
from typing import Any, Optional
import time
import sys
import copy
import numpy as np
import networkx as nx
import datetime

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network
from .partitioner import Partitioner, PartitionerFactory
from .partition_assigner import PartitionAssignerFactory
from .telegate_partitioner import TelegatePartitionerFactory
from .mapper import MapperFactory
from .navi_compiler import CompilationContext

class NAVIHybrid(Compiler):
    """
    Noise-Aware Distributed Quantum Compiler
    """
    compiler_id = "navihybrid"

    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "NAVI Hybrid"

    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit using the NADQC algorithm.
        """
        print(f"Compiling with [{self.name}]...")
        print(f"Compiling with [{self.name}]...", file=sys.stderr)
        
        if config is None:
            config = {}

        # print(f"[DEBUG] [compile] circuit: \n{circuit}")

        # 1. 解析配置
        circuit_name = config.get("circuit_name", "circ")
        
        # 2. 初始化组件
        partitioner_type = config.get("partitioner", "recursive_dp")
        max_option = config.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(partitioner_type, network, max_options=max_option)

        # partition_assigner_type = config.get("partition_assigner", "direct")
        partition_assigner_type = config.get("partition_assigner", "global_max_match")
        partition_assigner = PartitionAssignerFactory.create_assigner(partition_assigner_type)

        telegate_partitioner_type = config.get("telegate_partitioner", "cat")
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

        # Step 2: 构建分区表 (P Table)
        ctx.P_table = self._step_build_partition_table(ctx)

        # Step 3: 构建切片表 (S, T) 并获取子线路范围
        ctx.subc_ranges = self._step_build_slicing_table(ctx)

        # Step 4: 生成分区候选
        ctx.partition_candidates = [ctx.P_table[i][j] for (i, j) in ctx.subc_ranges]
        
        # Step 5: 分配分区计划
        assert ctx.partition_assigner is not None
        assign_time = time.time()
        assign_result = ctx.partition_assigner.assign(ctx.partition_candidates)
        ctx.partition_plan = assign_result["partition_plan"]
        assign_elapsed = time.time() - assign_time
        print(f"[partition assignment] Time: {assign_elapsed}s, candidates={len(ctx.partition_candidates)}")
        print(f"[partition assignment] Time: {assign_elapsed}s, candidates={len(ctx.partition_candidates)}", file=sys.stderr)

        # Step 6: 直接构建telegate和teledata混合的记录列表，供后续mapper使用
        hybrid_strategy = config.get("hybrid_strategy", "beam")
        if hybrid_strategy == "greedy":
            ctx.hybrid_records = self._step_construct_hybrid_records_greedy(ctx)
        elif hybrid_strategy == "beam":
            ctx.hybrid_records = self._step_construct_hybrid_records_beam(ctx)
        else:
            ctx.hybrid_records = self._step_construct_hybrid_records(ctx)

        total_cache_hits = ctx.telegate_cache_hits + ctx.telegate_cache_relaxed_hits
        total_telegate_calls = total_cache_hits + ctx.telegate_cache_misses
        if total_telegate_calls > 0:
            hit_rate = total_cache_hits / total_telegate_calls
            strict_unique = len(ctx.telegate_strict_seen_keys)
            range_unique = len(ctx.telegate_range_seen_keys)
            strict_repeat = max(0, ctx.telegate_total_calls - strict_unique)
            range_repeat = max(0, ctx.telegate_total_calls - range_unique)
            print(
                f"[telegate_cache] strict_hits={ctx.telegate_cache_hits}, relaxed_hits={ctx.telegate_cache_relaxed_hits}, "
                f"misses={ctx.telegate_cache_misses}, "
                f"hit_rate={hit_rate:.2%}"
            )
            print(
                f"[telegate_cache] strict_hits={ctx.telegate_cache_hits}, relaxed_hits={ctx.telegate_cache_relaxed_hits}, "
                f"misses={ctx.telegate_cache_misses}, "
                f"hit_rate={hit_rate:.2%}",
                file=sys.stderr,
            )
            print(
                f"[telegate_keys] calls={ctx.telegate_total_calls}, "
                f"strict_unique={strict_unique}, strict_repeat={strict_repeat}, "
                f"range_unique={range_unique}, range_repeat={range_repeat}"
            )
            print(
                f"[telegate_keys] calls={ctx.telegate_total_calls}, "
                f"strict_unique={strict_unique}, strict_repeat={strict_repeat}, "
                f"range_unique={range_unique}, range_repeat={range_repeat}",
                file=sys.stderr,
            )

        # 诊断：mapper 前混合候选的 CAT 使用情况
        ctx.hybrid_records.summarize_total_costs()
        pre_mapper_costs = ctx.hybrid_records.total_costs
        pre_mapper_path_epairs = pre_mapper_costs.epairs
        if len(ctx.hybrid_records.records) > 1:
            pre_mapper_path_epairs = 0.0
            for idx, rec in enumerate(ctx.hybrid_records.records):
                pre_mapper_path_epairs += rec.costs.epairs
                if idx > 0:
                    td_costs, _ = CompilerUtils.evaluate_teledata(
                        ctx.hybrid_records.records[idx - 1].partition,
                        rec.partition,
                        ctx.network,
                    )
                    pre_mapper_path_epairs += td_costs.epairs
        pre_mapper_telegate = sum(1 for rec in ctx.hybrid_records.records if rec.mapping_type in {"telegate", "cat"})
        pre_mapper_cat_records = sum(1 for rec in ctx.hybrid_records.records if rec.costs.cat_ents > 0)
        print(
            f"[cat_debug][hybrid_pre_mapper] records={len(ctx.hybrid_records.records)}, "
            f"telegate_records={pre_mapper_telegate}, cat_records={pre_mapper_cat_records}, "
            f"cat_ents={pre_mapper_costs.cat_ents}, epairs={pre_mapper_costs.epairs}, "
            f"path_epairs_with_teledata={pre_mapper_path_epairs}"
        )
        print(
            f"[cat_debug][hybrid_pre_mapper] records={len(ctx.hybrid_records.records)}, "
            f"telegate_records={pre_mapper_telegate}, cat_records={pre_mapper_cat_records}, "
            f"cat_ents={pre_mapper_costs.cat_ents}, epairs={pre_mapper_costs.epairs}, "
            f"path_epairs_with_teledata={pre_mapper_path_epairs}",
            file=sys.stderr,
        )

        # Step 7: 最终映射 (考虑噪声)
        assert ctx.mapper is not None
        final_result = ctx.mapper.map(
            ctx.hybrid_records, 
            ctx.circuit,
            ctx.circuit_layers, 
            ctx.network,
            config=ctx.config,
        )

        end_time = time.time()
        exec_time = end_time - start_time
        print(f"[INFO] NAVI Hybrid execution time: {exec_time} sec", file=sys.stderr)

        final_result.summarize_total_costs()
        post_mapper_costs = final_result.total_costs
        post_mapper_telegate = sum(1 for rec in final_result.records if rec.mapping_type in {"telegate", "cat"})
        post_mapper_cat_records = sum(1 for rec in final_result.records if rec.costs.cat_ents > 0)
        print(
            f"[cat_debug][hybrid_post_mapper] records={len(final_result.records)}, "
            f"telegate_records={post_mapper_telegate}, cat_records={post_mapper_cat_records}, "
            f"cat_ents={post_mapper_costs.cat_ents}, epairs={post_mapper_costs.epairs}"
        )
        print(
            f"[cat_debug][hybrid_post_mapper] records={len(final_result.records)}, "
            f"telegate_records={post_mapper_telegate}, cat_records={post_mapper_cat_records}, "
            f"cat_ents={post_mapper_costs.cat_ents}, epairs={post_mapper_costs.epairs}",
            file=sys.stderr,
        )
        final_result.update_total_costs(execution_time = exec_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_result.save_records(f"./outputs/{circuit_name}/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")

        return final_result

    # =========================================================================
    # 步骤实现 (Step Implementations)
    # =========================================================================

    def _step_remove_single_qubit_gates(self, ctx: CompilationContext):
        """
        Remove single qubit gates and calculate densities.
        Updates: ctx.multiq_layers, ctx.map_*, ctx.gate_density, ctx.twoq_gate_density
        """
        start_time = time.time()

        if bool(ctx.config.get("enable_cat_gate_reorder", True)):
            reordered_circuit, reorder_stats = self._reorder_cat_candidate_gates(ctx.circuit)
            ctx.circuit = reordered_circuit
            if bool(ctx.config.get("debug_cat_gate_reorder", True)):
                print(
                    f"[cat_debug][gate_reorder] runs={reorder_stats['runs']}, "
                    f"gates={reorder_stats['gates']}, moved={reorder_stats['moved']}"
                )
                print(
                    f"[cat_debug][gate_reorder] runs={reorder_stats['runs']}, "
                    f"gates={reorder_stats['gates']}, moved={reorder_stats['moved']}",
                    file=sys.stderr,
                )

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
        
        assert 0 in map_to_circuit_layer, "[ERROR] map_to_circuit_layer must contain the first layer mapping."

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
        return

    def _reorder_cat_candidate_gates(self, circuit: QuantumCircuit) -> tuple[QuantumCircuit, dict[str, int]]:
        """
        在语义安全前提下，对连续的 CZ/RZZ 双比特门做局部重排，
        让共享锚点（控制位）的门相邻，提高后续 CAT 机会。
        """
        reordered = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        qindex = {q: i for i, q in enumerate(circuit.qubits)}

        runs = 0
        total_gates = 0
        moved_gates = 0
        run_buffer: list[tuple[Any, list[Any], list[Any]]] = []

        def _unpack_instruction(inst: Any) -> tuple[Any, list[Any], list[Any]]:
            op = getattr(inst, "operation", inst[0])
            qargs = list(getattr(inst, "qubits", inst[1]))
            cargs = list(getattr(inst, "clbits", inst[2]))
            return op, qargs, cargs

        def _is_diag_twoq(op: Any, qargs: list[Any], cargs: list[Any]) -> bool:
            return len(qargs) == 2 and len(cargs) == 0 and getattr(op, "name", "") in {"cz", "rzz"}

        def _flush_run() -> None:
            nonlocal runs, total_gates, moved_gates, run_buffer
            if not run_buffer:
                return

            runs += 1
            total_gates += len(run_buffer)

            freq: dict[int, int] = {}
            for _, qargs, _ in run_buffer:
                q0 = qindex[qargs[0]]
                q1 = qindex[qargs[1]]
                freq[q0] = freq.get(q0, 0) + 1
                freq[q1] = freq.get(q1, 0) + 1

            decorated: list[tuple[tuple[int, int, int], tuple[Any, list[Any], list[Any]]]] = []
            for idx, item in enumerate(run_buffer):
                _, qargs, _ = item
                q0 = qindex[qargs[0]]
                q1 = qindex[qargs[1]]

                if (freq[q0] > freq[q1]) or (freq[q0] == freq[q1] and q0 <= q1):
                    anchor, target = q0, q1
                else:
                    anchor, target = q1, q0

                decorated.append(((anchor, target, idx), item))

            sorted_run = [item for _, item in sorted(decorated, key=lambda x: x[0])]

            moved_gates += sum(1 for idx, item in enumerate(sorted_run) if item is not run_buffer[idx])

            for op, qargs, cargs in sorted_run:
                reordered.append(op, qargs, cargs)

            run_buffer = []

        for inst in circuit.data:
            op, qargs, cargs = _unpack_instruction(inst)
            if _is_diag_twoq(op, qargs, cargs):
                run_buffer.append((op, qargs, cargs))
            else:
                _flush_run()
                reordered.append(op, qargs, cargs)

        _flush_run()

        stats = {
            "runs": runs,
            "gates": total_gates,
            "moved": moved_gates,
        }
        return reordered, stats

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

        # [disabled] 在P-table阶段预评估telegate hint。
        # 相关逻辑按需求临时停用，仅保留代码以便后续恢复。
        # self._build_telegate_hints_for_empty_partitions(ctx, P)

        print(f"[build_partition_table] Time: {time.time() - start_time} seconds")
        print(f"[build_partition_table] Time: {time.time() - start_time} seconds", file=sys.stderr)
        return P

    # def _build_telegate_hints_for_empty_partitions(self, ctx: CompilationContext, P: list[list[list[Any]]]) -> None:
    #     """
    #     针对无0-cut partition的区间，基于OEE进行telegate预评估并记录hint。
    #     hint将用于hybrid beam打分先验，提升CAT候选被保留概率。
    #     """
    #     if not bool(ctx.config.get("enable_ptable_telegate_hint", True)):
    #         ctx.ptable_telegate_hints = {}
    #         return

    #     num_depths = len(P)
    #     max_span = int(ctx.config.get("ptable_telegate_hint_max_span", max(1, num_depths)))
    #     max_span = max(1, max_span)
    #     debug_hint = bool(ctx.config.get("debug_ptable_telegate_hint", True))

    #     hints: dict[tuple[int, int], dict[str, float]] = {}
    #     total_empty = 0
    #     eval_count = 0
    #     cat_positive = 0

    #     for i in range(num_depths):
    #         for j in range(i, num_depths):
    #             if len(P[i][j]) > 0:
    #                 continue
    #             total_empty += 1
    #             if j - i + 1 > max_span:
    #                 continue

    #             ori_left, ori_right = self.get_original_layer_idx(ctx, (i, j))
    #             telegate_result = self._try_generate_telegate(ctx, ori_left, ori_right, None)
    #             tg_costs = telegate_result.total_costs
    #             hints[(ori_left, ori_right)] = {
    #                 "epairs": float(tg_costs.epairs),
    #                 "cat_ents": float(tg_costs.cat_ents),
    #                 "span": float(j - i + 1),
    #             }
    #             eval_count += 1
    #             if tg_costs.cat_ents > 0:
    #                 cat_positive += 1

    #     ctx.ptable_telegate_hints = hints

    #     if debug_hint:
    #         print(
    #             f"[cat_debug][ptable_hint] empty_ranges={total_empty}, evaluated={eval_count}, "
    #             f"cat_positive={cat_positive}, hint_max_span={max_span}"
    #         )
    #         print(
    #             f"[cat_debug][ptable_hint] empty_ranges={total_empty}, evaluated={eval_count}, "
    #             f"cat_positive={cat_positive}, hint_max_span={max_span}",
    #             file=sys.stderr,
    #         )

    def _step_build_slicing_table(self, ctx: CompilationContext) -> list[Any]:
        """
        Build S and T tables and extract subc_ranges.
        """
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
        print(f"[build_slicing_table] Time: {time.time() - start_time} seconds", file=sys.stderr)
        return subc_ranges

    def _step_construct_hybrid_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        直接获取telegate和teledata混合的记录列表，供后续mapper使用
        更新：ctx.hybrid_records
        """
        start_time = time.time()

        # 根据S-table，对于每一个分割点，尝试用telegate替换，并评估成本，如果更优则替换
        # 1. 获取原子块列表
        # blocks 是一个列表，每个元素是 (start_layer, end_layer)，对应 multiq_layer 的索引
        blocks = ctx.subc_ranges
        K = len(blocks) # K 是原子块的数量

        print(f"[construct_hybrid_records] Number of atomic blocks: {K}")
        print(f"[construct_hybrid_records] Number of atomic blocks: {K}", file=sys.stderr)

        if K == 0:
            return MappingRecordList()

        # 2. 预计算：为了方便，我们先把所有原子块的初始 Record 生成好
        # original_block_records[k] 对应 blocks[k] 的纯 teledata 记录
        original_block_records: list[MappingRecord] = []
        for idx, (s, e) in enumerate(blocks):
            # left = ctx.map_to_circuit_layer[s]
            # right = ctx.map_to_circuit_layer[e]
            left, right = self.get_original_layer_idx(ctx, (s, e))
            # logical_phy_map = CompilerUtils.init_logical_phy_map(ctx.partition_plan[idx])
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata" # ,
                # logical_phy_map=logical_phy_map
                # costs还没计算
            )
            original_block_records.append(record)

        # 3. 初始化 DP 表
        # dp[i][j] 表示合并处理 blocks[i...j] (闭区间) 的最优方案
        dp = [[MappingRecordList() for _ in range(K)] for _ in range(K)]

        # 4. 基准情况：长度为 1 (不合并，保持原样)
        for i in range(K):
            # 单个teledata-only块内部没有通信成本，也没有前序连接成本，基准成本设为 0
            dp[i][i].add_record(copy.deepcopy(original_block_records[i]))

        # 5. 自底向上填充 DP 表：长度从 2 到 K
        for length in range(2, K + 1):
            length_start_time = time.time()

            for i in range(K - length + 1):
                j = i + length - 1
                
                best_epairs = float('inf')
                best_record_list = None

                # --- 选项 1：切分 (Split) ---
                # 遍历所有可能的切分点 k，将 [i..j] 分为 [i..k] 和 [k+1..j]
                # 注意：因为我们的最小单位是块，所以切分只能发生在块与块之间
                # 只切分中间
                for k in range(i, j):
                # # --- 选项 1：由小区间最优解组合而来（试三个关键点）---
                # # 定义要试的切分点：左、中、右
                # candidate_ks = [i, (i + j) // 2, j - 1]
                # candidate_ks = list(set(candidate_ks)) # 去重（当区间长度很小时，这三个点可能重复）
                # candidate_ks.sort()
                # for k in candidate_ks:

                    left_entry = dp[i][k]
                    right_entry = dp[k+1][j]
                    
                    # 合并记录列表
                    combined_records = MappingRecordList()
                    combined_records.records = copy.deepcopy(left_entry.records) + \
                                               copy.deepcopy(right_entry.records)
                    
                    # 计算成本：左边成本 + 右边成本 + 拼接处的通信成本
                    # 拼接处通信成本 = 左边最后一个 Record 与右边第一个 Record 之间的 Teledata 成本
                    # 找到左右两边的连接处
                    left_last_idx = len(left_entry.records) - 1
                    prev_rec = combined_records.records[left_last_idx]
                    curr_rec = combined_records.records[left_last_idx + 1]
                    
                    # 计算这两个特定 Record 之间的切换成本
                    costs = CompilerUtils.evaluate_teledata(prev_rec, curr_rec, ctx.network)
                    # 更新combined_records的total_costs
                    combined_records.summarize_total_costs()

                    if combined_records.total_costs.epairs < best_epairs:
                        best_epairs = combined_records.total_costs.epairs
                        best_record_list = combined_records

                if length < max(int(0.3 * K), 20):
                    # --- 选项 2：整体替换 (Merge & Telegate) ---
                    # 尝试把 blocks[i] 到 blocks[j] 这一整段，全部用 telegate 来做
                    # 1. 确定物理层范围
                    global_start_layer = original_block_records[i].layer_start
                    global_end_layer = original_block_records[j].layer_end

                    # 2. 尝试生成 telegate
                    # initial_partition设成block[i-1]的partition（如果i>0），让telegate_partitioner有个初始方案可以基于它进行优化
                    prev_partition = None
                    if i > 0:
                        prev_partition = original_block_records[i-1].partition

                    telegate_result = self._try_generate_telegate(
                        ctx, global_start_layer, global_end_layer, prev_partition
                    )

                    # 比较
                    if telegate_result.total_costs.epairs < best_epairs:
                        best_epairs = telegate_result.total_costs.epairs
                        best_record_list = telegate_result

                assert best_record_list is not None, f"[ERROR] best_record_list should not be None for blocks[{i}..{j}]"
                dp[i][j] = best_record_list

            print(f"[INFO] length: {length} / {K} Time: {time.time() - length_start_time}")

        # 6. 最终结果        
        ctx.telegate_optimized = True

        end_time = time.time()
        print(f"[construct_hybrid_records] Time: {end_time - start_time} seconds")
        print(f"[construct_hybrid_records] Time: {end_time - start_time} seconds", file=sys.stderr)

        return dp[0][K-1]

    def _step_construct_hybrid_records_greedy(self, ctx: CompilationContext) -> MappingRecordList:
        """
        贪心构建混合记录：从左到右扫描，每次尽可能将连续的原子块合并为一个 telegate，
        仅当合并会恶化成本时才切分。telegate 划分调用次数为 O(K)。
        """
        start_time = time.time()
        blocks = ctx.subc_ranges
        K = len(blocks)
        print(f"[construct_hybrid_records] Number of atomic blocks: {K}")
        print(f"[construct_hybrid_records] Number of atomic blocks: {K}", file=sys.stderr)

        if K == 0:
            return MappingRecordList()

        # 预生成纯 teledata 记录（不含成本，仅保留分区信息）
        original_records: list[MappingRecord] = []
        for idx, (s, e) in enumerate(blocks):
            left, right = self.get_original_layer_idx(ctx, (s, e))
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata"
            )
            original_records.append(record)

        result = MappingRecordList()
        i = 0
        while i < K:
            length_start_time = time.time()

            start = i
            # 获取前一个记录的 partition，用于 telegate 初始划分
            prev_partition = result.records[-1].partition if result.records else None

            # 1. 生成仅包含 start 块的 telegate 候选
            s_layer = original_records[start].layer_start
            e_layer = original_records[start].layer_end
            current_telegate = self._try_generate_telegate(ctx, s_layer, e_layer, prev_partition)
            current_end = start

            # 2. 尝试向右扩展，每次扩展一步
            next_idx = current_end + 1
            while next_idx < K:

                print(f"[DEBUG] next_idx: {next_idx}")

                # 生成覆盖 [start, next_idx] 的新 telegate
                new_e_layer = original_records[next_idx].layer_end
                new_telegate = self._try_generate_telegate(
                    ctx, s_layer, new_e_layer, current_telegate.records[-1].partition
                )

                print(f"[DEBUG] new_telegate: epairs [{new_telegate.total_costs.epairs}]")

                # ###########################################################
                # 方案一：保持当前段为 telegate，下一个块单独作为 teledata
                # ###########################################################
                # 计算分开方案的成本：当前 telegate 内部成本 + 与下一个 teledata 块的连接成本
                teledata_costs, _ = CompilerUtils.evaluate_teledata(
                    current_telegate.records[-1].partition, original_records[next_idx].partition, ctx.network)
                split_cost = current_telegate.total_costs.epairs + teledata_costs.epairs

                # ###########################################################
                # 方案二：全部使用telegate
                # ###########################################################
                merge_cost = new_telegate.total_costs.epairs

                print(f"[DEBUG] split cost: {split_cost} = {current_telegate.total_costs.epairs} + {teledata_costs.epairs}, merge cost: {merge_cost}")

                if merge_cost < split_cost:
                    # 合并更优，继续扩展
                    current_telegate = new_telegate
                    current_end = next_idx
                    next_idx += 1
                else:
                    # 合并变差，停止扩展
                    break

            # 3. 决定最终 [start, current_end] 段使用 telegate 还是纯 teledata
            # 计算纯 teledata 方案的总成本（包含与左侧的连接）
            teledata_only_epairs = 0
            if result.records:
                teledata_costs, _ = CompilerUtils.evaluate_teledata(
                    result.records[-1].partition, original_records[start].partition, ctx.network)
                teledata_only_epairs += teledata_costs.epairs
                print(f"[DEBUG] teledata_only_epairs: {teledata_only_epairs}")
            for idx in range(start, current_end):
                teledata_costs, _ = CompilerUtils.evaluate_teledata(
                    original_records[idx].partition, original_records[idx+1].partition, ctx.network)
                teledata_only_epairs += teledata_costs.epairs
                print(f"[DEBUG] teledata_only_epairs: {teledata_only_epairs}")
            print(f"[DEBUG] teledata_only_epairs: {teledata_only_epairs}")

            # 计算 telegate 方案的总成本（包含与左侧的连接）
            telegate_epairs = 0
            if result.records:
                teledata_costs, _ = CompilerUtils.evaluate_teledata(
                    result.records[-1].partition, current_telegate.records[0].partition, ctx.network)
                telegate_epairs += teledata_costs.epairs

            if telegate_epairs + current_telegate.total_costs.epairs < teledata_only_epairs:
                result.add_record(current_telegate.records[0])
            else:
                for idx in range(start, current_end + 1):
                    result.add_record(original_records[idx])

            print(f"[INFO] i: {i} Time: {time.time() - length_start_time}")

            # 移动到下一段
            i = current_end + 1

        ctx.telegate_optimized = True

        end_time = time.time()
        print(f"[construct_hybrid_records] Time: {end_time - start_time} seconds")
        print(f"[construct_hybrid_records] Time: {end_time - start_time} seconds", file=sys.stderr)
        return result

    def _step_construct_hybrid_records_beam(self, ctx: CompilationContext) -> MappingRecordList:
        """
        束搜索构建混合记录

        配置项：
        - hybrid_beam_width: 每个位置保留的候选数
        - hybrid_max_merge_span: 单次 telegate 最长覆盖原子块数
        """
        start_time = time.time()
        blocks = ctx.subc_ranges
        K = len(blocks)

        beam_width = int(ctx.config.get("hybrid_beam_width", 3))
        max_merge_span = int(ctx.config.get("hybrid_max_merge_span", 12))
        # 仅当候选显著超出 beam_width 时才做前向剪枝，减少重复排序开销。
        prune_trigger_ratio = int(ctx.config.get("hybrid_prune_trigger_ratio", 5))
        beam_width = max(1, beam_width)
        max_merge_span = max(1, max_merge_span)
        prune_trigger_ratio = max(2, prune_trigger_ratio)

        print(
            f"[construct_hybrid_records_beam] blocks={K}, beam_width={beam_width}, "
            f"max_merge_span={max_merge_span}, objective=(min epairs, max cat_ents, max hint_hits)"
        )
        print(
            f"[construct_hybrid_records_beam] blocks={K}, beam_width={beam_width}, "
            f"max_merge_span={max_merge_span}, objective=(min epairs, max cat_ents, max hint_hits)",
            file=sys.stderr,
        )

        if K == 0:
            return MappingRecordList()

        original_records: list[MappingRecord] = []
        for idx, (s, e) in enumerate(blocks):
            left, right = self.get_original_layer_idx(ctx, (s, e))
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata",
            )
            original_records.append(record)

        # states[pos] = [(projected_epairs, projected_cat_ents, hint_hits, [records...]), ...]
        states: dict[int, list[tuple[float, int, int, list[MappingRecord]]]] = {0: [(0.0, 0, 0, [])]}
        # [disabled] ptable telegate hint 先验逻辑暂时停用。
        # ptable_hints = getattr(ctx, "ptable_telegate_hints", {})

        def _state_key(item: tuple[float, int, int, list[MappingRecord]]) -> tuple[float, int, int]:
            projected_epairs, projected_cat_ents, hint_hits, _ = item
            return (projected_epairs, -projected_cat_ents, -hint_hits)

        def _partition_sig(partition: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(sorted(part)) for part in partition)

        def _prune(candidates: list[tuple[float, int, int, list[MappingRecord]]]) -> list[tuple[float, int, int, list[MappingRecord]]]:
            # 先按“末尾划分签名”去重，保留每种签名成本最小者，再按总成本截断
            best_by_sig: dict[tuple[tuple[int, ...], ...], tuple[float, int, int, list[MappingRecord]]] = {}
            for item in candidates:
                _, _, _, recs = item
                if not recs:
                    sig = tuple()
                else:
                    sig = _partition_sig(recs[-1].partition)

                old = best_by_sig.get(sig)
                if old is None or _state_key(item) < _state_key(old):
                    best_by_sig[sig] = item

            pruned = sorted(best_by_sig.values(), key=_state_key)
            return pruned[:beam_width]

        for pos in range(K):
            # print(f"================= {pos} / {K} ==================")
            pos_start_time = time.time()
            # 消费前剪枝：保证本层展开输入受控。
            if pos in states and len(states[pos]) > beam_width:
                states[pos] = _prune(states[pos])
            current_states = states.get(pos, [])
            if not current_states:
                continue

            for base_epairs, base_cat_ents, base_hint_hits, base_records in current_states:
                prev_partition = base_records[-1].partition if base_records else None

                # if prev_partition:
                    # prev_layer_start = base_records[-1].layer_start
                    # prev_layer_end   = base_records[-1].layer_end
                    # print(f"\n\n[DEBUG] prev_partition for layer {prev_layer_start}-{prev_layer_end}\n{prev_partition}")

                # 选项 A：当前原子块保持 teledata
                td_record = copy.deepcopy(original_records[pos])
                transition_cost = 0.0
                if prev_partition is not None:
                    td_costs, _ = CompilerUtils.evaluate_teledata(
                        prev_partition, td_record.partition, ctx.network
                    )
                    transition_cost = td_costs.epairs

                next_epairs = base_epairs + transition_cost
                states.setdefault(pos + 1, []).append((next_epairs, base_cat_ents, base_hint_hits, base_records + [td_record]))

                # 选项 B：从 pos 开始合并为 telegate。
                start_layer = original_records[pos].layer_start
                upper_end = min(K - 1, pos + max_merge_span - 1)

                for end in range(pos, upper_end + 1):
                    # TODO: 根据线路增长的epairs信息，决定是否需要进一步延长telegate测试
                    end_layer = original_records[end].layer_end
                    # print(f"[DEBUG] ===== {start_layer} - {end_layer} =====")

                    telegate_result = self._try_generate_telegate(
                        ctx, start_layer, end_layer, prev_partition
                    )

                    tg_record = copy.deepcopy(telegate_result.records[0])
                    transition_cost = 0.0
                    if prev_partition is not None:
                        tg_link_costs, _ = CompilerUtils.evaluate_teledata(
                            prev_partition, tg_record.partition, ctx.network
                        )
                        transition_cost = tg_link_costs.epairs
                    
                    # print(f"[DEBUG] Telegate partition for layer {start_layer}-{end_layer}\n{tg_record.partition}")
                    # print(f"[DEBUG] Transition cost: {transition_cost}\n")

                    seg_cat_ents = int(telegate_result.total_costs.cat_ents)
                    merged_epairs = base_epairs + transition_cost + telegate_result.total_costs.epairs
                    merged_cat_ents = base_cat_ents + seg_cat_ents

                    # [disabled] hint命中统计暂时停用。
                    # hint = ptable_hints.get((start_layer, end_layer))
                    # hint_hit = 1 if hint is not None and float(hint.get("cat_ents", 0.0)) > 0.0 else 0
                    hint_hit = 0
                    merged_hint_hits = base_hint_hits + hint_hit

                    states.setdefault(end + 1, []).append((merged_epairs, merged_cat_ents, merged_hint_hits, base_records + [tg_record]))

            # 修剪不好的records路径，控制状态爆炸
            for prune_pos in range(pos + 1, min(K, pos + max_merge_span) + 1):
                if prune_pos in states and len(states[prune_pos]) > prune_trigger_ratio * beam_width:
                    states[prune_pos] = _prune(states[prune_pos])
            pos_end_time = time.time()
            print(f"[DEBUG] Position {pos} completed in {pos_end_time - pos_start_time} seconds")
            # print(f"[DEBUG] Position {pos} completed in {pos_end_time - pos_start_time} seconds", file=sys.stderr)

        final_candidates = states.get(K, [])
        if len(final_candidates) > beam_width:
            final_candidates = _prune(final_candidates)
        if not final_candidates:
            raise RuntimeError("[ERROR] beam search failed to produce any hybrid plan.")

        best_projected_epairs, best_projected_cat_ents, best_hint_hits, best_records = min(final_candidates, key=_state_key)
        result = MappingRecordList()
        result.records = best_records
        result.summarize_total_costs()

        ctx.telegate_optimized = True
        end_time = time.time()
        print(
            f"[construct_hybrid_records_beam] best_key=(epairs={best_projected_epairs}, cat_ents={best_projected_cat_ents}, hint_hits={best_hint_hits}), "
            f"best_epairs={result.total_costs.epairs}, best_cat_ents={result.total_costs.cat_ents}, "
            f"Time: {end_time - start_time} seconds"
        )
        print(
            f"[construct_hybrid_records_beam] best_key=(epairs={best_projected_epairs}, cat_ents={best_projected_cat_ents}, hint_hits={best_hint_hits}), "
            f"best_epairs={result.total_costs.epairs}, best_cat_ents={result.total_costs.cat_ents}, "
            f"Time: {end_time - start_time} seconds",
            file=sys.stderr,
        )
        return result

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

        subcircuit = transpile(subcircuit, optimization_level=0)

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

    def _try_generate_telegate(
            self, 
            ctx: CompilationContext, 
            ori_left: int, ori_right: int,
            prev_partition: Optional[list[list]] = None) -> MappingRecordList:
        """
        尝试将 multiq_layer 区间 [s_multi, e_multi] 作为一个整体生成 telegate。
        返回 (record_list, cost) 或 None。
        """
        # print(f"[DEBUG] _try_generate_telegate: {ori_left}-{ori_right}, prev_partition: {prev_partition}")
        # start_time = time.time()

        def _partition_signature(partition: Optional[list[list]]) -> tuple[Any, ...]:
            if partition is None:
                return ("NONE",)
            # 外层顺序保持不变（对应后端索引），内层排序避免同构列表顺序差异导致 miss。
            return tuple(tuple(sorted(part)) for part in partition)

        use_cache = bool(ctx.config.get("telegate_cache", True))
        cache_mode = str(ctx.config.get("telegate_cache_mode", "strict"))
        strict_key = (ori_left, ori_right, _partition_signature(prev_partition))
        range_key = (ori_left, ori_right, "ANY_PREV")

        ctx.telegate_total_calls += 1
        ctx.telegate_strict_seen_keys.add(strict_key)
        ctx.telegate_range_seen_keys.add(range_key)

        if use_cache:
            if cache_mode == "range":
                if range_key in ctx.telegate_cache:
                    ctx.telegate_cache_relaxed_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[range_key])
            else:
                # strict / range_fallback / unknown(按strict处理)
                if strict_key in ctx.telegate_cache:
                    ctx.telegate_cache_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[strict_key])

                if cache_mode == "range_fallback" and range_key in ctx.telegate_cache:
                    ctx.telegate_cache_relaxed_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[range_key])

            ctx.telegate_cache_misses += 1
        
        # 1. 提取子电路
        sub_qc = self._get_ori_subc(ctx, ori_left, ori_right)
        cat_controls = self._extract_cat_controls(sub_qc)

        # print(f"\n\n[DEBUG] try_generate_telegate\n")
        # print(sub_qc)

        # 2. 调用 partitioner
        assert ctx.telegate_partitioner is not None
        telegate_record = ctx.telegate_partitioner.partition(
            circuit = sub_qc,
            network = ctx.network,
            config = {
                "layer_start": ori_left,
                "layer_end": ori_right,
                "partition": prev_partition,
                "use_oee_init": bool(ctx.config.get("use_oee_init", True)),
                "iteration": int(ctx.config.get("iteration", 50)),
            }
        )

        if isinstance(telegate_record, MappingRecordList):
            record_list = telegate_record
            # 输出调试信息
            # print(f"[DEBUG] op_list:\n{record_list.records[0].extra_info['ops']}\nepairs: {record_list.records[0].costs.epairs}")
        else:
            if telegate_record.extra_info is None:
                telegate_record.extra_info = {}
            telegate_record.extra_info["cat_controls"] = list(cat_controls)
            
            record_list = MappingRecordList()
            record_list.add_record(telegate_record)
            record_list.summarize_total_costs()

        if use_cache:
            # strict模式只写strict key；range模式只写range key；fallback模式双写提高复用。
            if cache_mode == "range":
                ctx.telegate_cache[range_key] = copy.deepcopy(record_list)
            elif cache_mode == "range_fallback":
                cached = copy.deepcopy(record_list)
                ctx.telegate_cache[strict_key] = cached
                ctx.telegate_cache[range_key] = copy.deepcopy(cached)
            else:
                ctx.telegate_cache[strict_key] = copy.deepcopy(record_list)
        
        # print(f"[DEBUG] Time: {time.time() - start_time}")
        return record_list

    def _extract_cat_controls(self, circuit: QuantumCircuit) -> list[int]:
        """
        提取可用于CAT风格telegate复用的控制位候选。
        与 CompilerUtils 的双锚点规则保持一致（cz/rzz 两端都可作为锚点）。
        """
        return CompilerUtils._extract_cat_controls_for_circuit(
            circuit,
            support={"cx", "cz", "rzz"},
        )

    # Others
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