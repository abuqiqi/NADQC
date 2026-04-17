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

from ..compiler import Compiler, CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..utils import Network, log
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
        use_noise_aware_hybrid_records = bool(config.get("use_noise_aware_hybrid_records", False))
        
        # 2. 初始化组件
        partitioner_type = config.get("partitioner", "recursive_dp")
        max_option = config.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(partitioner_type, network, max_options=max_option)

        # partition_assigner_type = config.get("partition_assigner", "direct")
        partition_assigner_type = config.get("partition_assigner", "global_max_match")
        partition_assigner = PartitionAssignerFactory.create_assigner(partition_assigner_type)

        telegate_partitioner_type = config.get("telegate_partitioner", "cat") # "oee", "cat"
        telegate_partitioner = TelegatePartitionerFactory.create_telegate_partitioner(telegate_partitioner_type)

        # hybrid records模式与mapper绑定：
        # - noise-aware records: direct mapper
        # - beam records(all-to-all cost model): boundeddp_neighbor mapper
        # mapper_type = "direct" if use_noise_aware_hybrid_records else "direct" # "boundeddp_neighbor"
        # mapper = MapperFactory.create_mapper(mapper_type)
        # hybrid records模式默认使用single_record_debug，
        # 也支持通过config["mapper"]覆盖。
        mapper_type = config.get("mapper", "single_record_debug")
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

        # Step 6.1: 先构建teledata-only记录，供后续与hybrid records比较
        teledata_only_records = self._step_construct_teledata_only_records(ctx)
        teledata_eval_epairs = float(teledata_only_records.total_costs.epairs)

        # Step 6.2: 构建telegate和teledata混合记录（可选噪声感知映射协同版）
        if use_noise_aware_hybrid_records:
            ctx.hybrid_records = self._step_construct_noise_aware_hybrid_records(ctx)
        else:
            ctx.hybrid_records = self._step_construct_hybrid_records_beam(ctx)
        hybrid_eval_epairs = ctx.hybrid_records.total_costs.epairs

        # TODO: Step 6.3: 构建telegate-only记录

        # Step 6.3: 在all-to-all口径下比较epairs，选择更优者进入mapper
        chosen_records_tag = "hybrid"
        if teledata_eval_epairs < hybrid_eval_epairs:
            ctx.hybrid_records = teledata_only_records
            chosen_records_tag = "teledata_only"

        log(
            f"[records_compare][all_to_all] hybrid_epairs={hybrid_eval_epairs}, "
            f"teledata_only_epairs={teledata_eval_epairs}, chosen={chosen_records_tag}"
        )

        total_cache_hits = ctx.telegate_cache_hits + ctx.telegate_cache_relaxed_hits
        total_telegate_calls = total_cache_hits + ctx.telegate_cache_misses
        if total_telegate_calls > 0:
            hit_rate = total_cache_hits / total_telegate_calls
            strict_unique = len(ctx.telegate_strict_seen_keys)
            range_unique = len(ctx.telegate_range_seen_keys)
            strict_repeat = max(0, ctx.telegate_total_calls - strict_unique)
            range_repeat = max(0, ctx.telegate_total_calls - range_unique)
            log(
                f"[telegate_cache] strict_hits={ctx.telegate_cache_hits}, relaxed_hits={ctx.telegate_cache_relaxed_hits}, "
                f"misses={ctx.telegate_cache_misses}, "
                f"hit_rate={hit_rate:.2%}\n"
                f"[telegate_keys] calls={ctx.telegate_total_calls}, "
                f"strict_unique={strict_unique}, strict_repeat={strict_repeat}, "
                f"range_unique={range_unique}, range_repeat={range_repeat}"
            )

        # Step 7: 最终映射 (考虑噪声)
        assert ctx.mapper is not None

        # 单record场景下，同时输出贪心与全排列两种mapper的结果，便于直接比较。
        compare_single_record_mappers = bool(config.get("compare_single_record_mappers", True))
        if compare_single_record_mappers and len(ctx.hybrid_records.records) == 1:
            compare_mapper_ids = ["single_record_greedy", "single_record_refined", "single_record_debug"]
            compare_results: dict[str, MappingRecordList] = {}

            for mid in compare_mapper_ids:
                mapper_i = MapperFactory.create_mapper(mid)
                mapped_i = mapper_i.map(
                    copy.deepcopy(ctx.hybrid_records),
                    ctx.circuit,
                    ctx.circuit_layers,
                    ctx.network,
                    config=ctx.config,
                )
                mapped_i.summarize_total_costs()
                compare_results[mid] = mapped_i

                c = mapped_i.total_costs
                log(
                    f"[mapper_compare] mapper={mid}, "
                    f"epairs={c.epairs}, total_fidelity_loss={c.total_fidelity_loss}, "
                    f"num_comms={c.num_comms}, cat_ents={c.cat_ents}"
                )

            preferred_mapper = str(config.get("mapper", "single_record_refined")).lower()
            if preferred_mapper in compare_results:
                final_result = compare_results[preferred_mapper]
            else:
                best_mid = min(
                    compare_results,
                    key=lambda mid: (
                        compare_results[mid].total_costs.total_fidelity_loss,
                        compare_results[mid].total_costs.epairs,
                    ),
                )
                final_result = compare_results[best_mid]
                log(f"[mapper_compare] preferred mapper '{preferred_mapper}' not in compare set, fallback to best='{best_mid}'")

            c = final_result.total_costs
            log(
                f"[mapper_compare] selected_result mapper={preferred_mapper}, "
                f"epairs={c.epairs}, total_fidelity_loss={c.total_fidelity_loss}"
            )
        else:
            final_result = ctx.mapper.map(
                ctx.hybrid_records,
                ctx.circuit,
                ctx.circuit_layers,
                ctx.network,
                config=ctx.config,
            )

        end_time = time.time()
        exec_time = end_time - start_time
        log(f"[INFO] NAVI Hybrid execution time: {exec_time} sec")

        get_ori_subc_calls = int(getattr(ctx, "get_ori_subc_calls", 0))
        get_ori_subc_total_time = float(getattr(ctx, "get_ori_subc_total_time", 0.0))
        get_ori_subc_avg_time = (
            get_ori_subc_total_time / get_ori_subc_calls if get_ori_subc_calls > 0 else 0.0
        )
        get_ori_subc_ratio = (
            get_ori_subc_total_time / exec_time if exec_time > 0 else 0.0
        )
        log(
            f"[perf][_get_ori_subc] calls={get_ori_subc_calls}, "
            f"total={get_ori_subc_total_time:.6f}s, avg={get_ori_subc_avg_time:.6f}s, "
            f"ratio={get_ori_subc_ratio:.2%}"
        )

        final_result.summarize_total_costs()
        post_mapper_costs = final_result.total_costs
        post_mapper_telegate = sum(1 for rec in final_result.records if rec.mapping_type in {"telegate", "cat"})
        post_mapper_cat_records = sum(1 for rec in final_result.records if rec.costs.cat_ents > 0)
        log(
            f"[cat_debug][hybrid_post_mapper] records={len(final_result.records)}, "
            f"telegate_records={post_mapper_telegate}, cat_records={post_mapper_cat_records}, "
            f"cat_ents={post_mapper_costs.cat_ents}, epairs={post_mapper_costs.epairs}"
        )
        final_result.update_total_costs(execution_time = exec_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_result.save_records(f"./outputs/{circuit_name}/{circuit_name}_{network.name}_{self.name}_{timestamp}.json")

        return final_result

    # =========================================================================
    # 步骤实现 (Step Implementations)
    # =========================================================================

    def _get_all_to_all_eval_network(self, ctx: CompilationContext) -> Network:
        """
        获取预构建的all-to-all评估网络副本。
        """
        prebuilt = getattr(ctx.network, "all_to_all_copy", None)
        if prebuilt is not None:
            return prebuilt

        raise RuntimeError("[ERROR] ctx.network.all_to_all_copy is missing.")

    def _sample_budgeted_indices(self, start_idx: int, end_idx: int, budget: int) -> list[int]:
        """
        在[start_idx, end_idx]范围内按预算采样索引，保证包含头尾。
        """
        if end_idx < start_idx:
            return []

        budget = max(1, int(budget))
        span_len = end_idx - start_idx + 1
        if span_len <= budget:
            return list(range(start_idx, end_idx + 1))

        picked: set[int] = {start_idx, end_idx}
        if budget > 2:
            inner = budget - 2
            for t in range(1, inner + 1):
                idx = start_idx + (t * (span_len - 1)) // (inner + 1)
                picked.add(idx)

        return sorted(picked)

    def _step_construct_teledata_only_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        基于partition_plan构建teledata-only记录，并将all-to-all口径epairs写入record.costs。
        """
        mapping_record_list = MappingRecordList()
        eval_network = ctx.network # self._get_all_to_all_eval_network(ctx)
        prev_partition: Optional[list[list[int]]] = None

        for idx, (s, e) in enumerate(ctx.subc_ranges):
            left, right = self.get_original_layer_idx(ctx, (s, e))
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata",
            )

            if prev_partition is not None:
                td_costs, _ = CompilerUtils.evaluate_teledata(prev_partition, record.partition, eval_network)
                record.costs = copy.deepcopy(td_costs)
            else:
                record.costs = ExecCosts()
            prev_partition = record.partition

            mapping_record_list.add_record(record)

        mapping_record_list.summarize_total_costs()
        return mapping_record_list

    def _step_remove_single_qubit_gates(self, ctx: CompilationContext):
        """
        Remove single qubit gates and calculate densities.
        Updates: ctx.multiq_layers, ctx.map_*, ctx.gate_density, ctx.twoq_gate_density
        """
        start_time = time.time()

        if bool(ctx.config.get("enable_cat_gate_reorder", False)):
            reordered_circuit, reorder_stats = self._reorder_cat_candidate_gates(ctx.circuit)
            ctx.circuit = reordered_circuit
            if bool(ctx.config.get("debug_cat_gate_reorder", True)):
                log(
                    f"[cat_debug][gate_reorder] runs={reorder_stats['runs']}, "
                    f"gates={reorder_stats['gates']}, moved={reorder_stats['moved']}"
                )

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
        log(f"[INFO] remove_single_qubit_gates: {end_time - start_time} seconds")
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

        log(f"[build_partition_table] Partition calculation times: {cnt}.")

        # [disabled] 在P-table阶段预评估telegate hint。
        # 相关逻辑按需求临时停用，仅保留代码以便后续恢复。
        # self._build_telegate_hints_for_empty_partitions(ctx, P)

        log(f"[build_partition_table] Time: {time.time() - start_time} seconds")
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
        # 最多合并多少个block
        max_merge_span = int(ctx.config.get("hybrid_max_merge_span", 12))
        # 在每个pos，最多尝试几次_try_generate_telegate
        merge_probe_budget = int(ctx.config.get("hybrid_merge_probe_budget", 12))
        # 最多探查几个pos
        pos_probe_budget = int(ctx.config.get("hybrid_pos_probe_budget", 1))
        # 仅当候选显著超出 beam_width 时才做前向剪枝，减少重复排序开销。
        prune_trigger_ratio = int(ctx.config.get("hybrid_prune_trigger_ratio", 5))
        # 防止数值输入有误
        beam_width = max(1, beam_width)
        max_merge_span = max(1, max_merge_span)
        merge_probe_budget = max(1, merge_probe_budget)
        pos_probe_budget = max(1, pos_probe_budget)
        prune_trigger_ratio = max(2, prune_trigger_ratio)

        log(
            f"[construct_hybrid_records_beam] blocks={K}, beam_width={beam_width}, "
            f"max_merge_span={max_merge_span}, merge_probe_budget={merge_probe_budget}, "
            f"pos_probe_budget={pos_probe_budget}, "
            f"objective=(min epairs, max cat_ents)"
        )

        if K == 0:
            return MappingRecordList()

        eval_network = ctx.network # self._get_all_to_all_eval_network(ctx)

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

        telegate_probe_positions = set(self._sample_budgeted_indices(0, K - 1, pos_probe_budget))
        log(
            f"[construct_hybrid_records_beam] telegate_probe_positions: {telegate_probe_positions}\n"
            f"[construct_hybrid_records_beam] telegate_probe_positions={len(telegate_probe_positions)}/{K}"
        )

        for pos in range(K):
            print(f"\n================= {pos} / {K} ==================")
            pos_start_time = time.time()
            # 消费前剪枝：保证本层展开输入受控。
            if pos in states and len(states[pos]) > beam_width:
                states[pos] = _prune(states[pos])
            current_states = states.get(pos, [])

            if not current_states:
                log(f"[WARNING] pos {pos} has no states")
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
                        prev_partition, td_record.partition, eval_network
                    )
                    td_record.costs = copy.deepcopy(td_costs)
                else:
                    td_record.costs = ExecCosts()

                transition_cost = float(td_record.costs.epairs)

                next_epairs = base_epairs + transition_cost
                states.setdefault(pos + 1, []).append((next_epairs, base_cat_ents, base_hint_hits, base_records + [td_record]))

                # 选项 B：仅在采样到的pos上展开telegate，控制重计算规模。
                if pos in telegate_probe_positions:
                    start_layer = original_records[pos].layer_start
                    upper_end = min(K - 1, pos + max_merge_span - 1)
                    sampled_ends = self._sample_budgeted_indices(pos, upper_end, merge_probe_budget)

                    if pos == 0 and (K - 1) not in sampled_ends:
                        sampled_ends.append(K - 1)
                        sampled_ends = sorted(set(sampled_ends))

                    log(
                        f"[construct_hybrid_records_beam] Sampling telegate for position {pos}, sampled ends: {sampled_ends}"
                    )

                    for end in sampled_ends:
                        end_layer = original_records[end].layer_end
                        # print(f"[DEBUG] ===== {start_layer} - {end_layer} =====")

                        telegate_result = self._try_generate_telegate(
                            ctx,
                            start_layer,
                            end_layer,
                            prev_partition,
                        )

                        tg_record = copy.deepcopy(telegate_result.records[0])
                        transition_cost = 0.0
                        if prev_partition is not None:
                            tg_link_costs, _ = CompilerUtils.evaluate_teledata(
                                prev_partition, tg_record.partition, eval_network
                            )
                            tg_record.costs += tg_link_costs

                        # transition_cost = float(tg_record.costs.epairs) - float(telegate_result.total_costs.epairs)
                        
                        # print(f"[DEBUG] Telegate partition for layer {start_layer}-{end_layer}\n{tg_record.partition}")
                        # print(f"[DEBUG] Transition cost: {transition_cost}\n")

                        seg_cat_ents = int(telegate_result.total_costs.cat_ents)
                        merged_epairs = base_epairs + float(tg_record.costs.epairs)
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
            
            # 打印states
            for p, s in states.items():
                print(f"[DEBUG] Position {p}: ")
                for state in s:
                    print(f"[DEBUG] State: {state[0]}, {state[1]}, {state[2]}")
            
            pos_end_time = time.time()
            print(f"[DEBUG] Position {pos} completed in {pos_end_time - pos_start_time} seconds")

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
        log(
            f"[construct_hybrid_records_beam] best_key=(epairs={best_projected_epairs}, cat_ents={best_projected_cat_ents}, hint_hits={best_hint_hits}), "
            f"best_epairs={result.total_costs.epairs}, best_cat_ents={result.total_costs.cat_ents}, "
            f"Time: {end_time - start_time} seconds"
        )
        return result

    def _step_construct_noise_aware_hybrid_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        噪声感知束搜索构建混合记录：
        - 不替换现有 _step_construct_hybrid_records_beam；
        - 在每一步扩展时显式携带 logical_phy_map；
        - 用“当前映射下真实保真度损失和epairs增量”做打分。

        评分口径：
        score = 保真度, 历史累计真实epairs + 当前候选在(teledata + local/telegate)下的真实epairs增量
        """
        start_time = time.time()
        blocks = ctx.subc_ranges
        K = len(blocks)

        beam_width = int(ctx.config.get("hybrid_beam_width", 3))
        max_merge_span = int(ctx.config.get("hybrid_max_merge_span", 12))
        merge_probe_budget = int(ctx.config.get("hybrid_merge_probe_budget", 12))
        pos_probe_budget = int(ctx.config.get("hybrid_pos_probe_budget", 90))
        prune_trigger_ratio = int(ctx.config.get("hybrid_prune_trigger_ratio", 5))
        perm_budget = int(ctx.config.get("noise_aware_perm_budget", 6))
        perm_neighbor_swaps = int(ctx.config.get("noise_aware_perm_neighbor_swaps", 1))
        perm_refine_rounds = int(ctx.config.get("noise_aware_perm_refine_rounds", 2))
        hops_weight = float(ctx.config.get("noise_aware_hops_weight", 1.0))
        fidelity_loss_weight = float(ctx.config.get("noise_aware_fidelity_loss_weight", 1.0))
        beam_width = max(1, beam_width)
        max_merge_span = max(1, max_merge_span)
        merge_probe_budget = max(1, merge_probe_budget)
        pos_probe_budget = max(1, pos_probe_budget)
        prune_trigger_ratio = max(2, prune_trigger_ratio)
        perm_budget = max(1, perm_budget)
        perm_neighbor_swaps = max(0, perm_neighbor_swaps)
        perm_refine_rounds = max(0, perm_refine_rounds)

        log(
            f"[construct_noise_aware_hybrid_records] blocks={K}, beam_width={beam_width}, "
            f"max_merge_span={max_merge_span}, merge_probe_budget={merge_probe_budget}, "
            f"pos_probe_budget={pos_probe_budget}, perm_budget={perm_budget}, "
            f"perm_neighbor_swaps={perm_neighbor_swaps}, objective=(min real_epairs)"
        )

        if K == 0:
            return MappingRecordList()

        cost_eval_network = ctx.network
        n_backends = ctx.network.num_backends

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

        # states[pos] = [(total_fidelity_loss, epairs, [records...], logical_phy_map), ...]
        states: dict[int, list[tuple[float, float, list[MappingRecord], dict[int, tuple[int, int | None]]]]] = {
            0: [(0.0, 0.0, [], {})]
        }
        subcircuit_cache: dict[tuple[int, int], QuantumCircuit] = {}

        def _state_key(
            item: tuple[float, float, list[MappingRecord], dict[int, tuple[int, int | None]]]
        ) -> tuple[float, float]:
            total_fidelity_loss, epairs, _, _ = item
            return (total_fidelity_loss, epairs)

        def _partition_sig(partition: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(sorted(part)) for part in partition)

        def _prune(
            candidates: list[tuple[float, float, list[MappingRecord], dict[int, tuple[int, int | None]]]]
        ) -> list[tuple[float, float, list[MappingRecord], dict[int, tuple[int, int | None]]]]:
            # 先按“末尾划分签名”去重，保留每种签名成本最小者，再按总成本截断。
            best_by_sig: dict[
                tuple[tuple[int, ...], ...],
                tuple[float, float, list[MappingRecord], dict[int, tuple[int, int | None]]],
            ] = {}
            for item in candidates:
                _, _, recs, _ = item
                if not recs:
                    sig = tuple()
                else:
                    sig = _partition_sig(recs[-1].partition)

                old = best_by_sig.get(sig)
                if old is None or _state_key(item) < _state_key(old):
                    best_by_sig[sig] = item

            pruned = sorted(best_by_sig.values(), key=_state_key)
            return pruned[:beam_width]

        def _get_record_subcircuit(record: MappingRecord) -> QuantumCircuit:
            if record.extra_info is not None and "ops" in record.extra_info:
                return record.extra_info["ops"]

            key = (record.layer_start, record.layer_end)
            if key in subcircuit_cache:
                return subcircuit_cache[key]

            subc = CompilerUtils.get_subcircuit_by_level(
                num_qubits=ctx.circuit.num_qubits,
                circuit=ctx.circuit,
                circuit_layers=ctx.circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end,
            )
            subcircuit_cache[key] = subc
            return subc

        def _apply_perm(partition: list[list[int]], perm: tuple[int, ...]) -> list[list[int]]:
            return [partition[i] for i in perm]

        def _invert_perm(perm: tuple[int, ...]) -> list[int]:
            # perm[physical] = logical  -> inverse[logical] = physical
            inv = [-1 for _ in range(len(perm))]
            for physical, logical in enumerate(perm):
                inv[logical] = physical
            return inv

        def _build_prev_preferred_perm(
            prev_partition: Optional[list[list[int]]],
            base_logical_map: dict[int, tuple[int, int | None]],
        ) -> tuple[int, ...]:
            """
            从上一状态 logical_phy_map 提取一个“延续性”映射，作为 permutation 搜索起点。
            返回格式：perm[physical] = logical。
            """
            identity = tuple(range(n_backends))
            if prev_partition is None or len(base_logical_map) == 0:
                return identity

            vote: list[dict[int, int]] = [dict() for _ in range(n_backends)]
            for logical_id, group in enumerate(prev_partition):
                for q in group:
                    if q not in base_logical_map:
                        continue
                    physical_id = base_logical_map[q][0]
                    if physical_id is None:
                        continue
                    bucket = vote[logical_id]
                    bucket[physical_id] = bucket.get(physical_id, 0) + 1

            assigned_logical: set[int] = set()
            assigned_physical: set[int] = set()
            pairs: list[tuple[int, int, int]] = []  # (count, logical, physical)
            for logical_id in range(n_backends):
                for physical_id, cnt in vote[logical_id].items():
                    pairs.append((cnt, logical_id, physical_id))
            pairs.sort(reverse=True)

            logical_to_physical = [-1 for _ in range(n_backends)]
            for cnt, logical_id, physical_id in pairs:
                if cnt <= 0:
                    continue
                if logical_id in assigned_logical or physical_id in assigned_physical:
                    continue
                logical_to_physical[logical_id] = physical_id
                assigned_logical.add(logical_id)
                assigned_physical.add(physical_id)

            free_logical = [x for x in range(n_backends) if logical_to_physical[x] == -1]
            free_physical = [x for x in range(n_backends) if x not in assigned_physical]
            for logical_id, physical_id in zip(free_logical, free_physical):
                logical_to_physical[logical_id] = physical_id

            perm = [-1 for _ in range(n_backends)]
            for logical_id, physical_id in enumerate(logical_to_physical):
                perm[physical_id] = logical_id
            return tuple(perm)

        def _build_pair_demand(
            prev_partition: Optional[list[list[int]]],
            curr_partition: list[list[int]],
            subcircuit: QuantumCircuit,
        ) -> list[list[float]]:
            """
            构建逻辑分区间通信需求矩阵：
            - 加入 partition 切换造成的迁移需求；
            - 加入当前子线路中跨分区双比特/CommOp需求。
            """
            demand = [[0.0 for _ in range(n_backends)] for _ in range(n_backends)]

            q_to_curr: dict[int, int] = {}
            for pid, part in enumerate(curr_partition):
                for q in part:
                    q_to_curr[q] = pid

            if prev_partition is not None:
                q_to_prev: dict[int, int] = {}
                for pid, part in enumerate(prev_partition):
                    for q in part:
                        q_to_prev[q] = pid

                for q, prev_pid in q_to_prev.items():
                    curr_pid = q_to_curr.get(q)
                    if curr_pid is None or prev_pid == curr_pid:
                        continue
                    demand[prev_pid][curr_pid] += 1.0

            for inst in subcircuit:
                op = inst.operation
                op_name = getattr(op, "name", "")
                if op_name.startswith("comm_"):
                    src = getattr(op, "src_qpu", None)
                    dst = getattr(op, "dst_qpu", None)
                    if isinstance(src, int) and isinstance(dst, int) and src != dst:
                        w = 2.0 if getattr(op, "comm_type", "") == "rtp" else 1.0
                        demand[src][dst] += w
                    continue

                if len(inst.qubits) != 2:
                    continue
                q0 = subcircuit.qubits.index(inst.qubits[0])
                q1 = subcircuit.qubits.index(inst.qubits[1])
                p0 = q_to_curr.get(q0)
                p1 = q_to_curr.get(q1)
                if p0 is None or p1 is None or p0 == p1:
                    continue
                demand[p0][p1] += 1.0
                demand[p1][p0] += 1.0

            return demand

        def _mapping_penalty(physical_src: int, physical_dst: int) -> float:
            if physical_src == physical_dst:
                return 0.0
            hops = float(ctx.network.Hops[physical_src][physical_dst])
            loss = float(ctx.network.move_fidelity_loss[physical_src][physical_dst])
            return hops_weight * hops + fidelity_loss_weight * loss

        def _score_perm(perm: tuple[int, ...], demand: list[list[float]]) -> float:
            logical_to_physical = _invert_perm(perm)
            total = 0.0
            for lsrc in range(n_backends):
                psrc = logical_to_physical[lsrc]
                for ldst in range(n_backends):
                    w = demand[lsrc][ldst]
                    if w <= 0:
                        continue
                    pdst = logical_to_physical[ldst]
                    total += w * _mapping_penalty(psrc, pdst)
            return total

        def _refine_perm_by_swaps(base_perm: tuple[int, ...], demand: list[list[float]]) -> tuple[int, ...]:
            """
            对起始perm做小步邻域改进，不做全排列枚举。
            """
            best = base_perm
            best_score = _score_perm(best, demand)

            for _ in range(perm_refine_rounds):
                improved = False
                for i in range(n_backends):
                    for j in range(i + 1, n_backends):
                        cand = list(best)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand_t = tuple(cand)
                        cand_score = _score_perm(cand_t, demand)
                        if cand_score + 1e-12 < best_score:
                            best, best_score = cand_t, cand_score
                            improved = True
                if not improved:
                    break
            return best

        def _build_perm_candidates(
            prev_partition: Optional[list[list[int]]],
            curr_record: MappingRecord,
            base_logical_map: dict[int, tuple[int, int | None]],
        ) -> list[tuple[int, ...]]:
            identity = tuple(range(n_backends))
            subcircuit = _get_record_subcircuit(curr_record)
            demand = _build_pair_demand(prev_partition, curr_record.partition, subcircuit)

            seed = _build_prev_preferred_perm(prev_partition, base_logical_map)
            refined = _refine_perm_by_swaps(seed, demand)

            candidates: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()

            def _push(p: tuple[int, ...]) -> None:
                if p in seen:
                    return
                candidates.append(p)
                seen.add(p)

            _push(seed)
            _push(refined)
            _push(identity)

            frontier = list(candidates)
            for _ in range(perm_neighbor_swaps):
                if len(candidates) >= perm_budget:
                    break
                next_frontier: list[tuple[int, ...]] = []
                for perm in frontier:
                    if len(candidates) >= perm_budget:
                        break
                    for i in range(n_backends):
                        for j in range(i + 1, n_backends):
                            cand = list(perm)
                            cand[i], cand[j] = cand[j], cand[i]
                            cand_t = tuple(cand)
                            if cand_t in seen:
                                continue
                            _push(cand_t)
                            next_frontier.append(cand_t)
                            if len(candidates) >= perm_budget:
                                break
                        if len(candidates) >= perm_budget:
                            break
                frontier = next_frontier
                if not frontier:
                    break

            # 先按链路感知启发式排序，再截断到预算。
            candidates = sorted(candidates, key=lambda p: _score_perm(p, demand))
            return candidates[:perm_budget]

        def _evaluate_real_epairs_delta(
            prev_partition: Optional[list[list[int]]],
            curr_record: MappingRecord,
            base_logical_map: dict[int, tuple[int, int | None]],
            perm: Optional[tuple[int, ...]] = None,
        ) -> tuple[float, float, dict[int, tuple[int, int | None]], MappingRecord]:
            """
            计算在“当前映射状态”下，引入 curr_record 的真实epairs增量。
            返回: (delta_total_fidelity_loss, delta_epairs, next_logical_map, applied_record)
            """
            next_logical_map = copy.deepcopy(base_logical_map)
            applied_record = copy.deepcopy(curr_record)

            if perm is not None:
                applied_record.partition = _apply_perm(applied_record.partition, perm)

            delta_epairs = 0.0
            delta_total_fidelity_loss = 0.0

            if prev_partition is not None:
                td_costs, next_logical_map = CompilerUtils.evaluate_teledata(
                    prev_partition,
                    applied_record.partition,
                    cost_eval_network,
                    logical_phy_map=next_logical_map,
                )
                delta_epairs += td_costs.epairs
                delta_total_fidelity_loss += td_costs.total_fidelity_loss

            subcircuit = _get_record_subcircuit(applied_record)
            tg_costs, next_logical_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                applied_record.partition,
                subcircuit,
                cost_eval_network,
                logical_phy_map=next_logical_map,
                optimization_level=0,
            )
            delta_epairs += tg_costs.epairs
            delta_total_fidelity_loss += tg_costs.total_fidelity_loss

            return delta_total_fidelity_loss, delta_epairs, next_logical_map, applied_record

        def _select_best_permuted_candidate(
            prev_partition: Optional[list[list[int]]],
            candidate_record: MappingRecord,
            base_logical_map: dict[int, tuple[int, int | None]],
        ) -> tuple[float, float, dict[int, tuple[int, int | None]], MappingRecord]:
            """
            对单个候选记录进行“小预算 permutation 搜索”，返回真实epairs最优方案。
            """
            perm_candidates = _build_perm_candidates(prev_partition, candidate_record, base_logical_map)

            best_loss = float("inf")
            best_delta = float("inf")
            best_map: dict[int, tuple[int, int | None]] = {}
            best_record: Optional[MappingRecord] = None

            for perm in perm_candidates:
                delta_loss, delta_epairs, next_map, applied_record = _evaluate_real_epairs_delta(
                    prev_partition,
                    candidate_record,
                    base_logical_map,
                    perm=perm,
                )
                if (delta_loss, delta_epairs) < (best_loss, best_delta):
                    best_loss = delta_loss
                    best_delta = delta_epairs
                    best_map = next_map
                    best_record = applied_record

            assert best_record is not None
            return best_loss, best_delta, best_map, best_record

        def _build_noise_aware_teledata_baseline() -> tuple[float, float, list[MappingRecord]]:
            baseline_records: list[MappingRecord] = []
            baseline_total_loss = 0.0
            baseline_epairs = 0.0
            prev_part: Optional[list[list[int]]] = None
            logical_map: dict[int, tuple[int, int | None]] = {}

            for rec in original_records:
                curr = copy.deepcopy(rec)
                delta_loss, delta_epairs, logical_map, best_curr = _select_best_permuted_candidate(
                    prev_part,
                    curr,
                    logical_map,
                )
                baseline_total_loss += delta_loss
                baseline_epairs += delta_epairs
                baseline_records.append(best_curr)
                prev_part = best_curr.partition

            return baseline_total_loss, baseline_epairs, baseline_records

        td_only_total_loss, td_only_real_epairs, td_only_records = _build_noise_aware_teledata_baseline()
        telegate_probe_positions = set(self._sample_budgeted_indices(0, K - 1, pos_probe_budget))
        log(
            f"[construct_noise_aware_hybrid_records] telegate_probe_positions={len(telegate_probe_positions)}/{K}"
        )

        for pos in range(K):
            pos_start_time = time.time()
            if pos in states and len(states[pos]) > beam_width:
                states[pos] = _prune(states[pos])

            current_states = states.get(pos, [])
            if not current_states:
                continue

            for base_total_loss, base_epairs, base_records, base_map in current_states:
                prev_partition = base_records[-1].partition if base_records else None

                # 选项 A：当前原子块保持 teledata
                td_record = copy.deepcopy(original_records[pos])
                delta_loss, delta_epairs, next_map, best_td_record = _select_best_permuted_candidate(
                    prev_partition,
                    td_record,
                    base_map,
                )
                states.setdefault(pos + 1, []).append(
                    (
                        base_total_loss + delta_loss,
                        base_epairs + delta_epairs,
                        base_records + [best_td_record],
                        next_map,
                    )
                )

                # 选项 B：仅在采样到的pos上展开telegate
                if pos in telegate_probe_positions:
                    start_layer = original_records[pos].layer_start
                    upper_end = min(K - 1, pos + max_merge_span - 1)
                    sampled_ends = self._sample_budgeted_indices(pos, upper_end, merge_probe_budget)

                    for end in sampled_ends:
                        end_layer = original_records[end].layer_end
                        telegate_result = self._try_generate_telegate(
                            ctx,
                            start_layer,
                            end_layer,
                            prev_partition,
                        )

                        tg_record = copy.deepcopy(telegate_result.records[0])
                        delta_loss, delta_epairs, next_map, best_tg_record = _select_best_permuted_candidate(
                            prev_partition,
                            tg_record,
                            base_map,
                        )

                        states.setdefault(end + 1, []).append(
                            (
                                base_total_loss + delta_loss,
                                base_epairs + delta_epairs,
                                base_records + [best_tg_record],
                                next_map,
                            )
                        )

            for prune_pos in range(pos + 1, min(K, pos + max_merge_span) + 1):
                if prune_pos in states and len(states[prune_pos]) > prune_trigger_ratio * beam_width:
                    states[prune_pos] = _prune(states[prune_pos])

            pos_end_time = time.time()
            log(f"[DEBUG] NoiseAware Position {pos} completed in {pos_end_time - pos_start_time} seconds")

        final_candidates = states.get(K, [])
        if len(final_candidates) > beam_width:
            final_candidates = _prune(final_candidates)
        if not final_candidates:
            raise RuntimeError("[ERROR] noise-aware beam search failed to produce any hybrid plan.")

        best_total_loss, best_real_epairs, best_records, _ = min(
            final_candidates,
            key=_state_key,
        )

        # 兜底保证：noise-aware hybrid输出不应比teledata-only更差（按(total_fidelity_loss, epairs)口径比较）。
        if (best_total_loss, best_real_epairs) > (td_only_total_loss, td_only_real_epairs):
            log(
                f"[noise_aware_hybrid_fallback] hybrid=(loss={best_total_loss}, epairs={best_real_epairs}) > "
                f"teledata_only=(loss={td_only_total_loss}, epairs={td_only_real_epairs}), fallback to teledata-only"
            )
            best_total_loss = td_only_total_loss
            best_real_epairs = td_only_real_epairs
            best_records = td_only_records

        result = MappingRecordList()
        result.records = best_records
        result.summarize_total_costs()

        ctx.telegate_optimized = True
        end_time = time.time()
        log(
            f"[construct_noise_aware_hybrid_records] best_key=(loss={best_total_loss}, real_epairs={best_real_epairs}), "
            f"best_epairs={result.total_costs.epairs}, best_cat_ents={result.total_costs.cat_ents}, "
            f"Time: {end_time - start_time} seconds"
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
        start_time = time.perf_counter()
        try:
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
        finally:
            ctx.get_ori_subc_calls += 1
            ctx.get_ori_subc_total_time += time.perf_counter() - start_time

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
        # _try_generate_telegate内部统一使用all-to-all副本作为代价评估网络。
        eval_network = ctx.network # self._get_all_to_all_eval_network(ctx)
        assert ctx.telegate_partitioner is not None
        telegate_record = ctx.telegate_partitioner.partition(
            circuit = sub_qc,
            network = eval_network,
            config = {
                "layer_start": ori_left,
                "layer_end": ori_right,
                "partition": prev_partition,
                "use_oee_init": bool(ctx.config.get("use_oee_init", True)),
                "iteration": int(ctx.config.get("iteration", 50)),
                "add_teledata_costs": False,
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