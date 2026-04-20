from typing import Any, Optional
import copy
import itertools
import time

import numpy as np

from qiskit import QuantumCircuit

from ..compiler import CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..compiler.compiler_utils import CommOp
from ..utils import Network
from ..utils import log
from .navi_compiler import CompilationContext
from .navi_hybrid import HybridSearchState, NAVIHybrid


class NAVIHybridDirectNoiseAware(NAVIHybrid):
    """
    将 direct noise-aware hybrid records 构造路径独立成单独编译器，
    便于与原始 NAVIHybrid 的 beam + mapper 流程并排比较。
    """

    compiler_id = "navihybriddirect"

    @property
    def name(self) -> str:
        return "NAVI Hybrid Direct"

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> MappingRecordList:
        return self._compile_impl(
            circuit,
            network,
            config,
            use_direct_noise_aware=True,
        )

    def _step_construct_direct_noise_aware_hybrid_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        直接噪声感知 hybrid 构造：
        - 保留 hybrid beam 的 records 构造流程；
        - 在构造阶段直接确定每条 record 的 mapping；
        - 中间搜索仅用近似指标做状态打分；
        - 最终真实成本由 compile() 外层的 Direct Mapper 统一重算。
        """
        start_time = time.time()
        blocks = ctx.subc_ranges
        K = len(blocks)

        beam_width = int(ctx.config.get("hybrid_beam_width", 1))
        max_merge_span = int(ctx.config.get("hybrid_max_merge_span", 12))
        merge_probe_budget = int(ctx.config.get("hybrid_merge_probe_budget", max_merge_span))
        pos_probe_budget = int(ctx.config.get("hybrid_pos_probe_budget", K))
        prune_trigger_ratio = int(ctx.config.get("hybrid_prune_trigger_ratio", 5))
        exact_perm_max_backends = int(
            ctx.config.get("direct_noise_aware_exact_perm_max_backends", 1)
        )
        beam_width = max(1, beam_width)
        max_merge_span = max(1, max_merge_span)
        merge_probe_budget = max(1, merge_probe_budget)
        pos_probe_budget = max(1, pos_probe_budget)
        prune_trigger_ratio = max(2, prune_trigger_ratio)
        exact_perm_max_backends = max(1, exact_perm_max_backends)

        log(
            f"[construct_direct_noise_aware_hybrid_records] blocks={K}, beam_width={beam_width}, "
            f"max_merge_span={max_merge_span}, merge_probe_budget={merge_probe_budget}, "
            f"pos_probe_budget={pos_probe_budget}, exact_perm_max_backends={exact_perm_max_backends}, "
            f"objective=(min estimated_total_fidelity_loss, estimated_epairs)"
        )
        setattr(ctx, "runtime_hybrid_beam_width", beam_width)

        if K == 0:
            return MappingRecordList()

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

        states: dict[int, list[HybridSearchState]] = {
            0: [
                HybridSearchState(
                    total_fidelity_loss=0.0,
                    epairs=0.0,
                    records=[],
                    logical_phy_map={},
                )
            ]
        }
        subcircuit_cache: dict[tuple[int, int], QuantumCircuit] = {}

        def _state_key(item: HybridSearchState) -> tuple[float, float]:
            return (item.total_fidelity_loss, item.epairs)

        def _partition_sig(partition: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(sorted(part)) for part in partition)

        def _apply_perm(partition: list[list[int]], perm: tuple[int, ...]) -> list[list[int]]:
            return [partition[i] for i in perm]

        def _invert_perm(perm: tuple[int, ...]) -> list[int]:
            inv = [-1 for _ in range(len(perm))]
            for physical, logical in enumerate(perm):
                inv[logical] = physical
            return inv

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

        def _build_prev_preferred_perm(
            prev_perm: Optional[tuple[int, ...]],
        ) -> tuple[int, ...]:
            if prev_perm is None:
                return tuple(range(n_backends))
            return tuple(prev_perm)

        def _get_prev_perm_from_records(records: list[MappingRecord]) -> tuple[int, ...]:
            if not records:
                return tuple(range(n_backends))

            extra_info = records[-1].extra_info or {}
            perm = extra_info.get("perm")
            if isinstance(perm, (list, tuple)):
                return tuple(int(x) for x in perm)
            return tuple(range(n_backends))

        def _build_pair_demand(
            prev_partition: Optional[list[list[int]]],
            curr_partition: list[list[int]],
            subcircuit: QuantumCircuit,
        ) -> list[list[float]]:
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
                    curr_pid = q_to_curr[q]
                    demand[prev_pid][curr_pid] += 1.0

            for inst in subcircuit:
                op = inst.operation
                if isinstance(op, CommOp):
                    src = int(op.src_qpu)
                    dst = int(op.dst_qpu)
                    weight = 2.0 if op.comm_type == "rtp" else 1.0
                    demand[src][dst] += weight
                    continue

                if len(inst.qubits) != 2:
                    continue
                q0 = subcircuit.qubits.index(inst.qubits[0])
                q1 = subcircuit.qubits.index(inst.qubits[1])
                p0 = q_to_curr[q0]
                p1 = q_to_curr[q1]
                demand[p0][p1] += 1.0
                demand[p1][p0] += 1.0

            return demand

        def _mapping_penalty(physical_src: int, physical_dst: int) -> float:
            if physical_src == physical_dst:
                return 0.0
            loss = float(ctx.network.move_fidelity_loss[physical_src][physical_dst])
            return loss

        def _estimate_perm_metrics(
            perm: tuple[int, ...],
            demand: list[list[float]],
            applied_partition: list[list[int]],
            subcircuit: QuantumCircuit,
        ) -> ExecCosts:
            logical_to_physical = _invert_perm(perm)
            est_costs = ExecCosts()
            q_to_qpu: dict[int, int] = {}

            for qpu_id, part in enumerate(applied_partition):
                for q in part:
                    q_to_qpu[q] = qpu_id

            for lsrc in range(n_backends):
                psrc = logical_to_physical[lsrc]
                for ldst in range(n_backends):
                    weight = demand[lsrc][ldst]
                    if weight <= 0:
                        continue
                    pdst = logical_to_physical[ldst]
                    if psrc == pdst:
                        continue
                    est_costs.epairs += int(weight)
                    est_costs.remote_hops += weight * ctx.network.Hops[psrc][pdst]
                    est_costs.remote_fidelity_loss += weight * _mapping_penalty(psrc, pdst)

            for inst in subcircuit:
                op = inst.operation
                if isinstance(op, CommOp) and op.comm_type == "cat":
                    est_costs.cat_ents += 1
                    dst_qpu = int(op.dst_qpu)
                    for gate_op in op.gate_list:
                        gate_name = gate_op.name
                        gate_error = ctx.network.backends[dst_qpu].gate_dict.get(gate_name, {}).get("gate_error_value")
                        if gate_error is None or np.isnan(gate_error):
                            continue
                        est_costs.local_gate_num += 1
                        est_costs.local_fidelity_loss += float(gate_error)
                        est_costs.local_fidelity *= (1 - float(gate_error))
                        est_costs.local_fidelity_log_sum += np.log(max(1e-12, 1 - float(gate_error)))
                    continue

                if isinstance(op, CommOp):
                    continue

                if len(inst.qubits) == 0:
                    continue

                global_qids = [subcircuit.qubits.index(qubit) for qubit in inst.qubits]
                qpu_ids = {q_to_qpu[q] for q in global_qids if q in q_to_qpu}
                if len(qpu_ids) != 1:
                    continue

                qpu_id = next(iter(qpu_ids))
                gate_name = op.name
                gate_error = ctx.network.backends[qpu_id].gate_dict.get(gate_name, {}).get("gate_error_value")
                if gate_error is None or np.isnan(gate_error):
                    continue
                est_costs.local_gate_num += 1
                est_costs.local_fidelity_loss += float(gate_error)
                est_costs.local_fidelity *= (1 - float(gate_error))
                est_costs.local_fidelity_log_sum += np.log(max(1e-12, 1 - float(gate_error)))

            return est_costs

        def _remap_commop_endpoints_by_perm(
            subcircuit: QuantumCircuit,
            perm: tuple[int, ...],
        ) -> QuantumCircuit:
            if len(perm) == 0:
                return subcircuit

            old_to_new_qpu: dict[int, int] = {}
            for new_qpu, old_qpu in enumerate(perm):
                old_to_new_qpu[int(old_qpu)] = int(new_qpu)

            for instruction in subcircuit:
                op = instruction.operation
                if not isinstance(op, CommOp):
                    continue
                op.src_qpu = old_to_new_qpu[op.src_qpu]
                op.dst_qpu = old_to_new_qpu[op.dst_qpu]

            return subcircuit

        def _prepare_subcircuit_for_perm(
            record: MappingRecord,
            perm: tuple[int, ...],
        ) -> QuantumCircuit:
            subcircuit = copy.deepcopy(_get_record_subcircuit(record))
            if record.extra_info is not None and "ops" in record.extra_info:
                subcircuit = _remap_commop_endpoints_by_perm(subcircuit, perm)
            return subcircuit

        def _state_sig(
            records: list[MappingRecord],
        ) -> tuple[Any, ...]:
            if not records:
                return tuple()
            last_partition = records[-1].partition
            return _partition_sig(last_partition)

        def _prune(candidates: list[HybridSearchState]) -> list[HybridSearchState]:
            best_by_sig: dict[tuple[Any, ...], HybridSearchState] = {}
            for item in candidates:
                sig = _state_sig(item.records)
                old = best_by_sig.get(sig)
                if old is None or _state_key(item) < _state_key(old):
                    best_by_sig[sig] = item

            pruned = sorted(best_by_sig.values(), key=_state_key)
            return pruned[:beam_width]

        all_perms: list[tuple[int, ...]] = []
        if n_backends <= exact_perm_max_backends:
            all_perms = list(itertools.permutations(range(n_backends)))
        else:
            log(
                f"[construct_direct_noise_aware_hybrid_records] skip exact perm enumeration "
                f"because n_backends={n_backends} > {exact_perm_max_backends}"
            )

        best_complete_key: list[tuple[float, float]] = [(float("inf"), float("inf"))]
        prune_stats = {
            "prefix_pruned": 0,
            "append_pruned": 0,
            "complete_updates": 0,
        }

        def _update_best_complete(key: tuple[float, float]) -> None:
            if key < best_complete_key[0]:
                old_key = best_complete_key[0]
                best_complete_key[0] = key
                prune_stats["complete_updates"] += 1
                log(
                    f"[direct_noise_aware_bound_update] old=(estimated_loss={old_key[0]}, estimated_epairs={old_key[1]}), "
                    f"new=(estimated_loss={key[0]}, estimated_epairs={key[1]})"
                )

        def _is_prunable_prefix(key: tuple[float, float]) -> bool:
            return key >= best_complete_key[0]

        def _append_state(
            next_pos: int,
            total_loss: float,
            epairs: float,
            records: list[MappingRecord],
            logical_map: dict[int, tuple[int, int | None]],
        ) -> None:
            key = (float(total_loss), float(epairs))
            next_state = HybridSearchState(
                total_fidelity_loss=key[0],
                epairs=key[1],
                records=records,
                logical_phy_map=logical_map,
            )
            if next_pos >= K:
                _update_best_complete(key)
                states.setdefault(K, []).append(next_state)
                return

            if _is_prunable_prefix(key):
                prune_stats["append_pruned"] += 1
                return

            states.setdefault(next_pos, []).append(next_state)

        def _score_perm(perm: tuple[int, ...], demand: list[list[float]]) -> float:
            logical_to_physical = _invert_perm(perm)
            total = 0.0
            for lsrc in range(n_backends):
                psrc = logical_to_physical[lsrc]
                for ldst in range(n_backends):
                    weight = demand[lsrc][ldst]
                    if weight <= 0:
                        continue
                    pdst = logical_to_physical[ldst]
                    total += weight * _mapping_penalty(psrc, pdst)
            return total

        def _select_direct_perm(
            prev_partition: Optional[list[list[int]]],
            curr_record: MappingRecord,
            prev_perm: tuple[int, ...],
        ) -> tuple[int, ...]:
            baseline_perm = _build_prev_preferred_perm(prev_perm)
            if not all_perms:
                return baseline_perm

            demand = _build_pair_demand(prev_partition, curr_record.partition, _get_record_subcircuit(curr_record))
            best_perm = baseline_perm
            best_score = _score_perm(baseline_perm, demand)

            for perm in all_perms:
                score = _score_perm(perm, demand)
                if score + 1e-12 < best_score:
                    best_score = score
                    best_perm = perm

            return best_perm

        def _solve_linear_assignment_dp(cost_matrix: list[list[float]]) -> tuple[int, ...]:
            n = len(cost_matrix)
            if n == 0:
                return tuple()

            full_mask = (1 << n) - 1
            dp = [float("inf")] * (1 << n)
            parent: list[tuple[int, int] | None] = [None] * (1 << n)
            dp[0] = 0.0

            for mask in range(1 << n):
                logical_idx = mask.bit_count()
                if logical_idx >= n:
                    continue

                base = dp[mask]
                if base == float("inf"):
                    continue

                for physical_idx in range(n):
                    if mask & (1 << physical_idx):
                        continue
                    next_mask = mask | (1 << physical_idx)
                    cand = base + float(cost_matrix[logical_idx][physical_idx])
                    if cand + 1e-12 < dp[next_mask]:
                        dp[next_mask] = cand
                        parent[next_mask] = (mask, physical_idx)

            if dp[full_mask] == float("inf"):
                raise RuntimeError("[assignment_dp] failed to find a complete assignment")

            logical_to_physical = [-1] * n
            mask = full_mask
            logical_idx = n - 1
            while mask:
                prev = parent[mask]
                if prev is None:
                    raise RuntimeError("[assignment_dp] broken backtrace")
                prev_mask, physical_idx = prev
                logical_to_physical[logical_idx] = physical_idx
                mask = prev_mask
                logical_idx -= 1

            perm = [-1] * n
            for logical_idx, physical_idx in enumerate(logical_to_physical):
                perm[physical_idx] = logical_idx

            return tuple(perm)

        def _select_direct_perm_assignment(
            prev_partition: Optional[list[list[int]]],
            curr_record: MappingRecord,
            prev_perm: tuple[int, ...],
        ) -> tuple[int, ...]:
            baseline_perm = _build_prev_preferred_perm(prev_perm)
            subcircuit = _get_record_subcircuit(curr_record)
            demand = _build_pair_demand(prev_partition, curr_record.partition, subcircuit)
            logical_to_prev_physical = _invert_perm(baseline_perm)

            continuity_weight = float(
                ctx.config.get("direct_noise_aware_assignment_continuity_weight", 1.0)
            )
            demand_weight = float(
                ctx.config.get("direct_noise_aware_assignment_demand_weight", 1.0)
            )

            physical_affinity = [0.0 for _ in range(n_backends)]
            for physical_idx in range(n_backends):
                total = 0.0
                for other_idx in range(n_backends):
                    if other_idx == physical_idx:
                        continue
                    total += _mapping_penalty(physical_idx, other_idx)
                physical_affinity[physical_idx] = total

            logical_pressure = [0.0 for _ in range(n_backends)]
            for logical_idx in range(n_backends):
                total = 0.0
                for other_idx in range(n_backends):
                    total += float(demand[logical_idx][other_idx])
                    total += float(demand[other_idx][logical_idx])
                logical_pressure[logical_idx] = total

            cost_matrix = [[0.0 for _ in range(n_backends)] for _ in range(n_backends)]
            for logical_idx in range(n_backends):
                prev_physical = logical_to_prev_physical[logical_idx]
                for physical_idx in range(n_backends):
                    continuity_cost = continuity_weight * _mapping_penalty(prev_physical, physical_idx)
                    traffic_cost = demand_weight * logical_pressure[logical_idx] * physical_affinity[physical_idx]
                    cost_matrix[logical_idx][physical_idx] = continuity_cost + traffic_cost

            return _solve_linear_assignment_dp(cost_matrix)

        def _estimate_candidate_with_perm(
            prev_partition: Optional[list[list[int]]],
            candidate_record: MappingRecord,
            base_logical_map: dict[int, tuple[int, int | None]],
            perm: tuple[int, ...],
        ) -> tuple[float, float, dict[int, tuple[int, int | None]], MappingRecord]:
            applied_record = copy.deepcopy(candidate_record)
            applied_record.partition = _apply_perm(applied_record.partition, perm)
            applied_record.costs = ExecCosts()
            applied_record.logical_phy_map = copy.deepcopy(base_logical_map)

            if applied_record.extra_info is None:
                applied_record.extra_info = {}
            applied_record.extra_info["perm"] = list(perm)

            subcircuit = _prepare_subcircuit_for_perm(candidate_record, perm)
            applied_record.extra_info["ops"] = subcircuit
            demand = _build_pair_demand(prev_partition, candidate_record.partition, subcircuit)
            est_costs = _estimate_perm_metrics(perm, demand, applied_record.partition, subcircuit)
            applied_record.costs = est_costs

            return (
                float(applied_record.costs.total_fidelity_loss),
                float(applied_record.costs.epairs),
                copy.deepcopy(base_logical_map),
                applied_record,
            )

        def _select_candidate_with_shared_perm(
            prev_partition: Optional[list[list[int]]],
            candidate_record: MappingRecord,
            base_logical_map: dict[int, tuple[int, int | None]],
            prev_perm: tuple[int, ...],
            shared_perm: tuple[int, ...],
        ) -> tuple[float, float, dict[int, tuple[int, int | None]], MappingRecord, tuple[int, ...]]:
            baseline_perm = _build_prev_preferred_perm(prev_perm)
            baseline_eval = _estimate_candidate_with_perm(
                prev_partition,
                candidate_record,
                base_logical_map,
                baseline_perm,
            )

            if shared_perm == baseline_perm:
                return (*baseline_eval, baseline_perm)

            shared_eval = _estimate_candidate_with_perm(
                prev_partition,
                candidate_record,
                base_logical_map,
                shared_perm,
            )
            if shared_eval[:2] <= baseline_eval[:2]:
                return (*shared_eval, shared_perm)
            return (*baseline_eval, baseline_perm)

        def _build_direct_noise_aware_teledata_baseline() -> tuple[float, float, list[MappingRecord]]:
            baseline_records: list[MappingRecord] = []
            baseline_total_loss = 0.0
            baseline_epairs = 0.0
            prev_part: Optional[list[list[int]]] = None
            logical_map: dict[int, tuple[int, int | None]] = {}
            prev_perm = tuple(range(n_backends))

            for rec in original_records:
                shared_perm = _select_direct_perm_assignment(prev_part, rec, prev_perm)
                delta_loss, delta_epairs, logical_map, best_curr, chosen_perm = _select_candidate_with_shared_perm(
                    prev_part,
                    rec,
                    logical_map,
                    prev_perm,
                    shared_perm,
                )
                baseline_total_loss += delta_loss
                baseline_epairs += delta_epairs
                baseline_records.append(best_curr)
                prev_part = best_curr.partition
                prev_perm = chosen_perm

            return baseline_total_loss, baseline_epairs, baseline_records

        td_only_total_loss, td_only_real_epairs, td_only_records = _build_direct_noise_aware_teledata_baseline()
        _update_best_complete((td_only_total_loss, td_only_real_epairs))
        telegate_probe_positions = set(self._sample_budgeted_indices(0, K - 1, pos_probe_budget))
        log(
            f"[construct_direct_noise_aware_hybrid_records] telegate_probe_positions={len(telegate_probe_positions)}/{K}, "
            f"initial_complete_bound=(estimated_loss={best_complete_key[0][0]}, estimated_epairs={best_complete_key[0][1]})"
        )

        for pos in range(K):
            pos_start_time = time.time()
            if pos in states and len(states[pos]) > beam_width:
                states[pos] = _prune(states[pos])

            current_states = states.get(pos, [])
            if not current_states:
                continue

            for state in current_states:
                if _is_prunable_prefix((state.total_fidelity_loss, state.epairs)):
                    prune_stats["prefix_pruned"] += 1
                    continue

                prev_partition = state.records[-1].partition if state.records else None
                pos_shared_perm = _select_direct_perm_assignment(
                    prev_partition,
                    original_records[pos],
                    _get_prev_perm_from_records(state.records),
                )

                td_record = copy.deepcopy(original_records[pos])
                delta_loss, delta_epairs, next_map, best_td_record, _ = _select_candidate_with_shared_perm(
                    prev_partition,
                    td_record,
                    state.logical_phy_map,
                    _get_prev_perm_from_records(state.records),
                    pos_shared_perm,
                )
                _append_state(
                    pos + 1,
                    state.total_fidelity_loss + delta_loss,
                    state.epairs + delta_epairs,
                    state.records + [best_td_record],
                    next_map,
                )

                if pos in telegate_probe_positions:
                    start_layer = original_records[pos].layer_start
                    upper_end = min(K - 1, pos + max_merge_span - 1)
                    sampled_ends = self._sample_budgeted_indices(pos, upper_end, merge_probe_budget)

                    if pos == 0 and (K - 1) not in sampled_ends:
                        sampled_ends.append(K - 1)
                        sampled_ends = sorted(set(sampled_ends))

                    for end in sampled_ends:
                        end_layer = original_records[end].layer_end
                        telegate_result = self._try_generate_telegate(
                            ctx,
                            start_layer,
                            end_layer,
                            prev_partition,
                        )
                        if not telegate_result.records:
                            continue

                        tg_record = copy.deepcopy(telegate_result.records[0])
                        delta_loss, delta_epairs, next_map, best_tg_record, _ = _select_candidate_with_shared_perm(
                            prev_partition,
                            tg_record,
                            state.logical_phy_map,
                            _get_prev_perm_from_records(state.records),
                            pos_shared_perm,
                        )
                        _append_state(
                            end + 1,
                            state.total_fidelity_loss + delta_loss,
                            state.epairs + delta_epairs,
                            state.records + [best_tg_record],
                            next_map,
                        )

            for prune_pos in range(pos + 1, min(K, pos + max_merge_span) + 1):
                if prune_pos in states and len(states[prune_pos]) > prune_trigger_ratio * beam_width:
                    states[prune_pos] = _prune(states[prune_pos])

            log(
                f"[DEBUG] DirectNoiseAware Position {pos} completed in "
                f"{time.time() - pos_start_time} seconds"
            )

        final_candidates = states.get(K, [])
        if len(final_candidates) > beam_width:
            final_candidates = _prune(final_candidates)
        if not final_candidates:
            raise RuntimeError("[ERROR] direct noise-aware beam search failed to produce any hybrid plan.")

        best_state = min(final_candidates, key=_state_key)
        best_total_loss = best_state.total_fidelity_loss
        best_real_epairs = best_state.epairs
        best_records = best_state.records
        if (best_total_loss, best_real_epairs) > (td_only_total_loss, td_only_real_epairs):
            log(
                f"[direct_noise_aware_hybrid_fallback] hybrid=(estimated_loss={best_total_loss}, estimated_epairs={best_real_epairs}) > "
                f"teledata_only=(estimated_loss={td_only_total_loss}, estimated_epairs={td_only_real_epairs}), fallback to teledata-only"
            )
            best_total_loss = td_only_total_loss
            best_real_epairs = td_only_real_epairs
            best_records = td_only_records

        result = MappingRecordList()
        result.records = best_records
        result.summarize_total_costs()
        ctx.hybrid_records = result
        ctx.final_records = result
        ctx.telegate_optimized = True

        log(
            f"[construct_direct_noise_aware_hybrid_records] best_key=(estimated_loss={best_total_loss}, estimated_epairs={best_real_epairs}), "
            f"estimated_result_epairs={result.total_costs.epairs}, estimated_cat_ents={result.total_costs.cat_ents}, "
            f"prune_stats={prune_stats}, "
            f"Time: {time.time() - start_time} seconds"
        )
        return result
