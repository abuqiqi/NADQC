from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from .compiler_utils import CommOp, CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..utils import Network


@dataclass(frozen=True)
class EvaluationPolicy:
    """
    Final evaluation policy for replaying compiled mapping records.

    Remote communication cost and continuous initial layout are always included
    by the evaluator. This policy controls which local operations are routed
    through the backend coupling map and whether communication-introduced local
    primitives are included.
    """
    name: str
    route_payload_gates: bool = True # 原始线路上的量子门要route，无论是comp-comp还是cat-ent的comm-comp
    route_comm_gates: bool = True # comm额外引入的通信本地门要route
    route_entanglement_swap: bool = False

    strict_flush_on_remote: bool = True
    flush_each_comm_gate: bool = False
    fill_initial_layout: bool = False
    optimization_level: int = 3

    @staticmethod
    def local_all_to_all() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="local_all_to_all",
            route_payload_gates=False,
            route_comm_gates=False,
            route_entanglement_swap=False
        )

    @staticmethod
    def comm_to_all() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="comm_to_all",
            route_payload_gates=True,
            route_comm_gates=False,
            route_entanglement_swap=False
        )

    @staticmethod
    def full_realistic() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="full_realistic",
            route_payload_gates=True,
            route_comm_gates=True,
            route_entanglement_swap=True
        )


class LocalGateKind(str, Enum):
    """
    Source category for a local gate introduced during replay.

    Policy decisions should be made from this category, not from gate name alone.
    """
    PAYLOAD = "payload"
    COMM_PRIMITIVE = "comm_primitive"
    ENTANGLEMENT_SWAP = "entanglement_swap"


@dataclass
class RuntimeLocation:
    """
    Current physical runtime location of one logical qubit.

    logical_pos tracks resident logical qubits on computation slots. Communication
    slots are tracked separately by EvaluationState.comm_phy_map.
    """
    qpu_id: int
    local_slot: int
    physical_slot: int | None = None


@dataclass
class EvaluationState:
    """
    Mutable replay state carried across records.

    This is intentionally local to MappingEvaluator. MappingRecord keeps the old
    public logical_phy_map/comm_phy_map representation; this state gives the new
    evaluator enough detail to model TP/CommOp behavior cleanly.
    """
    costs: ExecCosts = field(default_factory=ExecCosts)
    logical_pos: dict[int, RuntimeLocation] = field(default_factory=dict)
    comm_phy_map: dict[int, list[int | None]] = field(default_factory=dict)
    routed_buffers: list[QuantumCircuit] = field(default_factory=list)


class MappingEvaluator:
    """
    Replay a finished mapping strategy under a selected final evaluation policy.

    This class should not choose or modify the mapping strategy itself. It only
    walks the existing MappingRecordList in order and recomputes costs.
    """

    def __init__(self):
        # Keep evaluator stateless. All experiment conditions should be explicit
        # evaluate(...) arguments so repeated runs cannot leak state.
        pass

    def evaluate(
        self,
        mapping_record_list: MappingRecordList,
        circuit: QuantumCircuit,
        circuit_layers: list[Any],
        network: Network,
        policy: EvaluationPolicy,
    ) -> MappingRecordList:
        """
        Recompute costs for an existing MappingRecordList using policy.

        Input/output intentionally mirror Mapper._reevaluate_mapping_record_list:
        the input MappingRecordList is updated in place and returned.

        High-level replay order:
        1. Initialize the logical-to-physical state from the first record.
        2. Keep EvaluationState as the authoritative runtime state throughout
           this evaluate() call. MappingRecord fields are output snapshots.
        3. For each following record, first evaluate teledata between previous
           and current partitions.
        4. Evaluate the record body:
           - use record.extra_info["ops"] when present, e.g. AutoComm/NAVI
             telegate records with explicit CommOps;
           - otherwise slice the original circuit by layer_start/layer_end.
        5. Apply policy when accounting local routing:
           - route_payload_gates controls original/payload gates;
           - route_comm_gates controls communication-introduced local gates;
           - remote communication cost is always included.
        6. Summarize total costs before returning.
        """
        state: Optional[EvaluationState] = None

        for t, record in enumerate(mapping_record_list.records):
            if t == 0:
                state = self._initialize_state_from_partition(record.partition, network)
            else:
                assert state is not None

            # Start each replayed record from a clean cost object. This avoids
            # double-counting costs accumulated during compilation/mapping, but
            # keep logical_pos/comm_phy_map continuous across records.
            self._start_record_replay(state, record.partition, network)

            subcircuit = self._get_record_subcircuit(
                record,
                circuit,
                circuit_layers,
            )

            if t != 0:
                state = self._evaluate_teledata(
                    record.partition,
                    network,
                    policy,
                    state,
                )

            state = self._evaluate_record_body(
                record,
                subcircuit,
                network,
                policy,
                state,
            )
            self._commit_state_to_record(record, state)

        mapping_record_list.summarize_total_costs()
        return mapping_record_list

    def _start_record_replay(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
    ) -> None:
        """
        Reset per-record replay state without reconstructing logical positions.

        logical_pos and comm_phy_map are the continuous runtime state. costs and
        routed_buffers are local to the record currently being evaluated.
        """
        self._validate_partition_capacity(partition, network)
        state.costs = ExecCosts()
        state.routed_buffers = self._new_routed_buffers(partition, network)

    def _initialize_state_from_partition(
        self,
        partition: list[list[int]],
        network: Network,
    ) -> EvaluationState:
        """
        Build the initial runtime state for the first mapping record.

        Computation slots follow the partition order. Physical slots may later be
        refined by routed transpilation and written back to MappingRecord.
        """
        self._validate_partition_capacity(partition, network)

        logical_pos: dict[int, RuntimeLocation] = {}
        for qpu_id, group in enumerate(partition):
            for slot, logical_qid in enumerate(group):
                logical_pos[int(logical_qid)] = RuntimeLocation(
                    qpu_id=int(qpu_id),
                    local_slot=int(slot),
                    physical_slot=None,
                )

        comm_phy_map = self._normalize_comm_phy_map({}, network)
        routed_buffers = self._new_routed_buffers(partition, network)
        return EvaluationState(
            logical_pos=logical_pos,
            comm_phy_map=comm_phy_map,
            routed_buffers=routed_buffers,
        )

    def _validate_partition_capacity(
        self,
        partition: list[list[int]],
        network: Network,
    ) -> None:
        if len(partition) != network.num_backends:
            raise ValueError(
                f"[EVALUATOR] partition/backend size mismatch: "
                f"partition={len(partition)}, backends={network.num_backends}"
            )

        backend_sizes = getattr(network, "backend_sizes", [])
        for qpu_id, group in enumerate(partition):
            if qpu_id < len(backend_sizes) and len(group) > int(backend_sizes[qpu_id]):
                raise ValueError(
                    f"[EVALUATOR] QPU {qpu_id} partition size exceeds computation capacity: "
                    f"size={len(group)}, capacity={backend_sizes[qpu_id]}"
                )

    def _normalize_comm_phy_map(
        self,
        comm_phy_map: dict[Any, list[int | None]],
        network: Network,
    ) -> dict[int, list[int | None]]:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        normalized: dict[int, list[int | None]] = {}
        for qpu_id in range(network.num_backends):
            saved = comm_phy_map.get(qpu_id, comm_phy_map.get(str(qpu_id), []))
            saved = list(saved) if saved is not None else []
            slots: list[int | None] = []
            for i in range(reserve):
                value = saved[i] if i < len(saved) else None
                slots.append(int(value) if value is not None else None)
            normalized[qpu_id] = slots
        return normalized

    def _commit_state_to_record(
        self,
        record: MappingRecord,
        state: EvaluationState,
    ) -> None:
        """
        Project internal replay state back to MappingRecord's public fields.
        """
        record.costs = state.costs
        record.logical_phy_map = {
            int(q): (int(loc.qpu_id), loc.physical_slot)
            for q, loc in state.logical_pos.items()
        }
        record.comm_phy_map = {
            int(qpu_id): list(phy_slots)
            for qpu_id, phy_slots in state.comm_phy_map.items()
        }

    def _new_routed_buffers(
        self,
        partition: list[list[int]],
        network: Network,
    ) -> list[QuantumCircuit]:
        """
        Allocate per-QPU buffers for gates that policy says must be routed.

        Unrouted gates never enter these buffers; their cost is accumulated
        directly under local all-to-all assumptions.
        """
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        return [
            QuantumCircuit(len(group) + reserve)
            for group in partition
        ]

    def _get_record_subcircuit(
        self,
        record: MappingRecord,
        circuit: QuantumCircuit,
        circuit_layers: list[Any],
    ) -> QuantumCircuit:
        """
        Pick the circuit fragment represented by one mapping record.

        Records produced by telegate partitioners or AutoComm can already carry
        an executable op list in extra_info["ops"]. Other records are layer
        ranges over the original circuit and must be sliced here.
        """
        if record.extra_info is not None and "ops" in record.extra_info:
            return record.extra_info["ops"]

        if record.mapping_type == "cat":
            raise ValueError("[EVALUATOR] cat-type record requires extra_info['ops'].")

        return CompilerUtils.get_subcircuit_by_level(
            num_qubits=circuit.num_qubits,
            circuit=circuit,
            circuit_layers=circuit_layers,
            layer_start=record.layer_start,
            layer_end=record.layer_end,
        )

    def _evaluate_teledata(
        self,
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        """
        Evaluate partition transition cost.

        This method must not call CompilerUtils.evaluate_teledata_with_local.

        The migration scheduling follows the old teledata flow:
        1. Pair bidirectional moves first.
        2. Resolve longer directed cycles.
        3. Charge remaining one-way moves.

        The cost model differs from the old helper: no remote swap primitive is
        used. A bidirectional pair is charged as two one-way moves, while still
        exchanging physical slots so the next local routing pass has a concrete
        layout whenever the transfer is structurally paired.

        TODO:
        - add local landing/entanglement-swap gates under
          policy.route_entanglement_swap.
        """
        current_locations = self._partition_locations(target_partition)
        previous_logical_ids = set(state.logical_pos.keys())
        current_logical_ids = set(current_locations.keys())
        if previous_logical_ids != current_logical_ids:
            raise ValueError(
                f"[EVALUATOR] partition logical qubits changed across records: "
                f"prev_only={sorted(previous_logical_ids - current_logical_ids)}, "
                f"curr_only={sorted(current_logical_ids - previous_logical_ids)}"
            )

        graph = nx.DiGraph()
        graph.add_nodes_from(range(network.num_backends))

        old_locations = {
            int(q): RuntimeLocation(
                qpu_id=int(loc.qpu_id),
                local_slot=int(loc.local_slot),
                physical_slot=loc.physical_slot,
            )
            for q, loc in state.logical_pos.items()
        }
        final_physical: dict[int, int | None] = {
            int(q): loc.physical_slot
            for q, loc in old_locations.items()
        }

        for logical_qid, (target_qpu, _target_local_slot) in current_locations.items():
            source_qpu = old_locations[logical_qid].qpu_id
            target_qpu = int(target_qpu)
            if source_qpu == target_qpu:
                continue

            if graph.has_edge(target_qpu, source_qpu):
                reverse_qubits = graph[target_qpu][source_qpu]["qubits"]
                if len(reverse_qubits) > 0:
                    swap_partner = int(reverse_qubits.pop(0))
                    graph[target_qpu][source_qpu]["weight"] -= 1
                    if graph[target_qpu][source_qpu]["weight"] == 0:
                        graph.remove_edge(target_qpu, source_qpu)

                    state.costs = CompilerUtils.update_remote_move_costs(
                        state.costs,
                        source_qpu,
                        target_qpu,
                        1,
                        network,
                    )
                    state.costs = CompilerUtils.update_remote_move_costs(
                        state.costs,
                        target_qpu,
                        source_qpu,
                        1,
                        network,
                    )

                    final_physical[logical_qid] = old_locations[swap_partner].physical_slot
                    final_physical[swap_partner] = old_locations[logical_qid].physical_slot
                    continue

            if graph.has_edge(source_qpu, target_qpu):
                graph[source_qpu][target_qpu]["qubits"].append(logical_qid)
                graph[source_qpu][target_qpu]["weight"] += 1
            else:
                graph.add_edge(
                    source_qpu,
                    target_qpu,
                    weight=1,
                    qubits=[logical_qid],
                )

        cycles_by_length: dict[int, list[list[int]]] = defaultdict(list)
        for cycle in nx.simple_cycles(graph):
            length = len(cycle)
            if length >= 3:
                cycles_by_length[length].append([int(qpu_id) for qpu_id in cycle])

        for length in sorted(cycles_by_length.keys()):
            for cycle in cycles_by_length[length]:
                min_weight: int | None = None
                valid = True
                for i in range(length):
                    source_qpu = cycle[i]
                    target_qpu = cycle[(i + 1) % length]
                    if not graph.has_edge(source_qpu, target_qpu):
                        valid = False
                        break
                    weight = int(graph[source_qpu][target_qpu]["weight"])
                    min_weight = weight if min_weight is None else min(min_weight, weight)

                if not valid or min_weight is None or min_weight <= 0:
                    continue
                move_count = int(min_weight)

                for _ in range(move_count):
                    cycle_qubits: list[int] = []
                    for i in range(length):
                        source_qpu = cycle[i]
                        target_qpu = cycle[(i + 1) % length]
                        qubit = int(graph[source_qpu][target_qpu]["qubits"].pop(0))
                        cycle_qubits.append(qubit)

                    first_qubit = cycle_qubits[0]
                    first_physical_slot = old_locations[first_qubit].physical_slot
                    for i in range(length - 1):
                        curr_qubit = cycle_qubits[i]
                        next_qubit = cycle_qubits[i + 1]
                        final_physical[curr_qubit] = old_locations[next_qubit].physical_slot
                    final_physical[cycle_qubits[-1]] = first_physical_slot

                for i in range(length):
                    source_qpu = cycle[i]
                    target_qpu = cycle[(i + 1) % length]
                    graph[source_qpu][target_qpu]["weight"] -= move_count
                    if graph[source_qpu][target_qpu]["weight"] == 0:
                        graph.remove_edge(source_qpu, target_qpu)

                    state.costs = CompilerUtils.update_remote_move_costs(
                        state.costs,
                        source_qpu,
                        target_qpu,
                        move_count,
                        network,
                    )

        for source_qpu, target_qpu, data in list(graph.edges(data=True)):
            qubits_to_move = [int(q) for q in data["qubits"]]
            for logical_qid in qubits_to_move:
                final_physical[logical_qid] = None
            state.costs = CompilerUtils.update_remote_move_costs(
                state.costs,
                int(source_qpu),
                int(target_qpu),
                int(data["weight"]),
                network,
            )

        for logical_qid, (target_qpu, target_local_slot) in current_locations.items():
            state.logical_pos[logical_qid] = RuntimeLocation(
                qpu_id=int(target_qpu),
                local_slot=int(target_local_slot),
                physical_slot=final_physical[logical_qid],
            )

        return state

    def _partition_locations(
        self,
        partition: list[list[int]],
    ) -> dict[int, tuple[int, int]]:
        """
        Return logical_qid -> (qpu_id, local_slot) for a partition.
        """
        locations: dict[int, tuple[int, int]] = {}
        for qpu_id, group in enumerate(partition):
            for local_slot, logical_qid in enumerate(group):
                logical_qid = int(logical_qid)
                if logical_qid in locations:
                    raise ValueError(
                        f"[EVALUATOR] logical qubit appears in multiple partition groups: q{logical_qid}"
                    )
                locations[logical_qid] = (int(qpu_id), int(local_slot))
        return locations

    def _evaluate_record_body(
        self,
        record: MappingRecord,
        subcircuit: QuantumCircuit,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        """
        Evaluate local gates, telegate events, and CommOp bodies for one record.

        This method must not call
        CompilerUtils.evaluate_local_and_telegate_with_cat.

        Intended implementation:
        - Normal in-QPU gates are PAYLOAD gates.
        - Cross-QPU native gates become synthetic communication events whose body
          gate is PAYLOAD.
        - CommOp.gate_list entries are PAYLOAD gates executed at the destination
          using the source qubit's runtime comm slot when applicable.
        - CAT/TP/RTP helper cx/h/landing operations are COMM_PRIMITIVE gates.
        - PAYLOAD routing follows policy.route_payload_gates.
        - COMM_PRIMITIVE routing follows policy.route_comm_gates.
        - Remote communication costs are always included.
        """
        for instruction in subcircuit:
            op = instruction.operation
            global_qids = [
                subcircuit.qubits.index(qubit)
                for qubit in instruction.qubits
            ]

            if isinstance(op, CommOp):
                state = self._process_commop(record.partition, op, network, policy, state)
                continue

            involved_qpus = {state.logical_pos[q].qpu_id for q in global_qids}
            if len(involved_qpus) == 1:
                qpu_id = next(iter(involved_qpus))
                local_slots = [state.logical_pos[q].local_slot for q in global_qids]
                state = self._add_local_gate(
                    state,
                    qpu_id=qpu_id,
                    gate=op,
                    local_slots=local_slots,
                    kind=LocalGateKind.PAYLOAD,
                    network=network,
                    policy=policy,
                )
            else:
                state = self._process_synthetic_telegate(
                    record,
                    op,
                    global_qids,
                    network,
                    policy,
                    state,
                )

        return self._flush_local_buffers(state, record.partition, network, policy)

    def _process_commop(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        """
        Replay an explicit AutoComm/NAVI CommOp.

        CAT/RTP/TP should be implemented with a shared state machine:
        - allocate or reuse communication slots;
        - add remote move cost;
        - add protocol local gates as COMM_PRIMITIVE;
        - add gate_list gates as PAYLOAD;
        - update source runtime location for one-way TP, but restore it for RTP.
        """
        # src_qpu, dst_qpu = self._resolve_comm_qpu_endpoints(state, comm_op)
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)

        if policy.strict_flush_on_remote:
            state = self._flush_local_buffers(state, partition, network, policy)

        remote_loss_before = state.costs.remote_fidelity_loss
        remote_log_before = state.costs.remote_fidelity_log_sum

        state.costs.comm_block_events += 1
        state.costs.payload_gate_num += len(comm_op.gate_list)

        if comm_op.comm_type == "cat":
            state.costs = CompilerUtils.update_remote_move_costs(
                state.costs,
                src_qpu,
                dst_qpu,
                1,
                network,
            )
            state.costs.cat_ents += 1

            src_comm_slot = self._comm_local_slot(partition, network, src_qpu, 0)
            dst_comm_slot = self._comm_local_slot(partition, network, dst_qpu, 0)
            src_payload_slot = state.logical_pos[comm_op.source_qubit].local_slot

            state = self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=Gate("cx", 2, []),
                local_slots=[src_payload_slot, src_comm_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._append_comm_payload_gate_list(
                state,
                partition,
                comm_op,
                dst_qpu,
                dst_comm_slot,
                network,
                policy,
            )
            state = self._add_local_gate(
                state,
                qpu_id=dst_qpu,
                gate=Gate("h", 1, []),
                local_slots=[dst_comm_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )

        elif comm_op.comm_type == "rtp":
            state.costs = CompilerUtils.update_remote_move_costs(
                state.costs,
                src_qpu,
                dst_qpu,
                2,
                network,
            )

            src_comm_slot = self._comm_local_slot(partition, network, src_qpu, 0)
            dst_comm_slot = self._comm_local_slot(partition, network, dst_qpu, 0)
            dst_return_slot = self._comm_local_slot(partition, network, dst_qpu, 1)
            src_payload_slot = state.logical_pos[comm_op.source_qubit].local_slot

            state = self._add_local_gate( # src comp, src comm -> dst comm0
                state,
                qpu_id=src_qpu,
                gate=Gate("cx", 2, []),
                local_slots=[src_payload_slot, src_comm_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=Gate("h", 1, []),
                local_slots=[src_payload_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._append_comm_payload_gate_list(
                state,
                partition,
                comm_op,
                dst_qpu,
                dst_comm_slot,
                network,
                policy,
            )
            state = self._add_local_gate( # dst comm0, dst comm1 -> src comm0
                state,
                qpu_id=dst_qpu,
                gate=Gate("cx", 2, []),
                local_slots=[dst_comm_slot, dst_return_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._add_local_gate(
                state,
                qpu_id=dst_qpu,
                gate=Gate("h", 1, []),
                local_slots=[dst_comm_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=Gate("swap", 2, []),
                local_slots=[src_comm_slot, src_payload_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )

        elif comm_op.comm_type == "tp":
            state.costs = CompilerUtils.update_remote_move_costs(
                state.costs,
                src_qpu,
                dst_qpu,
                1,
                network,
            )

            src_comm_slot = self._comm_local_slot(partition, network, src_qpu, 0)
            dst_comm_slot = self._comm_local_slot(partition, network, dst_qpu, 0)
            src_payload_slot = state.logical_pos[comm_op.source_qubit].local_slot

            state = self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=Gate("cx", 2, []),
                local_slots=[src_payload_slot, src_comm_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=Gate("h", 1, []),
                local_slots=[src_payload_slot],
                kind=LocalGateKind.COMM_PRIMITIVE,
                network=network,
                policy=policy,
            )
            state = self._append_comm_payload_gate_list(
                state,
                partition,
                comm_op,
                dst_qpu,
                dst_comm_slot,
                network,
                policy,
            )
            if len(comm_op.gate_list) == 0:
                home_loc = state.logical_pos[comm_op.source_qubit]
                if int(home_loc.qpu_id) != int(dst_qpu):
                    raise RuntimeError(
                        "[EVALUATOR] Empty TP return block destination does not match source home QPU: "
                        f"source={comm_op.source_qubit}, home_qpu={home_loc.qpu_id}, dst_qpu={dst_qpu}"
                    )
                state = self._add_local_gate(
                    state,
                    qpu_id=dst_qpu,
                    gate=Gate("swap", 2, []),
                    local_slots=[dst_comm_slot, int(home_loc.local_slot)],
                    kind=LocalGateKind.COMM_PRIMITIVE,
                    network=network,
                    policy=policy,
                )

        else:
            raise ValueError(f"Unsupported CommOp type: {comm_op.comm_type}")

        remote_loss_delta = state.costs.remote_fidelity_loss - remote_loss_before
        remote_log_delta = state.costs.remote_fidelity_log_sum - remote_log_before
        state.costs.comm_block_remote_fidelity_loss += remote_loss_delta
        state.costs.comm_block_remote_fidelity_log_sum += remote_log_delta

        if policy.strict_flush_on_remote:
            state = self._flush_local_buffers(state, partition, network, policy)

        return state

    def _resolve_comm_qpu_endpoints(
        self,
        state: EvaluationState,
        comm_op: CommOp,
    ) -> tuple[int, int]:
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)

        source_loc = state.logical_pos.get(comm_op.source_qubit)
        if source_loc is None:
            raise RuntimeError(
                "[EVALUATOR] CommOp source qubit missing from runtime state: "
                f"source={comm_op.source_qubit}, metadata_src={comm_op.src_qpu}"
            )
        if int(source_loc.qpu_id) != src_qpu:
            raise RuntimeError(
                "[EVALUATOR] CommOp src_qpu metadata inconsistent with runtime state: "
                f"source={comm_op.source_qubit}, metadata_src={src_qpu}, "
                f"runtime_src={source_loc.qpu_id}"
            )

        dst_candidates: set[int] = set()
        for logical_qid in comm_op.involved_qubits:
            if logical_qid == comm_op.source_qubit:
                continue
            loc = state.logical_pos.get(logical_qid)
            if loc is None:
                raise RuntimeError(
                    "[EVALUATOR] CommOp involved qubit missing from runtime state: "
                    f"source={comm_op.source_qubit}, logical_qid={logical_qid}, "
                    f"involved={comm_op.involved_qubits}"
                )
            dst_candidates.add(int(loc.qpu_id))

        if len(dst_candidates) == 1:
            runtime_dst = next(iter(dst_candidates))
            if runtime_dst != dst_qpu:
                raise RuntimeError(
                    "[EVALUATOR] CommOp dst_qpu metadata inconsistent with runtime state: "
                    f"source={comm_op.source_qubit}, involved={comm_op.involved_qubits}, "
                    f"metadata_dst={dst_qpu}, runtime_dst={runtime_dst}"
                )

        return src_qpu, dst_qpu

    def _comm_local_slot(
        self,
        partition: list[list[int]],
        network: Network,
        qpu_id: int,
        offset: int,
    ) -> int:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        if offset >= reserve:
            raise RuntimeError(
                f"[EVALUATOR] CommOp requires communication slot offset {offset}, "
                f"but QPU {qpu_id} only reserves {reserve} comm slots."
            )
        return len(partition[qpu_id]) + int(offset)

    def _append_comm_payload_gate_list(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        comm_op: CommOp,
        dst_qpu: int,
        source_dst_comm_slot: int,
        network: Network,
        policy: EvaluationPolicy,
    ) -> EvaluationState:
        for gate_op in comm_op.gate_list:
            global_lqids = getattr(gate_op, "_global_lqids", None)
            if global_lqids is None:
                raise RuntimeError(
                    f"[EVALUATOR] CommOp gate_list gate missing _global_lqids metadata: gate={gate_op}"
                )

            local_slots: list[int] = []
            gate_qpu = dst_qpu
            global_lqids = [int(logical_qid) for logical_qid in global_lqids]

            if comm_op.source_qubit in global_lqids:
                for logical_qid in global_lqids:
                    if logical_qid == comm_op.source_qubit:
                        local_slots.append(int(source_dst_comm_slot)) # 加到dst comm slot
                        continue

                    loc = state.logical_pos[logical_qid]
                    if int(loc.qpu_id) != int(dst_qpu):
                        raise RuntimeError(
                            "[EVALUATOR] CommOp payload gate contains non-source qubit outside dst QPU: "
                            f"gate={gate_op}, logical_qid={logical_qid}, "
                            f"runtime_qpu={loc.qpu_id}, dst_qpu={dst_qpu}"
                        )
                    local_slots.append(int(loc.local_slot)) # dst comp slot
            else: # 混进来的本地量子门
                gate_qpus: set[int] = set()
                for logical_qid in global_lqids:
                    loc = state.logical_pos[logical_qid]
                    gate_qpus.add(int(loc.qpu_id))
                    local_slots.append(int(loc.local_slot))

                if len(gate_qpus) != 1:
                    raise RuntimeError(
                        "[EVALUATOR] CommOp packed local gate spans multiple runtime QPUs: "
                        f"gate={gate_op}, logical_qids={global_lqids}, "
                        f"runtime_qpus={sorted(gate_qpus)}"
                    )
                gate_qpu = next(iter(gate_qpus))

            state = self._add_local_gate(
                state,
                qpu_id=gate_qpu,
                gate=gate_op,
                local_slots=local_slots,
                kind=LocalGateKind.PAYLOAD,
                network=network,
                policy=policy,
            )

            if policy.flush_each_comm_gate:
                state = self._flush_local_buffers(state, partition, network, policy)

        return state

    def _process_synthetic_telegate(
        self,
        record: MappingRecord,
        gate: Gate,
        global_qids: list[int],
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        """
        Replay a native cross-QPU gate that was not already wrapped as CommOp.

        This should synthesize the same abstract communication style used for
        direct telegate evaluation, while still tagging the actual remote gate as
        PAYLOAD and helper operations as COMM_PRIMITIVE.
        """
        if len(global_qids) != 2:
            if policy.strict_flush_on_remote:
                state = self._flush_local_buffers(state, record.partition, network, policy)

            remote_loss_before = state.costs.remote_fidelity_loss
            remote_log_before = state.costs.remote_fidelity_log_sum
            state.costs.telegate_exec_events += 1

            for idx in range(len(global_qids) - 1):
                q1 = int(global_qids[idx])
                q2 = int(global_qids[idx + 1])
                p1 = int(state.logical_pos[q1].qpu_id)
                p2 = int(state.logical_pos[q2].qpu_id)
                if p1 != p2:
                    state.costs = CompilerUtils.update_remote_move_costs(
                        state.costs,
                        p1,
                        p2,
                        1,
                        network,
                    )

            remote_loss_delta = state.costs.remote_fidelity_loss - remote_loss_before
            remote_log_delta = state.costs.remote_fidelity_log_sum - remote_log_before
            state.costs.telegate_exec_remote_fidelity_loss += remote_loss_delta
            state.costs.telegate_exec_remote_fidelity_log_sum += remote_log_delta

            if policy.strict_flush_on_remote:
                state = self._flush_local_buffers(state, record.partition, network, policy)
            return state

        src_global = int(global_qids[0])
        dst_global = int(global_qids[1])
        src_qpu = int(state.logical_pos[src_global].qpu_id)
        dst_qpu = int(state.logical_pos[dst_global].qpu_id)
        if src_qpu == dst_qpu:
            return self._add_local_gate(
                state,
                qpu_id=src_qpu,
                gate=gate,
                local_slots=[
                    state.logical_pos[src_global].local_slot,
                    state.logical_pos[dst_global].local_slot,
                ],
                kind=LocalGateKind.PAYLOAD,
                network=network,
                policy=policy,
            )

        if policy.strict_flush_on_remote:
            state = self._flush_local_buffers(state, record.partition, network, policy)

        remote_loss_before = state.costs.remote_fidelity_loss
        remote_log_before = state.costs.remote_fidelity_log_sum

        state.costs.telegate_exec_events += 1
        state.costs.payload_gate_num += 1
        state.costs = CompilerUtils.update_remote_move_costs(
            state.costs,
            src_qpu,
            dst_qpu,
            1,
            network,
        )
        state.costs.cat_ents += 1

        src_comm_slot = self._comm_local_slot(record.partition, network, src_qpu, 0)
        dst_comm_slot = self._comm_local_slot(record.partition, network, dst_qpu, 0)
        src_payload_slot = state.logical_pos[src_global].local_slot
        dst_payload_slot = state.logical_pos[dst_global].local_slot

        state = self._add_local_gate(
            state,
            qpu_id=src_qpu,
            gate=Gate("cx", 2, []),
            local_slots=[src_payload_slot, src_comm_slot],
            kind=LocalGateKind.COMM_PRIMITIVE,
            network=network,
            policy=policy,
        )
        state = self._add_local_gate(
            state,
            qpu_id=dst_qpu,
            gate=gate,
            local_slots=[dst_comm_slot, dst_payload_slot],
            kind=LocalGateKind.PAYLOAD,
            network=network,
            policy=policy,
        )
        state = self._add_local_gate(
            state,
            qpu_id=dst_qpu,
            gate=Gate("h", 1, []),
            local_slots=[dst_comm_slot],
            kind=LocalGateKind.COMM_PRIMITIVE,
            network=network,
            policy=policy,
        )

        remote_loss_delta = state.costs.remote_fidelity_loss - remote_loss_before
        remote_log_delta = state.costs.remote_fidelity_log_sum - remote_log_before
        state.costs.telegate_exec_remote_fidelity_loss += remote_loss_delta
        state.costs.telegate_exec_remote_fidelity_log_sum += remote_log_delta

        if policy.strict_flush_on_remote:
            state = self._flush_local_buffers(state, record.partition, network, policy)
        return state

    def _add_local_gate(
        self,
        state: EvaluationState,
        qpu_id: int,
        gate: Gate,
        local_slots: Sequence[int | None],
        kind: LocalGateKind,
        network: Network,
        policy: EvaluationPolicy,
    ) -> EvaluationState:
        """
        Add one local gate according to policy.

        Routed gates go into state.routed_buffers[qpu_id] and are charged during
        _flush_local_buffers. Unrouted gates are charged immediately under the
        local all-to-all assumption.
        """
        route_gate = self._should_route_local_gate(kind, policy)
        if route_gate:
            self._append_routed_gate(state, qpu_id, gate, local_slots)
        else:
            self._accumulate_unrouted_gate_cost(state, qpu_id, gate, local_slots, network)
        return state

    def _should_route_local_gate(
        self,
        kind: LocalGateKind,
        policy: EvaluationPolicy,
    ) -> bool:
        if kind == LocalGateKind.PAYLOAD:
            return policy.route_payload_gates
        if kind == LocalGateKind.COMM_PRIMITIVE:
            return policy.route_comm_gates
        if kind == LocalGateKind.ENTANGLEMENT_SWAP:
            return policy.route_entanglement_swap
        raise ValueError(f"Unsupported local gate kind: {kind}")

    def _append_routed_gate(
        self,
        state: EvaluationState,
        qpu_id: int,
        gate: Gate,
        local_slots: Sequence[int | None],
    ) -> None:
        """
        Append a gate to the QPU buffer that will later be transpiled.
        """
        int_slots: list[int] = []
        for slot in local_slots:
            if slot is None:
                raise RuntimeError(
                    f"[EVALUATOR] Cannot route gate with unknown local slot: gate={gate}, slots={local_slots}"
                )
            int_slots.append(int(slot))

        state.routed_buffers[qpu_id].append(
            gate,
            [state.routed_buffers[qpu_id].qubits[slot] for slot in int_slots],
        )

    def _accumulate_unrouted_gate_cost(
        self,
        state: EvaluationState,
        qpu_id: int,
        gate: Gate,
        local_slots: Sequence[int | None],
        network: Network,
    ) -> None:
        """
        Charge a local gate without routing.

        Under the unrouted model, QPU-local qubits are treated as all-to-all.
        Local slots are used as the physical qubit ids for intrinsic gate error
        lookup, so no extra routing gates are introduced.
        """
        physical_slots: list[int] = []
        for local_slot in local_slots:
            if local_slot is None:
                raise RuntimeError(
                    f"[EVALUATOR] Cannot charge unrouted gate with unknown local slot: "
                    f"gate={gate}, slots={local_slots}"
                )
            physical_slots.append(int(local_slot))

        gate_error = CompilerUtils._get_sampled_backend_gate_error(
            network.backends[qpu_id],
            gate.name,
            physical_slots,
        )
        state.costs.local_gate_num += 1
        state.costs.local_fidelity_loss += gate_error
        state.costs.local_fidelity *= (1 - gate_error)
        state.costs.local_fidelity_log_sum += float(np.log(1 - gate_error))

    def _flush_local_buffers(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> EvaluationState:
        """
        Transpile and charge all routed local buffers.

        Routed buffers contain only local gates that the selected policy says
        should see the backend coupling map. The final layout becomes the next
        runtime physical state for resident and communication slots.
        """
        for qpu_id, subcircuit in enumerate(state.routed_buffers):
            if subcircuit.size() == 0:
                continue

            backend = network.backends[qpu_id]
            initial_layout = self._build_initial_layout_for_qpu(
                state,
                qpu_id,
                subcircuit,
                partition,
                network,
                policy,
            )
            transpile_initial_layout = initial_layout if initial_layout else None

            transpiled_circuit = transpile(
                subcircuit,
                coupling_map=backend.coupling_map,
                basis_gates=backend.basis_gates,
                initial_layout=transpile_initial_layout,
                optimization_level=policy.optimization_level,
                seed_transpiler=42,
            )

            state.costs.local_transpile_calls += 1
            for instruction in transpiled_circuit:
                gate_name = instruction.operation.name
                physical_slots = [
                    transpiled_circuit.qubits.index(qubit)
                    for qubit in instruction.qubits
                ]
                if len(physical_slots) == 0:
                    continue
                gate_error = CompilerUtils._get_sampled_backend_gate_error(
                    backend,
                    gate_name,
                    physical_slots,
                )
                state.costs.local_gate_num += 1
                state.costs.local_fidelity_loss += gate_error
                state.costs.local_fidelity *= (1 - gate_error)
                state.costs.local_fidelity_log_sum += float(np.log(1 - gate_error))

            self._update_state_from_transpiled_layout(
                state,
                qpu_id,
                transpiled_circuit,
                partition,
            )

        state.routed_buffers = self._new_routed_buffers(partition, network)
        return state

    def _build_initial_layout_for_qpu(
        self,
        state: EvaluationState,
        qpu_id: int,
        subcircuit: QuantumCircuit,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> dict[Any, int]:
        """
        Build Qiskit initial_layout for known resident/comm physical slots.
        """
        initial_layout: dict[Any, int] = {}
        used_phys: set[int] = set()

        for loc in state.logical_pos.values():
            if loc.qpu_id != qpu_id or loc.physical_slot is None:
                continue
            if not (0 <= loc.local_slot < subcircuit.num_qubits):
                raise RuntimeError(
                    f"[EVALUATOR] local slot out of routed buffer range while building initial layout: "
                    f"qpu={qpu_id}, local_slot={loc.local_slot}, buffer_qubits={subcircuit.num_qubits}, "
                    f"physical_slot={loc.physical_slot}"
                )
            phy = int(loc.physical_slot)
            if phy in used_phys:
                raise RuntimeError(
                    f"[EVALUATOR] duplicate initial physical slot on QPU {qpu_id}: phy={phy}"
                )
            initial_layout[subcircuit.qubits[loc.local_slot]] = phy
            used_phys.add(phy)

        comm_slots = state.comm_phy_map.get(qpu_id, [])
        comm_slot_start = len(partition[qpu_id])
        for offset, physical_slot in enumerate(comm_slots):
            if physical_slot is None:
                continue
            local_slot = comm_slot_start + offset # logical comm qid
            if not (0 <= local_slot < subcircuit.num_qubits):
                raise RuntimeError(
                    f"[EVALUATOR] comm local slot out of routed buffer range while building initial layout: "
                    f"qpu={qpu_id}, local_slot={local_slot}, buffer_qubits={subcircuit.num_qubits}, "
                    f"comm_slot_start={comm_slot_start}, offset={offset}, physical_slot={physical_slot}"
                )
            phy = int(physical_slot)
            if phy in used_phys:
                raise RuntimeError(
                    f"[EVALUATOR] duplicate comm initial physical slot on QPU {qpu_id}: phy={phy}"
                )
            initial_layout[subcircuit.qubits[local_slot]] = phy
            used_phys.add(phy)

        if policy.fill_initial_layout:
            available_phys = [
                phy
                for phy in range(int(network.backends[qpu_id].num_qubits))
                if phy not in used_phys
            ]
            for local_slot in range(subcircuit.num_qubits):
                qobj = subcircuit.qubits[local_slot]
                if qobj in initial_layout:
                    continue
                if len(available_phys) == 0:
                    raise RuntimeError(
                        f"[EVALUATOR] no available physical slot to fill initial layout: "
                        f"qpu={qpu_id}, local_slot={local_slot}"
                    )
                phy = int(available_phys.pop(0))
                initial_layout[qobj] = phy
                used_phys.add(phy)

        return initial_layout

    def _update_state_from_transpiled_layout(
        self,
        state: EvaluationState,
        qpu_id: int,
        transpiled_circuit: QuantumCircuit,
        partition: list[list[int]],
    ) -> None:
        """
        Persist final local-slot to physical-slot mapping after routed flush.
        """
        try:
            local_to_physical = CompilerUtils.get_local_to_physical_map(transpiled_circuit)
        except ValueError:
            local_to_physical = {
                local_slot: local_slot
                for local_slot in range(transpiled_circuit.num_qubits)
            }

        # update comp qubits' physical_slot
        for loc in state.logical_pos.values():
            if loc.qpu_id != qpu_id:
                continue
            physical_slot = local_to_physical.get(loc.local_slot)
            if physical_slot is None:
                raise RuntimeError(
                    f"[EVALUATOR] missing physical slot for resident local slot after transpile: "
                    f"qpu={qpu_id}, local_slot={loc.local_slot}, "
                    f"local_to_physical={local_to_physical}"
                )
            loc.physical_slot = int(physical_slot)

        comm_slots = state.comm_phy_map.get(qpu_id, [])
        comm_slot_start = len(partition[qpu_id])
        for offset in range(len(comm_slots)):
            local_slot = comm_slot_start + offset
            physical_slot = local_to_physical.get(local_slot)
            if physical_slot is None:
                raise RuntimeError(
                    f"[EVALUATOR] missing physical slot for comm local slot after transpile: "
                    f"qpu={qpu_id}, local_slot={local_slot}, offset={offset}, "
                    f"local_to_physical={local_to_physical}"
                )
            comm_slots[offset] = int(physical_slot)
        state.comm_phy_map[qpu_id] = comm_slots
