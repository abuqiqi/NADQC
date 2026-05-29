from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from .compiler_utils import CommOp, CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..utils import Network


@dataclass(frozen=True)
class EvaluationPolicy:
    name: str
    route_payload_gates: bool = True
    route_comm_gates: bool = True
    route_entanglement_swap: bool = False
    strict_flush_on_remote: bool = True
    flush_each_comm_gate: bool = False
    # Kept for config compatibility. MappingEvaluator always supplies a full
    # initial layout to Qiskit so replay is deterministic.
    fill_initial_layout: bool = False
    optimization_level: int = 3
    local_eval_mode: Literal["immediate", "deferred"] = "immediate"
    deferred_route_local_gates: bool = True
    deferred_initial_layout: Literal["fixed", "free"] = "fixed"

    @staticmethod
    def local_all_to_all() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="local_all_to_all",
            route_payload_gates=False,
            route_comm_gates=False,
            route_entanglement_swap=False,
        )

    @staticmethod
    def comm_to_all() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="comm_to_all",
            route_payload_gates=True,
            route_comm_gates=False,
            route_entanglement_swap=False,
        )

    @staticmethod
    def full_realistic() -> "EvaluationPolicy":
        return EvaluationPolicy(
            name="full_realistic",
            route_payload_gates=True,
            route_comm_gates=True,
            route_entanglement_swap=True,
        )


class LocalGateKind(str, Enum):
    PAYLOAD = "payload"
    COMM_PRIMITIVE = "comm_primitive"
    TELEDATA = "teledata"
    ENTANGLEMENT_SWAP = "entanglement_swap"


class CommunicationStatsKind(str, Enum):
    COMM_BLOCK = "comm_block"
    TELEGATE_EXEC = "telegate_exec"


WireOwnerKind = Literal["resident", "entangled_copy", "protocol"]


@dataclass(frozen=True)
class RuntimeLocation:
    """
    Current runtime local wire of one logical qubit.

    local_wire is indexed in the QPU's full local-wire space.
    """
    qpu_id: int
    local_wire: int

    def is_comp(self, comp_wire_count: int) -> bool:
        return int(self.local_wire) < int(comp_wire_count)

    def is_comm(self, comp_wire_count: int) -> bool:
        return int(self.local_wire) >= int(comp_wire_count)

    def comm_offset(self, comp_wire_count: int) -> int:
        if self.is_comp(comp_wire_count):
            raise ValueError("computation wire has no communication offset")
        return int(self.local_wire) - int(comp_wire_count)


@dataclass(frozen=True)
class WireOwner:
    kind: WireOwnerKind
    logical_qid: int | None = None
    label: str | None = None


@dataclass
class EvaluationState:
    costs: ExecCosts = field(default_factory=ExecCosts)
    logical_pos: dict[int, RuntimeLocation] = field(default_factory=dict)
    wire_phy_map: dict[int, list[int | None]] = field(default_factory=dict)
    wire_owners: dict[int, list[WireOwner | None]] = field(default_factory=dict)
    routed_buffers: list[QuantumCircuit] = field(default_factory=list)
    active_buffer_use_coupling_map: list[bool | None] = field(default_factory=list)
    active_buffer_kind: list[LocalGateKind | None] = field(default_factory=list)


@dataclass(frozen=True)
class TeledataMove:
    logical_qid: int
    src_qpu: int
    dst_qpu: int
    dst_wire: int | None = None


class MappingEvaluator:
    def evaluate(
        self,
        mapping_record_list: MappingRecordList,
        network: Network,
        policy: EvaluationPolicy,
    ) -> MappingRecordList:
        self._validate_policy(policy)
        state: EvaluationState | None = None

        for t, record in enumerate(mapping_record_list.records):
            if t == 0:
                state = self._initialize_state_from_partition(record.partition, network, policy)
            assert state is not None

            if t == 0:
                self._start_record_replay(state, record.partition, network, policy)
            else:
                state.costs = ExecCosts()
                self._evaluate_partition_transition(
                    mapping_record_list.records[t - 1].partition,
                    record.partition,
                    network,
                    policy,
                    state,
                )
            subcircuit = self._get_record_subcircuit(record)

            self._evaluate_record_body(record, subcircuit, network, policy, state)
            self.flush_local_ops(state, record.partition, network, policy)
            self._require_all_residents_in_comp_space(state, record.partition, "record-end")
            self._commit_state_to_record(record, state, record.partition)

        if self._is_deferred(policy) and state is not None and mapping_record_list.records:
            last_record = mapping_record_list.records[-1]
            self.flush_local_ops(state, last_record.partition, network, policy, final=True)
            self._commit_state_to_record(last_record, state, last_record.partition)

        mapping_record_list.summarize_total_costs()
        return mapping_record_list

    def _start_record_replay(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        self._validate_partition_capacity(partition, network)
        state.costs = ExecCosts()
        self._prepare_local_buffers(state, partition, network, policy)
        self._validate_physical_state(state, partition, network, "record-start")

    def _prepare_local_buffers(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy | None = None,
    ) -> None:
        if self._is_deferred(policy) and state.routed_buffers:
            return
        state.routed_buffers = self._new_routed_buffers(partition, network, policy)
        state.active_buffer_use_coupling_map = [None for _ in range(network.num_backends)]
        state.active_buffer_kind = [None for _ in range(network.num_backends)]

    def _initialize_state_from_partition(
        self,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy | None = None,
    ) -> EvaluationState:
        self._validate_partition_capacity(partition, network)
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        state = EvaluationState(
            routed_buffers=self._new_routed_buffers(partition, network, policy),
            active_buffer_use_coupling_map=[None for _ in range(network.num_backends)],
            active_buffer_kind=[None for _ in range(network.num_backends)],
        )
        for qpu_id, group in enumerate(partition):
            wire_count = len(group) + reserve
            state.wire_phy_map[qpu_id] = [None for _ in range(wire_count)]
            state.wire_owners[qpu_id] = [None for _ in range(wire_count)]
            for local_wire, logical_qid in enumerate(group):
                q = int(logical_qid)
                state.logical_pos[q] = RuntimeLocation(qpu_id=qpu_id, local_wire=local_wire)
                state.wire_owners[qpu_id][local_wire] = WireOwner("resident", q)
        return state

    def _validate_partition_capacity(self, partition: list[list[int]], network: Network) -> None:
        if len(partition) != network.num_backends:
            raise ValueError(
                f"[EVALUATOR] partition/backend size mismatch: "
                f"partition={len(partition)}, backends={network.num_backends}"
            )
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        for qpu_id, group in enumerate(partition):
            if qpu_id < len(network.backend_sizes) and len(group) > int(network.backend_sizes[qpu_id]):
                raise ValueError(
                    f"[EVALUATOR] QPU {qpu_id} partition size exceeds computation capacity: "
                    f"size={len(group)}, capacity={network.backend_sizes[qpu_id]}"
                )
            backend_qubits = int(network.backends[qpu_id].num_qubits)
            required_wires = int(len(group)) + reserve
            if required_wires > backend_qubits:
                raise ValueError(
                    f"[EVALUATOR] QPU {qpu_id} partition+comm reserve exceeds backend qubits: "
                    f"partition={len(group)}, reserve={reserve}, required={required_wires}, backend_qubits={backend_qubits}"
                )

    def _is_deferred(self, policy: EvaluationPolicy | None) -> bool:
        return bool(policy is not None and policy.local_eval_mode == "deferred")

    def _validate_policy(self, policy: EvaluationPolicy) -> None:
        if policy.local_eval_mode not in {"immediate", "deferred"}:
            raise ValueError(f"unknown local_eval_mode: {policy.local_eval_mode}")
        if policy.deferred_initial_layout not in {"fixed", "free"}:
            raise ValueError(f"unknown deferred_initial_layout: {policy.deferred_initial_layout}")

    def _new_routed_buffers(
        self,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy | None = None,
    ) -> list[QuantumCircuit]:
        if self._is_deferred(policy):
            return [QuantumCircuit(int(backend.num_qubits)) for backend in network.backends]
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        return [QuantumCircuit(len(group) + reserve) for group in partition]

    def _get_record_subcircuit(
        self,
        record: MappingRecord,
    ) -> QuantumCircuit:
        if record.extra_info is not None and "ops" in record.extra_info:
            return record.extra_info["ops"]
        raise ValueError(
            "[EVALUATOR] record requires extra_info['ops']; fallback to circuit slicing is disabled."
        )

    def resolve_free_or_explicit_wire(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        qpu_id: int,
        local_wire: int | None,
        wire_kind: Literal["comp", "comm"] | None,
    ) -> int:
        comp_wire_count = len(partition[qpu_id])
        wire_count = len(state.wire_owners[qpu_id])
        if local_wire is None:
            if wire_kind is None:
                raise ValueError("local_wire and wire_kind cannot both be None")
            candidates = range(0, comp_wire_count) if wire_kind == "comp" else range(comp_wire_count, wire_count)
            for candidate in candidates:
                if state.wire_owners[qpu_id][candidate] is None:
                    return candidate
            raise RuntimeError(f"no free {wire_kind} wire on qpu {qpu_id}")

        local_wire = int(local_wire)
        if local_wire < 0 or local_wire >= wire_count:
            raise ValueError(f"local_wire out of range: {local_wire}")
        if wire_kind == "comp" and local_wire >= comp_wire_count:
            raise ValueError("expected comp wire, got comm wire")
        if wire_kind == "comm" and local_wire < comp_wire_count:
            raise ValueError("expected comm wire, got comp wire")
        owner = state.wire_owners[qpu_id][local_wire]
        if owner is not None:
            raise RuntimeError(
                f"target wire already occupied: qpu={qpu_id}, wire={local_wire}, owner={owner}"
            )
        return local_wire

    def release_old_resident_owner(self, state: EvaluationState, logical_qid: int) -> None:
        old_loc = state.logical_pos.get(int(logical_qid))
        if old_loc is None:
            return
        old_owner = state.wire_owners[old_loc.qpu_id][old_loc.local_wire]
        if old_owner is None or old_owner.kind != "resident" or old_owner.logical_qid != int(logical_qid):
            raise RuntimeError(
                f"logical_pos and wire_owners inconsistent for logical qubit "
                f"{logical_qid}: loc={old_loc}, owner={old_owner}"
            )
        state.wire_owners[old_loc.qpu_id][old_loc.local_wire] = None

    def reserve_wire(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        qpu_id: int,
        local_wire: int | None = None,
        wire_kind: Literal["comp", "comm"] | None = None,
        owner_kind: WireOwnerKind = "protocol",
        logical_qid: int | None = None,
        label: str | None = None,
    ) -> RuntimeLocation:
        target_wire = self.resolve_free_or_explicit_wire(state, partition, qpu_id, local_wire, wire_kind)
        target_loc = RuntimeLocation(qpu_id=qpu_id, local_wire=target_wire)
        if owner_kind == "resident":
            if logical_qid is None:
                raise ValueError("resident owner requires logical_qid")
            self.release_old_resident_owner(state, int(logical_qid))
            state.wire_owners[qpu_id][target_wire] = WireOwner("resident", int(logical_qid), label)
            state.logical_pos[int(logical_qid)] = target_loc
            return target_loc
        if owner_kind == "entangled_copy":
            if logical_qid is None:
                raise ValueError("entangled_copy owner requires logical_qid")
            if int(logical_qid) not in state.logical_pos:
                raise RuntimeError(f"cannot create entangled_copy for unknown logical qubit {logical_qid}")
            state.wire_owners[qpu_id][target_wire] = WireOwner("entangled_copy", int(logical_qid), label)
            return target_loc
        if owner_kind == "protocol":
            state.wire_owners[qpu_id][target_wire] = WireOwner("protocol", logical_qid, label)
            return target_loc
        raise ValueError(f"unknown owner_kind: {owner_kind}")

    def release_wire(
        self,
        state: EvaluationState,
        qpu_id: int,
        local_wire: int,
        expected_owner_kind: Literal["entangled_copy", "protocol"] | None = None,
    ) -> None:
        owner = state.wire_owners[qpu_id][local_wire]
        if owner is None:
            raise RuntimeError(f"wire is already free: qpu={qpu_id}, wire={local_wire}")
        if expected_owner_kind is not None and owner.kind != expected_owner_kind:
            raise RuntimeError(
                f"unexpected owner kind on qpu={qpu_id}, wire={local_wire}: "
                f"expected={expected_owner_kind}, actual={owner}"
            )
        if owner.kind == "resident":
            raise RuntimeError("resident owner must be moved, not released")
        state.wire_owners[qpu_id][local_wire] = None

    def _should_use_coupling_map(self, kind: LocalGateKind, policy: EvaluationPolicy) -> bool:
        if self._is_deferred(policy):
            return bool(policy.deferred_route_local_gates)
        if kind == LocalGateKind.PAYLOAD:
            return bool(policy.route_payload_gates)
        if kind == LocalGateKind.COMM_PRIMITIVE:
            return bool(policy.route_comm_gates)
        if kind == LocalGateKind.TELEDATA:
            return bool(policy.route_comm_gates)
        if kind == LocalGateKind.ENTANGLEMENT_SWAP:
            return bool(policy.route_entanglement_swap)
        raise ValueError(f"unknown local gate kind: {kind}")

    def add_local_ops(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
        qpu_id: int,
        ops: Sequence[tuple[Gate, Sequence[int]]],
        kind: LocalGateKind,
    ) -> None:
        qpu_id = int(qpu_id)
        use_coupling_map = self._should_use_coupling_map(kind, policy)
        current_mode = state.active_buffer_use_coupling_map[qpu_id]
        current_kind = state.active_buffer_kind[qpu_id]
        if self._is_deferred(policy):
            if current_mode is None:
                state.active_buffer_use_coupling_map[qpu_id] = use_coupling_map
            elif current_mode != use_coupling_map:
                raise RuntimeError("[EVALUATOR] deferred local routing mode changed within one QPU buffer")
            state.active_buffer_kind[qpu_id] = None
        else:
            if current_mode is None and current_kind is None:
                state.active_buffer_use_coupling_map[qpu_id] = use_coupling_map
                state.active_buffer_kind[qpu_id] = kind
            elif current_mode != use_coupling_map or current_kind != kind:
                self.flush_local_ops(state, partition, network, policy, qpu_ids=[qpu_id])
                state.active_buffer_use_coupling_map[qpu_id] = use_coupling_map
                state.active_buffer_kind[qpu_id] = kind

        buffer = state.routed_buffers[qpu_id]
        for gate, wires in ops:
            wire_list = [int(w) for w in wires]
            if len(wire_list) == 0:
                continue
            if any(w < 0 or w >= buffer.num_qubits for w in wire_list):
                raise RuntimeError(f"[EVALUATOR] local wire out of range: qpu={qpu_id}, wires={wire_list}")
            buffer.append(gate, wire_list)
            if kind == LocalGateKind.PAYLOAD:
                state.costs.payload_gate_num += 1

    def flush_local_ops(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
        qpu_ids: Sequence[int] | None = None,
        final: bool = False,
    ) -> EvaluationState:
        targets = range(network.num_backends) if qpu_ids is None else [int(q) for q in qpu_ids]
        state.costs.flush_calls += 1
        if self._is_deferred(policy) and not final:
            self._append_deferred_barriers(state, targets)
            return state

        flushed_any = False
        for qpu_id in targets:
            buffer = state.routed_buffers[qpu_id]
            if buffer.size() == 0:
                state.active_buffer_use_coupling_map[qpu_id] = None
                state.active_buffer_kind[qpu_id] = None
                continue

            flushed_any = True
            backend = network.backends[qpu_id]
            use_coupling_map = bool(state.active_buffer_use_coupling_map[qpu_id])
            buffer_kind = state.active_buffer_kind[qpu_id]
            if self._is_deferred(policy):
                use_coupling_map = bool(policy.deferred_route_local_gates)
                buffer_kind = None
            initial_layout = self._get_transpile_initial_layout(state, qpu_id, buffer, backend, policy)
            pre_transpile_gate_count = self._count_quantum_ops(buffer)
            transpile_kwargs: dict[str, Any] = {
                "basis_gates": backend.basis_gates,
                "optimization_level": int(policy.optimization_level),
                "seed_transpiler": 42,
            }
            if initial_layout is not None:
                transpile_kwargs["initial_layout"] = initial_layout
            if use_coupling_map:
                transpile_kwargs["coupling_map"] = backend.coupling_map
            transpiled_circuit = transpile(buffer, **transpile_kwargs)
            state.costs.local_transpile_calls += 1
            state.costs.local_pre_transpile_gate_num += pre_transpile_gate_count
            post_transpile_gate_count = self._accumulate_local_circuit_costs(
                state,
                backend,
                transpiled_circuit,
                buffer_kind,
            )
            state.costs.local_transpile_added_gate_num += post_transpile_gate_count - pre_transpile_gate_count
            if not self._is_deferred(policy):
                state.wire_phy_map[qpu_id] = self._extract_final_wire_layout(
                    transpiled_circuit,
                    len(state.wire_phy_map[qpu_id]),
                    backend,
                )
            state.routed_buffers[qpu_id] = self._new_routed_buffers(partition, network, policy)[qpu_id]
            state.active_buffer_use_coupling_map[qpu_id] = None
            state.active_buffer_kind[qpu_id] = None

        if flushed_any:
            state.costs.nonempty_flushes += 1
        self._validate_physical_state(state, partition, network, "post-flush")
        return state

    def _append_deferred_barriers(self, state: EvaluationState, qpu_ids: Sequence[int]) -> None:
        for qpu_id in qpu_ids:
            buffer = state.routed_buffers[int(qpu_id)]
            if buffer.size() == 0:
                continue
            buffer.barrier(*buffer.qubits)

    def _get_transpile_initial_layout(
        self,
        state: EvaluationState,
        qpu_id: int,
        circuit: QuantumCircuit,
        backend: Any,
        policy: EvaluationPolicy,
    ) -> dict[Any, int] | None:
        if self._is_deferred(policy) and policy.deferred_initial_layout == "free":
            return None
        initial_layout = self._build_initial_layout(state, qpu_id, circuit, backend, policy)
        return initial_layout if initial_layout else None

    def _count_quantum_ops(self, circuit: QuantumCircuit) -> int:
        count = 0
        for instruction in circuit:
            gate_name = instruction.operation.name
            if gate_name in {"barrier", "delay"} or len(instruction.qubits) == 0:
                continue
            count += 1
        return count

    def _build_initial_layout(
        self,
        state: EvaluationState,
        qpu_id: int,
        circuit: QuantumCircuit,
        backend: Any,
        policy: EvaluationPolicy,
    ) -> dict[Any, int]:
        initial_layout: dict[Any, int] = {}
        used: set[int] = set()
        for local_wire, phy in enumerate(state.wire_phy_map[qpu_id]):
            if phy is None:
                continue
            if local_wire >= circuit.num_qubits:
                continue
            phy = int(phy)
            if phy < 0 or phy >= int(backend.num_qubits):
                raise RuntimeError(f"[LAYOUT] physical qubit out of range: qpu={qpu_id}, wire={local_wire}, phy={phy}")
            if phy in used:
                raise RuntimeError(f"[LAYOUT] duplicate initial physical qubit: qpu={qpu_id}, phy={phy}")
            initial_layout[circuit.qubits[local_wire]] = phy
            used.add(phy)
        del policy
        free_phys = [p for p in range(int(backend.num_qubits)) if p not in used]
        for local_wire in range(circuit.num_qubits):
            qobj = circuit.qubits[local_wire]
            if qobj in initial_layout:
                continue
            if not free_phys:
                raise RuntimeError(f"[LAYOUT] no free physical qubit for qpu={qpu_id}, wire={local_wire}")
            initial_layout[qobj] = free_phys.pop(0)
        return initial_layout

    def _extract_final_wire_layout(
        self,
        transpiled_circuit: QuantumCircuit,
        wire_count: int,
        backend: Any,
    ) -> list[int | None]:
        local_to_phy = self._get_local_to_physical_map(transpiled_circuit)
        result: list[int | None] = [None for _ in range(wire_count)]
        for local_wire in range(wire_count):
            phy = local_to_phy.get(local_wire)
            if phy is None and local_wire < transpiled_circuit.num_qubits:
                phy = local_wire
            if phy is not None:
                phy = int(phy)
                if phy >= int(backend.num_qubits):
                    raise RuntimeError(f"[LAYOUT] final physical qubit out of range: wire={local_wire}, phy={phy}")
            result[local_wire] = phy
        return result

    def _get_local_to_physical_map(
        self,
        transpiled_circuit: QuantumCircuit,
    ) -> dict[int, int | None]:
        local_to_phy: dict[int, int | None] = {
            local_wire: None
            for local_wire in range(transpiled_circuit.num_qubits)
        }
        layout_obj = getattr(transpiled_circuit, "layout", None)
        if layout_obj is None:
            return {
                local_wire: local_wire
                for local_wire in range(transpiled_circuit.num_qubits)
            }

        layout = layout_obj.final_layout
        if layout is None:
            layout = layout_obj.initial_layout
        if layout is None:
            return {
                local_wire: local_wire
                for local_wire in range(transpiled_circuit.num_qubits)
            }

        for phy_qid, logic_qubit in layout.get_physical_bits().items():
            reg = getattr(logic_qubit, "register", None)
            if reg is None:
                reg = getattr(logic_qubit, "_register", None)
            logic_idx = getattr(logic_qubit, "index", None)
            if logic_idx is None:
                logic_idx = getattr(logic_qubit, "_index", None)
            if reg is None or logic_idx is None:
                continue
            if reg.name == "q" and int(logic_idx) in local_to_phy:
                local_to_phy[int(logic_idx)] = int(phy_qid)
        return local_to_phy

    def _accumulate_local_circuit_costs(
        self,
        state: EvaluationState,
        backend: Any,
        transpiled_circuit: QuantumCircuit,
        kind: LocalGateKind | None,
    ) -> int:
        counted = 0
        for instruction in transpiled_circuit:
            gate_name = instruction.operation.name
            if gate_name in {"barrier", "delay"} or len(instruction.qubits) == 0:
                continue
            physical_slots = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
            gate_error = CompilerUtils._get_sampled_backend_gate_error(backend, gate_name, physical_slots)
            counted += 1
            state.costs.local_gate_num += 1
            if kind == LocalGateKind.PAYLOAD:
                state.costs.local_payload_gate_num += 1
            elif kind == LocalGateKind.TELEDATA:
                state.costs.local_teledata_gate_num += 1
            elif kind in {LocalGateKind.COMM_PRIMITIVE, LocalGateKind.ENTANGLEMENT_SWAP}:
                state.costs.local_comm_protocol_gate_num += 1
            else:
                state.costs.local_uncategorized_gate_num += 1
            state.costs.local_fidelity_loss += gate_error
            state.costs.local_fidelity *= 1 - gate_error
            state.costs.local_fidelity_log_sum += float(np.log(1 - gate_error))
        return counted

    def _evaluate_record_body(
        self,
        record: MappingRecord,
        subcircuit: QuantumCircuit,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        for instruction in subcircuit:
            op = instruction.operation
            global_qids = [subcircuit.qubits.index(qubit) for qubit in instruction.qubits]

            if isinstance(op, CommOp):
                self._process_commop(record.partition, op, network, policy, state)
                continue

            if len(global_qids) == 0:
                continue

            involved_qpus = {state.logical_pos[int(q)].qpu_id for q in global_qids}
            if len(involved_qpus) == 1:
                qpu_id = next(iter(involved_qpus))
                self.add_local_ops(
                    state,
                    record.partition,
                    network,
                    policy,
                    qpu_id=qpu_id,
                    ops=[(op, [state.logical_pos[int(q)].local_wire for q in global_qids])],
                    kind=LocalGateKind.PAYLOAD,
                )
            else:
                self._process_synthetic_telegate(record.partition, op, global_qids, network, policy, state)

        return state

    def _process_synthetic_telegate(
        self,
        partition: list[list[int]],
        op: Any,
        global_qids: list[int],
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        if not isinstance(op, Gate):
            raise RuntimeError(f"[EVALUATOR] cannot synthesize telegate for non-gate op: {op}")
        if len(global_qids) < 2:
            raise RuntimeError(f"[EVALUATOR] cross-QPU gate has too few operands: {op}, qids={global_qids}")

        source = int(global_qids[0])
        src_qpu = state.logical_pos[source].qpu_id
        dst_candidates = [state.logical_pos[int(q)].qpu_id for q in global_qids[1:]]
        dst_qpu = next((qpu for qpu in dst_candidates if qpu != src_qpu), dst_candidates[0])
        gate_copy = op.to_mutable() if hasattr(op, "to_mutable") else copy.deepcopy(op)
        setattr(gate_copy, "_global_lqids", [int(q) for q in global_qids])
        synthetic = CommOp(
            comm_type="cat",
            source_qubit=source,
            src_qpu=src_qpu,
            dst_qpu=dst_qpu,
            involved_qubits=[int(q) for q in global_qids],
            gate_list=[gate_copy],
        )
        self._process_comm_like_op(partition, synthetic, network, policy, state, CommunicationStatsKind.TELEGATE_EXEC)

    def _process_commop(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        self._process_comm_like_op(partition, comm_op, network, policy, state, CommunicationStatsKind.COMM_BLOCK)

    def _process_comm_like_op(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
        stats_kind: CommunicationStatsKind,
    ) -> None:
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)
        self._validate_commop_runtime_endpoints(comm_op, src_qpu, dst_qpu, state, partition)

        if policy.strict_flush_on_remote:
            self.flush_local_ops(state, partition, network, policy)

        remote_loss_before = state.costs.remote_fidelity_loss
        remote_log_before = state.costs.remote_fidelity_log_sum

        if stats_kind == CommunicationStatsKind.COMM_BLOCK:
            state.costs.comm_block_events += 1
        elif stats_kind == CommunicationStatsKind.TELEGATE_EXEC:
            state.costs.telegate_exec_events += 1
        else:
            raise ValueError(f"unsupported stats kind: {stats_kind}")

        if comm_op.comm_type == "cat":
            self._process_cat(partition, comm_op, network, policy, state)
        elif comm_op.comm_type == "tp":
            if len(comm_op.gate_list) == 0:
                self._process_empty_tp_return(partition, comm_op, network, policy, state)
            else:
                self._process_tp_payload(partition, comm_op, network, policy, state)
        elif comm_op.comm_type == "rtp":
            self._process_rtp(partition, comm_op, network, policy, state)
        else:
            raise ValueError(f"unsupported CommOp type: {comm_op.comm_type}")

        remote_loss_delta = state.costs.remote_fidelity_loss - remote_loss_before
        remote_log_delta = state.costs.remote_fidelity_log_sum - remote_log_before
        if stats_kind == CommunicationStatsKind.COMM_BLOCK:
            state.costs.comm_block_remote_fidelity_loss += remote_loss_delta
            state.costs.comm_block_remote_fidelity_log_sum += remote_log_delta
        else:
            state.costs.telegate_exec_remote_fidelity_loss += remote_loss_delta
            state.costs.telegate_exec_remote_fidelity_log_sum += remote_log_delta

        if policy.strict_flush_on_remote:
            self.flush_local_ops(state, partition, network, policy)

    def _process_cat(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        source = int(comm_op.source_qubit)
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)
        src_loc = state.logical_pos[source]
        if src_loc.qpu_id != src_qpu:
            raise RuntimeError(f"[EVALUATOR] CAT source runtime qpu mismatch: source={source}")

        src_comm = self.reserve_wire(state, partition, src_qpu, wire_kind="comm", owner_kind="protocol", label="cat-src-comm")
        dst_comm = self.reserve_wire(
            state,
            partition,
            dst_qpu,
            wire_kind="comm",
            owner_kind="entangled_copy",
            logical_qid=source,
            label="cat-dst-comm",
        )
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            src_qpu,
            [(Gate("cx", 2, []), [src_loc.local_wire, src_comm.local_wire])],
            LocalGateKind.COMM_PRIMITIVE,
        )
        state.costs = CompilerUtils.update_remote_move_costs(state.costs, src_qpu, dst_qpu, 1, network)
        state.costs.cat_ents += 1
        self._append_comm_payload_gate_list(state, partition, comm_op, dst_qpu, dst_comm.local_wire, network, policy)
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            dst_qpu,
            [(Gate("h", 1, []), [dst_comm.local_wire])],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=sorted({src_qpu, dst_qpu}))
        self.release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")
        self.release_wire(state, dst_comm.qpu_id, dst_comm.local_wire, "entangled_copy")

    def _process_tp_payload(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        source = int(comm_op.source_qubit)
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)
        src_loc = state.logical_pos[source]
        src_comm = self.reserve_wire(state, partition, src_qpu, wire_kind="comm", owner_kind="protocol", label="tp-src-comm")
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            src_qpu,
            [
                (Gate("cx", 2, []), [src_loc.local_wire, src_comm.local_wire]),
                (Gate("h", 1, []), [src_loc.local_wire]),
            ],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[src_qpu])
        self.release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")
        state.costs = CompilerUtils.update_remote_move_costs(state.costs, src_qpu, dst_qpu, 1, network)
        self.reserve_wire(
            state,
            partition,
            dst_qpu,
            wire_kind="comm",
            owner_kind="resident",
            logical_qid=source,
            label="tp-dst-comm",
        )
        self._append_comm_payload_gate_list(state, partition, comm_op, dst_qpu, None, network, policy)

    def _process_empty_tp_return(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        source = int(comm_op.source_qubit)
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)
        src_loc = state.logical_pos[source]
        if src_loc.qpu_id != src_qpu or not src_loc.is_comm(len(partition[src_qpu])):
            raise RuntimeError(f"[EVALUATOR] empty TP source is not currently on source comm wire: source={source}")
        home_wire = self._select_free_target_comp_wire(state, partition, dst_qpu)
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[src_qpu])
        state.costs = CompilerUtils.update_remote_move_costs(state.costs, src_qpu, dst_qpu, 1, network)
        dst_comm = self.reserve_wire(
            state,
            partition,
            dst_qpu,
            wire_kind="comm",
            owner_kind="resident",
            logical_qid=source,
            label="tp-return-dst-comm",
        )
        self.require_comp_wire_available_for_landing(state, partition, dst_qpu, home_wire)
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            dst_qpu,
            [(Gate("swap", 2, []), [dst_comm.local_wire, home_wire])],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[dst_qpu])
        self.reserve_wire(
            state,
            partition,
            dst_qpu,
            local_wire=home_wire,
            wire_kind="comp",
            owner_kind="resident",
            logical_qid=source,
            label="tp-return-home",
        )

    def _process_rtp(
        self,
        partition: list[list[int]],
        comm_op: CommOp,
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> None:
        source = int(comm_op.source_qubit)
        src_qpu = int(comm_op.src_qpu)
        dst_qpu = int(comm_op.dst_qpu)
        src_loc = state.logical_pos[source]
        source_home_wire = src_loc.local_wire

        src_comm0 = self.reserve_wire(state, partition, src_qpu, wire_kind="comm", owner_kind="protocol", label="rtp-src-comm0")
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            src_qpu,
            [
                (Gate("cx", 2, []), [source_home_wire, src_comm0.local_wire]),
                (Gate("h", 1, []), [source_home_wire]),
            ],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[src_qpu])
        self.release_wire(state, src_comm0.qpu_id, src_comm0.local_wire, "protocol")
        state.costs = CompilerUtils.update_remote_move_costs(state.costs, src_qpu, dst_qpu, 1, network)
        dst_comm0 = self.reserve_wire(
            state,
            partition,
            dst_qpu,
            wire_kind="comm",
            owner_kind="resident",
            logical_qid=source,
            label="rtp-dst-comm0",
        )
        self._append_comm_payload_gate_list(state, partition, comm_op, dst_qpu, None, network, policy)

        dst_comm1 = self.reserve_wire(state, partition, dst_qpu, wire_kind="comm", owner_kind="protocol", label="rtp-dst-comm1")
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            dst_qpu,
            [
                (Gate("cx", 2, []), [dst_comm0.local_wire, dst_comm1.local_wire]),
                (Gate("h", 1, []), [dst_comm0.local_wire]),
            ],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[dst_qpu])
        self.release_wire(state, dst_comm1.qpu_id, dst_comm1.local_wire, "protocol")

        state.costs = CompilerUtils.update_remote_move_costs(state.costs, dst_qpu, src_qpu, 1, network)
        src_comm1 = self.reserve_wire(
            state,
            partition,
            src_qpu,
            wire_kind="comm",
            owner_kind="resident",
            logical_qid=source,
            label="rtp-src-comm1",
        )
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            src_qpu,
            [(Gate("swap", 2, []), [src_comm1.local_wire, source_home_wire])],
            LocalGateKind.COMM_PRIMITIVE,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[src_qpu])
        self.reserve_wire(
            state,
            partition,
            src_qpu,
            local_wire=source_home_wire,
            wire_kind="comp",
            owner_kind="resident",
            logical_qid=source,
            label="rtp-src-home",
        )

    def _append_comm_payload_gate_list(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        comm_op: CommOp,
        dst_qpu: int,
        source_dst_comm_wire: int | None,
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        source = int(comm_op.source_qubit)
        for gate_op in comm_op.gate_list:
            global_lqids = getattr(gate_op, "_global_lqids", None)
            if global_lqids is None:
                raise RuntimeError(f"[EVALUATOR] CommOp payload gate missing _global_lqids: gate={gate_op}")
            wires: list[int] = []
            qpu_id: int | None = None
            for q in [int(qid) for qid in global_lqids]:
                if q == source and source_dst_comm_wire is not None:
                    wires.append(int(source_dst_comm_wire))
                    qpu_id = dst_qpu if qpu_id is None else qpu_id
                    if qpu_id != dst_qpu:
                        raise RuntimeError("[EVALUATOR] CommOp payload spans multiple runtime QPUs")
                    continue
                loc = state.logical_pos[q]
                if qpu_id is None:
                    qpu_id = loc.qpu_id
                elif qpu_id != loc.qpu_id:
                    raise RuntimeError(
                        f"[EVALUATOR] CommOp payload operands not colocated: gate={gate_op}, qids={global_lqids}"
                    )
                wires.append(loc.local_wire)
            if qpu_id is None:
                continue
            if qpu_id != dst_qpu and source in [int(qid) for qid in global_lqids]:
                raise RuntimeError(f"[EVALUATOR] source payload did not execute on destination: gate={gate_op}")
            self.add_local_ops(state, partition, network, policy, qpu_id, [(gate_op, wires)], LocalGateKind.PAYLOAD)
            if policy.flush_each_comm_gate:
                self.flush_local_ops(state, partition, network, policy, qpu_ids=[qpu_id])

    def _validate_commop_runtime_endpoints(
        self,
        comm_op: CommOp,
        src_qpu: int,
        dst_qpu: int,
        state: EvaluationState,
        partition: list[list[int]] | None = None,
    ) -> None:
        source = int(comm_op.source_qubit)
        if source not in state.logical_pos:
            raise RuntimeError(
                f"[EVALUATOR] CommOp source qubit missing from runtime state: source={source}"
            )
        runtime_src_qpu = int(state.logical_pos[source].qpu_id)
        if comm_op.comm_type == "tp" and len(comm_op.gate_list) == 0:
            if runtime_src_qpu != int(src_qpu):
                raise RuntimeError(
                    "[EVALUATOR] Empty TP return block source does not match runtime QPU: "
                    f"source={source}, runtime_src_qpu={runtime_src_qpu}, op_src_qpu={src_qpu}"
                )
            if partition is None:
                return
            target_qpus = self._partition_qpus(partition)
            home_qpu = target_qpus.get(int(source))
            if home_qpu is None:
                raise RuntimeError(
                    "[EVALUATOR] Empty TP return block source missing from partition: "
                    f"source={source}, op_dst_qpu={dst_qpu}"
                )
            if int(dst_qpu) != int(home_qpu):
                raise RuntimeError(
                    "[EVALUATOR] Empty TP return block dst_qpu does not match partition home: "
                    f"source={source}, op_dst_qpu={dst_qpu}, home_qpu={home_qpu}"
                )
            return
        if runtime_src_qpu != int(src_qpu):
            raise RuntimeError(
                "[EVALUATOR] CommOp src_qpu metadata inconsistent with runtime state: "
                f"source={source}, op_src_qpu={src_qpu}, runtime_src_qpu={runtime_src_qpu}, dst_qpu={dst_qpu}"
            )

        dst_candidates: set[int] = set()
        for gate_op in comm_op.gate_list:
            global_lqids = getattr(gate_op, "_global_lqids", None)
            if global_lqids is None or source not in [int(q) for q in global_lqids]:
                continue
            for q in [int(qid) for qid in global_lqids]:
                if q != source and q in state.logical_pos:
                    dst_candidates.add(int(state.logical_pos[q].qpu_id))
        if not dst_candidates:
            for q in comm_op.involved_qubits:
                q = int(q)
                if q != source and q in state.logical_pos:
                    dst_candidates.add(int(state.logical_pos[q].qpu_id))
        if len(dst_candidates) == 1:
            runtime_dst_qpu = next(iter(dst_candidates))
            if runtime_dst_qpu != int(dst_qpu):
                raise RuntimeError(
                    "[EVALUATOR] CommOp dst_qpu metadata inconsistent with runtime state: "
                    f"source={source}, op_dst_qpu={dst_qpu}, runtime_dst_qpu={runtime_dst_qpu}, src_qpu={src_qpu}"
                )
        elif len(dst_candidates) > 1 and int(dst_qpu) not in dst_candidates:
            raise RuntimeError(
                "[EVALUATOR] CommOp dst_qpu metadata is not among runtime destination candidates: "
                f"source={source}, op_dst_qpu={dst_qpu}, runtime_dst_candidates={sorted(dst_candidates)}"
            )

    def _evaluate_partition_transition(
        self,
        prev_partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
        state: EvaluationState,
    ) -> EvaluationState:
        self._validate_partition_capacity(prev_partition, network)
        self._validate_partition_capacity(target_partition, network)
        transition_partition = self._build_transition_partition(prev_partition, target_partition)
        self._resize_state_to_transition_partition(state, prev_partition, transition_partition, network)
        self._prepare_local_buffers(state, transition_partition, network, policy)
        self._validate_physical_state(state, transition_partition, network, "transition-start")

        target_qpus = self._partition_qpus(target_partition)
        old_locations = dict(state.logical_pos)
        if set(old_locations.keys()) != set(target_qpus.keys()):
            raise ValueError(
                f"[EVALUATOR] partition logical qubits changed across records: "
                f"prev_only={sorted(set(old_locations) - set(target_qpus))}, "
                f"curr_only={sorted(set(target_qpus) - set(old_locations))}"
            )

        graph = nx.DiGraph()
        graph.add_nodes_from(range(network.num_backends))

        for q, dst_qpu in target_qpus.items():
            src_qpu = old_locations[q].qpu_id
            if src_qpu == dst_qpu:
                continue
            if graph.has_edge(dst_qpu, src_qpu):
                reverse_qubits = graph[dst_qpu][src_qpu]["qubits"]
                if reverse_qubits:
                    partner_q = int(reverse_qubits.pop(0))
                    self._decrement_or_remove_edge(graph, dst_qpu, src_qpu)
                    self._process_teledata_batch(
                        [
                            TeledataMove(q, src_qpu, dst_qpu),
                            TeledataMove(partner_q, dst_qpu, src_qpu),
                        ],
                        "teledata-pair",
                        state,
                        transition_partition,
                        target_partition,
                        network,
                        policy,
                    )
                    continue
            self._append_move_to_graph(graph, src_qpu, dst_qpu, q)

        self._process_cycle_teledata(
            graph,
            state,
            transition_partition,
            target_partition,
            network,
            policy,
        )
        self._process_remaining_teledata(
            graph,
            state,
            transition_partition,
            target_partition,
            network,
            policy,
        )
        self._repair_local_compaction(state, transition_partition, target_partition, network, policy)
        self._compact_state_to_target_partition(state, transition_partition, target_partition, network, policy)
        self._require_logical_positions_match_partition_membership(state, target_partition, "post-teledata")
        self._validate_physical_state(state, target_partition, network, "post-teledata")
        return state

    def _process_teledata_batch(
        self,
        moves: Sequence[TeledataMove],
        label_prefix: str,
        state: EvaluationState,
        partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        if not moves:
            return
        self._require_teledata_batch_comm_capacity(moves, network)
        src_comms: dict[int, RuntimeLocation] = {}
        dst_comms: dict[int, RuntimeLocation] = {}

        for move in moves:
            src_loc = state.logical_pos[move.logical_qid]
            if src_loc.qpu_id != move.src_qpu:
                raise RuntimeError(f"[EVALUATOR] teledata source mismatch: move={move}, loc={src_loc}")
            src_comm = self.reserve_wire(
                state,
                partition,
                move.src_qpu,
                wire_kind="comm",
                owner_kind="protocol",
                label=f"{label_prefix}-src-comm",
            )
            src_comms[move.logical_qid] = src_comm
            self.add_local_ops(
                state,
                partition,
                network,
                policy,
                move.src_qpu,
                [
                    (Gate("cx", 2, []), [src_loc.local_wire, src_comm.local_wire]),
                    (Gate("h", 1, []), [src_loc.local_wire]),
                ],
                LocalGateKind.TELEDATA,
            )

        self.flush_local_ops(state, partition, network, policy, qpu_ids=sorted({m.src_qpu for m in moves}))
        for src_comm in src_comms.values():
            self.release_wire(state, src_comm.qpu_id, src_comm.local_wire, "protocol")

        for move in moves:
            state.costs = CompilerUtils.update_remote_move_costs(
                state.costs,
                move.src_qpu,
                move.dst_qpu,
                1,
                network,
            )
            dst_comms[move.logical_qid] = self.reserve_wire(
                state,
                partition,
                move.dst_qpu,
                wire_kind="comm",
                owner_kind="resident",
                logical_qid=move.logical_qid,
                label=f"{label_prefix}-dst-comm",
            )

        for move in moves:
            dst_wire = move.dst_wire
            if dst_wire is None:
                dst_wire = self._select_free_target_comp_wire(state, target_partition, move.dst_qpu)
            self.require_comp_wire_available_for_landing(state, target_partition, move.dst_qpu, dst_wire)
            self.add_local_ops(
                state,
                partition,
                network,
                policy,
                move.dst_qpu,
                [(Gate("swap", 2, []), [dst_comms[move.logical_qid].local_wire, dst_wire])],
                LocalGateKind.TELEDATA,
            )
            dst_comms[move.logical_qid] = RuntimeLocation(move.dst_qpu, dst_wire)

        self.flush_local_ops(state, partition, network, policy, qpu_ids=sorted({m.dst_qpu for m in moves}))
        for move in moves:
            dst_wire = dst_comms[move.logical_qid].local_wire
            self.reserve_wire(
                state,
                partition,
                move.dst_qpu,
                local_wire=dst_wire,
                wire_kind="comp",
                owner_kind="resident",
                logical_qid=move.logical_qid,
                label=f"{label_prefix}-target",
            )

    def _require_teledata_batch_comm_capacity(
        self,
        moves: Sequence[TeledataMove],
        network: Network,
    ) -> None:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        outgoing: dict[int, int] = defaultdict(int)
        incoming: dict[int, int] = defaultdict(int)
        for move in moves:
            outgoing[int(move.src_qpu)] += 1
            incoming[int(move.dst_qpu)] += 1
        for qpu_id in sorted(set(outgoing) | set(incoming)):
            required = outgoing[qpu_id] + incoming[qpu_id]
            if reserve < required:
                raise RuntimeError(
                    f"[EVALUATOR] insufficient communication wires for teledata batch: "
                    f"qpu={qpu_id}, required={required}, available={reserve}, "
                    f"outgoing={outgoing[qpu_id]}, incoming={incoming[qpu_id]}"
                )

    def _move_resident_with_local_swap(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
        qpu_id: int,
        logical_qid: int,
        dst_wire: int,
        dst_wire_kind: Literal["comp", "comm"],
        label: str,
    ) -> None:
        src_loc = state.logical_pos[int(logical_qid)]
        if src_loc.qpu_id != int(qpu_id):
            raise RuntimeError(
                f"[EVALUATOR] local resident move qpu mismatch: q={logical_qid}, "
                f"qpu={qpu_id}, loc={src_loc}"
            )
        if src_loc.local_wire == int(dst_wire):
            return
        if state.wire_owners[int(qpu_id)][int(dst_wire)] is not None:
            raise RuntimeError(
                f"[EVALUATOR] local resident move target occupied: "
                f"qpu={qpu_id}, dst_wire={dst_wire}, owner={state.wire_owners[int(qpu_id)][int(dst_wire)]}"
            )
        self.add_local_ops(
            state,
            partition,
            network,
            policy,
            qpu_id=int(qpu_id),
            ops=[(Gate("swap", 2, []), [src_loc.local_wire, int(dst_wire)])],
            kind=LocalGateKind.TELEDATA,
        )
        self.flush_local_ops(state, partition, network, policy, qpu_ids=[int(qpu_id)])
        self.reserve_wire(
            state,
            partition,
            qpu_id=int(qpu_id),
            local_wire=int(dst_wire),
            wire_kind=dst_wire_kind,
            owner_kind="resident",
            logical_qid=int(logical_qid),
            label=label,
        )

    def _repair_local_compaction(
        self,
        state: EvaluationState,
        transition_partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        target_qpus = self._partition_qpus(target_partition)
        for qpu_id in range(network.num_backends):
            target_comp_count = len(target_partition[qpu_id])
            for q, expected_qpu in target_qpus.items():
                if expected_qpu != qpu_id:
                    continue
                loc = state.logical_pos[int(q)]
                if loc.qpu_id != qpu_id:
                    raise RuntimeError(
                        f"[EVALUATOR] logical qubit has not reached target QPU before compaction: "
                        f"q={q}, expected_qpu={qpu_id}, actual={loc}"
                    )
                if loc.local_wire < target_comp_count:
                    continue
                dst_wire = self._select_free_target_comp_wire(state, target_partition, qpu_id)
                self._move_resident_with_local_swap(
                    state,
                    transition_partition,
                    network,
                    policy,
                    qpu_id=qpu_id,
                    logical_qid=int(q),
                    dst_wire=dst_wire,
                    dst_wire_kind="comp",
                    label="local-compaction-target",
                )

    def _process_cycle_teledata(
        self,
        graph: nx.DiGraph,
        state: EvaluationState,
        partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        cycles_by_length: dict[int, list[list[int]]] = defaultdict(list)
        for cycle in nx.simple_cycles(graph):
            if len(cycle) >= 3:
                cycles_by_length[len(cycle)].append([int(qpu_id) for qpu_id in cycle])

        for length in sorted(cycles_by_length):
            for cycle in cycles_by_length[length]:
                min_weight: int | None = None
                valid = True
                for idx in range(length):
                    src_qpu = cycle[idx]
                    dst_qpu = cycle[(idx + 1) % length]
                    if not graph.has_edge(src_qpu, dst_qpu):
                        valid = False
                        break
                    weight = int(graph[src_qpu][dst_qpu]["weight"])
                    if weight <= 0:
                        valid = False
                        break
                    min_weight = weight if min_weight is None else min(min_weight, weight)
                if not valid or min_weight is None:
                    continue

                for _ in range(min_weight):
                    moves: list[TeledataMove] = []
                    for idx in range(length):
                        src_qpu = cycle[idx]
                        dst_qpu = cycle[(idx + 1) % length]
                        q = int(graph[src_qpu][dst_qpu]["qubits"].pop(0))
                        moves.append(TeledataMove(q, src_qpu, dst_qpu))
                    self._process_teledata_batch(
                        moves,
                        "teledata-cycle",
                        state,
                        partition,
                        target_partition,
                        network,
                        policy,
                    )

                for idx in range(length):
                    self._decrement_or_remove_edge(graph, cycle[idx], cycle[(idx + 1) % length], min_weight)

    def _process_remaining_teledata(
        self,
        graph: nx.DiGraph,
        state: EvaluationState,
        partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy,
    ) -> None:
        while graph.number_of_edges() > 0:
            path = self._find_landing_safe_unit_path(graph, target_partition, state)
            if path is None:
                raise RuntimeError("remaining teledata graph has no landing-safe path")
            moves: list[TeledataMove] = []
            for src_qpu, dst_qpu, q in path:
                graph[src_qpu][dst_qpu]["qubits"].remove(q)
                moves.append(TeledataMove(q, src_qpu, dst_qpu))
                self._decrement_or_remove_edge(graph, src_qpu, dst_qpu)
            self._process_teledata_batch(
                moves,
                "teledata-path",
                state,
                partition,
                target_partition,
                network,
                policy,
            )

    def _find_landing_safe_unit_path(
        self,
        graph: nx.DiGraph,
        target_partition: list[list[int]],
        state: EvaluationState,
    ) -> list[tuple[int, int, int]] | None:
        for src_qpu, dst_qpu in list(graph.edges()):
            path: list[tuple[int, int, int]] = []
            visited_qpus: set[int] = set()
            curr_src = int(src_qpu)
            curr_dst = int(dst_qpu)
            curr_q = int(graph[curr_src][curr_dst]["qubits"][0])
            while True:
                if curr_src in visited_qpus:
                    raise RuntimeError("remaining teledata path unexpectedly forms a cycle")
                visited_qpus.add(curr_src)
                path.append((curr_src, curr_dst, curr_q))

                if self._has_free_target_comp_wire(state, target_partition, curr_dst):
                    return path
                next_edges = list(graph.out_edges(curr_dst))
                if not next_edges:
                    raise RuntimeError(
                        f"remaining teledata destination has no free target comp wire and no outgoing move: qpu={curr_dst}"
                    )
                _next_src, next_dst = next_edges[0]
                blocking_q = int(graph[curr_dst][next_dst]["qubits"][0])
                curr_src = curr_dst
                curr_dst = int(next_dst)
                curr_q = blocking_q
        return None

    def require_comp_wire_available_for_landing(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        qpu_id: int,
        local_wire: int,
    ) -> None:
        if local_wire < 0 or local_wire >= len(partition[qpu_id]):
            raise RuntimeError(f"teledata landing target is not a comp wire: qpu={qpu_id}, wire={local_wire}")
        owner = state.wire_owners[qpu_id][local_wire]
        if owner is not None:
            raise RuntimeError(
                f"teledata landing target comp wire is occupied: qpu={qpu_id}, wire={local_wire}, owner={owner}"
            )

    def _has_free_target_comp_wire(
        self,
        state: EvaluationState,
        target_partition: list[list[int]],
        qpu_id: int,
    ) -> bool:
        target_comp_count = len(target_partition[int(qpu_id)])
        return any(state.wire_owners[int(qpu_id)][wire] is None for wire in range(target_comp_count))

    def _select_free_target_comp_wire(
        self,
        state: EvaluationState,
        target_partition: list[list[int]],
        qpu_id: int,
    ) -> int:
        target_comp_count = len(target_partition[int(qpu_id)])
        for wire in range(target_comp_count):
            if state.wire_owners[int(qpu_id)][wire] is None:
                return wire
        raise RuntimeError(f"[EVALUATOR] no free target comp wire on qpu={qpu_id}")

    def _append_move_to_graph(self, graph: nx.DiGraph, src_qpu: int, dst_qpu: int, logical_qid: int) -> None:
        if graph.has_edge(src_qpu, dst_qpu):
            graph[src_qpu][dst_qpu]["qubits"].append(int(logical_qid))
            graph[src_qpu][dst_qpu]["weight"] += 1
        else:
            graph.add_edge(src_qpu, dst_qpu, weight=1, qubits=[int(logical_qid)])

    def _decrement_or_remove_edge(
        self,
        graph: nx.DiGraph,
        src_qpu: int,
        dst_qpu: int,
        amount: int = 1,
    ) -> None:
        graph[src_qpu][dst_qpu]["weight"] -= int(amount)
        if graph[src_qpu][dst_qpu]["weight"] != len(graph[src_qpu][dst_qpu]["qubits"]):
            raise RuntimeError(
                f"teledata graph edge weight mismatch: edge={src_qpu}->{dst_qpu}, "
                f"weight={graph[src_qpu][dst_qpu]['weight']}, qubits={graph[src_qpu][dst_qpu]['qubits']}"
            )
        if graph[src_qpu][dst_qpu]["weight"] == 0:
            graph.remove_edge(src_qpu, dst_qpu)

    def _partition_qpus(self, partition: list[list[int]]) -> dict[int, int]:
        qpus: dict[int, int] = {}
        for qpu_id, group in enumerate(partition):
            for logical_qid in group:
                q = int(logical_qid)
                if q < 0:
                    continue
                if q in qpus:
                    raise ValueError(f"[EVALUATOR] logical qubit appears in multiple partition groups: q{q}")
                qpus[q] = int(qpu_id)
        return qpus

    def _partition_from_state_residents(self, state: EvaluationState) -> list[list[int]]:
        if not state.wire_owners:
            return []
        max_qpu = max(int(qpu_id) for qpu_id in state.wire_owners)
        partition: list[list[int]] = [[] for _ in range(max_qpu + 1)]
        for qpu_id, owners in state.wire_owners.items():
            for owner in owners:
                if owner is not None and owner.kind == "resident" and owner.logical_qid is not None:
                    partition[int(qpu_id)].append(int(owner.logical_qid))
        return partition

    def _build_transition_partition(
        self,
        prev_partition: list[list[int]],
        target_partition: list[list[int]],
    ) -> list[list[int]]:
        transition: list[list[int]] = []
        for qpu_id in range(len(target_partition)):
            prev_group = list(prev_partition[qpu_id])
            target_group = list(target_partition[qpu_id])
            comp_count = max(len(prev_group), len(target_group))
            group = [-1 for _ in range(comp_count)]
            for wire, q in enumerate(prev_group):
                group[wire] = int(q)
            transition.append(group)
        return transition

    def _resize_state_to_transition_partition(
        self,
        state: EvaluationState,
        prev_partition: list[list[int]],
        transition_partition: list[list[int]],
        network: Network,
    ) -> None:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        for qpu_id in range(network.num_backends):
            old_comp_count = len(prev_partition[qpu_id])
            new_comp_count = len(transition_partition[qpu_id])
            old_wire_count = old_comp_count + reserve
            new_wire_count = new_comp_count + reserve
            old_phy = list(state.wire_phy_map.get(qpu_id, []))
            old_owners = list(state.wire_owners.get(qpu_id, []))
            if len(old_phy) < old_wire_count:
                old_phy.extend([None] * (old_wire_count - len(old_phy)))
            if len(old_owners) < old_wire_count:
                old_owners.extend([None] * (old_wire_count - len(old_owners)))

            new_phy: list[int | None] = [None for _ in range(new_wire_count)]
            new_owners: list[WireOwner | None] = [None for _ in range(new_wire_count)]
            for wire in range(min(old_comp_count, new_comp_count)):
                new_phy[wire] = old_phy[wire]
                new_owners[wire] = old_owners[wire]
            for offset in range(reserve):
                old_wire = old_comp_count + offset
                new_wire = new_comp_count + offset
                if old_wire < len(old_phy) and new_wire < len(new_phy):
                    new_phy[new_wire] = old_phy[old_wire]
                    new_owners[new_wire] = old_owners[old_wire]

            state.wire_phy_map[qpu_id] = new_phy
            state.wire_owners[qpu_id] = new_owners

    def _compact_state_to_target_partition(
        self,
        state: EvaluationState,
        transition_partition: list[list[int]],
        target_partition: list[list[int]],
        network: Network,
        policy: EvaluationPolicy | None = None,
    ) -> None:
        deferred_mode = self._is_deferred(policy)
        for qpu_id, buffer in enumerate(state.routed_buffers):
            if not deferred_mode and buffer.size() != 0:
                raise RuntimeError(
                    f"[EVALUATOR] cannot compact state with pending local ops: "
                    f"qpu={qpu_id}, buffer_size={buffer.size()}"
                )
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        target_qpus = self._partition_qpus(target_partition)
        for q, target_qpu in target_qpus.items():
            actual_loc = state.logical_pos.get(q)
            if actual_loc is None or actual_loc.qpu_id != target_qpu or actual_loc.local_wire >= len(target_partition[target_qpu]):
                raise RuntimeError(
                    f"[EVALUATOR] cannot compact before logical qubit reaches target QPU comp space: "
                    f"q={q}, expected_qpu={target_qpu}, actual={actual_loc}"
                )

        for qpu_id in range(network.num_backends):
            transition_comp = len(transition_partition[qpu_id])
            target_comp = len(target_partition[qpu_id])
            new_wire_count = target_comp + reserve
            new_phy: list[int | None] = [None for _ in range(new_wire_count)]
            new_owners: list[WireOwner | None] = [None for _ in range(new_wire_count)]

            for wire in range(target_comp):
                new_phy[wire] = state.wire_phy_map[qpu_id][wire]
                new_owners[wire] = state.wire_owners[qpu_id][wire]

            for offset in range(reserve):
                old_wire = transition_comp + offset
                new_wire = target_comp + offset
                if old_wire < len(state.wire_phy_map[qpu_id]):
                    new_phy[new_wire] = state.wire_phy_map[qpu_id][old_wire]
                    new_owners[new_wire] = state.wire_owners[qpu_id][old_wire]

            for wire in range(target_comp, transition_comp):
                owner = state.wire_owners[qpu_id][wire]
                if owner is not None:
                    raise RuntimeError(
                        f"[EVALUATOR] cannot compact occupied extra transition comp wire: "
                        f"qpu={qpu_id}, wire={wire}, owner={owner}"
                    )

            state.wire_phy_map[qpu_id] = new_phy
            state.wire_owners[qpu_id] = new_owners

        if not deferred_mode:
            self._prepare_local_buffers(state, target_partition, network)

    def _require_all_residents_in_comp_space(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        context: str,
    ) -> None:
        for q, loc in state.logical_pos.items():
            comp_wire_count = len(partition[int(loc.qpu_id)])
            if int(loc.local_wire) >= int(comp_wire_count):
                raise RuntimeError(
                    f"[EVALUATOR][{context}] logical qubit is not in computation wire space at record boundary: "
                    f"q={q}, loc={loc}, comp_wire_count={comp_wire_count}"
                )

    def _commit_state_to_record(
        self,
        record: MappingRecord,
        state: EvaluationState,
        partition: list[list[int]],
    ) -> None:
        record.costs = state.costs
        record.logical_phy_map = {}
        for q, loc in state.logical_pos.items():
            phy = state.wire_phy_map[loc.qpu_id][loc.local_wire]
            record.logical_phy_map[int(q)] = (int(loc.qpu_id), None if phy is None else int(phy))

        record.comm_phy_map = {}
        for qpu_id, wires in state.wire_phy_map.items():
            comp_count = len(partition[int(qpu_id)])
            record.comm_phy_map[int(qpu_id)] = [
                None if phy is None else int(phy)
                for phy in wires[comp_count:]
            ]

    def _validate_physical_state(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        network: Network,
        context: str,
    ) -> None:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        backend_qubits = [int(backend.num_qubits) for backend in network.backends]

        for qpu_id in range(network.num_backends):
            expected_wire_count = len(partition[qpu_id]) + reserve
            if len(state.wire_phy_map.get(qpu_id, [])) != expected_wire_count:
                raise RuntimeError(
                    f"[EVALUATOR][{context}] wire_phy_map length mismatch: "
                    f"qpu={qpu_id}, actual={len(state.wire_phy_map.get(qpu_id, []))}, expected={expected_wire_count}"
                )
            if len(state.wire_owners.get(qpu_id, [])) != expected_wire_count:
                raise RuntimeError(
                    f"[EVALUATOR][{context}] wire_owners length mismatch: "
                    f"qpu={qpu_id}, actual={len(state.wire_owners.get(qpu_id, []))}, expected={expected_wire_count}"
                )

            seen_phy: dict[int, int] = {}
            for local_wire, phy in enumerate(state.wire_phy_map[qpu_id]):
                if phy is None:
                    continue
                phy = int(phy)
                if phy < 0 or phy >= backend_qubits[qpu_id]:
                    raise RuntimeError(
                        f"[EVALUATOR][{context}] physical qubit out of range: "
                        f"qpu={qpu_id}, wire={local_wire}, phy={phy}, backend_qubits={backend_qubits[qpu_id]}"
                    )
                if phy in seen_phy:
                    raise RuntimeError(
                        f"[EVALUATOR][{context}] duplicate physical qubit on qpu: "
                        f"qpu={qpu_id}, phy={phy}, wires=({seen_phy[phy]}, {local_wire})"
                    )
                seen_phy[phy] = local_wire

        resident_reverse: dict[int, RuntimeLocation] = {}
        for qpu_id, owners in state.wire_owners.items():
            comp_wire_count = len(partition[int(qpu_id)])
            for local_wire, owner in enumerate(owners):
                if owner is None:
                    continue
                if local_wire >= comp_wire_count and owner.kind not in {"resident", "entangled_copy", "protocol"}:
                    raise RuntimeError(f"[EVALUATOR][{context}] invalid communication wire owner: {owner}")
                if owner.kind == "resident":
                    if owner.logical_qid is None:
                        raise RuntimeError(f"[EVALUATOR][{context}] resident owner missing logical_qid")
                    q = int(owner.logical_qid)
                    if q in resident_reverse:
                        raise RuntimeError(f"[EVALUATOR][{context}] duplicate resident owner for q{q}")
                    resident_reverse[q] = RuntimeLocation(qpu_id=int(qpu_id), local_wire=int(local_wire))
                elif owner.kind == "entangled_copy":
                    if owner.logical_qid is None or int(owner.logical_qid) not in state.logical_pos:
                        raise RuntimeError(f"[EVALUATOR][{context}] entangled copy references unknown logical qubit: {owner}")
                    if local_wire < comp_wire_count:
                        raise RuntimeError(
                            f"[EVALUATOR][{context}] entangled_copy must occupy comm wire: "
                            f"qpu={qpu_id}, wire={local_wire}, comp_wire_count={comp_wire_count}"
                        )
                elif owner.kind != "protocol":
                    raise RuntimeError(f"[EVALUATOR][{context}] unknown owner kind: {owner}")
                elif local_wire < comp_wire_count:
                    raise RuntimeError(
                        f"[EVALUATOR][{context}] protocol owner must occupy comm wire: "
                        f"qpu={qpu_id}, wire={local_wire}, comp_wire_count={comp_wire_count}"
                    )

        if set(resident_reverse.keys()) != set(state.logical_pos.keys()):
            raise RuntimeError(
                f"[EVALUATOR][{context}] resident/logical_pos key mismatch: "
                f"resident_only={sorted(set(resident_reverse) - set(state.logical_pos))}, "
                f"logical_only={sorted(set(state.logical_pos) - set(resident_reverse))}"
            )
        for q, loc in state.logical_pos.items():
            if loc.qpu_id < 0 or loc.qpu_id >= network.num_backends:
                raise RuntimeError(
                    f"[EVALUATOR][{context}] logical_pos qpu out of range: q={q}, loc={loc}"
                )
            if loc.local_wire < 0 or loc.local_wire >= len(state.wire_owners[loc.qpu_id]):
                raise RuntimeError(
                    f"[EVALUATOR][{context}] logical_pos wire out of range: q={q}, loc={loc}"
                )
            if loc != resident_reverse[int(q)]:
                raise RuntimeError(
                    f"[EVALUATOR][{context}] logical_pos and resident owner disagree: "
                    f"q={q}, loc={loc}, owner_loc={resident_reverse[int(q)]}"
                )

        target_qpus = self._partition_qpus(partition)
        if set(target_qpus.keys()) != set(state.logical_pos.keys()):
            raise RuntimeError(
                f"[EVALUATOR][{context}] partition/logical_pos qubit mismatch: "
                f"partition_only={sorted(set(target_qpus) - set(state.logical_pos))}, "
                f"state_only={sorted(set(state.logical_pos) - set(target_qpus))}"
            )

    def _require_logical_positions_match_partition_membership(
        self,
        state: EvaluationState,
        partition: list[list[int]],
        context: str,
    ) -> None:
        target_qpus = self._partition_qpus(partition)
        for q, target_qpu in target_qpus.items():
            actual_loc = state.logical_pos.get(q)
            if actual_loc is None or actual_loc.qpu_id != target_qpu or actual_loc.local_wire >= len(partition[target_qpu]):
                raise RuntimeError(
                    f"[EVALUATOR][{context}] logical location does not match target partition membership: "
                    f"q={q}, expected_qpu={target_qpu}, actual={actual_loc}"
                )
