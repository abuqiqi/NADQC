import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.compiler import MappingRecord, MappingRecordList
from src.compiler.compiler_utils import CommOp
from src.compiler.evaluator import (
    EvaluationPolicy,
    LocalGateKind,
    MappingEvaluator,
    RuntimeLocation,
    WireOwner,
)


class _Backend:
    num_qubits = 16

    def __init__(self, coupling_map=None):
        self.coupling_map = coupling_map
        self.basis_gates = ["x", "h", "cx", "swap"]
        self.gate_dict = {
            "x0": {"gate_error_value": 0.1},
            "x": {"gate_error_value": 0.3},
            "h": {"gate_error_value": 0.05},
            "cx0_1": {"gate_error_value": 0.2},
            "cx": {"gate_error_value": 0.4},
            "swap": {"gate_error_value": 0.6},
        }


class _Network:
    def __init__(self, num_backends=2, comm_slot_reserve=0, coupling_map=None):
        self.backends = [_Backend(coupling_map=coupling_map) for _ in range(num_backends)]
        self.num_backends = num_backends
        self.comm_slot_reserve = comm_slot_reserve
        self.backend_sizes = [16 for _ in range(num_backends)]
        self.Hops = [
            [0 if i == j else 1 for j in range(num_backends)]
            for i in range(num_backends)
        ]
        self.move_fidelity_loss = [
            [0.0 if i == j else 0.25 for j in range(num_backends)]
            for i in range(num_backends)
        ]
        self.move_fidelity = [
            [1.0 if i == j else 0.75 for j in range(num_backends)]
            for i in range(num_backends)
        ]


def _phy(state, q):
    loc = state.logical_pos[q]
    return state.wire_phy_map[loc.qpu_id][loc.local_wire]


def test_runtime_location_uses_only_local_wire():
    loc = RuntimeLocation(qpu_id=2, local_wire=3)

    assert loc.qpu_id == 2
    assert loc.local_wire == 3
    assert loc.is_comm(comp_wire_count=2)
    assert loc.comm_offset(comp_wire_count=2) == 1


def test_add_and_flush_payload_ops_updates_wire_physical_map():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0], []]
    state = evaluator._initialize_state_from_partition(partition, network)

    evaluator.add_local_ops(
        state,
        partition,
        network,
        EvaluationPolicy.full_realistic(),
        qpu_id=0,
        ops=[(Gate("x", 1, []), [0])],
        kind=LocalGateKind.PAYLOAD,
    )
    evaluator.flush_local_ops(state, partition, network, EvaluationPolicy.full_realistic())

    assert state.costs.local_transpile_calls == 1
    assert state.costs.local_gate_num == 1
    assert state.costs.local_pre_transpile_gate_num == 1
    assert state.costs.local_transpile_added_gate_num == 0
    assert state.routed_buffers[0].size() == 0
    assert _phy(state, 0) == 0


def test_deferred_uses_physical_sized_buffers_and_delays_transpile():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0], []]
    policy = EvaluationPolicy(
        name="deferred_test",
        route_payload_gates=False,
        route_comm_gates=False,
        local_eval_mode="deferred",
        deferred_route_local_gates=False,
    )
    state = evaluator._initialize_state_from_partition(partition, network, policy)

    assert state.routed_buffers[0].num_qubits == network.backends[0].num_qubits
    assert state.routed_buffers[1].num_qubits == network.backends[1].num_qubits

    evaluator.add_local_ops(
        state,
        partition,
        network,
        policy,
        qpu_id=0,
        ops=[(Gate("x", 1, []), [0])],
        kind=LocalGateKind.PAYLOAD,
    )
    evaluator.flush_local_ops(state, partition, network, policy)

    assert state.costs.local_transpile_calls == 0
    assert state.costs.local_gate_num == 0
    assert state.routed_buffers[0].size() > 0

    evaluator.flush_local_ops(state, partition, network, policy, final=True)

    assert state.costs.local_transpile_calls == 1
    assert state.costs.local_gate_num == 1
    assert state.costs.local_pre_transpile_gate_num == 1
    assert state.costs.local_transpile_added_gate_num == 0
    assert state.costs.local_uncategorized_gate_num == 1
    assert state.costs.local_gate_breakdown_gap == 0
    assert state.costs.payload_gate_num == 1
    assert state.routed_buffers[0].num_qubits == network.backends[0].num_qubits


def test_build_initial_layout_always_fills_unknown_wire_slots():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0, 1], []]
    state = evaluator._initialize_state_from_partition(partition, network)
    circuit = QuantumCircuit(2)

    layout = evaluator._build_initial_layout(
        state,
        qpu_id=0,
        circuit=circuit,
        backend=network.backends[0],
        policy=EvaluationPolicy.full_realistic(),
    )

    assert layout[circuit.qubits[0]] == 0
    assert layout[circuit.qubits[1]] == 1


def test_deferred_free_initial_layout_omits_transpile_initial_layout():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0], []]
    state = evaluator._initialize_state_from_partition(
        partition,
        network,
        EvaluationPolicy(
            name="deferred_free",
            local_eval_mode="deferred",
            deferred_initial_layout="free",
        ),
    )
    circuit = QuantumCircuit(network.backends[0].num_qubits)

    layout = evaluator._get_transpile_initial_layout(
        state,
        qpu_id=0,
        circuit=circuit,
        backend=network.backends[0],
        policy=EvaluationPolicy(
            name="deferred_free",
            local_eval_mode="deferred",
            deferred_initial_layout="free",
        ),
    )

    assert layout is None


def test_deferred_fixed_initial_layout_supplies_transpile_initial_layout():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0], []]
    state = evaluator._initialize_state_from_partition(
        partition,
        network,
        EvaluationPolicy(
            name="deferred_fixed",
            local_eval_mode="deferred",
            deferred_initial_layout="fixed",
        ),
    )
    circuit = QuantumCircuit(network.backends[0].num_qubits)

    layout = evaluator._get_transpile_initial_layout(
        state,
        qpu_id=0,
        circuit=circuit,
        backend=network.backends[0],
        policy=EvaluationPolicy(
            name="deferred_fixed",
            local_eval_mode="deferred",
            deferred_initial_layout="fixed",
        ),
    )

    assert layout is not None
    assert layout[circuit.qubits[0]] == 0


def test_cat_commop_uses_entangled_copy_without_moving_source():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    partition = [[0], [1]]
    state = evaluator._initialize_state_from_partition(partition, network)

    payload_gate = Gate("cx", 2, [])
    setattr(payload_gate, "_global_lqids", [0, 1])
    comm_op = CommOp(
        comm_type="cat",
        source_qubit=0,
        src_qpu=0,
        dst_qpu=1,
        involved_qubits=[0, 1],
        gate_list=[payload_gate],
    )

    evaluator._process_commop(
        partition,
        comm_op,
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.logical_pos[0] == RuntimeLocation(qpu_id=0, local_wire=0)
    assert state.costs.comm_block_events == 1
    assert state.costs.cat_ents == 1
    assert state.costs.payload_gate_num == 1
    assert state.costs.local_gate_num == 3
    assert math.isclose(state.costs.comm_block_remote_fidelity_loss, 0.25)
    assert state.wire_owners[0][1] is None
    assert state.wire_owners[1][1] is None


def test_tp_payload_moves_source_resident_to_destination_comm_wire():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    partition = [[0], [1]]
    state = evaluator._initialize_state_from_partition(partition, network)

    payload_gate = Gate("cx", 2, [])
    setattr(payload_gate, "_global_lqids", [0, 1])
    comm_op = CommOp(
        comm_type="tp",
        source_qubit=0,
        src_qpu=0,
        dst_qpu=1,
        involved_qubits=[0, 1],
        gate_list=[payload_gate],
    )

    evaluator._process_commop(
        partition,
        comm_op,
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.logical_pos[0] == RuntimeLocation(qpu_id=1, local_wire=1)
    assert state.wire_owners[0][0] is None
    assert state.wire_owners[1][1] == WireOwner(kind="resident", logical_qid=0, label="tp-dst-comm")
    assert state.costs.epairs == 1
    assert state.costs.payload_gate_num == 1


def test_synthetic_telegate_replays_cross_qpu_gate_as_cat_block():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    partition = [[0], [1]]
    state = evaluator._initialize_state_from_partition(partition, network)

    evaluator._process_synthetic_telegate(
        partition,
        Gate("cx", 2, []),
        [0, 1],
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.costs.epairs == 1
    assert state.costs.cat_ents == 1
    assert state.costs.comm_block_events == 0
    assert state.costs.telegate_exec_events == 1
    assert state.costs.payload_gate_num == 1
    assert state.costs.local_gate_num == 3


def test_evaluate_keeps_runtime_state_continuous_between_records():
    class ContinuousStateEvaluator(MappingEvaluator):
        def _get_record_subcircuit(self, record):
            return QuantumCircuit(1)

        def _evaluate_partition_transition(self, prev_partition, target_partition, network, policy, state):
            loc = state.logical_pos[0]
            state.wire_phy_map[loc.qpu_id][loc.local_wire] = 9
            state.costs.epairs += 1
            return state

        def _evaluate_record_body(self, record, subcircuit, network, policy, state):
            state.costs.local_gate_num += 1
            return state

    records = MappingRecordList()
    records.add_record(MappingRecord(partition=[[0], []]))
    records.add_record(MappingRecord(partition=[[0], []]))

    result = ContinuousStateEvaluator().evaluate(
        records,
        network=_Network(),
        policy=EvaluationPolicy.full_realistic(),
    )

    assert result.records[0].costs.local_gate_num == 1
    assert result.records[1].costs.local_gate_num == 1
    assert result.records[1].costs.epairs == 1
    assert result.records[1].logical_phy_map[0] == (0, 9)
    assert result.total_costs.local_gate_num == 2
    assert result.total_costs.epairs == 1


def test_partition_transition_handles_changed_local_wire_counts():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=1)
    state = evaluator._initialize_state_from_partition([[0, 1], [2, 3]], network)

    evaluator._evaluate_partition_transition(
        prev_partition=[[0, 1], [2, 3]],
        target_partition=[[0], [1, 2, 3]],
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos[0] == RuntimeLocation(qpu_id=0, local_wire=0)
    assert {state.logical_pos[q].qpu_id for q in [1, 2, 3]} == {1}
    assert {state.logical_pos[q].local_wire for q in [1, 2, 3]} == {0, 1, 2}
    assert len(state.wire_phy_map[0]) == 2
    assert len(state.wire_phy_map[1]) == 4
    assert state.wire_owners[0][0] == WireOwner(kind="resident", logical_qid=0)
    assert {
        state.wire_owners[1][wire].logical_qid
        for wire in range(3)
        if state.wire_owners[1][wire] is not None
    } == {1, 2, 3}
    assert state.costs.epairs == 1


def test_deferred_partition_transition_keeps_pending_buffer_until_final_flush():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=2)
    policy = EvaluationPolicy(
        name="deferred_test",
        route_payload_gates=False,
        route_comm_gates=False,
        local_eval_mode="deferred",
        deferred_route_local_gates=False,
    )
    state = evaluator._initialize_state_from_partition([[0], [1]], network, policy)

    evaluator._evaluate_partition_transition(
        prev_partition=[[0], [1]],
        target_partition=[[1], [0]],
        network=network,
        policy=policy,
        state=state,
    )

    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[1].qpu_id == 0
    assert state.costs.epairs == 2
    assert state.costs.local_transpile_calls == 0
    assert any(buffer.size() > 0 for buffer in state.routed_buffers)

    evaluator.flush_local_ops(state, [[1], [0]], network, policy, final=True)

    assert state.costs.local_transpile_calls == 2
    assert state.costs.local_gate_num > 0
    assert state.costs.local_pre_transpile_gate_num > 0
    assert state.costs.local_transpile_added_gate_num == (
        state.costs.local_gate_num - state.costs.local_pre_transpile_gate_num
    )
    assert state.costs.local_uncategorized_gate_num == state.costs.local_gate_num
    assert state.costs.local_gate_breakdown_gap == 0


def test_bidirectional_partition_transition_requires_two_comm_wires_per_qpu():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=1)
    state = evaluator._initialize_state_from_partition([[0], [1]], network)

    with pytest.raises(RuntimeError, match="insufficient communication wires"):
        evaluator._evaluate_partition_transition(
            prev_partition=[[0], [1]],
            target_partition=[[1], [0]],
            network=network,
            policy=EvaluationPolicy.local_all_to_all(),
            state=state,
        )


def test_bidirectional_partition_transition_with_two_comm_wires_per_qpu():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=2)
    state = evaluator._initialize_state_from_partition([[0], [1]], network)

    evaluator._evaluate_partition_transition(
        prev_partition=[[0], [1]],
        target_partition=[[1], [0]],
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[1].qpu_id == 0
    assert state.costs.epairs == 2


def test_partition_order_change_does_not_force_local_reindex():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=1, comm_slot_reserve=1)
    state = evaluator._initialize_state_from_partition([[0, 1, 2]], network)

    evaluator._evaluate_partition_transition(
        prev_partition=[[0, 1, 2]],
        target_partition=[[2, 0, 1]],
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos == {
        0: RuntimeLocation(qpu_id=0, local_wire=0),
        1: RuntimeLocation(qpu_id=0, local_wire=1),
        2: RuntimeLocation(qpu_id=0, local_wire=2),
    }
    assert state.costs.local_gate_num == 0
    assert state.costs.epairs == 0
