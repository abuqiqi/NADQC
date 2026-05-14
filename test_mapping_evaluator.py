import math

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.compiler import MappingRecord, MappingRecordList
from src.compiler.compiler_utils import CommOp
from src.compiler.evaluator import EvaluationPolicy, EvaluationState, MappingEvaluator, RuntimeLocation


class _Backend:
    num_qubits = 16

    def __init__(self):
        self.coupling_map = None
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
    def __init__(self, num_backends=2, comm_slot_reserve=0):
        self.backends = [_Backend() for _ in range(num_backends)]
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


def test_unrouted_gate_cost_uses_exact_physical_calibration():
    state = EvaluationState(
        logical_pos={
            0: RuntimeLocation(qpu_id=0, local_slot=0, physical_slot=5),
            1: RuntimeLocation(qpu_id=0, local_slot=1, physical_slot=7),
        },
        comm_phy_map={0: []},
        routed_buffers=[QuantumCircuit(2)],
    )

    MappingEvaluator()._accumulate_unrouted_gate_cost(
        state,
        qpu_id=0,
        gate=Gate("cx", 2, []),
        local_slots=[0, 1],
        network=_Network(),
    )

    assert state.costs.local_gate_num == 1
    assert math.isclose(state.costs.local_fidelity_loss, 0.2)
    assert math.isclose(state.costs.local_fidelity, 0.8)


def test_unrouted_gate_cost_falls_back_to_average_when_physical_unknown():
    state = EvaluationState(
        logical_pos={
            0: RuntimeLocation(qpu_id=0, local_slot=1, physical_slot=None),
            1: RuntimeLocation(qpu_id=0, local_slot=2, physical_slot=None),
        },
        comm_phy_map={0: []},
        routed_buffers=[QuantumCircuit(3)],
    )

    MappingEvaluator()._accumulate_unrouted_gate_cost(
        state,
        qpu_id=0,
        gate=Gate("cx", 2, []),
        local_slots=[1, 2],
        network=_Network(),
    )

    assert state.costs.local_gate_num == 1
    assert math.isclose(state.costs.local_fidelity_loss, 0.4)
    assert math.isclose(state.costs.local_fidelity, 0.6)


def test_evaluate_keeps_state_continuous_between_records():
    class ContinuousStateEvaluator(MappingEvaluator):
        def _get_record_subcircuit(self, record, circuit, circuit_layers):
            return QuantumCircuit(1)

        def _evaluate_teledata(self, target_partition, network, policy, state):
            state.logical_pos[0].physical_slot = 9
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
        circuit=QuantumCircuit(1),
        circuit_layers=[],
        network=_Network(),
        policy=EvaluationPolicy.full_realistic(),
    )

    assert result.records[0].costs.local_gate_num == 1
    assert result.records[0].costs.epairs == 0
    assert result.records[1].costs.local_gate_num == 1
    assert result.records[1].costs.epairs == 1
    assert result.records[1].logical_phy_map[0] == (0, 9)
    assert result.total_costs.local_gate_num == 2
    assert result.total_costs.epairs == 1


def test_flush_local_buffers_charges_routed_gates_and_resets_buffer():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0], []]
    state = evaluator._initialize_state_from_partition(partition, network)

    evaluator._append_routed_gate(
        state,
        qpu_id=0,
        gate=Gate("x", 1, []),
        local_slots=[0],
    )

    evaluator._flush_local_buffers(
        state,
        partition,
        network,
        EvaluationPolicy.full_realistic(),
    )

    assert state.costs.local_transpile_calls == 1
    assert state.costs.local_gate_num == 1
    assert math.isclose(state.costs.local_fidelity_loss, 0.1)
    assert state.logical_pos[0].physical_slot == 0
    assert state.routed_buffers[0].size() == 0


def test_build_initial_layout_can_fill_unknown_physical_slots():
    evaluator = MappingEvaluator()
    network = _Network()
    partition = [[0, 1], []]
    state = evaluator._initialize_state_from_partition(partition, network)
    subcircuit = QuantumCircuit(2)

    partial_layout = evaluator._build_initial_layout_for_qpu(
        state,
        qpu_id=0,
        subcircuit=subcircuit,
        partition=partition,
        network=network,
        policy=EvaluationPolicy.full_realistic(),
    )
    assert partial_layout == {}

    filled_layout = evaluator._build_initial_layout_for_qpu(
        state,
        qpu_id=0,
        subcircuit=subcircuit,
        partition=partition,
        network=network,
        policy=EvaluationPolicy(
            name="filled",
            fill_initial_layout=True,
        ),
    )
    assert filled_layout[subcircuit.qubits[0]] == 0
    assert filled_layout[subcircuit.qubits[1]] == 1


def test_evaluate_teledata_minimal_move_cost_and_reindexing():
    evaluator = MappingEvaluator()
    network = _Network()
    curr_record = MappingRecord(partition=[[1], [0]])
    state = evaluator._initialize_state_from_partition([[0, 1], []], network)
    state.logical_pos[0].physical_slot = 5
    state.logical_pos[1].physical_slot = 6

    evaluator._evaluate_teledata(
        curr_record.partition,
        network,
        EvaluationPolicy.full_realistic(),
        state,
    )

    assert state.costs.epairs == 1
    assert state.costs.remote_hops == 1
    assert math.isclose(state.costs.remote_fidelity_loss, 0.25)
    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[0].local_slot == 0
    assert state.logical_pos[0].physical_slot is None
    assert state.logical_pos[1].qpu_id == 0
    assert state.logical_pos[1].local_slot == 0
    assert state.logical_pos[1].physical_slot == 6


def test_evaluate_teledata_uses_one_way_moves_for_reciprocal_exchange():
    evaluator = MappingEvaluator()
    network = _Network()
    state = evaluator._initialize_state_from_partition([[0], [1]], network)
    state.logical_pos[0].physical_slot = 5
    state.logical_pos[1].physical_slot = 6

    evaluator._evaluate_teledata(
        [[1], [0]],
        network,
        EvaluationPolicy.full_realistic(),
        state,
    )

    assert state.costs.epairs == 2
    assert state.costs.remote_hops == 2
    assert state.costs.remote_swaps == 0
    assert math.isclose(state.costs.remote_fidelity_loss, 0.5)
    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[0].physical_slot == 6
    assert state.logical_pos[1].qpu_id == 0
    assert state.logical_pos[1].physical_slot == 5


def test_evaluate_teledata_rotates_physical_slots_for_long_cycle():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=3)
    state = evaluator._initialize_state_from_partition([[0], [1], [2]], network)
    state.logical_pos[0].physical_slot = 5
    state.logical_pos[1].physical_slot = 6
    state.logical_pos[2].physical_slot = 7

    evaluator._evaluate_teledata(
        [[2], [0], [1]],
        network,
        EvaluationPolicy.full_realistic(),
        state,
    )

    assert state.costs.epairs == 3
    assert state.costs.remote_hops == 3
    assert state.costs.remote_swaps == 0
    assert math.isclose(state.costs.remote_fidelity_loss, 0.75)
    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[0].physical_slot == 6
    assert state.logical_pos[1].qpu_id == 2
    assert state.logical_pos[1].physical_slot == 7
    assert state.logical_pos[2].qpu_id == 0
    assert state.logical_pos[2].physical_slot == 5


def test_commop_cat_routes_comm_primitives_when_enabled():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    record = MappingRecord(partition=[[0], [1]])
    state = evaluator._initialize_state_from_partition(record.partition, network)

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
        record.partition,
        comm_op,
        network,
        EvaluationPolicy(
            name="route_comm_only",
            route_payload_gates=False,
            route_comm_gates=True,
        ),
        state,
    )

    assert state.costs.epairs == 1
    assert state.costs.cat_ents == 1
    assert state.costs.comm_block_events == 1
    assert state.costs.payload_gate_num == 1
    assert state.costs.local_transpile_calls == 2
    assert state.costs.local_gate_num == 3
    assert math.isclose(state.costs.remote_fidelity_loss, 0.25)
    assert math.isclose(state.costs.comm_block_remote_fidelity_loss, 0.25)


def test_commop_packed_local_gate_uses_its_runtime_qpu():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    record = MappingRecord(partition=[[0, 2], [1]])
    state = evaluator._initialize_state_from_partition(record.partition, network)

    payload_gate = Gate("cx", 2, [])
    setattr(payload_gate, "_global_lqids", [0, 1])
    packed_local_gate = Gate("x", 1, [])
    setattr(packed_local_gate, "_global_lqids", [2])
    comm_op = CommOp(
        comm_type="cat",
        source_qubit=0,
        src_qpu=0,
        dst_qpu=1,
        involved_qubits=[0, 1],
        gate_list=[payload_gate, packed_local_gate],
    )

    evaluator._process_commop(
        record.partition,
        comm_op,
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.costs.epairs == 1
    assert state.costs.cat_ents == 1
    assert state.costs.payload_gate_num == 2
    assert state.costs.local_gate_num == 4


def test_commop_tp_keeps_source_runtime_location():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    record = MappingRecord(partition=[[0], [1]])
    state = evaluator._initialize_state_from_partition(record.partition, network)
    original_qpu = state.logical_pos[0].qpu_id
    original_local_slot = state.logical_pos[0].local_slot

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
        record.partition,
        comm_op,
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.logical_pos[0].qpu_id == original_qpu
    assert state.logical_pos[0].local_slot == original_local_slot
    assert state.logical_pos[0].physical_slot is None


def test_empty_tp_return_block_lands_on_source_home_slot():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    record = MappingRecord(partition=[[0], [1]])
    state = evaluator._initialize_state_from_partition(record.partition, network)

    comm_op = CommOp(
        comm_type="tp",
        source_qubit=0,
        src_qpu=1,
        dst_qpu=0,
        involved_qubits=[0],
        gate_list=[],
    )

    evaluator._process_commop(
        record.partition,
        comm_op,
        network,
        EvaluationPolicy.local_all_to_all(),
        state,
    )

    assert state.costs.epairs == 1
    assert state.costs.payload_gate_num == 0
    assert state.costs.local_gate_num == 3
    assert state.logical_pos[0].qpu_id == 0
    assert state.logical_pos[0].local_slot == 0


def test_synthetic_telegate_replays_cross_qpu_gate_as_cat_block():
    evaluator = MappingEvaluator()
    network = _Network(comm_slot_reserve=1)
    record = MappingRecord(partition=[[0], [1]])
    state = evaluator._initialize_state_from_partition(record.partition, network)

    evaluator._process_synthetic_telegate(
        record,
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
    assert math.isclose(state.costs.telegate_exec_remote_fidelity_loss, 0.25)
