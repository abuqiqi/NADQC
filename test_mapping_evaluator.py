import dataclasses
import math

import networkx as nx
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.compiler import MappingRecord, MappingRecordList
from src.compiler.compiler_utils import CommOp, CompilerUtils, ExecCosts
from src.compiler.evaluator import (
    EvaluationPolicy,
    LocalGateKind,
    MappingEvaluator,
    RuntimeLocation,
    WireOwner,
)
from src.navi.navi_compiler import CompilationContext, NAVI
from src.navi.telegate_partitioner import CatEntPartitioner, OEEPartitioner


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

    def get_backend_qubit_counts(self, include_comm_slot=False):
        del include_comm_slot
        return self.backend_sizes


def _phy(state, q):
    loc = state.logical_pos[q]
    return state.wire_phy_map[loc.qpu_id][loc.local_wire]


def _immediate(policy):
    return dataclasses.replace(policy, local_eval_mode="immediate")


def test_evaluate_raw_mapping_records_reads_optimization_level_config(monkeypatch):
    captured = {}

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured["optimization_level"] = policy.optimization_level
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    records = MappingRecordList(records=[])
    result = CompilerUtils.evaluate_raw_mapping_records(
        records,
        _Network(),
        config={"optimization_level": 2},
    )

    assert result is records
    assert captured["optimization_level"] == 2


def test_evaluate_raw_mapping_records_keeps_default_optimization_level(monkeypatch):
    captured = {}

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured["optimization_level"] = policy.optimization_level
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    CompilerUtils.evaluate_raw_mapping_records(MappingRecordList(records=[]), _Network())

    assert captured["optimization_level"] == 3


def test_evaluate_raw_mapping_records_defaults_to_deferred_local_eval(monkeypatch):
    captured = {}

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured["local_eval_mode"] = policy.local_eval_mode
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    CompilerUtils.evaluate_raw_mapping_records(MappingRecordList(records=[]), _Network())

    assert captured["local_eval_mode"] == "deferred"


def test_evaluate_raw_mapping_records_allows_immediate_local_eval_override(monkeypatch):
    captured = {}

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured["local_eval_mode"] = policy.local_eval_mode
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    CompilerUtils.evaluate_raw_mapping_records(
        MappingRecordList(records=[]),
        _Network(),
        config={"local_eval_mode": "immediate"},
    )

    assert captured["local_eval_mode"] == "immediate"


def test_evaluate_raw_mapping_records_defaults_telegate_choice_only_for_navi(monkeypatch):
    captured = []

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured.append(policy.evaluate_telegate_execution_choice)
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    CompilerUtils.evaluate_raw_mapping_records(
        MappingRecordList(records=[]),
        _Network(),
        config={"_compiler_id": "staticoee"},
    )
    CompilerUtils.evaluate_raw_mapping_records(
        MappingRecordList(records=[]),
        _Network(),
        config={"_compiler_id": "navi"},
    )

    assert captured == [False, True]


def test_evaluation_policy_defaults_to_telegate_execution_choice():
    assert EvaluationPolicy(name="test").evaluate_telegate_execution_choice is True


def test_evaluate_raw_mapping_records_allows_disabling_navi_telegate_choice(monkeypatch):
    captured = {}

    def fake_evaluate(self, mapping_record_list, network, policy):
        del self, network
        captured["enabled"] = policy.evaluate_telegate_execution_choice
        return mapping_record_list

    monkeypatch.setattr(MappingEvaluator, "evaluate", fake_evaluate)

    CompilerUtils.evaluate_raw_mapping_records(
        MappingRecordList(records=[]),
        _Network(),
        config={
            "_compiler_id": "navi",
            "evaluate_telegate_execution_choice": False,
        },
    )

    assert captured["enabled"] is False


def test_evaluate_raw_mapping_records_rejects_unknown_optimization_level():
    with pytest.raises(ValueError, match="unknown optimization_level: 4"):
        CompilerUtils.evaluate_raw_mapping_records(
            MappingRecordList(records=[]),
            _Network(),
            config={"optimization_level": 4},
        )


def _commop_record(mapping_type="cat", with_global_lqids=True, simple_telegate_ops=None):
    gate = Gate("x", 1, [])
    if with_global_lqids:
        setattr(gate, "_global_lqids", [0])
    comm = CommOp(
        comm_type="cat",
        source_qubit=0,
        src_qpu=0,
        dst_qpu=1,
        involved_qubits=[0, 1],
        gate_list=[gate],
    )
    circuit = QuantumCircuit(2)
    circuit.append(comm, [0, 1])
    extra_info = {"ops": circuit}
    if simple_telegate_ops is not None:
        extra_info["simple_telegate_ops"] = simple_telegate_ops
    return MappingRecord(
        partition=[[0], [1]],
        mapping_type=mapping_type,
        extra_info=extra_info,
    )


def test_oee_partitioner_does_not_save_simple_telegate_ops(monkeypatch):
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    cat_ops = QuantumCircuit(2)
    cat_ops.append(CommOp("cat", 0, 0, 1, [0, 1], []), [0, 1])

    def fake_partition(partition, qig, network, iteration_count):
        del qig, network, iteration_count
        return partition

    def fake_build_cat_comm_ops(self, subcircuit, partition):
        del self, subcircuit, partition
        return cat_ops, {"folded_blocks": 1}

    def fake_evaluate_telegate_with_cat(record_or_partition, subcircuit, network):
        del record_or_partition, subcircuit, network
        return ExecCosts()

    monkeypatch.setattr("src.partitioning.OEE.partition", fake_partition)
    monkeypatch.setattr(OEEPartitioner, "_build_cat_comm_ops", fake_build_cat_comm_ops)
    monkeypatch.setattr(
        CompilerUtils,
        "evaluate_telegate_with_cat",
        fake_evaluate_telegate_with_cat,
    )

    record = OEEPartitioner().partition(
        circuit,
        _Network(comm_slot_reserve=1),
        {
            "partition": [[0], [1]],
            "enable_cat_telegate": True,
            "debug_cat_telegate": False,
        },
    )

    assert record.extra_info["ops"] is cat_ops
    assert "simple_telegate_ops" not in record.extra_info


def test_cat_ent_partitioner_does_not_save_simple_telegate_ops(monkeypatch):
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    ops = QuantumCircuit(2)
    ops.append(CommOp("cat", 0, 0, 1, [0, 1], []), [0, 1])

    def fake_build(circuit, network, partition=None):
        del circuit, network
        assert partition == [[0], [1]]
        return MappingRecordList(
            records=[
                MappingRecord(
                    partition=[[0], [1]],
                    mapping_type="cat",
                    extra_info={"ops": ops},
                )
            ]
        )

    def fake_comm_only_evaluate(record, record_ops, network):
        del record_ops, network
        record.costs.epairs += 1
        record.costs.cat_ents += 1
        return record.costs

    monkeypatch.setattr(
        "src.navi.telegate_partitioner.build_autocomm_records",
        fake_build,
    )
    monkeypatch.setattr(
        CompilerUtils,
        "evaluate_telegate_with_cat",
        fake_comm_only_evaluate,
    )
    monkeypatch.setattr(
        CompilerUtils,
        "evaluate_raw_mapping_records",
        lambda *args, **kwargs: pytest.fail("CatEntPartitioner used the full evaluator"),
    )

    records = CatEntPartitioner().partition(
        circuit,
        _Network(comm_slot_reserve=1),
        {"partition": [[0], [1]], "use_oee_init": False},
    )

    record = records.records[0]
    assert any(
        isinstance(instruction.operation, CommOp)
        for instruction in record.extra_info["ops"]
    )
    assert "simple_telegate_ops" not in record.extra_info
    assert record.extra_info["comm_only_costs"] is True
    assert record.costs.epairs == 1
    assert records.total_costs.epairs == 1


def test_navi_full_circuit_range_sets_simple_telegate_ops_from_static_oee_view():
    ctx_circuit = QuantumCircuit(2)
    ctx_circuit.x(0)
    ctx_circuit.cz(0, 1)
    ctx_circuit.x(1)
    partitioner_circuit = QuantumCircuit(2)
    partitioner_circuit.cz(0, 1)
    record = MappingRecord(
        partition=[[0], [1]],
        mapping_type="cat",
        extra_info={"ops": partitioner_circuit},
    )
    records = MappingRecordList(records=[record])
    ctx = CompilationContext(
        circuit=ctx_circuit,
        network=_Network(comm_slot_reserve=1),
        config={},
        circuit_layers=[[], [], []],
    )

    NAVI()._set_simple_telegate_ops_for_full_circuit_record(
        ctx,
        records,
        ori_left=0,
        ori_right=2,
    )

    saved_simple = record.extra_info["simple_telegate_ops"]
    assert saved_simple is not ctx_circuit
    static_oee_layers = CompilerUtils.build_circuit_layers(ctx_circuit)
    expected_simple = CompilerUtils.get_subcircuit_by_level(
        num_qubits=ctx_circuit.num_qubits,
        circuit=ctx_circuit,
        circuit_layers=static_oee_layers,
        layer_start=0,
        layer_end=len(static_oee_layers) - 1,
    )
    assert saved_simple.count_ops() == {"x": 2, "cz": 1}
    assert [
        (
            instruction.operation.name,
            tuple(saved_simple.qubits.index(qubit) for qubit in instruction.qubits),
        )
        for instruction in saved_simple
    ] == [
        (
            instruction.operation.name,
            tuple(expected_simple.qubits.index(qubit) for qubit in instruction.qubits),
        )
        for instruction in expected_simple
    ]
    assert record.extra_info["ops"].count_ops() == {"cz": 1}


def test_navi_non_full_range_removes_simple_telegate_ops_and_keeps_ops():
    ctx_circuit = QuantumCircuit(2)
    ctx_circuit.x(0)
    ctx_circuit.cz(0, 1)
    partitioner_circuit = QuantumCircuit(2)
    partitioner_circuit.append(CommOp("cat", 0, 0, 1, [0, 1], []), [0, 1])
    stale_simple = QuantumCircuit(2)
    stale_simple.x(1)
    record = MappingRecord(
        partition=[[0], [1]],
        mapping_type="cat",
        extra_info={
            "ops": partitioner_circuit,
            "simple_telegate_ops": stale_simple,
        },
    )
    records = MappingRecordList(records=[record])
    ctx = CompilationContext(
        circuit=ctx_circuit,
        network=_Network(comm_slot_reserve=1),
        config={},
        circuit_layers=[[], [], []],
    )

    NAVI()._set_simple_telegate_ops_for_full_circuit_record(
        ctx,
        records,
        ori_left=0,
        ori_right=1,
    )

    assert "simple_telegate_ops" not in record.extra_info
    assert any(
        isinstance(instruction.operation, CommOp)
        for instruction in record.extra_info["ops"]
    )


def test_simple_telegate_ops_survives_extra_info_json_roundtrip():
    simple_ops = QuantumCircuit(2)
    simple_ops.rzz(0.25, 0, 1)

    encoded = MappingRecordList._serialize_extra_info(
        {
            "ops": QuantumCircuit(2),
            "simple_telegate_ops": simple_ops,
        }
    )
    decoded = MappingRecordList._deserialize_extra_info(encoded)

    assert decoded["simple_telegate_ops"].count_ops() == {"rzz": 1}


def test_single_record_telegate_choice_requires_saved_simple_ops(monkeypatch):
    calls = 0

    def fake_choice(self, *args, **kwargs):
        del self, args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("telegate choice should not run without simple_telegate_ops")

    monkeypatch.setattr(MappingEvaluator, "_evaluate_single_record_telegate_execution_choice", fake_choice)
    records = MappingRecordList(records=[_commop_record()])
    result = CompilerUtils.evaluate_raw_mapping_records(
        records,
        _Network(comm_slot_reserve=1),
        config={
            "_compiler_id": "navi",
            "evaluate_telegate_execution_choice": True,
            "deferred_route_local_gates": False,
        },
    )

    record = result.records[0]
    assert calls == 0
    assert "telegate_execution_choice" not in record.extra_info


def test_single_record_telegate_choice_prefers_saved_simple_ops(monkeypatch):
    saved_simple = QuantumCircuit(2)
    saved_simple.x(1)

    def fake_evaluate_body(self, record, subcircuit, network, policy, state):
        del self, record, network, policy
        if subcircuit is saved_simple or subcircuit.count_ops().get("x", 0) == 1:
            state.costs.local_fidelity_loss += 1.0
        elif any(isinstance(instruction.operation, CommOp) for instruction in subcircuit):
            state.costs.local_fidelity_loss += 10.0
        else:
            state.costs.local_fidelity_loss += 5.0

    monkeypatch.setattr(MappingEvaluator, "_evaluate_record_body", fake_evaluate_body)

    records = MappingRecordList(records=[_commop_record(simple_telegate_ops=saved_simple)])
    result = CompilerUtils.evaluate_raw_mapping_records(
        records,
        _Network(comm_slot_reserve=1),
        config={
            "_compiler_id": "navi",
            "evaluate_telegate_execution_choice": True,
            "deferred_route_local_gates": False,
        },
    )

    record = result.records[0]
    assert record.costs.total_fidelity_loss == 1.0
    assert record.extra_info["telegate_execution_choice"] == "simple"
    assert record.extra_info["telegate_execution_simple_source"] == "saved_original"
    assert record.extra_info["telegate_execution_choice_losses"] == {
        "commop": 10.0,
        "simple": 1.0,
    }


def test_telegate_choice_does_not_trigger_for_multiple_records(monkeypatch):
    calls = 0

    def fake_choice(self, *args, **kwargs):
        del self, args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("telegate choice should not run for multiple records")

    monkeypatch.setattr(MappingEvaluator, "_evaluate_single_record_telegate_execution_choice", fake_choice)

    saved_simple = QuantumCircuit(2)
    saved_simple.x(0)
    records = MappingRecordList(
        records=[
            _commop_record(simple_telegate_ops=saved_simple),
            _commop_record(simple_telegate_ops=saved_simple),
        ]
    )
    CompilerUtils.evaluate_raw_mapping_records(
        records,
        _Network(comm_slot_reserve=1),
        config={
            "_compiler_id": "navi",
            "evaluate_telegate_execution_choice": True,
            "deferred_route_local_gates": False,
        },
    )

    assert calls == 0


def test_telegate_choice_without_saved_simple_ops_evaluates_commop(monkeypatch):
    def fake_evaluate_body(self, record, subcircuit, network, policy, state):
        del self, record, subcircuit, network, policy
        state.costs.local_fidelity_loss += 3.0

    monkeypatch.setattr(MappingEvaluator, "_evaluate_record_body", fake_evaluate_body)

    records = MappingRecordList(records=[_commop_record(with_global_lqids=False)])
    result = CompilerUtils.evaluate_raw_mapping_records(
        records,
        _Network(comm_slot_reserve=1),
        config={
            "_compiler_id": "navi",
            "evaluate_telegate_execution_choice": True,
            "deferred_route_local_gates": False,
        },
    )

    record = result.records[0]
    assert record.costs.total_fidelity_loss == 3.0
    assert "telegate_execution_choice" not in record.extra_info


def test_evaluate_local_and_telegate_defaults_to_sequential_initial_layout():
    partition = [[0, 1], []]
    record = MappingRecord(
        partition=partition,
        logical_phy_map={0: (0, None), 1: (0, None)},
    )
    circuit = QuantumCircuit(2)

    _, logical_phy_map, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(
        record,
        circuit,
        _Network(),
    )

    assert logical_phy_map[0] == (0, 0)
    assert logical_phy_map[1] == (0, 1)


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
    policy = _immediate(EvaluationPolicy.full_realistic())
    state = evaluator._initialize_state_from_partition(partition, network, policy)

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
    policy = _immediate(EvaluationPolicy.local_all_to_all())
    state = evaluator._initialize_state_from_partition(partition, network, policy)

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
        policy,
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
    policy = _immediate(EvaluationPolicy.local_all_to_all())
    state = evaluator._initialize_state_from_partition(partition, network, policy)

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
        policy,
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
    policy = _immediate(EvaluationPolicy.local_all_to_all())
    state = evaluator._initialize_state_from_partition(partition, network, policy)

    evaluator._process_synthetic_telegate(
        partition,
        Gate("cx", 2, []),
        [0, 1],
        network,
        policy,
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


def test_full_cycle_partition_transition_lands_in_target_qpus_without_free_comp_space():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=3, comm_slot_reserve=2)
    prev_partition = [[0], [1], [2]]
    target_partition = [[2], [0], [1]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)

    evaluator._evaluate_partition_transition(
        prev_partition=prev_partition,
        target_partition=target_partition,
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos == {
        0: RuntimeLocation(qpu_id=1, local_wire=0),
        1: RuntimeLocation(qpu_id=2, local_wire=0),
        2: RuntimeLocation(qpu_id=0, local_wire=0),
    }
    assert state.costs.epairs == 3
    for qpu_id, expected_q in enumerate([2, 0, 1]):
        owner = state.wire_owners[qpu_id][0]
        assert owner.kind == "resident"
        assert owner.logical_qid == expected_q


def test_partition_transition_pair_swap_does_not_require_target_wire_order():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=2)
    prev_partition = [[0, 1], [2, 3]]
    target_partition = [[3, 1], [2, 0]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)

    evaluator._evaluate_partition_transition(
        prev_partition=prev_partition,
        target_partition=target_partition,
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos[3] == RuntimeLocation(qpu_id=0, local_wire=0)
    assert state.logical_pos[1] == RuntimeLocation(qpu_id=0, local_wire=1)
    assert state.logical_pos[2] == RuntimeLocation(qpu_id=1, local_wire=0)
    assert state.logical_pos[0] == RuntimeLocation(qpu_id=1, local_wire=1)
    assert state.wire_owners[0][0].kind == "resident"
    assert state.wire_owners[0][0].logical_qid == 3
    assert state.wire_owners[1][1].kind == "resident"
    assert state.wire_owners[1][1].logical_qid == 0
    assert state.costs.epairs == 2


def test_multi_weight_full_cycle_partition_transition_runs_unit_cycles():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=3, comm_slot_reserve=2)
    prev_partition = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    target_partition = [[6, 7, 8], [0, 1, 2], [3, 4, 5]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)

    evaluator._evaluate_partition_transition(
        prev_partition=prev_partition,
        target_partition=target_partition,
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    expected_qpus = {
        0: 1,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 0,
        7: 0,
        8: 0,
    }
    assert {q: loc.qpu_id for q, loc in state.logical_pos.items()} == expected_qpus
    assert all(loc.local_wire < 3 for loc in state.logical_pos.values())
    assert state.costs.epairs == 9


def test_one_way_partition_transition_to_free_destination_slot_succeeds():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=1)
    prev_partition = [[0, 1], [2]]
    target_partition = [[1], [2, 0]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)

    evaluator._evaluate_partition_transition(
        prev_partition=prev_partition,
        target_partition=target_partition,
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[0].local_wire < 2
    assert state.costs.epairs == 1


def test_one_way_partition_transition_to_full_destination_without_path_fails():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=2, comm_slot_reserve=1)
    prev_partition = [[0], [1]]
    target_partition = [[0], [1]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(network.num_backends))
    evaluator._append_move_to_graph(graph, 0, 1, 0)

    with pytest.raises(RuntimeError, match="no free target comp wire"):
        evaluator._process_remaining_teledata(
            graph=graph,
            state=state,
            partition=prev_partition,
            target_partition=target_partition,
            network=network,
            policy=EvaluationPolicy.local_all_to_all(),
        )


def test_one_way_partition_transition_uses_path_to_release_intermediate_slot():
    evaluator = MappingEvaluator()
    network = _Network(num_backends=3, comm_slot_reserve=2)
    prev_partition = [[0], [1], []]
    target_partition = [[], [0], [1]]
    state = evaluator._initialize_state_from_partition(prev_partition, network)

    evaluator._evaluate_partition_transition(
        prev_partition=prev_partition,
        target_partition=target_partition,
        network=network,
        policy=EvaluationPolicy.local_all_to_all(),
        state=state,
    )

    assert state.logical_pos[0].qpu_id == 1
    assert state.logical_pos[0].local_wire < 1
    assert state.logical_pos[1].qpu_id == 2
    assert state.logical_pos[1].local_wire < 1
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
