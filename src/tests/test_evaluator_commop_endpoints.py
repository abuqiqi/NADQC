import pytest

from qiskit.circuit import Gate

from src.compiler.compiler_utils import CommOp
from src.compiler.evaluator import EvaluationState, MappingEvaluator, RuntimeLocation, WireOwner


def _state() -> EvaluationState:
    return EvaluationState(
        logical_pos={
            0: RuntimeLocation(qpu_id=0, local_wire=0),
            1: RuntimeLocation(qpu_id=1, local_wire=0),
        },
        wire_phy_map={0: [0], 1: [0]},
        wire_owners={
            0: [WireOwner(kind="resident", logical_qid=0)],
            1: [WireOwner(kind="resident", logical_qid=1)],
        },
    )


def _tp_return_state() -> EvaluationState:
    return EvaluationState(
        logical_pos={
            0: RuntimeLocation(qpu_id=1, local_wire=1),
            1: RuntimeLocation(qpu_id=1, local_wire=0),
        },
        wire_phy_map={0: [None, None], 1: [0, 1]},
        wire_owners={
            0: [None, None],
            1: [
                WireOwner(kind="resident", logical_qid=1),
                WireOwner(kind="resident", logical_qid=0),
            ],
        },
    )


def _comm_op(src_qpu: int = 0, dst_qpu: int = 1) -> CommOp:
    gate = Gate("cx", 2, [])
    setattr(gate, "_global_lqids", [0, 1])
    return CommOp(
        comm_type="cat",
        source_qubit=0,
        src_qpu=src_qpu,
        dst_qpu=dst_qpu,
        involved_qubits=[0, 1],
        gate_list=[gate],
    )


def test_commop_runtime_endpoint_validation_accepts_current_state():
    evaluator = MappingEvaluator()
    comm_op = _comm_op()

    evaluator._validate_commop_runtime_endpoints(comm_op, 0, 1, _state())


def test_commop_runtime_endpoint_validation_rejects_stale_src_qpu():
    evaluator = MappingEvaluator()
    comm_op = _comm_op(src_qpu=1, dst_qpu=1)

    with pytest.raises(RuntimeError, match="src_qpu metadata inconsistent"):
        evaluator._validate_commop_runtime_endpoints(comm_op, 1, 1, _state())


def test_commop_runtime_endpoint_validation_rejects_stale_dst_qpu():
    evaluator = MappingEvaluator()
    comm_op = _comm_op(src_qpu=0, dst_qpu=0)

    with pytest.raises(RuntimeError, match="dst_qpu metadata inconsistent"):
        evaluator._validate_commop_runtime_endpoints(comm_op, 0, 0, _state())


def test_empty_tp_return_endpoint_validation_accepts_home_destination():
    evaluator = MappingEvaluator()
    comm_op = CommOp(
        comm_type="tp",
        source_qubit=0,
        src_qpu=1,
        dst_qpu=0,
        involved_qubits=[0],
        gate_list=[],
    )

    evaluator._validate_commop_runtime_endpoints(comm_op, 1, 0, _tp_return_state())


def test_empty_tp_return_endpoint_validation_rejects_non_home_destination():
    evaluator = MappingEvaluator()
    comm_op = CommOp(
        comm_type="tp",
        source_qubit=0,
        src_qpu=0,
        dst_qpu=1,
        involved_qubits=[0],
        gate_list=[],
    )

    with pytest.raises(RuntimeError, match="source does not match runtime QPU"):
        evaluator._validate_commop_runtime_endpoints(comm_op, 0, 1, _tp_return_state())
