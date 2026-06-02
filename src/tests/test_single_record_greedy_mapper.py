import copy
import math

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.compiler import CompilerUtils, MappingRecord, MappingRecordList
from src.compiler.compiler_utils import CommOp
from src.navi.mapper import SingleRecordGreedyMapper


class _Backend:
    num_qubits = 8
    coupling_map = None
    basis_gates = ["x", "h", "cx", "swap", "ccx"]
    gate_dict = {
        "x": {"gate_error_value": 0.01},
        "h": {"gate_error_value": 0.01},
        "cx": {"gate_error_value": 0.02},
        "swap": {"gate_error_value": 0.03},
        "ccx": {"gate_error_value": 0.04},
    }


class _Network:
    def __init__(self):
        self.backends = [_Backend(), _Backend(), _Backend()]
        self.num_backends = 3
        self.comm_slot_reserve = 1
        self.backend_sizes = [7, 7, 7]
        self.Hops = [
            [0, 1, 5],
            [1, 0, 1],
            [5, 1, 0],
        ]
        self.move_fidelity_loss = [
            [0.0, 0.1, 0.5],
            [0.1, 0.0, 0.1],
            [0.5, 0.1, 0.0],
        ]
        self.move_fidelity = [
            [1.0, 0.9, 0.5],
            [0.9, 1.0, 0.9],
            [0.5, 0.9, 1.0],
        ]


def _payload_gate(name: str, qids: list[int]) -> Gate:
    gate = Gate(name, len(qids), [])
    setattr(gate, "_global_lqids", [int(q) for q in qids])
    return gate


def test_single_record_estimator_uses_synthetic_telegate_source_to_first_remote_target():
    mapper = SingleRecordGreedyMapper()
    circuit = QuantumCircuit(3)
    circuit.append(Gate("ccx", 3, []), [0, 1, 2])

    comm = mapper._estimate_partition_comm_matrix_for_single_record(
        circuit,
        [[0], [1], [2]],
    )

    assert comm[0][1] == 1.0
    assert comm[1][0] == 1.0
    assert comm[1][2] == 0.0
    assert comm[0][2] == 0.0


def test_single_record_estimator_uses_commop_metadata_endpoints_and_weights():
    mapper = SingleRecordGreedyMapper()
    circuit = QuantumCircuit(3)
    circuit.append(
        CommOp(
            comm_type="cat",
            source_qubit=0,
            src_qpu=2,
            dst_qpu=0,
            involved_qubits=[0, 1, 2],
            gate_list=[_payload_gate("cx", [0, 1])],
        ),
        [0, 1, 2],
    )
    circuit.append(
        CommOp(
            comm_type="rtp",
            source_qubit=1,
            src_qpu=1,
            dst_qpu=2,
            involved_qubits=[1, 2],
            gate_list=[_payload_gate("cx", [1, 2])],
        ),
        [1, 2],
    )
    circuit.append(
        CommOp(
            comm_type="tp",
            source_qubit=2,
            src_qpu=2,
            dst_qpu=1,
            involved_qubits=[2],
            gate_list=[],
        ),
        [2],
    )

    comm = mapper._estimate_partition_comm_matrix_for_single_record(
        circuit,
        [[0], [1], [2]],
    )

    assert comm[0][2] == 1.0
    assert comm[2][0] == 1.0
    assert comm[1][2] == 3.0
    assert comm[2][1] == 3.0
    assert comm[0][1] == 0.0


def test_single_record_greedy_returns_evaluator_costs_and_remapped_ops():
    mapper = SingleRecordGreedyMapper()
    network = _Network()
    ops = QuantumCircuit(3)
    ops.append(
        CommOp(
            comm_type="cat",
            source_qubit=0,
            src_qpu=0,
            dst_qpu=2,
            involved_qubits=[0, 2],
            gate_list=[_payload_gate("cx", [0, 2])],
        ),
        [0, 2],
    )
    records = MappingRecordList()
    records.add_record(
        MappingRecord(
            layer_start=0,
            layer_end=0,
            partition=[[0], [1], [2]],
            mapping_type="cat",
            extra_info={"ops": ops},
        )
    )

    mapped = mapper.map(
        records,
        QuantumCircuit(3),
        [],
        network,
        config={"evaluator_policy": "local_all_to_all"},
    )
    mapped_record = mapped.records[0]

    assert mapped_record.extra_info is not None
    assert mapped_record.extra_info["perm"] == [1, 0, 2]
    remapped_op = mapped_record.extra_info["ops"].data[0].operation
    assert isinstance(remapped_op, CommOp)
    assert remapped_op.src_qpu == 1
    assert remapped_op.dst_qpu == 2

    replayed = CompilerUtils.evaluate_raw_mapping_records(
        copy.deepcopy(mapped),
        network,
        policy_name="local_all_to_all",
        config={"evaluator_policy": "local_all_to_all"},
    )
    assert mapped.total_costs.epairs == replayed.total_costs.epairs
    assert math.isclose(
        mapped.total_costs.total_fidelity_loss,
        replayed.total_costs.total_fidelity_loss,
    )
