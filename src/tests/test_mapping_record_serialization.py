from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.compiler.compiler_utils import MappingRecordList


def test_quantum_circuit_parameter_expression_round_trip():
    gamma = ParameterVector("γ", 1)
    beta = ParameterVector("β", 1)
    circuit = QuantumCircuit(2)
    circuit.rzz(2.0 * gamma[0], 0, 1)
    circuit.rx(2.0 * beta[0], 0)

    payload = MappingRecordList._encode_quantum_circuit(circuit)
    decoded = MappingRecordList._decode_quantum_circuit(payload)

    assert decoded.data[0].operation.name == "rzz"
    assert str(decoded.data[0].operation.params[0]) == "2.0*γ[0]"
    assert decoded.data[1].operation.name == "rx"
    assert str(decoded.data[1].operation.params[0]) == "2.0*β[0]"


def test_legacy_repr_parameter_expression_decodes_for_parameterized_gate():
    payload = {
        "__type__": "gate",
        "name": "rzz",
        "num_qubits": 2,
        "params": [{"__type__": "repr", "value": "2.0*γ[0]"}],
    }

    gate = MappingRecordList._decode_gate(payload)

    assert gate.name == "rzz"
    assert str(gate.params[0]) == "2.0*γ[0]"
