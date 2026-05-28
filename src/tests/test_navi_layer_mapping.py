from qiskit import QuantumCircuit

from src.compiler import CompilerUtils
from src.navi.navi_compiler import CompilationContext, NAVI
from src.navi.navi_hybrid import NAVIHybrid


class _Network:
    num_backends = 2


def _build_context():
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.z(0)

    ctx = CompilationContext(circuit=circuit, network=_Network(), config={})
    compiler = NAVIHybrid()
    compiler._step_remove_single_qubit_gates(ctx)
    return compiler, ctx


def _gate_count_for_ranges(ctx, ranges):
    total = 0
    for left, right in ranges:
        subcircuit = CompilerUtils.get_subcircuit_by_level(
            num_qubits=ctx.circuit.num_qubits,
            circuit=ctx.circuit,
            circuit_layers=ctx.circuit_layers,
            layer_start=left,
            layer_end=right,
        )
        total += subcircuit.size()
    return total


def test_navi_hybrid_original_layer_ranges_restore_single_qubit_gaps():
    compiler, ctx = _build_context()

    ranges = [
        compiler.get_original_layer_idx(ctx, (i, i))
        for i in range(len(ctx.multiq_layers))
    ]

    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(ctx.circuit_layers) - 1
    for prev, curr in zip(ranges, ranges[1:]):
        assert curr[0] == prev[1] + 1
    assert _gate_count_for_ranges(ctx, ranges) == ctx.circuit.size()


def test_navi_compiler_original_layer_ranges_restore_single_qubit_gaps():
    _, ctx = _build_context()
    compiler = NAVI()

    ranges = [
        compiler.get_original_layer_idx(ctx, (i, i))
        for i in range(len(ctx.multiq_layers))
    ]

    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(ctx.circuit_layers) - 1
    for prev, curr in zip(ranges, ranges[1:]):
        assert curr[0] == prev[1] + 1
    assert _gate_count_for_ranges(ctx, ranges) == ctx.circuit.size()
