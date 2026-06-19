from qiskit import QuantumCircuit

from main import _build_compile_config
from src.compiler import CompilerUtils, MappingRecord, MappingRecordList
from src.navi.navi_compiler import CompilationContext, NAVI


class _Network:
    num_backends = 2


class _Compiler:
    compiler_id = "navi"


class _CountingTelegatePartitioner:
    def __init__(self):
        self.calls = 0

    def partition(self, circuit, network, config):
        self.calls += 1
        return MappingRecord(partition=[[0], [1]], mapping_type="telegate")


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
    compiler = NAVI()
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


def test_navi_original_layer_ranges_restore_single_qubit_gaps():
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


def _capture_finish_mapper(monkeypatch, record_count, config=None, override_mapper_id=None):
    compiler = NAVI()
    records = MappingRecordList(
        records=[MappingRecord() for _ in range(record_count)],
    )
    ctx = CompilationContext(
        circuit=QuantumCircuit(1),
        network=_Network(),
        config=dict(config or {}),
    )
    captured = {}

    monkeypatch.setattr(
        compiler,
        "_construct_records_from_ctx",
        lambda ctx, shared_prefix_time=0.0: (records, 0.25),
    )

    def fake_map_records(
        ctx,
        selected_records,
        mapper_id,
        shared_prefix_time=0.0,
        records_time=0.0,
    ):
        captured["mapper_id"] = mapper_id
        captured["shared_prefix_time"] = shared_prefix_time
        captured["records_time"] = records_time
        return selected_records

    monkeypatch.setattr(compiler, "_map_records_with_mapper", fake_map_records)
    result = compiler._finish_compile_from_ctx(
        ctx,
        shared_prefix_time=1.5,
        override_mapper_id=override_mapper_id,
    )
    return captured, records, result


def test_navi_single_record_defaults_to_single_record_greedy(monkeypatch):
    captured, records, result = _capture_finish_mapper(monkeypatch, record_count=1)

    assert result is records
    assert captured == {
        "mapper_id": "single_record_greedy",
        "shared_prefix_time": 1.5,
        "records_time": 0.25,
    }


def test_navi_multiple_records_default_to_bounded_dp_neighbor(monkeypatch):
    captured, _, _ = _capture_finish_mapper(monkeypatch, record_count=2)

    assert captured["mapper_id"] == "boundeddp_neighbor"


def test_navi_mapper_override_takes_precedence(monkeypatch):
    captured, _, _ = _capture_finish_mapper(
        monkeypatch,
        record_count=1,
        override_mapper_id="direct",
    )

    assert captured["mapper_id"] == "direct"


def _telegate_cache_partition_calls(monkeypatch, config):
    compiler = NAVI()
    partitioner = _CountingTelegatePartitioner()
    ctx = CompilationContext(
        circuit=QuantumCircuit(2),
        network=_Network(),
        config=config,
        telegate_partitioner=partitioner,
    )
    ctx.runtime_hybrid_beam_width = 2
    monkeypatch.setattr(compiler, "_get_ori_subc", lambda *args: QuantumCircuit(2))
    monkeypatch.setattr(
        compiler,
        "_set_simple_telegate_ops_for_full_circuit_record",
        lambda *args: None,
    )

    compiler._try_generate_telegate(ctx, 0, 0, prev_partition=[[0], [1]])
    compiler._try_generate_telegate(ctx, 0, 0, prev_partition=[[1], [0]])
    return partitioner.calls


def test_navi_telegate_cache_mode_defaults_to_range(monkeypatch):
    assert _telegate_cache_partition_calls(monkeypatch, {}) == 1


def test_navi_telegate_cache_mode_allows_strict_override(monkeypatch):
    assert _telegate_cache_partition_calls(
        monkeypatch,
        {"telegate_cache_mode": "strict"},
    ) == 2


def test_build_compile_config_allows_missing_compiler_config():
    assert _build_compile_config({}, _Compiler(), "example") == {
        "circuit_name": "example",
    }
