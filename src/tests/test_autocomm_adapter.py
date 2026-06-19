import pytest
from qiskit import QuantumCircuit

from src.baselines.autocomm import QAutoComm
from src.compiler import ExecCosts, MappingRecord, MappingRecordList
from src.third_party.AutoComm import adapter


class _Network:
    num_backends = 2
    backend_sizes = [2, 2]


def _fake_autocomm_result(captured):
    def fake_autocomm_full(
        gate_list,
        qubit_node_mapping,
        aggregate_iter_cnt,
        schedule_iter_cnt,
        return_ops,
    ):
        captured["gate_list"] = gate_list
        captured["mapping"] = qubit_node_mapping
        captured["aggregate_iter_cnt"] = aggregate_iter_cnt
        captured["schedule_iter_cnt"] = schedule_iter_cnt
        captured["return_ops"] = return_ops
        ops = QuantumCircuit(len(qubit_node_mapping))
        return 1, 3.5, [["block"]], [1, 0], ops

    return fake_autocomm_full


def test_build_autocomm_records_uses_default_partition(monkeypatch):
    captured = {}
    monkeypatch.setattr(adapter, "autocomm_full", _fake_autocomm_result(captured))
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 2)

    records = adapter.build_autocomm_records(circuit, _Network())

    record = records.records[0]
    assert record.partition == [[0, 1], [2]]
    assert captured["mapping"] == [0, 0, 1]
    assert captured["return_ops"] is True
    assert record.extra_info["autocomm_latency"] == 3.5
    assert record.extra_info["autocomm_assigned_block_count"] == 1
    assert "comm_only_costs" not in record.extra_info
    assert record.costs == ExecCosts()
    assert records.total_costs == ExecCosts()


def test_build_autocomm_records_accepts_explicit_partition(monkeypatch):
    captured = {}
    monkeypatch.setattr(adapter, "autocomm_full", _fake_autocomm_result(captured))

    records = adapter.build_autocomm_records(
        QuantumCircuit(3),
        _Network(),
        partition=[[0, 2], [1]],
    )

    assert records.records[0].partition == [[0, 2], [1]]
    assert captured["mapping"] == [0, 1, 0]


@pytest.mark.parametrize(
    ("partition", "message"),
    [
        ([[0, 2], [1]], "Invalid qubit index"),
        ([[0, 1], [1]], "multiple partition blocks"),
        ([[0], []], "does not cover all circuit qubits"),
    ],
)
def test_build_autocomm_records_rejects_invalid_partition(partition, message):
    with pytest.raises(ValueError, match=message):
        adapter.build_autocomm_records(
            QuantumCircuit(2),
            _Network(),
            partition=partition,
        )


def test_qautocomm_always_uses_full_evaluator(monkeypatch):
    raw_records = MappingRecordList(records=[MappingRecord(partition=[[0], [1]])])
    captured = {"saved": []}

    def fake_build(circuit, network, partition=None):
        del circuit, network
        captured["partition"] = partition
        return raw_records

    def fake_evaluate(
        self,
        records,
        network,
        policy_name=None,
        config=None,
        start_time=None,
        pre_eval_time=None,
    ):
        del self, network, start_time, pre_eval_time
        captured["policy_name"] = policy_name
        captured["config"] = config
        records.summarize_total_costs()
        return records, 0.0, 0.0

    def fake_save(self, path, dump_type="evaluated"):
        del self
        captured["saved"].append((path, dump_type))

    monkeypatch.setattr("src.baselines.autocomm.build_autocomm_records", fake_build)
    monkeypatch.setattr(QAutoComm, "_evaluate_raw_mapping_records_with_timing", fake_evaluate)
    monkeypatch.setattr(MappingRecordList, "save_records", fake_save)
    monkeypatch.setattr(
        "src.baselines.autocomm.CompilerUtils.build_record_base_path",
        lambda *args, **kwargs: "/tmp/autocomm-test",
    )

    result = QAutoComm().compile(
        QuantumCircuit(2),
        _Network(),
        {
            "partition": [[0], [1]],
            "save_records": True,
            "evaluator_policy": "full_realistic",
        },
    )

    assert result is raw_records
    assert captured["partition"] == [[0], [1]]
    assert captured["policy_name"] == "full_realistic"
    assert captured["saved"] == [
        ("/tmp/autocomm-test_raw.json", "raw"),
        ("/tmp/autocomm-test_evaluated.json", "evaluated"),
    ]
    assert result.total_costs.execution_time >= 0.0
