from typing import Any, Optional
import datetime
import time

from qiskit import QuantumCircuit

from ..compiler import Compiler, CompilerUtils, MappingRecordList
from ..third_party.AutoComm import build_autocomm_records
from ..utils import Network


class QAutoComm(Compiler):
    """AutoComm baseline wrapped as the NADQC Compiler interface."""

    compiler_id = "autocomm"

    @property
    def name(self) -> str:
        return "AutoComm"

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> MappingRecordList:
        cfg = config or {}
        circuit_name = cfg.get("circuit_name", "circ")
        start_time = time.time()

        records = build_autocomm_records(
            circuit,
            network,
            partition=cfg.get("partition") or None,
        )

        save_records = bool(cfg.get("save_records", True))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if save_records else None
        base_path = (
            CompilerUtils.build_record_base_path(circuit_name, network, self.name, timestamp)
            if save_records and timestamp is not None
            else None
        )
        if save_records:
            assert base_path is not None
            records.save_records(f"{base_path}_raw.json", dump_type="raw")

        records, _, _ = self._evaluate_raw_mapping_records_with_timing(
            records,
            network,
            policy_name=cfg.get("evaluator_policy"),
            config=cfg,
            start_time=start_time,
        )
        records.update_total_costs(execution_time=time.time() - start_time)

        if save_records:
            assert base_path is not None
            records.save_records(f"{base_path}_evaluated.json", dump_type="evaluated")
        return records
