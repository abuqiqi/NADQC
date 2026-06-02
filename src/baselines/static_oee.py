from qiskit import QuantumCircuit
from typing import Any, Optional
import datetime
import time

from .oee import OEE
from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network

class StaticOEE(Compiler):
    """
    Static OEE
    """
    compiler_id = "staticoee"

    def __init__(self):
        super().__init__()
        return

    @property
    def name(self) -> str:
        return "Static OEE"

    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        Compile the circuit using Static OEE algorithm
        """
        print(f"Compiling with [{self.name}]...")
        
        start_time = time.time()
        iteration_count = config.get("iteration", 50) if config else 50
        circuit_name = config.get("circuit_name", "circ") if config else "circ"

        partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network) # initialize partition
        qig = CompilerUtils.build_qubit_interaction_graph(circuit)
        partition = OEE.partition(partition, qig, network, iteration_count)
        # Raw-only construction: MappingEvaluator owns runtime logical/physical state.
        # logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        record = MappingRecord(
            layer_start = 0, 
            layer_end = circuit.depth() - 1,
            partition = partition,
            mapping_type = "telegate",
            # Legacy evaluator path populated logical_phy_map here:
            # logical_phy_map = logical_phy_map,
        )

        # Legacy evaluator path, kept for comparison/rollback. It eagerly
        # charges costs before raw records are replayed by MappingEvaluator.
        # _ = CompilerUtils.evaluate_local_and_telegate_with_cat(
        #     record,
        #     circuit,
        #     network,
        #     config=config,
        # )

        mapping_record_list = MappingRecordList()
        mapping_record_list.add_record(record)

        circuit_layers = CompilerUtils.build_circuit_layers(circuit)
        mapping_record_list.ensure_record_ops(circuit, circuit_layers)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = CompilerUtils.build_record_base_path(circuit_name, network, self.name, timestamp)
        mapping_record_list.save_records(f"{base_path}_raw.json", dump_type="raw")

        policy_name = config.get("evaluator_policy") if config else None
        mapping_record_list, _, _ = self._evaluate_raw_mapping_records_with_timing(
            mapping_record_list,
            network,
            policy_name=policy_name,
            config=config,
            start_time=start_time,
        )

        end_time = time.time()
        mapping_record_list.update_total_costs(execution_time = end_time - start_time)
        
        mapping_record_list.save_records(f"{base_path}_evaluated.json", dump_type="evaluated")
        return mapping_record_list
