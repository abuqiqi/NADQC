from qiskit import QuantumCircuit
from typing import Any, Optional
import networkx as nx
import numpy as np
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

        record = MappingRecord(
            layer_start = 0, 
            layer_end = circuit.depth() - 1,
            partition = partition,
            mapping_type = "telegate"
        )

        _ = CompilerUtils.evaluate_local_and_telegate(record, circuit, network)
        
        mapping_record_list = MappingRecordList()
        mapping_record_list.add_record(record)
        mapping_record_list.summarize_total_costs()

        end_time = time.time()
        mapping_record_list.update_total_costs(execution_time = end_time - start_time)
        
        mapping_record_list.save_records(f"./outputs/{circuit_name}_{network.name}_{self.name}.json")
        return mapping_record_list
