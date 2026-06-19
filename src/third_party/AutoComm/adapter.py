from typing import Any, Optional

from qiskit import QuantumCircuit

from ...compiler import CompilerUtils, MappingRecord, MappingRecordList
from ...utils import Network
from .autocomm import autocomm_full
from .gate_util import (
    build_CU1_gate,
    build_CRZ_gate,
    build_CX_gate,
    build_CZ_gate,
    build_H_gate,
    build_RZ_gate,
    build_gate,
)


def _to_autocomm_gate_list(circuit: QuantumCircuit) -> list[list[Any]]:
    gate_list: list[list[Any]] = []
    global_phase = circuit.global_phase

    for instruction in circuit:
        gate = instruction.operation
        gate_name = gate.name
        qubits = [circuit.qubits.index(q) for q in instruction.qubits]
        params = list(gate.params) if gate.params else []

        if gate_name == "u3":
            converted = build_gate("U3", qubits, params, global_phase=global_phase)
        elif gate_name == "rz":
            converted = build_RZ_gate(qubits[0], angle=params[0], global_phase=global_phase)
        elif gate_name == "rx":
            converted = build_gate("RX", [qubits[0]], [params[0]], global_phase=global_phase)
        elif gate_name == "x":
            converted = build_gate("X", [qubits[0]], [], global_phase=global_phase)
        elif gate_name == "sx":
            converted = build_gate("SX", [qubits[0]], [], global_phase=global_phase)
        elif gate_name == "id":
            converted = build_gate("ID", [qubits[0]], [], global_phase=global_phase)
        elif gate_name == "reset":
            converted = build_gate("RESET", [qubits[0]], [], global_phase=global_phase)
        elif gate_name == "cx":
            converted = build_CX_gate(qubits[0], qubits[1])
        elif gate_name == "cz":
            converted = build_CZ_gate(qubits[0], qubits[1])
        elif gate_name == "rzz":
            converted = build_gate("RZZ", qubits, [params[0]], global_phase=global_phase)
        elif gate_name == "crz":
            converted = build_CRZ_gate(qubits[0], qubits[1], angle=params[0])
        elif gate_name == "cu1":
            converted = build_CU1_gate(qubits[0], qubits[1], angle=params[0])
        elif gate_name == "h":
            converted = build_H_gate(qubits[0])
        elif gate_name in {"barrier", "measure"}:
            continue
        else:
            raise ValueError(f"Unsupported gate type for AutoComm: {gate_name}")

        gate_list.append(converted)
    return gate_list


def _mapping_from_partition(
    num_qubits: int,
    partition: list[list[int]],
    num_backends: int,
) -> list[int]:
    if len(partition) > num_backends:
        raise ValueError(
            f"Partition has {len(partition)} blocks for {num_backends} backends"
        )

    qubit_node_mapping = [-1] * num_qubits
    for node_idx, part in enumerate(partition):
        for raw_qubit in part:
            qubit = int(raw_qubit)
            if qubit < 0 or qubit >= num_qubits:
                raise ValueError(f"Invalid qubit index in partition: {qubit}")
            if qubit_node_mapping[qubit] != -1:
                raise ValueError(f"Qubit {qubit} appears in multiple partition blocks")
            qubit_node_mapping[qubit] = node_idx

    if any(node == -1 for node in qubit_node_mapping):
        raise ValueError("Partition does not cover all circuit qubits")
    return qubit_node_mapping


def _default_partition(num_qubits: int, capacities: list[int]) -> tuple[list[list[int]], list[int]]:
    if sum(capacities) < num_qubits:
        raise ValueError(
            f"Network capacity {sum(capacities)} is smaller than circuit size {num_qubits}"
        )

    partition = [[] for _ in capacities]
    mapping: list[int] = []
    node_idx = 0
    remaining = int(capacities[node_idx]) if capacities else 0
    for qubit in range(num_qubits):
        while remaining == 0:
            node_idx += 1
            remaining = int(capacities[node_idx])
        partition[node_idx].append(qubit)
        mapping.append(node_idx)
        remaining -= 1
    return partition, mapping


def build_autocomm_records(
    circuit: QuantumCircuit,
    network: Network,
    partition: Optional[list[list[int]]] = None,
) -> MappingRecordList:
    """Build raw NADQC records from AutoComm without evaluating or saving them."""
    if partition is None:
        resolved_partition, qubit_node_mapping = _default_partition(
            circuit.num_qubits,
            list(network.backend_sizes),
        )
    else:
        resolved_partition = [list(part) for part in partition]
        qubit_node_mapping = _mapping_from_partition(
            circuit.num_qubits,
            resolved_partition,
            network.num_backends,
        )

    gate_list = _to_autocomm_gate_list(circuit)
    qubits_per_node = max(network.backend_sizes) if network.backend_sizes else 1
    iteration_count = max(1, circuit.num_qubits // max(1, qubits_per_node))
    result = autocomm_full(
        gate_list,
        qubit_node_mapping,
        aggregate_iter_cnt=iteration_count,
        schedule_iter_cnt=iteration_count,
        return_ops=True,
    )
    if len(result) == 5:
        _, latency, assigned_blocks, _, op_list = result
    else:
        _, latency, assigned_blocks, _ = result
        op_list = QuantumCircuit(circuit.num_qubits)

    record = MappingRecord(
        layer_start=0,
        layer_end=max(0, circuit.depth() - 1),
        partition=resolved_partition,
        mapping_type="cat",
        logical_phy_map=CompilerUtils.init_logical_phy_map(resolved_partition),
        extra_info={
            "ops": op_list,
            "autocomm_latency": float(latency),
            "autocomm_assigned_block_count": len(assigned_blocks),
        },
    )
    return MappingRecordList(records=[record])
