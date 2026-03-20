import warnings
from typing import Any, Optional, Union
import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ..compiler import MappingRecord
from .backend import Backend
from .network import Network

class NoiseEvaluator:
    """
    噪声评估器：根据量子比特划分、线路、后端噪声和网络信息，评估在给定硬件上执行线路的总体保真度。
    """

    def __init__(self):
        pass

    @classmethod
    def evaluate_local_and_telegate(
        cls,
        mapping_record: MappingRecord,
        circuit: QuantumCircuit,
        network: Network
    ) -> MappingRecord:
        """
        评估在给定划分和后端分配下执行线路的总体保真度
        """

        # 获取每个划分对应的子线路，并记录所有的跨分区操作
        partition = mapping_record.partition

        # 建立每个分区的子线路
        subcircuits = [QuantumCircuit(len(group)) for group in partition]
        telegate_operations = []  # 记录跨分区的操作

        # 建立一个反向索引，用于快速查询每个量子比特属于哪个分区
        qubit_to_partition = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        # 对每个分区内，要将量子比特编号映射到0,1,...,len(group)-1，以便构建子线路
        # 例如，如果分区是 [0,2,5]，则子线路中的量子比特0对应原线路的0，量子比特1对应原线路的2，量子比特2对应原线路的5
        qubit_to_subcircuits = {}
        for group in partition:
            mapping = {original_qubit: idx for idx, original_qubit in enumerate(group)}
            qubit_to_subcircuits.update(mapping)

        # 遍历circuit上的每个操作，如果操作完全属于某个group，则添加到对应的subcircuit；如果操作跨越多个group，则记录为telegate操作
        for instruction in circuit:
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # 检查操作涉及的量子比特属于哪个分区
            involved_partitions = set()
            for qubit in qubits:
                involved_partitions.add(qubit_to_partition[qubit])

            if len(involved_partitions) == 1:
                # 操作完全属于一个分区，添加到对应的subcircuit
                partition_idx = involved_partitions.pop()
                # 更新操作中的量子比特编号为子线路中的编号
                mapped_qubits = [qubit_to_subcircuits[qubit] for qubit in qubits]
                subcircuits[partition_idx].append(instruction.operation, mapped_qubits)
                print(f"[DEBUG] Added instruction {instruction.operation.name} on qubits {mapped_qubits} to subcircuit {partition_idx}")
            else:
                # 操作跨越多个分区，记录为telegate操作
                telegate_operations.append(instruction)
                
                # TODO: 直接统计跨分区的telegate保真度损失


        # 遍历每个子线路

            # 获取每个分区单独的保真度损失

            
        
        
        

        return mapping_record

    @classmethod
    def evaluate_circuit_to_backend(
        cls,
        circuit: QuantumCircuit,
        backend: Backend
    ) -> float:
        """
        评估线路在给定后端上的保真度
        """
        # Implementation for evaluating circuit fidelity on a specific backend
        pass