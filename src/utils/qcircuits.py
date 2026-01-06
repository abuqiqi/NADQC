from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import *
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    UnitarySynthesis,
)
import random
import numpy as np
from math import *
import sys
from qiskit.quantum_info import Pauli
from QASMBench.interface.qiskit import QASMBenchmark

# basis_gates = ["rx", "ry", "rxx"]
# basis_gates = ["ecr", "u3"]
basis_gates = ["cu1", "u3"]
# basis_gates=["cu1", "rz", "h"] # Pytket-DQC
# basis_gates = ["crz", "rz", "h"] # AutoComm rebuttal
two_qubit_gates = ["ecr", "cx", "cz", "cu1", "cp", "crz", "swap"]
transpiler = PassManager(
    [
        Collect2qBlocks(),
        ConsolidateBlocks(basis_gates=basis_gates),
        UnitarySynthesis(basis_gates),
    ]
)

def select_circuit(name, num_qubits, num_qpus, qpus, basis_gates=["cu1", "u3"]):
    assert num_qpus == len(qpus), "[ERROR] num_qpus != len(qpus)"
    
    circ = None
    if name == "QV":
        circ = QuantumVolume(num_qubits, seed=26).decompose()
    elif name == "QFT":
        circ = QFT(num_qubits).decompose()
    elif name == "CuccaroAdder":
        circ = CDKMRippleCarryAdder(num_qubits).decompose()
    elif name == "DraperQFTAdder":
        circ = DraperQFTAdder(num_qubits).decompose()
    elif name == "Permutation":
        circ = Permutation(num_qubits, seed=21)
    elif "Fraction" in name: # CP_p_d
        parts = name.split("_")
        assert len(parts) >= 2, "[ERROR] Too few arguments for Fraction_p_depth."
        p = float(parts[1])
        depth = num_qubits if len(parts) == 2 else int(parts[2])
        circ = CPFraction(num_qubits, depth, p)
    elif name == "QAOA":
        circ = QAOA(num_qubits)
    elif name == "QFTAdder":
        circ = QFTAdder(num_qubits)
    elif name == "Grover":
        circ = Grover(num_qubits).decompose()
    elif name == "Pauli":
        circ = PauliGadget(num_qubits).decompose()
    elif name == "Toffoli":
        circ = Toffoli(num_qubits).decompose()
    elif name == "MCMT":
        circ = MCMTcircuit(num_qubits).decompose()
    else:
        trans_bm = QASMbm(do_transpile=True,
                          basis_gates=basis_gates,
                          category="large")
        circ = trans_bm.bm.get(name).decompose()

    assert circ != None, "[ERROR] Unknown circuit name."

    # 检查qpus是否能容纳下circ.num_qubits
    if sum(qpus) < circ.num_qubits:
        print(f"[WARNING] sum(qpus) ({sum(qpus)}) < circ.num_qubits ({circ.num_qubits})")
        print(f"[WARNING] Reallocate QPU capacities.")
        qpu_capacity = circ.num_qubits // num_qpus + 1
        if qpu_capacity % 2 == 1:
            qpu_capacity += 1
        qpus = [qpu_capacity] * num_qpus

    # 将线路转换到basis gates
    trans_circ = transpile(circ, basis_gates=basis_gates, optimization_level=0)
    # trans_circ = remove_single_qubit_gates(trans_circ)
    # print(trans_circ)
    # 输出线路和QPU信息
    gate_counts = trans_circ.count_ops()
    total_gates = sum(gate_counts.values())
    assert total_gates > 0, "[ERROR] An empty circuit."
    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}")
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}")
    print(f"[INFO] {name} #Gates: {total_gates}")
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n")
    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}", file=sys.stderr)
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}", file=sys.stderr)
    print(f"[INFO] {name} #Gates: {total_gates}", file=sys.stderr)
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n", file=sys.stderr)
    return circ, trans_circ, qpus

class QASMbm:
    def __init__(self, 
                 do_transpile=True, 
                 basis_gates=basis_gates, 
                 path="./QASMBench", 
                 category="small"):
        transpile_args = {
            "basis_gates": basis_gates
        }
        self.bm = QASMBenchmark(path, category, 
                                num_qubits_list=None, 
                                remove_final_measurements=True, 
                                do_transpile=do_transpile, 
                                **transpile_args)
        return

    def set_category(self, category):
        self.category = category
        return

    def get_circuit(self, circ_name, num_qpus=2):
        circ = self.bm.get(circ_name)
        qpu_capacity = circ.num_qubits // num_qpus + 1
        if qpu_capacity % 2 == 1:
            qpu_capacity += 1
        qpus = [qpu_capacity] * num_qpus
        return circ, qpus

    def get_circuit_list(self, circ_list):
        circ_list = self.bm.get(circ_list)
        return circ_list

def PauliGadget(num_qubits):
    np.random.seed(26)
    circuit = QuantumCircuit(num_qubits)
    for t in range(num_qubits):
        # 选择随机字符串 s^t ∈ {I, X, Y, Z}^n
        s_t = np.random.choice(['I', 'X', 'Y', 'Z'], size=num_qubits)
        # print(s_t)
        # 生成随机角度 α^t ∈ [0, 2π]
        alpha_t = np.random.uniform(0, 2 * np.pi)
        # 构建泡利算子字符串
        pauli_string = ''.join(s_t)
        # 在量子比特上 enact exp(i⊗_j s^t_j α^t)
        pauli_op = Pauli(pauli_string)
        evolution_gate = PauliEvolutionGate(pauli_op, time=alpha_t)
        circuit.append(evolution_gate, range(num_qubits))
    return circuit

def Fraction(num_qubits: int, p: float):
    np.random.seed(26)
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits):  # depth = num_qubits
        # 以概率 (1-p) 对每个量子比特应用 U 门
        applied = []
        for qubit in range(num_qubits):
            if np.random.random() > p:  # 概率 1-p
                qc.u(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), qubit)
                # qc.h(qubit)
                applied.append(qubit)
        # 随机配对未应用 U 门的量子比特
        remaining_qubits = [q for q in range(num_qubits) if q not in applied]
        np.random.shuffle(remaining_qubits)  # 随机打乱
        # add 2-qubit gates
        for i in range(0, len(remaining_qubits)-1, 2):
            if i+1 < len(remaining_qubits):  # 确保成对
                # qc.cp(random.uniform(0, 2 * pi), remaining_qubits[i], remaining_qubits[i+1])
                qc.cu(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), 0,
                      remaining_qubits[i], remaining_qubits[i+1])
    return qc

def CPFraction(num_qubits: int, depth: int, p: float):
    """
    构建 CZ Fraction 电路的实例
    参数:
        n (int): 量子比特数 (宽度)
        d (int): 电路深度 (层数)
        p (float): 跳过 H 门的概率 (fraction)
    返回:
        QuantumCircuit: 生成的量子电路
    """
    np.random.seed(21)
    qc = QuantumCircuit(num_qubits)
    cnt = 0
    # for _ in range(depth):  # 每层循环
    #     for _ in range(num_qubits): # 每层施加num_qubits个量子门
    for _ in range(depth*depth):  # 每层循环
        # 以概率 (1-p) 对每个量子比特应用 H 门
        if np.random.random() > p:
            # 随机选择一个量子比特施加 U3 门
            j = np.random.randint(0, num_qubits)
            # 以一半的概率施加h，一半的概率施加rz
            # roll = np.random.randint(0, 2)
            # if roll == 0:
            #     qc.h(j)
            # else:
            #     qc.rz(random.uniform(0, 2 * pi), j)
            qc.u(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), j)
            # qc.rz(random.uniform(0, 2 * pi), j)
        else: # 施加cp门
            # 随机选择两个量子比特施加 CP 门
            j = np.random.randint(0, num_qubits)
            while j == num_qubits-1:
                j = np.random.randint(0, num_qubits)
            # j = 0
            # k = np.random.randint(0, num_qubits)
            # while k == j:  # 确保选择不同的量子比特
            #     k = np.random.randint(0, num_qubits)
            # roll = np.random.randint(0, 2)
            # if roll == 0:
            #     qc.cp(random.uniform(0, 2 * pi), j, k)
            # else:
            #     qc.swap(j, k)
            # qc.ecr(j, k)
            cnt += 1
            qc.cp(random.uniform(0, 2 * pi), j, j+1)
            # qc.cu(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), 0, j, j+1)
            # qc.cu(random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), random.uniform(0, 2 * pi), 0, k, j)
    print(f"[DEBUG] [Fraction] #2-qubit gates: {cnt}")
    return qc

    # for _ in range(depth):  # 每层循环
        # # 以概率 (1-p) 对每个量子比特应用 H 门
        # applied_h = []
        # for qubit in range(num_qubits):
        #     if np.random.random() > p:  # 概率 1-p
        #         qc.h(qubit)
        #         applied_h.append(qubit)
        # # 随机配对未应用 H 门的量子比特
        # remaining_qubits = [q for q in range(num_qubits) if q not in applied_h]
        # np.random.shuffle(remaining_qubits)  # 随机打乱
        # # 对每对应用 CP 门
        # for i in range(0, len(remaining_qubits)-1, 2):
        #     if i+1 < len(remaining_qubits):  # 确保成对
        #         qc.cp(random.uniform(0, 2 * pi), remaining_qubits[i], remaining_qubits[i+1])
        #         # qc.swap(remaining_qubits[i], remaining_qubits[i+1])
    return qc

def QAOA(numQubits):
    qc = QuantumCircuit(numQubits)
    qc.h(range(numQubits))
    for j in range(numQubits):
        for i in range(j+1, numQubits):
            qc.cx(j, i)
            qc.rz(random.uniform(0, 2 * pi), i)
            qc.cx(j, i)
    qc.h(range(numQubits))
    qc.rz(random.uniform(0, 2 * pi), range(numQubits))
    qc.h(range(numQubits))
    return qc

def Toffoli(num_qubits):
    qc = QuantumCircuit(num_qubits)
    oracle = XGate().control(num_qubits-1)
    oracle_qubits = [i for i in range(num_qubits)]
    qc.append(oracle, oracle_qubits)
    return qc

def QFTAdder(num_qubits):
    # 创建量子线路
    qc = QuantumCircuit(num_qubits, name="Adder")
    
    # 应用Hadamard门到前两个量子比特
    qc.h(0)
    qc.h(1)
    
    # 应用QFT到后一半的量子比特
    qft = QFT(num_qubits // 2, do_swaps=False)
    qc.append(qft, range(num_qubits // 2, num_qubits))
    
    # 加法操作
    for i in range(num_qubits // 2):
        targ = i + num_qubits // 2
        k = 0
        for j in range(i, num_qubits // 2):
            k += 1
            qc.cp(2 * np.pi / (2 ** k), j, targ)
    
    # 应用逆QFT到后一半的量子比特
    iqft = qft.inverse()
    qc.append(iqft, range(num_qubits // 2, num_qubits))
    return qc.decompose()

def MCMTcircuit(num_qubits, t=10):
    qc = QuantumCircuit(num_qubits)

    for _ in range(1):
        for i in range(0, num_qubits, t):
            if i + t <= num_qubits:
                # 随机选择基础门
                base_gate = random.choice([HGate(), RZGate(np.random.uniform(0, 2 * np.pi))])
                num_ctrl = t // 2
                num_target = t - num_ctrl
                mcmt_gate = MCMT(base_gate, num_ctrl, num_target)
                qc.append(mcmt_gate, range(i, i + t))

    return qc

def test():
    qc = QuantumCircuit(6)
    qc.ecr(0, 1)
    qc.ecr(2, 3)
    qc.ecr(4, 5)
    # qc.ecr(0, 4)
    # qc.ecr(1, 3)
    # qc.ecr(0, 3)
    # qc.ecr(1, 2)
    # qc.ecr(1, 4)
    # qc.ecr(2, 3)
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    return qc, transpiled_qc

def remove_single_qubit_gates(circuit):
    new_circuit = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit:
        # print(instruction.qubits)
        gate = instruction.operation
        # print(gate)
        qubits = [qubit._index for qubit in instruction.qubits]
        if qubits[0] == None:
            qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
        # print(qubits)
        if len(qubits) > 1:
            new_circuit.append(gate, qubits)
    return new_circuit

def remove_single_qubit_gates_by_instr(circuit):
    new_circuit = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit:
        gate = instruction.operation
        qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
        print(qubits)
        if len(qubits) > 1:
            assert(gate.name in two_qubit_gates)
            new_circuit.append(gate, qubits)
    return new_circuit

def Grover(numQubits, k=0):
    ''' One implementation of Grover\'s algorithm '''
    qc = QuantumCircuit(numQubits)
    oracle = ZGate().control(numQubits-1-k)
    oracle_qubits = [i for i in range(k, numQubits)]

    cz = ZGate().control(numQubits-1)
    qubits = [i for i in range(numQubits)]

    # 1. Apply Hadamard gates to all qubits
    # for j in range(numQubits): # |00...0> => |u>
    #     qc.h(j)

    # 2. Start the Grover iteration
    # numIterations = int(round(pi / 4 * sqrt(2**numQubits / 2**k), 0))
    numIterations = 1

    for _ in range(numIterations):
        # 2.1. Apply the phase inversion use the controlled-Z gate
        # i.e., transform the target state |x*> into -|x*>
        qc.append(oracle, oracle_qubits)  # CC...Z gate: |11...1> => -|11...1>
        # qc.barrier()

        # 2.2. Apply the diffusion operator
        # 2.2.1. Apply H, X to all qubits
        # for j in range(numQubits):
        #     qc.h(j) # |u> => |00...0>
        #     qc.x(j) # |00...0> => |11...1>

        # 2.2.2. Apply the controlled-Z gate
        qc.append(cz, qubits) # |11...1> => -|11...1>

        # 2.2.3. Recover X, H
        # for j in range(numQubits):
        #     qc.x(j) # |11...1> => |00...0>
        #     qc.h(j) # |00...0> => |u>
    return qc

def myQFT(num_qubits):
    print(f"[QFT] #Qubits: [{num_qubits}]")
    qc = QFT(num_qubits).decompose()
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    # transpiled_qc = transpiler.run(qc)
    # single_removed_qc = remove_single_qubit_gates(transpiled_qc)
    return qc, transpiled_qc #, single_removed_qc

def myQV(num_qubits):
    print(f"[QV] #Qubits: [{num_qubits}]")
    qc = QuantumVolume(num_qubits, seed=26).decompose()
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    # transpiled_qc = transpiler.run(qc)
    # single_removed_qc = remove_single_qubit_gates(transpiled_qc)
    return qc, transpiled_qc #, single_removed_qc

# def myCZFraction(num_qubits: int, depth: int, p: float):
#     qc = CPFraction(num_qubits, depth, p)
#     transpiled_qc = transpile(qc, basis_gates=basis_gates)
#     return qc, transpiled_qc

def myCuccaroAdder(num_qubits, num_qpus):
    qc = CDKMRippleCarryAdder(num_qubits).decompose()
    print(f"[CuccaroAdder] #Qubits: [{num_qubits}] [{qc.num_qubits}]")
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    # 构造QPU集合
    qpu_capacity = qc.num_qubits // num_qpus + 1
    if qpu_capacity % 2 == 1:
        qpu_capacity += 1
    qpus = [qpu_capacity] * num_qpus
    # print(transpiled_qc)
    return qc, transpiled_qc, qpus

def myGrover(num_qubits):
    print(f"[Grover] #Qubits: [{num_qubits}]")
    qc = Grover(num_qubits).decompose()
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    return qc, transpiled_qc

def myIQP(num_qubits):
    print(f"[IQP] #Qubits: [{num_qubits}]")
    A = np.random.randint(0, 10, size=(num_qubits, num_qubits))
    symmetric_matrix = (A + A.T) // 2 # 生成对称矩阵
    # print(symmetric_matrix)
    qc = IQP(symmetric_matrix).decompose()
    transpiled_qc = transpile(qc, basis_gates=basis_gates)
    return qc, transpiled_qc

def VQC_AA(numQubits):
    qc = QuantumCircuit(numQubits)
    for _ in range(1):
        # 对每个量子比特应用随机的 RX 门
        for i in range(numQubits):
            angle_rx = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rx(angle_rx, i)

        # 对每个量子比特应用随机的 RZ 门
        for i in range(numQubits):
            angle_rz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rz(angle_rz, i)

        # 双比特的 CRZ 门
        for i in range(numQubits):
            for j in range(numQubits):
                if i == j:
                    continue
                qc.barrier()  # 加入 barrier 隔离量子门
                angle_crz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
                qc.crz(angle_crz, i, j)  # 应用 CRZ 门
    return qc
