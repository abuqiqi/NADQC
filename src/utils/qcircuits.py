from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    QuantumVolume, 
    QFT, 
    PauliEvolutionGate, 
    CDKMRippleCarryAdder, 
    DraperQFTAdder, 
    Permutation,
    IQP,
    TwoLocal,
    ZZFeatureMap,
    QAOAAnsatz,
    MCMT,
    XGate, HGate, ZGate, RZGate
)
from qiskit.circuit.random import random_circuit
import random
import numpy as np
from math import pi
import os
import sys
from qiskit.quantum_info import Pauli, SparsePauliOp
from QASMBench.interface.qiskit import QASMBenchmark
from qiskit.converters import circuit_to_dag

# basis_gates = ["rx", "ry", "rxx"]
# basis_gates = ["ecr", "u3"]
# basis_gates = ["cu1", "u3"]
# basis_gates=["cu1", "rz", "h"] # Pytket-DQC
# basis_gates = ["crz", "rz", "h"] # AutoComm rebuttal
# two_qubit_gates = ["ecr", "cx", "cz", "cu1", "cp", "crz", "swap"]

def select_circuit(name, num_qubits, num_qpus, qpus, basis_gates, two_qubit_gates):
    assert num_qpus == len(qpus), "[ERROR] num_qpus != len(qpus)"
    
    circ = None
    optimization_level = 3
    if name == "QV":
        circ = QuantumVolume(num_qubits, 50, seed=42).decompose()
    elif name == "BV":
        circ = BV(num_qubits).decompose()
    elif name == "QFT":
        circ = QFT(num_qubits).decompose()
    elif name == "CuccaroAdder":
        circ = CuccaroAdder(num_qubits).decompose()
    elif name == "DraperQFTAdder":
        circ = MyDraperQFTAdder(num_qubits).decompose()
    elif name == "Permutation":
        optimization_level = 0
        circ = Permutation(num_qubits, seed=21)
    elif name == "Random":
        circ = random_circuit(num_qubits, 50, max_operands=4, seed=42)
    # elif name == "QPE":
    elif name == "QAOA":
        circ = QAOA(num_qubits)
        # circ = QiskitQAOA(num_qubits)

    elif "Fraction" in name: # CP_p_d
        parts = name.split("_")
        assert len(parts) >= 2, "[ERROR] Too few arguments for Fraction_p_depth."
        p = float(parts[1])
        depth = num_qubits if len(parts) == 2 else int(parts[2])
        circ = CPFraction(num_qubits, depth, p)
    elif name == "QFTAdder":
        circ = QFTAdder(num_qubits)
    elif name == "Grover":
        circ = Grover(num_qubits).decompose()
    elif name == "GraphState":
        circ = GraphState(num_qubits).decompose()
    elif name == "Pauli":
        circ = PauliGadget(num_qubits).decompose()
    elif name == "Toffoli":
        circ = Toffoli(num_qubits).decompose()
    elif name == "MCMT":
        circ = MCMTcircuit(num_qubits).decompose()
    elif name == "IQP":
        circ = myIQP(num_qubits).decompose()
    elif name == "VQC_AA":
        circ = VQC_AA(num_qubits).decompose()
    elif name == "VQC":
        circ = VQC(num_qubits).decompose()
    elif name == "VQE" or name.startswith("VQE_"):
        circ = build_vqe_benchmark(name, num_qubits).decompose()
    elif name == "QKNN" or name.startswith("QKNN_"):
        circ = build_qknn_benchmark(name, num_qubits).decompose()
    elif name == "test":
        circ = QuantumCircuit(4)
        circ.h(range(4))
        circ.rzz(0.5, 0, 1)
        circ.rzz(0.4, 2, 3)
        circ.rzz(0.1, 0, 2)
        circ.rzz(0.2, 0, 3)
        circ.rzz(0.5, 0, 1)
        circ.rzz(0.4, 2, 3)
        circ.rzz(0.5, 0, 1)
        circ.rzz(0.4, 2, 3)
        print(circ)
    else:
        circ = load_circ_from_qasm("./data/benchmarks", name)
        # trans_bm = QASMbm(do_transpile=True,
        #                   basis_gates=basis_gates,
        #                   category="large")
        # circ = trans_bm.bm.get(name).decompose()

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
    trans_circ = transpile(circ, basis_gates=basis_gates, optimization_level=optimization_level, seed_transpiler=42)
    # MAX_ALLOWED_DEPTH = 1000
    # if trans_circ.depth() > MAX_ALLOWED_DEPTH:
    #     print(f"[INFO] Circuit depth ({circ.depth()}) exceeds threshold ({MAX_ALLOWED_DEPTH}). Truncating...")
    #     trans_circ = truncate_circuit_by_depth(trans_circ, MAX_ALLOWED_DEPTH)
    #     print(f"[INFO] Truncated circuit depth: {trans_circ.depth()}")

    # print(trans_circ)
    # 输出线路和QPU信息
    gate_counts = trans_circ.count_ops()
    total_gates = sum(gate_counts.values())
    assert total_gates > 0, "[ERROR] An empty circuit."
    
    # 计算2-qubit门数量
    two_qubit_gate_counts = {gate: count for gate, count in gate_counts.items() if gate in two_qubit_gates}
    num_2q_gates = sum(two_qubit_gate_counts.values())

    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}")
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}")
    print(f"[INFO] {name} #Gates: {total_gates}")
    print(f"[INFO] {name} #2Q Gates: {num_2q_gates}")
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n")
    print(f"[INFO] {name} #Qubits: {trans_circ.num_qubits}", file=sys.stderr)
    print(f"[INFO] {name} #Depths: {trans_circ.depth()}", file=sys.stderr)
    print(f"[INFO] {name} #Gates: {total_gates}", file=sys.stderr)
    print(f"[INFO] {name} #2Q Gates: {num_2q_gates}", file=sys.stderr)
    print(f"[INFO] {num_qpus} QPUs: {qpus}\n\n", file=sys.stderr)

    task_info = {
        "Circuit": name,
        "#Qubits": trans_circ.num_qubits,
        "#Gates": total_gates,
        "#2Q Gates": num_2q_gates,
        "#Depth": trans_circ.depth(),
        "#QPUs": num_qpus,
        "QPUs": qpus,
        "Basis Gates": basis_gates,
        "2Q Gate Names": two_qubit_gates
    }

    # print(trans_circ)

    return circ, trans_circ, task_info

def truncate_circuit_by_depth(original_circuit: QuantumCircuit, max_depth: int) -> QuantumCircuit:
    """
    截断量子线路，仅保留前 max_depth 层 (不使用 dag_to_circuit)。
    """
    # 0. 如果不需要截断，直接返回副本
    if original_circuit.depth() <= max_depth:
        return original_circuit

    # 1. 完美复制原电路的寄存器结构
    qregs = original_circuit.qregs
    cregs = original_circuit.cregs
    new_circ = QuantumCircuit(*qregs, *cregs)

    # 2. 建立「原比特对象」到「新电路比特对象」的映射
    # 通过索引匹配，这是最稳健的方式
    q_map = {old_q: new_q for old_q, new_q in zip(original_circuit.qubits, new_circ.qubits)}
    c_map = {old_c: new_c for old_c, new_c in zip(original_circuit.clbits, new_circ.clbits)}

    # 3. 仅使用 DAG 来获取层级信息 (不做反向转换)
    dag = circuit_to_dag(original_circuit)

    # 4. 遍历层级，手动 append 到新电路
    for i, layer in enumerate(dag.layers()):
        if i >= max_depth:
            break
        
        # layer['graph'] 是该层的子 DAG，遍历其所有操作节点
        for node in layer['graph'].op_nodes():
            # 跳过 barrier 等非门操作（可选，如果想保留 barrier 可以去掉这行）
            if node.op.name in ['barrier']:
                continue
                
            # 映射比特
            new_qargs = [q_map[q] for q in node.qargs]
            new_cargs = [c_map[c] for c in node.cargs]
            
            # 追加指令
            new_circ.append(node.op, new_qargs, new_cargs)

    return new_circ

def truncate_circuit_by_instructions(original_circuit: QuantumCircuit, max_instructions: int) -> QuantumCircuit:
    """
    简单截断：直接保留前 N 个指令门。
    """
    new_circ = original_circuit.copy()
    # 清空 data，然后只把前 N 个加回来
    new_circ.data = new_circ.data[:max_instructions]
    return new_circ

def load_circ_from_qasm(path, circ_name=None):
    if circ_name == None:
        return None
    # load the .qasm file
    filename = os.path.join(path, circ_name + ".qasm")
    circ = QuantumCircuit.from_qasm_file(filename)
    return circ

def CuccaroAdder(num_qubits):
    """
    创建总量子比特数尽可能接近 num_qubits 的全加法器线路。
    若 num_qubits 为奇数，自动减 1 调整为偶数。

    参数:
        num_qubits: 期望的总量子比特数（若奇数则自动减1）

    返回:
        CDKMRippleCarryAdder 实例，实际量子比特数等于调整后的 actual_qubits
    """
    actual_qubits = num_qubits
    if actual_qubits < 4:
        raise ValueError("full 加法器至少需要 4 个量子比特")
    if actual_qubits % 2 != 0:
        actual_qubits -= 1
        print(
            f"警告: Full adder要求偶数个量子比特，已自动将 {num_qubits} 调整为 {actual_qubits}"
        )

    n = (actual_qubits - 2) // 2
    adder = CDKMRippleCarryAdder(num_state_qubits=n, kind="full")

    # full adder 的总量子比特数应为 2 * n + 2
    assert actual_qubits == adder.num_qubits
    return adder

def MyDraperQFTAdder(num_qubits):
    """
    创建总量子比特数尽可能接近 num_qubits 的 Draper QFT Adder 线路。
    若 num_qubits 为奇数，自动减 1 调整为偶数。

    参数:
        num_qubits: 期望的总量子比特数（若奇数则自动减1）

    返回:
        DraperQFTAdder 实例，实际量子比特数等于调整后的 actual_qubits
    """
    actual_qubits = num_qubits
    if actual_qubits < 2:
        raise ValueError("DraperQFTAdder 至少需要 2 个量子比特")
    if actual_qubits % 2 != 0:
        actual_qubits -= 1
        print(
            f"警告: DraperQFTAdder 需要偶数个量子比特，已自动将 {num_qubits} 调整为 {actual_qubits}"
        )

    n = actual_qubits // 2
    adder = DraperQFTAdder(num_state_qubits=n)

    # fixed DraperQFTAdder 的总量子比特数应为 2 * n
    assert actual_qubits == adder.num_qubits
    return adder

# class QASMbm:
#     def __init__(self, 
#                  do_transpile=True, 
#                  basis_gates=basis_gates, 
#                  path="./QASMBench", 
#                  category="small"):
#         transpile_args = {
#             "basis_gates": basis_gates
#         }
#         self.bm = QASMBenchmark(path, category, 
#                                 num_qubits_list=None, 
#                                 remove_final_measurements=True, 
#                                 do_transpile=do_transpile, 
#                                 **transpile_args)
#         return

#     def set_category(self, category):
#         self.category = category
#         return

#     def get_circuit(self, circ_name, num_qpus=2):
#         circ = self.bm.get(circ_name)
#         qpu_capacity = circ.num_qubits // num_qpus + 1
#         if qpu_capacity % 2 == 1:
#             qpu_capacity += 1
#         qpus = [qpu_capacity] * num_qpus
#         return circ, qpus

#     def get_circuit_list(self, circ_list):
#         circ_list = self.bm.get(circ_list)
#         return circ_list

# def PauliGadget(num_qubits, reps=2, max_pauli_weight=10):
#     if num_qubits < 1:
#         raise ValueError("num_qubits must be >= 1")
#     if reps < 1:
#         raise ValueError("reps must be >= 1")
#     if max_pauli_weight < 1:
#         raise ValueError("max_pauli_weight must be >= 1")

#     rng = np.random.default_rng(26)
#     circuit = QuantumCircuit(num_qubits)
#     pauli_weight = min(max_pauli_weight, num_qubits)

#     if pauli_weight == 1:
#         for _ in range(reps):
#             pauli_string = rng.choice(['X', 'Y', 'Z'])
#             alpha_t = rng.uniform(0, 2 * np.pi)
#             pauli_op = Pauli(pauli_string)
#             evolution_gate = PauliEvolutionGate(pauli_op, time=alpha_t)
#             circuit.append(evolution_gate, [0])
#         return circuit

#     for layer in range(reps):
#         # 生成 bounded k-local Pauli gadget，在全宽随机 Pauli 和 2-local 之间折中线路规模。
#         stride = max(1, pauli_weight // 2)
#         start = (layer * stride) % num_qubits
#         for offset in range(0, num_qubits, stride):
#             qubits = [
#                 (start + offset + delta) % num_qubits
#                 for delta in range(pauli_weight)
#             ]
#             s_t = ['I'] * num_qubits
#             for qubit in qubits:
#                 s_t[qubit] = rng.choice(['X', 'Y', 'Z'])
#             alpha_t = rng.uniform(0, 2 * np.pi)
#             pauli_string = ''.join(s_t)
#             pauli_op = Pauli(pauli_string)
#             evolution_gate = PauliEvolutionGate(pauli_op, time=alpha_t)
#             circuit.append(evolution_gate, range(num_qubits))
#     return circuit

def PauliGadget(num_qubits, reps = 20):
    np.random.seed(26)
    circuit = QuantumCircuit(num_qubits)
    for t in range(reps):
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

def QAOA(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    # Cost Hamiltonian
    for j in range(num_qubits):
        for i in range(j+1, num_qubits):
            qc.cx(j, i)
            qc.rz(random.uniform(0, 2 * pi), i)
            qc.cx(j, i)

    # Mixer Hamiltonian
    for i in range(num_qubits):
        qc.rx(random.uniform(0, 2 * pi), i)
    # qc.h(range(num_qubits))
    # qc.rz(random.uniform(0, 2 * pi), range(num_qubits))
    # qc.h(range(num_qubits))
    return qc

def QiskitQAOA(num_qubits: int, reps: int = 1, seed: int = 26):
    """
    使用 Qiskit QAOAAnsatz 构造一个确定性的 QAOA benchmark 电路。
    cost Hamiltonian 采用全连接 ZZ，相比线性链更接近原始 QAOA() 的结构。
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    terms = []
    if num_qubits == 1:
        terms.append(("Z", 1.0))
    else:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                label = ["I"] * num_qubits
                label[num_qubits - 1 - i] = "Z"
                label[num_qubits - 1 - j] = "Z"
                terms.append(("".join(label), 1.0))

    cost_operator = SparsePauliOp.from_list(terms)
    ansatz = QAOAAnsatz(
        cost_operator=cost_operator,
        reps=reps,
        flatten=True,
        name="QAOA",
    )

    return ansatz

    rng = np.random.default_rng(seed)
    parameter_values = {
        parameter: float(rng.uniform(0, 2 * np.pi))
        for parameter in ansatz.parameters
    }
    return ansatz.assign_parameters(parameter_values)

def GraphState(num_qubits):
    """
    构造一个近似方形二维网格上的 graph state。
    先对所有量子比特施加 H，再对网格中的右邻/下邻施加 CZ。
    当 num_qubits 不是完全平方数时，最后一行自动部分填充。
    """
    if num_qubits < 1:
        raise ValueError("GraphState 至少需要 1 个量子比特")

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    n_cols = int(np.ceil(np.sqrt(num_qubits)))
    n_rows = int(np.ceil(num_qubits / n_cols))

    def _idx(row: int, col: int) -> int:
        return row * n_cols + col

    for row in range(n_rows):
        for col in range(n_cols):
            q = _idx(row, col)
            if q >= num_qubits:
                continue

            right = _idx(row, col + 1)
            if col + 1 < n_cols and right < num_qubits:
                qc.cz(q, right)

            down = _idx(row + 1, col)
            if row + 1 < n_rows and down < num_qubits:
                qc.cz(q, down)

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

    for _ in range(2):
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
    qc.h(range(6))
    qc.ecr(0, 1)
    qc.ecr(2, 3)
    qc.ecr(4, 5)
    # qc.ecr(0, 4)
    # qc.ecr(1, 3)
    # qc.ecr(0, 3)
    # qc.ecr(1, 2)
    # qc.ecr(1, 4)
    # qc.ecr(2, 3)
    # transpiled_qc = transpile(qc, basis_gates=basis_gates)
    return qc #, transpiled_qc

def Grover(num_qubits, k=0):
    ''' One implementation of Grover\'s algorithm '''
    qc = QuantumCircuit(num_qubits)
    oracle = ZGate().control(num_qubits-1-k)
    oracle_qubits = [i for i in range(k, num_qubits)]

    cz = ZGate().control(num_qubits-1)
    qubits = [i for i in range(num_qubits)]

    # 1. Apply Hadamard gates to all qubits
    # for j in range(num_qubits): # |00...0> => |u>
    #     qc.h(j)

    # 2. Start the Grover iteration
    # numIterations = int(round(pi / 4 * sqrt(2**num_qubits / 2**k), 0))
    numIterations = 1

    for _ in range(numIterations):
        # 2.1. Apply the phase inversion use the controlled-Z gate
        # i.e., transform the target state |x*> into -|x*>
        qc.append(oracle, oracle_qubits)  # CC...Z gate: |11...1> => -|11...1>
        # qc.barrier()

        # 2.2. Apply the diffusion operator
        # 2.2.1. Apply H, X to all qubits
        # for j in range(num_qubits):
        #     qc.h(j) # |u> => |00...0>
        #     qc.x(j) # |00...0> => |11...1>

        # 2.2.2. Apply the controlled-Z gate
        qc.append(cz, qubits) # |11...1> => -|11...1>

        # 2.2.3. Recover X, H
        # for j in range(num_qubits):
        #     qc.x(j) # |11...1> => |00...0>
        #     qc.h(j) # |00...0> => |u>
    return qc

def myIQP(num_qubits):
    print(f"[IQP] #Qubits: [{num_qubits}]")
    A = np.random.randint(0, 10, size=(num_qubits, num_qubits))
    symmetric_matrix = (A + A.T) // 2 # 生成对称矩阵
    # print(symmetric_matrix)
    qc = IQP(symmetric_matrix)
    return qc

def VQC_AA(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for _ in range(1):
        # 对每个量子比特应用随机的 RX 门
        # for i in range(num_qubits):
        #     angle_rx = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
        #     qc.rx(angle_rx, i)

        # 对每个量子比特应用随机的 RZ 门
        for i in range(num_qubits):
            angle_rz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rz(angle_rz, i)

        # 双比特的 CRZ 门
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i == j:
                    continue
                qc.barrier()  # 加入 barrier 隔离量子门
                angle_crz = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
                qc.crz(angle_crz, i, j)  # 应用 CRZ 门
    return qc

def VQC(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue
            # qc.barrier()  # 加入 barrier 隔离量子门
            angle = np.random.rand() * 2 * pi  # 随机生成 0 到 2π 的角度
            qc.rzz(angle, i, j)
    return qc

def build_vqe_benchmark(name: str, num_qubits: int):
    """
    构造一个基于 Qiskit QuantumCircuit 接口的 VQE benchmark 电路。

    支持以下名称格式：
    - VQE: 使用默认配置 reps=2, entanglement="linear"
    - VQE_<reps>: 例如 VQE_4
    - VQE_<reps>_<entanglement>: 例如 VQE_3_ring, VQE_2_full

    entanglement 目前支持:
    - linear: 0-1-2-... 链式纠缠
    - ring: 在线性基础上补一条首尾纠缠
    - full: 全连接纠缠
    """
    parts = name.split("_")
    reps = 2
    entanglement = "linear"

    if len(parts) >= 2 and parts[1] != "":
        try:
            reps = int(parts[1])
        except ValueError:
            entanglement = parts[1].lower()

    if len(parts) >= 3 and parts[2] != "":
        entanglement = parts[2].lower()

    return VQE(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

def VQE(num_qubits: int, reps: int = 2, entanglement: str = "linear", seed: int = 26):
    """
    使用 Qiskit TwoLocal 构造一个用于 benchmark 的 VQE ansatz。

    设计目标：
    - 直接复用 Qiskit TwoLocal 模板，接口和 VQE 习惯更一致；
    - 电路结构稳定、参数可复现，适合作为编译/划分 benchmark；
    - 通过 reps 和 entanglement 控制线路规模。
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    entanglement = entanglement.lower()
    if entanglement not in {"linear", "ring", "full"}:
        raise ValueError(
            f"Unsupported VQE entanglement pattern: {entanglement}. "
            "Expected one of {'linear', 'ring', 'full'}."
        )
    two_local_entanglement = "circular" if entanglement == "ring" else entanglement

    rng = np.random.default_rng(seed)

    # 使用一个确定性的 Hartree-Fock 风格初态，避免生成过于稀疏的前缀层。
    initial_state = QuantumCircuit(num_qubits, name="HF")
    for qubit in range(num_qubits // 2):
        initial_state.x(qubit)

    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement=two_local_entanglement,
        reps=reps,
        skip_final_rotation_layer=False,
        initial_state=initial_state,
        parameter_prefix="theta",
        flatten=True,
        name=f"VQE_{reps}_{entanglement}",
    )

    parameter_values = {
        parameter: float(rng.uniform(0, 2 * np.pi))
        for parameter in ansatz.parameters
    }
    return ansatz.assign_parameters(parameter_values)

def build_qknn_benchmark(name: str, num_qubits: int):
    """
    构造一个 fidelity-based QKNN benchmark 电路。

    支持以下名称格式：
    - QKNN: 使用默认配置 reps=2
    - QKNN_<reps>: 例如 QKNN_4
    """
    parts = name.split("_")
    reps = 2

    if len(parts) >= 2 and parts[1] != "":
        reps = int(parts[1])

    return QKNN(num_qubits=num_qubits, reps=reps)

def QKNN(num_qubits: int, reps: int = 2, seed: int = 26) -> QuantumCircuit:
    """
    Generate a fidelity-based Quantum KNN (QKNN) benchmark circuit.

    The circuit compares a training sample and a test sample by applying
    U_phi(x_train) followed by U_phi(x_test)^dagger. The probability of
    measuring the all-zero state estimates the kernel similarity:

        K(x_train, x_test) = |<phi(x_test)|phi(x_train)>|^2

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the generated QKNN circuit.
    reps : int
        Number of repetitions in the ZZFeatureMap.
    seed : int
        Random seed for generating deterministic feature vectors.

    Returns
    -------
    QuantumCircuit
        A QKNN benchmark circuit with exactly num_qubits qubits.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")

    rng = np.random.default_rng(seed)

    x_train = rng.uniform(0, 2 * np.pi, size=num_qubits)
    x_test = rng.uniform(0, 2 * np.pi, size=num_qubits)

    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement="linear",
    )

    train_circuit = feature_map.assign_parameters(x_train)
    test_circuit_inv = feature_map.assign_parameters(x_test).inverse()

    qc = QuantumCircuit(num_qubits, name="QKNN")
    qc.compose(train_circuit, qubits=range(num_qubits), inplace=True)
    qc.compose(test_circuit_inv, qubits=range(num_qubits), inplace=True)

    return qc

def BV(num_qubits, secret_bitstring=None):
    """
    为给定的秘密比特串构建Bernstein-Vazirani算法电路。
    
    参数:
        num_qubits (int): 电路中总量子比特数（包括辅助比特）
        secret_bitstring (str, optional): 由'0'和'1'组成的秘密字符串。
                                         长度应为 num_qubits - 1。
                                         如果为None，则生成101010...。
        
    返回:
        QuantumCircuit: 完整的BV算法量子电路。
    """
    # 数据比特的数量 = 总量子比特数 - 1（辅助比特）
    n = num_qubits - 1
    
    # 如果secret_bitstring是None，生成101010...模式
    if secret_bitstring is None:
        # ''.join(np.random.choice(['0', '1'], size=n))
        # secret_bitstring = ("10" * ((n + 1) // 2))[:n]
        secret_bitstring = "1" * n
    
    # 如果secret_bitstring太长，截取前n位
    if len(secret_bitstring) > n:
        secret_bitstring = secret_bitstring[:n]
        # print(f"秘密比特串过长，已截取前{n}位: {secret_bitstring}")
    
    # 如果secret_bitstring太短，在后面补0
    if len(secret_bitstring) < n:
        secret_bitstring = secret_bitstring.ljust(n, '0')
        # print(f"秘密比特串过短，已补充0至{n}位: {secret_bitstring}")
    
    # 需要 n 个数据比特 + 1个辅助比特
    circuit = QuantumCircuit(num_qubits) # , n
    
    # 1. 对辅助比特应用X门，使其进入 |1> 态
    # 辅助比特是最后一个量子比特（索引为n）
    circuit.x(n)
    
    # 2. 对所有比特应用Hadamard门
    circuit.h(range(num_qubits))
    
    # 3. 构建Oracle：根据秘密比特串，在对应位置添加CNOT门
    # 如果秘密比特串的第i位是'1'，则在第i个数据比特和辅助比特之间添加CNOT门
    for i, bit in enumerate(secret_bitstring):
        if bit == '1':
            circuit.cx(i, n)
    
    # 4. 再次对所有数据比特应用Hadamard门（不包括辅助比特）
    circuit.h(range(n))
    
    # 5. 测量数据比特
    # circuit.measure(range(n), range(n))
    
    return circuit

# def QPE(num_qubits, unitary_matrix=None):
#     # 如果unitary_matrix
#     # 随机
#     num_eval_qubits = num_qubits - 1

#     circ = PhaseEstimation(
#         num_evaluation_qubits: int,  # 计数/估计比特数 t
#         unitary: Gate | QuantumCircuit,  # 待估计的酉算子 U
#         iqft: QuantumCircuit | None = None,  # 自定义逆QFT，默认内置标准IQFT
#         name: str = "QPE"
#     )
