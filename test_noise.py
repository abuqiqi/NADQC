import datetime
from pprint import pprint
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from src.utils import get_config, Backend, Network

def extract_backend_info(backend: Backend):
    """
    从 Backend 实例中提取编译所需信息：耦合列表、基门集。
    返回：
        coupling_list: 两比特门支持的物理比特对列表（双向）
        basis_gates: 基门集列表
    """
    coupling_set = set()
    basis_set = set()

    print("Extracting backend information...")
    # pprint(backend.gate_info)

    for gate in backend.gate_info:
        # print(f"Processing gate: {gate}")
        gate_name = gate["gate"]
        qubits_str = gate["qubits"]
        # 解析量子比特列表（例如 "0,1"）
        try:
            qubits = [int(q) for q in qubits_str.split(",")]
        except:
            print(f"Invalid qubit string: {qubits_str}")
            continue

        basis_set.add(gate_name)

        # 两比特门加入耦合图（双向）
        if len(qubits) == 2 and gate_name in {"cx", "cz", "ecr"}:
            coupling_set.add(tuple(qubits))
            coupling_set.add(tuple(reversed(qubits)))

    coupling_list = [list(pair) for pair in coupling_set]
    basis_gates = list(basis_set)
    return coupling_list, basis_gates

def main():
    # 1. 配置参数
    num_qubits_per_qpu = 10
    global_config = get_config()
    backend_config = {
        'backend_name': f'ibm_torino_sampled_{num_qubits_per_qpu}q',
        'date': datetime.datetime(2025, 11, 9)
    }

    # 2. 创建采样后端
    backend = Backend(global_config, backend_config)
    backend.print()
    print(f"[DEBUG] Backend basis gates: {backend.basis_gates}")
    print(f"[DEBUG] Backend coupling map: {backend.coupling_map}")

    # 3. 构建一个简单的量子线路（逻辑比特数 <= 物理比特数）
    circuit = QuantumCircuit(3, 3)
    circuit.sx(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)

    print("\nOriginal circuit:")
    print(circuit.draw())

    # 4. 设置初始布局：逻辑比特 0,1,2 映射到物理比特 0,1,2
    initial_layout = [0, 2, 3]  # 逻辑 i -> 物理 initial_layout[i]

    # 5. 编译线路
    transpiled = transpile(
        circuit,
        coupling_map=backend.coupling_map,
        basis_gates=backend.basis_gates,
        initial_layout=initial_layout,
        optimization_level=0,   # 最小优化，尽可能保留布局
        # routing_method='sabre'   # 允许路由插入 SWAP
    )

    print("\nTranspiled circuit:")
    print(transpiled.draw())

    # # 7. 输出编译后的一些统计信息
    # print(f"Depth: {transpiled.depth()}")
    # print(f"Total gates: {transpiled.size()}")
    # # 统计特定门数量
    # gate_counts = transpiled.count_ops()
    # print("Gate counts:", gate_counts)

    # 可选：将编译后线路保存为图片
    # transpiled.draw(output='mpl', filename='transpiled_circuit.png')

if __name__ == "__main__":
    main()