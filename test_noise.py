import datetime
from pprint import pprint
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from src.utils import get_config, Backend, Network


def noise():
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
    # initial_layout = [0, 2, 3]  # 逻辑 i -> 物理 initial_layout[i]

    # 5. 编译线路
    transpiled = transpile(
        circuit,
        coupling_map=backend.coupling_map,
        basis_gates=backend.basis_gates,
        # initial_layout=initial_layout,
        # optimization_level=0,   # 最小优化，尽可能保留布局
        # routing_method='sabre'   # 允许路由插入 SWAP
    )

    print("\nTranspiled circuit:")
    print(transpiled.draw())

    # print(backend.gate_info, type(backend.gate_info))

    # 6. 评估每一个操作的保真度和保真度损失
    # backend.gate_info里面有每个量子门的gate_error_value
    # key是量子门名字_量子比特，例如 'cx_0_1'，表示在物理比特0和1之间的CX门
    print("\nGate fidelities:")

    for instruction in transpiled:
        gate_name = instruction.operation.name
        qubits = [qubit._index for qubit in instruction.qubits]
        if qubits[0] == None:
            qubits = [transpiled.qubits.index(qubit) for qubit in instruction.qubits]

        print(f"Instruction: {gate_name} on qubits {qubits}")
        gate_key = f"{gate_name}{'_'.join(map(str, qubits))}"
        print(f"[DEBUG] Looking for gate_key: {gate_key} in backend.gate_dict")
        gate_error = backend.gate_dict.get(gate_key, {}).get("gate_error_value", None)
        print(f"[DEBUG] Retrieved gate error: {gate_error} for gate_key: {gate_key}")
        # if gate_error is not None:
        #     print(f"Gate: {gate_key}, Gate Error: {gate_error} ({type(gate_error)})")
        # else:
        #     print(f"Gate: {gate_key}, Gate Error: N/A (not found in backend)")

    # for gate in transpiled.count_ops():
        # print(f"Gate: {gate}")
        # count = transpiled.count_ops()[gate]
        # 获取qubit

        # gate_key = f"{gate}{}_1"  # 这是一个简化的假设
        # fidelity = backend.gate_dict[gate_key]["gate_error_value"]

        # if fidelity is not None:
        #     print(f"Gate: {gate}, Count: {count}, Fidelity: {fidelity} ({type(fidelity)})")
        # else:
        #     print(f"Gate: {gate}, Count: {count}, Fidelity: N/A (not found in backend)")

    # # 7. 输出编译后的一些统计信息
    # print(f"Depth: {transpiled.depth()}")
    # print(f"Total gates: {transpiled.size()}")
    # # 统计特定门数量
    # gate_counts = transpiled.count_ops()
    # print("Gate counts:", gate_counts)

    # 可选：将编译后线路保存为图片
    # transpiled.draw(output='mpl', filename='transpiled_circuit.png')

if __name__ == "__main__":
    noise()