from pprint import pprint

def test_partition():
    """测试最优路径计算"""
    try:
        from nadqc import Backend, Network, NADQC

        # 自定义网络配置
        network_config = {
            'type': 'self_defined',
            'network_coupling': {
                (0, 1): 0.99,
                (1, 2): 0.98,
                (0, 2): 0.93
            }
        }

        backend = Backend(config={"num_qubits": 5})
        backend_config = [backend for _ in range(3)]

        net = Network(network_config, backend_config)

        # 创建一个简单的量子电路
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import QuantumVolume, QFT
        qc = QuantumVolume(15, seed=26).decompose()
        # qc = QFT(15).decompose()
        qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
        # print(qc)

        # 分配
        nadqc = NADQC(circ=qc, network=net)
        nadqc.distribute()
        partition_plan = nadqc.get_partition_plan()
        print("Partition Plan:")
        pprint(partition_plan)

        return True, "Partition test passed"
    except Exception as e:
        return False, f"Partition test failed: {str(e)}"