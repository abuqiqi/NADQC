def test_static_oee():
    try:
        import datetime
        from pprint import pprint

        from utils import get_config, Backend, Network
        from baselines import CompilerFactory

        global_config = get_config()
        backend_config = {
            'backend_name': 'ibm_torino_sampled_10q',
            'date': datetime.datetime(2025, 11, 9)
        }

        backend = Backend(global_config, backend_config)
        backend.print()

        backend_config = [backend for _ in range(3)]

        # 自定义网络配置
        network_config = {
            'type': 'self_defined',
            'network_coupling': {
                (0, 1): 0.979,
                (1, 2): 0.98,
                (0, 2): 0.981
            }
        }

        net = Network(network_config, backend_config)

        # 创建一个简单的量子电路
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import QuantumVolume, QFT
        qc = QuantumVolume(30, seed=42).decompose()
        # qc = QFT(30).decompose()
        qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
        # print(qc)

        compiler_names = ["staticoee"]
        compilers = []

        for name in compiler_names:
            compilers.append(CompilerFactory.create_compiler(name))

        for compiler in compilers:
            result = compiler.compile(qc, net)
            pprint(result)

        return True, "Static OEE test passed"
    except Exception as e:
        return False, f"Static OEE test failed: {str(e)}"