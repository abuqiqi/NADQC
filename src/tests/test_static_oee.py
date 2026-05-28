import datetime
from pprint import pprint

from qiskit import transpile
from qiskit.circuit.library import QuantumVolume

from src.compiler import CompilerFactory
from src.utils import Backend, Network, get_config


def test_static_oee():
    global_config = get_config()
    backend_config = {
        'backend_name': 'ibm_torino_sampled_11q',
        'date': datetime.datetime(2025, 11, 9)
    }

    backend = Backend(global_config, backend_config)
    backend.print()

    backend_config = [backend for _ in range(3)]

    # 自定义网络配置
    network_config = {
        'type': 'self_defined',
        'comm_slot_reserve': 1,
        'network_coupling': {
            (0, 1): 0.979,
            (1, 2): 0.98,
            (0, 2): 0.981
        }
    }

    net = Network(network_config, backend_config)

    # 创建一个简单的量子电路
    qc = QuantumVolume(30, seed=42).decompose()
    # qc = QFT(30).decompose()
    qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
    # print(qc)

    compiler_names = ["staticoee"]
    compilers = []
    CompilerFactory.register_compilers(global_config.get("compiler_modules"))

    for name in compiler_names:
        compilers.append(CompilerFactory.get_compiler(name)())

    for compiler in compilers:
        result = compiler.compile(qc, net)
        pprint(result)
