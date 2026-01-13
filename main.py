import datetime
from pprint import pprint
from src.nadqc import Backend, Network, NADQC
from src.utils import get_config
from src.nadqc import SimpleMapper, LinkOrientedMapper

# 自定义网络配置
network_config = {
    'type': 'self_defined',
    'network_coupling': {
        (0, 1): 0.99,
        (1, 2): 0.985,
        (0, 2): 0.977
    }
}

# backend = Backend(config={"num_qubits": 10})

# config = get_config()

# backend = Backend()
# date = datetime.datetime(2025, 10, 20)
# backend.load_properties(config, "ibm_torino", date)

# backend.sample_and_export(10, config["output_folder"])

global_config = get_config()
backend_config = {
    'backend_name': 'ibm_torino_sampled_3q',
    'date': datetime.datetime(2025, 11, 9)
}

backend = Backend(global_config, backend_config)

backend_config = [backend for _ in range(2)]

network = Network(network_config, backend_config)

# 创建一个简单的量子电路
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume, QFT
qc = QuantumVolume(6, seed=26).decompose()
# qc = QFT(15).decompose()
qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=2)
# print(qc)

# 分配
nadqc = NADQC(circ=qc, network=network)
nadqc.distribute()
partition_plan = nadqc.get_partition_plan()

# # 测试基线映射器
# simple_mapper = SimpleMapper()
# result_baseline = simple_mapper.map_circuit(partition_plan, network)

# # 测试链路导向映射器
# link_oriented_mapper = LinkOrientedMapper()
# result_link_oriented = link_oriented_mapper.map_circuit(partition_plan, network)


# # 输出比较结果
# print(f"Simple Mapper: {simple_mapper.get_name()}")
# print(f"Metrics: {result_baseline['metrics']}")
# print(f"Mapping Seq: {result_baseline['mapping_sequence']}")

# print(f"\nLink Oriented Mapper: {link_oriented_mapper.get_name()}")
# print(f"Metrics: {result_link_oriented['metrics']}")
# print(f"Mapping Seq: {result_link_oriented['mapping_sequence']}")
