import datetime
from pprint import pprint
from src.nadqc import Backend, Network, NADQC
from src.utils import get_config

# 自定义网络配置
network_config = {
    'type': 'self_defined',
    'network_coupling': {
        (0, 1): 0.99,
        (1, 2): 0.98,
        (0, 2): 0.97
    }
}

# backend = Backend(config={"num_qubits": 10})

config = get_config()

backend = Backend()
date = datetime.datetime(2025, 10, 20)
backend.load_properties(config, "ibm_torino", date)

backend.sample_and_export(10, config["output_folder"])

backend_config = [backend for _ in range(3)]

net = Network(network_config, backend_config)

# 创建一个简单的量子电路
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume, QFT
qc = QuantumVolume(30, seed=26).decompose()
# qc = QFT(15).decompose()
qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
# print(qc)

# 分配
nadqc = NADQC(circ=qc, network=net)
nadqc.distribute()
partition_plan = nadqc.get_partition_plan()
total_comm_cost, mapping_sequence = nadqc.calculate_comm_cost_dynamic(partition_plan)
print("Partition Plan:")
pprint(partition_plan)
print(f"Total Communication Cost: {total_comm_cost:.4f}")
print("Mapping Sequence:")
pprint(mapping_sequence)