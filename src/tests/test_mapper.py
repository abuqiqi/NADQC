# import datetime
# from pprint import pprint

# def test_mock_network():
#     try:
#         import numpy as np
#         from navi import MapperFactory

#         # 创建一个模拟网络对象
#         class MockNetwork:
#             def __init__(self, num_backends=4):
#                 self.num_backends = num_backends
#                 # 模拟保真度矩阵
#                 self.move_fidelity = np.random.rand(num_backends, num_backends)
#                 # 对角线设为1（自我传输保真度最高）
#                 np.fill_diagonal(self.move_fidelity, 1.0)

#         # 创建一个模拟划分计划
#         partition_plan = [
#             [[0, 1], [2, 3]],  # 时间片0：逻辑QPU 0管理qubits [0,1]，逻辑QPU 1管理qubits [2,3]
#             [[0, 2], [1, 3]],  # 时间片1：重新分配
#             [[0, 3], [1, 2]]   # 时间片2：再次重新分配
#         ]

#         network = MockNetwork(num_backends=2)
#         # pprint(network.move_fidelity)
        
#         # 测试基线映射器
#         simple_mapper = MapperFactory.create_mapper("simple")
#         result_baseline = simple_mapper.map_circuit(partition_plan, network)

#         # 测试链路导向映射器
#         link_oriented_mapper = MapperFactory.create_mapper("link_oriented")
#         result_link_oriented = link_oriented_mapper.map_circuit(partition_plan, network)

#         # 输出比较结果
#         print(f"Simple Mapper: {simple_mapper.get_name()}")
#         print(f"Metrics: {result_baseline['metrics']}")
#         print(f"Mapping Seq: {result_baseline['mapping_sequence']}")

#         print(f"\nLink Oriented Mapper: {link_oriented_mapper.get_name()}")
#         print(f"Metrics: {result_link_oriented['metrics']}")
#         print(f"Mapping Seq: {result_link_oriented['mapping_sequence']}")

#         return True, "Mapper test passed"
#     except Exception as e:
#         return False, f"Mapper test failed: {str(e)}"


# def test_mapper():
#     """测试 Mapper 类"""
#     try:
#         from utils import get_config, Backend, Network
#         from navi import NAVI, MapperFactory, PartitionAssignerFactory

#         global_config = get_config()
#         backend_config = {
#             'backend_name': 'ibm_torino_sampled_10q',
#             'date': datetime.datetime(2025, 11, 9)
#         }

#         backend = Backend(global_config, backend_config)
#         backend.print()

#         backend_config = [backend for _ in range(3)]

#         # 自定义网络配置
#         network_config = {
#             'type': 'self_defined',
#             'network_coupling': {
#                 (0, 1): 0.979,
#                 (1, 2): 0.98,
#                 (0, 2): 0.981
#             }
#         }

#         net = Network(network_config, backend_config)

#         # 创建一个简单的量子电路
#         from qiskit import QuantumCircuit, transpile
#         from qiskit.circuit.library import QuantumVolume, QFT
#         qc = QuantumVolume(30, seed=42).decompose()
#         # qc = QFT(30).decompose()
#         # qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
#         # print(qc)

#         # 分配
#         nadqc = NADQC(circ=qc, network=net)
#         nadqc.distribute()

#         # partitioner = PartitionAssignerFactory.create_assigner("global_max_match")
#         partitioner = PartitionAssignerFactory.create_assigner("max_match")
#         partition_candidates = nadqc.get_partition_candidates()
#         partition_plan = partitioner.assign_partitions(partition_candidates)["partition_plan"]
#         # print("Partition Plan:")
#         # pprint(partition_plan)

#         mapper_names = ["simple", "link_oriented", "exact", "greedy"]
#         mappers = []

#         for name in mapper_names:
#             mappers.append(MapperFactory.create_mapper(name))

#         for mapper in mappers:
#             result = mapper.map_circuit(partition_plan, net)
#             print(f"\nMapper: {mapper.get_name()}")
#             print(f"Metrics: {result['metrics']}")
#             print(f"Mapping Seq: {result['mapping_sequence']}")

#         return True, "Mapper test passed"
#     except Exception as e:
#         return False, f"Mapper test failed: {str(e)}"

