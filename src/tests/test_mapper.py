import datetime
from pprint import pprint

def test_mock_network():
    try:
        import numpy as np
        from nadqc import MapperFactory

        # 创建一个模拟网络对象
        class MockNetwork:
            def __init__(self, num_backends=4):
                self.num_backends = num_backends
                # 模拟保真度矩阵
                self.W_eff = np.random.rand(num_backends, num_backends)
                # 对角线设为1（自我传输保真度最高）
                np.fill_diagonal(self.W_eff, 1.0)

        # 创建一个模拟划分计划
        partition_plan = [
            [[0, 1], [2, 3]],  # 时间片0：逻辑QPU 0管理qubits [0,1]，逻辑QPU 1管理qubits [2,3]
            [[0, 2], [1, 3]],  # 时间片1：重新分配
            [[0, 3], [1, 2]]   # 时间片2：再次重新分配
        ]

        network = MockNetwork(num_backends=2)
        # pprint(network.W_eff)
        
        # 测试基线映射器
        simple_mapper = MapperFactory.create_mapper("simple")
        result_baseline = simple_mapper.map_circuit(partition_plan, network)

        # 测试链路导向映射器
        link_oriented_mapper = MapperFactory.create_mapper("link_oriented")
        result_link_oriented = link_oriented_mapper.map_circuit(partition_plan, network)

        # 输出比较结果
        print(f"Simple Mapper: {simple_mapper.get_name()}")
        print(f"Metrics: {result_baseline['metrics']}")
        print(f"Mapping Seq: {result_baseline['mapping_sequence']}")

        print(f"\nLink Oriented Mapper: {link_oriented_mapper.get_name()}")
        print(f"Metrics: {result_link_oriented['metrics']}")
        print(f"Mapping Seq: {result_link_oriented['mapping_sequence']}")

        return True, "Mapper test passed"
    except Exception as e:
        return False, f"Mapper test failed: {str(e)}"


def test_mapper():
    """测试 Mapper 类"""
    try:
        from utils import get_config
        from nadqc import Backend, Network, NADQC, MapperFactory, PartitionAssignerFactory

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
                (0, 1): 0.99,
                (1, 2): 0.98,
                (0, 2): 0.97
            }
        }

        net = Network(network_config, backend_config)

        # 创建一个简单的量子电路
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import QuantumVolume, QFT
        qc = QuantumVolume(30, seed=42).decompose()
        # qc = QFT(15).decompose()
        qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
        # print(qc)

        # 分配
        nadqc = NADQC(circ=qc, network=net)
        nadqc.distribute()

        # partitioner = PartitionAssignerFactory.create_assigner("global_max_match")
        # partition_candidates = nadqc.get_partition_candidates()
        # partition_plan = partitioner.assign_partitions(partition_candidates)["partition_plan"]
        # print("Partition Plan:")
        # pprint(partition_plan)

        partitioner = PartitionAssignerFactory.create_assigner("max_match")
        partition_candidates = nadqc.get_partition_candidates()
        partition_plan = partitioner.assign_partitions(partition_candidates)["partition_plan"]

        simple_mapper  = MapperFactory.create_mapper("simple")
        link_oriented_mapper = MapperFactory.create_mapper("link_oriented")

        # total_comm_cost, total_comm_cost_mm, total_comm_cost_gmm = 0, 0, 0

        # for t in range(len(partition_plan) - 1):
        #     current_partition = partition_plan[t]
        #     next_partition = partition_plan[t + 1]
        #     switch_demand, switch_mapping = link_oriented_mapper._compute_switch_demand(current_partition, next_partition)
        #     # print(f"Time Step {t} to {t+1}:")
        #     # print("Switch Demand Matrix:")
        #     # print(switch_demand)
        #     # print("Switch Mapping:")
        #     # pprint(switch_mapping)

        #     current_partition_max_match = partition_plan_max_match[t]
        #     next_partition_max_match = partition_plan_max_match[t + 1]
        #     switch_demand_mm, switch_mapping_mm = link_oriented_mapper._compute_switch_demand(current_partition_max_match, next_partition_max_match)
        #     # print(f"Time Step {t} to {t+1} (Max Match):")
        #     # print("Switch Demand Matrix (Max Match):")
        #     # print(switch_demand_mm)

        #     current_partition_global_max_match = partition_plan_global_max_match[t]
        #     next_partition_global_max_match = partition_plan_global_max_match[t + 1]
        #     switch_demand_gmm, switch_mapping_gmm = link_oriented_mapper._compute_switch_demand(current_partition_global_max_match, next_partition_global_max_match)
        #     # print(f"Time Step {t} to {t+1} (Global Max Match):")
        #     # print("Switch Demand Matrix (Global Max Match):")
        #     # print(switch_demand_gmm)

        #     # 计算总通信开销，即switch_demand、switch_demand_mm的总和
        #     # print(f"Total Communication Cost Calculation:{switch_demand.sum()} VS {switch_demand_mm.sum()}")
        #     total_comm_cost += switch_demand.sum()
        #     total_comm_cost_mm += switch_demand_mm.sum()
        #     total_comm_cost_gmm += switch_demand_gmm.sum()

        # print(f"Total Communication Cost: {total_comm_cost}")
        # print(f"Total Communication Cost (Max Match): {total_comm_cost_mm}")
        # print(f"Total Communication Cost (Global Max Match): {total_comm_cost_gmm}")


        result_baseline = simple_mapper.map_circuit(partition_plan, net)
        total_comm_cost_baseline = result_baseline['metrics']['total_comm_cost']
        result_dynamic = link_oriented_mapper.map_circuit(partition_plan, net)
        total_comm_cost_dynamic = result_dynamic['metrics']['total_comm_cost']

        print(f"Total Communication Cost (Simple): {total_comm_cost_baseline}")
        print(f"Total Communication Cost (Dynamic): {total_comm_cost_dynamic}")

        print("Simple Mapper:")
        pprint(result_baseline['metrics'])
        print("Link-Oriented Mapper:")
        pprint(result_dynamic['metrics'])

        return True, "Mapper test passed"
    except Exception as e:
        return False, f"Mapper test failed: {str(e)}"

