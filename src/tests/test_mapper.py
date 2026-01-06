import datetime
from pprint import pprint

def test_mapper():
    """测试 Mapper 类"""
    try:
        from utils import get_config
        from nadqc import Backend, Network, NADQC, BaselineMapper, LinkOrientedMapper

        global_config = get_config()
        backend_config = {
            'backend_name': 'ibm_torino_sampled_5q',
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
        qc = QuantumVolume(15, seed=821).decompose()
        # qc = QFT(15).decompose()
        qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
        # print(qc)

        # 分配
        nadqc = NADQC(circ=qc, network=net)
        nadqc.distribute()
        partition_plan = nadqc.get_partition_plan()
        # print("Partition Plan:")
        # pprint(partition_plan)
        partition_plan_max_match = nadqc.get_partition_plan_max_match()
        # print("Partition Plan (Max Match):")
        # pprint(partition_plan_max_match)
        partition_plan_global_max_match = nadqc.get_partition_plan_global_max_match()
        # print("Partition Plan (Global Max Match):")
        # pprint(partition_plan_global_max_match)

        bm  = BaselineMapper(net)
        lom = LinkOrientedMapper(net)

        total_comm_cost, total_comm_cost_mm, total_comm_cost_gmm = 0, 0, 0

        for t in range(len(partition_plan) - 1):
            current_partition = partition_plan[t]
            next_partition = partition_plan[t + 1]
            switch_demand, switch_mapping = lom._compute_switch_demand(current_partition, next_partition)
            # print(f"Time Step {t} to {t+1}:")
            # print("Switch Demand Matrix:")
            # print(switch_demand)
            # print("Switch Mapping:")
            # pprint(switch_mapping)

            current_partition_max_match = partition_plan_max_match[t]
            next_partition_max_match = partition_plan_max_match[t + 1]
            switch_demand_mm, switch_mapping_mm = lom._compute_switch_demand(current_partition_max_match, next_partition_max_match)
            # print(f"Time Step {t} to {t+1} (Max Match):")
            # print("Switch Demand Matrix (Max Match):")
            # print(switch_demand_mm)

            current_partition_global_max_match = partition_plan_global_max_match[t]
            next_partition_global_max_match = partition_plan_global_max_match[t + 1]
            switch_demand_gmm, switch_mapping_gmm = lom._compute_switch_demand(current_partition_global_max_match, next_partition_global_max_match)
            # print(f"Time Step {t} to {t+1} (Global Max Match):")
            # print("Switch Demand Matrix (Global Max Match):")
            # print(switch_demand_gmm)

            # 计算总通信开销，即switch_demand、switch_demand_mm的总和
            # print(f"Total Communication Cost Calculation:{switch_demand.sum()} VS {switch_demand_mm.sum()}")
            total_comm_cost += switch_demand.sum()
            total_comm_cost_mm += switch_demand_mm.sum()
            total_comm_cost_gmm += switch_demand_gmm.sum()

        print(f"Total Communication Cost: {total_comm_cost}")
        print(f"Total Communication Cost (Max Match): {total_comm_cost_mm}")
        print(f"Total Communication Cost (Global Max Match): {total_comm_cost_gmm}")

        # 测试calculate_comm_cost_baseline和calculate_comm_cost_dynamic
        total_comm_cost_baseline, mapping_sequence_baseline = bm.calculate_comm_cost_baseline(partition_plan)
        total_comm_cost_dynamic, mapping_sequence_dynamic = lom.calculate_comm_cost_dynamic(partition_plan)

        print(f"Total Communication Cost (Baseline): {total_comm_cost_baseline}")
        print(f"Total Communication Cost (Dynamic): {total_comm_cost_dynamic}")

        return True, "Mapper test passed"
    except Exception as e:
        return False, f"Mapper test failed: {str(e)}"
