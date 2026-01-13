from pprint import pprint
import random

def test_mock_component_partitioner():
    try:
        from nadqc import PartitionerFactory

        # 模拟网络对象
        class MockNetwork:
            def __init__(self, qpu_capacities):
                self.qpu_capacities = qpu_capacities
            
            def get_backend_qubit_counts(self):
                return self.qpu_capacities
        
        # 创建模拟网络（3个QPU，容量分别为4, 4, 2）
        mock_network = MockNetwork([4, 4, 2])
        
        # 创建组件（模拟连通分量）
        components = [[0, 1], [2, 3], [4], [5], [6, 7, 8]]
        
        # 测试不同类型的分区器
        partitioners = ["greedy", "dynamic_programming", "recursive_dp"]

        print("Testing different component partitioners:\n")
        
        for ptype in partitioners:
            partitioner = PartitionerFactory.create_partitioner(ptype, mock_network)
            result = partitioner.partition(components)
            print(f"Number of partition options: {len(result)}")
            for i, partition in enumerate(result):
                print(f"Option {i+1}: {partition}")
            print("-" * 50)

        return True, "Component Partitioner test passed"
    except Exception as e:
        return False, f"Component Partitioner test failed: {str(e)}"


def test_mock_partition_assigner():
    try:
        from nadqc import PartitionAssignerFactory

        # 创建示例数据（模拟P和subc_ranges）
        class MockDataStructure:
            def __init__(self):
                # self.partition_candidates = [
                #     [[[0, 1], [2, 3]]],  # 时间片0-0的划分方案
                #     [[[0, 2], [1, 3]]],  # 时间片1-1的划分方案
                #     [[[0, 3], [1, 2]]]   # 时间片2-2的划分方案
                # ] * 3  # 扩展为3个时间段
                self.partition_candidates = []
                # 随机生成一些partition_candidates数据以模拟更复杂的情况
                num_qubits = 12
                num_qpus = 4
                num_time_slices = 5
                group_size = num_qubits // num_qpus
                qubits = list(range(num_qubits))
                for _ in range(num_time_slices):
                    partition_candidates = []

                    for _ in range(2):  # 每个时间片有两个划分方案
                        random.shuffle(qubits)
                        partition = [
                            qubits[i * group_size : (i + 1) * group_size]
                            for i in range(num_qpus)
                        ]
                        partition_candidates.append(partition)
                    self.partition_candidates.append(partition_candidates)
                pprint(self.partition_candidates)
        
        mock_data = MockDataStructure()
        
        # 创建不同的分配器实例
        assigners = ["direct", "max_match", "global_max_match"]
        
        print("Testing different partition assigners:\n")
        
        for assigner_type in assigners:
            assigner = PartitionAssignerFactory.create_assigner(assigner_type)
            print(f"Processing with {assigner.get_name()}:")
            result = assigner.assign_partitions(mock_data.partition_candidates)
            print("Partition Plan:")
            pprint(result['partition_plan'])
            print()
            pprint(f"Metrics: {result['metrics']}")
            print("-" * 50)

        return True, "Partition Assigner test passed"
    except Exception as e:
        return False, f"Partition Assigner test failed: {str(e)}"
