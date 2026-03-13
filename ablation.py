import sys
import time
import datetime
from pprint import pprint

from src.utils import get_args, get_config, select_circuit, write_compiler_results_to_csv, Backend, Network
from src.nadqc import NADQC, PartitionAssignerFactory, MapperFactory
from src.compiler import CompilerFactory

def ablation(args):
    global_config = get_config(args.global_config_path)

    circ, trans_circ, task_info = select_circuit(args.circuit_name,
                                                 args.qubit_count,
                                                 args.core_count,
                                                 args.core_capacity,
                                                 args.gate_set)
    # TODO: basis gate set换成硬件机器的实际gate set

    # TODO: 根据task_info里面的QPU信息，构建对应的Backend和Network

    backend_list = []
    for i in range(args.core_count):
        # 将字符串列表转换为datetime对象列表
        backend_config = {
            'backend_name': f'{args.backend_name[i]}_sampled_{args.core_capacity[i]}q',
            'date': datetime.datetime.strptime(args.date[i], "%Y-%m-%d")
        }
        backend = Backend(global_config, backend_config)
        backend_list.append(backend)

    network_config = global_config.get('network', {})

    network = Network(network_config, backend_list)

    result_info = {}

    compiler_ids = CompilerFactory.register_compilers(global_config.get("compiler_modules"))
    # compilers = []
    # for compiler_id in compiler_ids:
    #     compilers.append(CompilerFactory.get_compiler(compiler_id)())

    compiler = CompilerFactory.get_compiler("nadqc")()

    circuit_name = f"{args.circuit_name}{args.qubit_count}"

    # partition_assigner: direct, max_match, global_max_match
    # mapper: simple, link_oriented, exact, greedy
    # 构建compiler configs
    partition_assigners = ["direct", "max_match", "global_max_match"]
    mappers = ["simple", "link_oriented", "exact", "greedy"]
    compiler_configs = []
    for partition_assigner in partition_assigners:
        for mapper in mappers:
            compiler_configs.append({
                "circuit_name": circuit_name,
                "partition_assigner": partition_assigner,
                "mapper": mapper
            })

    print(f"Compiler: [{compiler.name}]")
    for compiler_config in compiler_configs:
        result = compiler.compile(trans_circ, network, compiler_config)
        # pprint(result.total_costs)
        result_info[f"{compiler.name}_{compiler_config['partition_assigner']}_{compiler_config['mapper']}"] = result.total_costs

    # Write results to CSV
    write_compiler_results_to_csv(task_info, result_info, f"{global_config.get('output_folder')}compiler_results.csv")

    return

if __name__ == "__main__":
    # 获取全局配置
    args = get_args()
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}"
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'a') as f:
        # sys.stdout = f
        start_time = time.time()
        ablation(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout



# # 自定义网络配置
# network_config = {
#     'type': 'self_defined',
#     'network_coupling': {
#         (0, 1): 0.99,
#         (1, 2): 0.985,
#         (0, 2): 0.977
#     }
# }

# # backend = Backend(config={"num_qubits": 10})

# # config = get_config()

# # backend = Backend()
# # date = datetime.datetime(2025, 10, 20)
# # backend.load_properties(config, "ibm_torino", date)

# # backend.sample_and_export(10, config["output_folder"])

# global_config = get_config()
# backend_config = {
#     'backend_name': 'ibm_torino_sampled_3q',
#     'date': datetime.datetime(2025, 11, 9)
# }

# backend = Backend(global_config, backend_config)

# backend_config = [backend for _ in range(2)]

# network = Network(network_config, backend_config)

# # 创建一个简单的量子电路
# from qiskit import QuantumCircuit, transpile
# from qiskit.circuit.library import QuantumVolume, QFT
# qc = QuantumVolume(6, seed=26).decompose()
# # qc = QFT(15).decompose()
# qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=2)
# # print(qc)

# # 分配
# nadqc = NADQC(circ=qc, network=network)
# nadqc.distribute()
# partition_plan = nadqc.get_partition_plan()

# # # 测试基线映射器
# # simple_mapper = SimpleMapper()
# # result_baseline = simple_mapper.map_circuit(partition_plan, network)

# # # 测试链路导向映射器
# # link_oriented_mapper = LinkOrientedMapper()
# # result_link_oriented = link_oriented_mapper.map_circuit(partition_plan, network)


# # # 输出比较结果
# # print(f"Simple Mapper: {simple_mapper.get_name()}")
# # print(f"Metrics: {result_baseline['metrics']}")
# # print(f"Mapping Seq: {result_baseline['mapping_sequence']}")

# # print(f"\nLink Oriented Mapper: {link_oriented_mapper.get_name()}")
# # print(f"Metrics: {result_link_oriented['metrics']}")
# # print(f"Mapping Seq: {result_link_oriented['mapping_sequence']}")

# from src.utils import get_config, Backend, Network
# from src.nadqc import NADQC, MapperFactory, PartitionAssignerFactory
# from src.baselines import OEE

# for num_qpus in range(5, 11):
#     print(f"\n=== Number of QPUs: {num_qpus} ===")

#     global_config = get_config()
#     backend_config = {
#         'backend_name': 'ibm_torino_sampled_10q',
#         'date': datetime.datetime(2025, 11, 9)
#     }

#     backend = Backend(global_config, backend_config)
#     backend.print()

#     backend_config = [backend for _ in range(num_qpus)]

#     # 自定义网络配置
#     # network_config = {
#     #     'type': 'self_defined',
#     #     'network_coupling': {
#     #         (0, 1): 0.979,
#     #         (1, 2): 0.98,
#     #         (0, 2): 0.981
#     #     }
#     # }
#     network_config = {
#         'type': 'all_to_all',
#     }

#     net = Network(network_config, backend_config)

#     # 创建一个简单的量子电路
#     from qiskit import QuantumCircuit, transpile
#     from qiskit.circuit.library import QuantumVolume, QFT
#     qc = QuantumVolume(num_qpus * 10, seed=26).decompose()
#     # qc = QFT(num_qpus * 10).decompose()
#     qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
#     # print(qc)

#     # 分配
#     nadqc = NADQC(circ=qc, network=net)
#     nadqc.distribute()

#     # partitioner = PartitionAssignerFactory.create_assigner("global_max_match")
#     partitioner = PartitionAssignerFactory.create_assigner("max_match")
#     partition_candidates = nadqc.get_partition_candidates()
#     partition_plan = partitioner.assign_partitions(partition_candidates)["partition_plan"]
#     # print("Partition Plan:")
#     # pprint(partition_plan)

#     oee = OEE(circ=qc, network=net)
#     oee.distribute()

#     # mapper_names = ["simple", "link_oriented", "exact", "greedy"]
#     mapper_names = ["greedy"]
#     mappers = []

#     for name in mapper_names:
#         mappers.append(MapperFactory.create_mapper(name))

#     for mapper in mappers:
#         result = mapper.map_circuit(partition_plan, net)
#         print(f"\nMapper: {mapper.get_name()}")
#         print(f"Metrics (total_fidelity_loss): {result['metrics']['total_fidelity_loss']}")
#         # print(f"Mapping Seq: {result['mapping_sequence']}")
