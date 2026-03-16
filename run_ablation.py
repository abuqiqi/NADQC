import sys
import time
import datetime
from pprint import pprint
import copy
import pandas as pd
from typing import Any, Optional
from tqdm import tqdm # 建议安装 tqdm 以显示进度条

from qiskit import QuantumCircuit

from src.utils import get_args, get_config, select_circuit, write_compiler_results_to_csv, Backend, Network
from src.nadqc import NAVI
from src.compiler import CompilerUtils, MappingRecordList


class NAVIAblationStudy:
    """
    专门用于运行 NAVI 编译器消融实验的管理器。
    负责配置管理、上下文隔离（Deep Copy）、结果收集与分析。
    """

    def __init__(self, compiler: NAVI):
        self.compiler = compiler
        self.results: dict[str, MappingRecordList] = {}

    def run_full_combination_study(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: dict[str, Any],
        partitioners: list[str],
        partition_assigners: list[str],
        telegate_partitioners: list[str],
        mappers: list[str]
    ) -> dict[str, Any]:
        """
        运行全排列组合实验 (A x B x C x D)
        """
        print(f"Starting Full Combination Study: {len(partitioners)} x {len(partition_assigners)} x {len(telegate_partitioners)} x {len(mappers)}")
        
        # --- Phase 0: Preprocess 保持不变 ---
        base_ctx = self.compiler.preprocess(circuit, network, config)
        
        # 定义上下文池：(ctx, trace_dict)
        # trace_dict 用于记录当前分支用了哪些配置
        ctx_pool = [(base_ctx, {"status": "preprocessed"})]

        # --- Phase 1: Partitioner ---
        # 注意：这一步通常需要重建 P_table，所以必须传参
        new_pool = []
        for p_type in partitioners:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                ctx_new = self.compiler.generate_partition_candidates(ctx_copy, partitioner_type=p_type)
                new_pool.append((ctx_new, {**trace, "partitioner": p_type}))
        ctx_pool = new_pool

        # --- Phase 2: Partition Assigner ---
        new_pool = []
        for pa_type in partition_assigners:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                ctx_new = self.compiler.generate_partition_plan(ctx_copy, pa_type)
                new_pool.append((ctx_new, {**trace, "partition_assigner": pa_type}))
        ctx_pool = new_pool

        # --- Phase 3: Telegate ---
        new_pool = []
        for tp_type in telegate_partitioners:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                ctx_new = self.compiler.optimize_with_telegate(ctx_copy, tp_type)
                new_pool.append((ctx_new, {**trace, "telegate_partitioner": tp_type}))
                # 输出每个ctx下的hybrid_records的total_costs，便于调试
                # if hasattr(ctx_new, 'hybrid_records') and ctx_new.hybrid_records is not None:
                #     print(f"Trace: {trace}, Hybrid Records:")
                #     for record in ctx_new.hybrid_records.records:
                #         print(record.costs)
                # else:
                #     print(f"Trace: {trace}, No hybrid records found.")
        ctx_pool = new_pool

        # --- Phase 4: Mapper ---
        new_pool = []
        for m_type in mappers:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                ctx_new = self.compiler.optimize_mapping(ctx_copy, m_type)
                new_pool.append((ctx_new, {**trace, "mapper": m_type}))
        ctx_pool = new_pool

        # --- Phase 5: 统一收集结果 ---
        results = {}

        # debug: 展开每个ctx里面的total_costs
        assert len(ctx_pool) == len(partitioners) * len(partition_assigners) * len(telegate_partitioners) * len(mappers), "Context pool size mismatch with expected combinations"
        for ctx, trace in ctx_pool:
            # 输出ctx的类型
            print(f"Context Type: {type(ctx)}")
            # 输出ctx所有的属性
            print(f"Context Attributes: {ctx.__dict__.keys()}")
            print(f"Final records type in context: {type(ctx.final_records)}")
            # if hasattr(ctx, 'final_records') and ctx.final_records is not None:
            #     print(f"Trace: {trace}, Total Costs: {ctx.final_records}")
            # else:
            #     print(f"Trace: {trace}, No final records found.")
        
        for ctx, trace in ctx_pool:
            combo_name = "_".join([
                trace.get("partitioner", "default"),
                trace.get("partition_assigner", "default"),
                trace.get("telegate_partitioner", "default"),
                trace.get("mapper", "default")
            ])
            
            # 提取结果
            if hasattr(ctx, 'final_records') and ctx.final_records is not None:
                results[combo_name] = ctx.hybrid_records.total_costs # TODO

        return results

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

    partitioner = ["recursive_dp"]
    partition_assigners = ["direct", "global_max_match"]
    telegate_partitioners = ["direct"]
    mappers = ["direct"]

    ablation = NAVIAblationStudy(NAVI())

    result_info = ablation.run_full_combination_study(
        circuit = trans_circ,
        network = network,
        config = {},
        partitioners = partitioner,
        partition_assigners = partition_assigners,
        telegate_partitioners = telegate_partitioners,
        mappers = mappers
    )

    print("task_info:")
    pprint(task_info)

    print("Final Ablation Results:")
    pprint(result_info)

    # Write results to CSV
    write_compiler_results_to_csv(task_info, result_info, f"{global_config.get('output_folder')}navi_ablation_results.csv")

    return

if __name__ == "__main__":
    # 获取全局配置
    args = get_args()
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}"
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'a') as f:
        sys.stdout = f
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
