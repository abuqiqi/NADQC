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
                start_time = time.time()
                ctx_new = self.compiler.generate_partition_candidates(ctx_copy, partitioner_type=p_type)
                new_pool.append((ctx_new, {**trace, 
                                           "partitioner": p_type, 
                                           "partition_time (sec)": time.time() - start_time}))
        ctx_pool = new_pool

        # --- Phase 2: Partition Assigner ---
        new_pool = []
        for pa_type in partition_assigners:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                start_time = time.time()
                ctx_new = self.compiler.generate_partition_plan(ctx_copy, pa_type)
                new_pool.append((ctx_new, {**trace, 
                                           "partition_assigner": pa_type, 
                                           "assign_time (sec)": time.time() - start_time}))
        ctx_pool = new_pool

        # --- Phase 3: Telegate ---
        new_pool = []
        for tp_type in telegate_partitioners:
            for ctx, trace in ctx_pool:
                ctx_copy = copy.deepcopy(ctx)
                start_time = time.time()
                ctx_new = self.compiler.optimize_with_telegate(ctx_copy, tp_type)
                new_pool.append((ctx_new, {**trace, 
                                           "telegate_partitioner": tp_type, 
                                           "telegate_time (sec)": time.time() - start_time}))
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
                start_time = time.time()
                ctx_new = self.compiler.optimize_mapping(ctx_copy, m_type)
                new_pool.append((ctx_new, {**trace, 
                                           "mapper": m_type, 
                                           "mapper_time (sec)": time.time() - start_time}))
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
                results[combo_name] = ctx.final_records.total_costs.to_dict() # TODO

        return results

def ablation(args):
    global_config = get_config(args.global_config_path)

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

    # 获取basis gate set
    basis_gates = network.basis_gates
    two_qubit_gates = network.two_qubit_gates

    # 调用量子线路
    circ, trans_circ, task_info = select_circuit(args.circuit_name,
                                                 args.qubit_count,
                                                 args.core_count,
                                                 args.core_capacity,
                                                #  args.gate_set
                                                 basis_gates,
                                                 two_qubit_gates
                                                 )

    # partitioner = ["recursive_dp"]
    # partition_assigners = ["direct", "max_match", "global_max_match"]
    # telegate_partitioners = ["direct"]
    # mappers = ["direct", "greedy", "dp"]
    partitioner = ["recursive_dp"]
    partition_assigners = ["direct", "max_match", "global_max_match"]
    telegate_partitioners = ["direct"]
    mappers = ["direct", "greedy", "dp"]

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
    with open(f'outputs/{filename}.txt', 'w') as f:
        sys.stdout = f
        start_time = time.time()
        ablation(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout
