import argparse
from typing import List
import pandas as pd
from typing import Any
import numpy as np
import os

def parse_int_list(input_str: str) -> List[int]:
    """Convert a comma-separated string to a list of integers (e.g., '4,6,8' -> [4,6,8])"""
    try:
        return [int(item.strip()) for item in input_str.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list format: '{input_str}'. Please use comma-separated integers, e.g., '4,6,8'")

def get_args():
    parser = argparse.ArgumentParser(description='Distributed quantum circuit mapping parameter configuration')

    # Required arguments
    parser.add_argument('--core_count', '-core', type=int, required=True,
                        help='Number of QPUs (integer)')
    parser.add_argument('--core_capacity', '-cap', type=parse_int_list, required=True,
                        help='Capacity of each QPU (comma-separated integers, e.g., "4" or "4,6,8")')
    parser.add_argument('--circuit_name', '-cname', type=str, required=True,
                        help='Name of the quantum circuit (string)')
    parser.add_argument('--qubit_count', '-nq', type=int, required=True,
                        help='Number of qubits in the quantum circuit (integer)')
    parser.add_argument('--gate_set', '-gset', type=str, required=False, default='cu1,u3',
                        help='Comma-separated list of basis gates (string)')
    parser.add_argument('--network', '-net', type=str, required=False, default='fc',
                        help='Name of the network (string)')

    args = parser.parse_args()

    # Unify the capacity list for each QPU
    if len(args.core_capacity) == 1:
        core_capacities = args.core_capacity * args.core_count
    else:
        core_capacities = args.core_capacity

    # Validate the array length
    if len(core_capacities) != 1 and len(core_capacities) != args.core_count:
        raise ValueError(
            f"The QPU capacity must be a single integer (for all QPUs) or {args.core_count} comma-separated values (specified individually for each QPU). "
            f"Current input: {args.core_capacity}"
        )
    args.core_capacities = core_capacities

    # Validate the basis gate set
    if not args.gate_set:
        raise ValueError("The basis gate set must not be empty.")
    args.gate_set = args.gate_set.split(",")

    print("[INFO] Configuration parameters:")
    print(f"[INFO] Number of QPUs: {args.core_count}")
    print(f"[INFO] Capacity of each QPU: {args.core_capacities}")
    print(f"[INFO] Name of the quantum circuit: {args.circuit_name}")
    print(f"[INFO] Number of qubits in the quantum circuit: {args.qubit_count}")
    print(f"[INFO] Basis gate set: {args.gate_set}")
    print(f"[INFO] Name of the network: {args.network}")
    return args

def write_compiler_results_to_csv(
    task_info: dict[str, Any],
    result_info: dict[str, dict[str, Any]],
    output_path: str
):
    """
    通用化编译器结果写入CSV函数
    动态读取task_info的所有字段作为基础列，自动识别result_info的指标和编译器名称
    
    参数:
        task_info: 任务信息字典，包含#Qubits、#Depths、#Gates、#2Q Gates、#QPUs等字段
        result_info: 结果信息字典，格式 {compiler_name: 指标字典}
        output_path: CSV输出路径
        circuit_name: 电路名称（可选，默认Unknown_Circuit）
    """
    # -------------------------- 1. 预处理和提取核心信息 --------------------------
    # 1.1 提取所有编译器名称
    compiler_names = list(result_info.keys())
    if not compiler_names:
        print("警告：result_info为空，无数据可写入")
        return
    
    # 1.2 提取所有指标名称（取第一个编译器的指标作为基准，确保所有编译器指标一致）
    first_compiler = compiler_names[0]
    metrics = list(result_info[first_compiler].keys())
    
    # 1.3 整理基础列（circuit_name + task_info的所有键）
    base_columns = list(task_info.keys()) + ["Metrics"]
    
    # -------------------------- 2. 初始化数据字典 --------------------------
    # 完整表头 = 基础列 + 所有编译器名称
    full_headers = base_columns + compiler_names
    data = {header: [] for header in full_headers}
    
    # -------------------------- 3. 填充数据 --------------------------
    for metric in metrics:
        # 3.1 填充task_info的所有字段（保持和task_info一致的顺序）
        for task_key in task_info.keys():
            data[task_key].append(task_info[task_key])
        
        # 3.2 填充当前指标名称
        data["Metrics"].append(metric)
        
        # 3.3 填充每个编译器的对应指标值
        for compiler in compiler_names:
            # 获取当前编译器的当前指标值
            metric_value = result_info[compiler].get(metric, np.nan)
            
            # 统一数据类型（转换numpy类型为原生Python类型，避免CSV显示异常）
            # if isinstance(metric_value, (np.float64, np.float32, np.float16)):
            #     metric_value = float(metric_value)
            # elif isinstance(metric_value, (np.int64, np.int32, np.int16, np.int8)):
            #     metric_value = int(metric_value)
            # elif isinstance(metric_value, np.bool_):
            #     metric_value = bool(metric_value)
            if isinstance(metric_value, np.floating):  # 匹配所有numpy浮点类型（float16/32/64）
                metric_value = float(metric_value)
            elif isinstance(metric_value, np.integer):  # 匹配所有numpy整数类型（int8/16/32/64）
                metric_value = int(metric_value)
            elif isinstance(metric_value, np.bool_):    # numpy布尔类型
                metric_value = bool(metric_value)
            
            data[compiler].append(metric_value)
    
    # -------------------------- 4. 写入CSV文件 --------------------------
    # 自动创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 构建DataFrame
    df = pd.DataFrame(data)
    
    # 判断文件是否存在，决定是否写入表头
    file_exists = os.path.exists(output_path)
    
    # 写入CSV（追加模式，编码为utf-8确保中文/特殊字符正常）
    df.to_csv(
        output_path,
        mode = "a",
        header = True, # not file_exists,
        index = False,
        encoding = "utf-8"
    )
    
    print(f"✅ 数据已成功写入: {output_path}")
    print(f"📊 本次写入数据预览:\n{df}")
    return df

def output_results(cname, circ, qpus, distributors):
    """
    将数据写入.csv文件
    """
    headers = ["Circuit", "#Qubits", "#Depths", "#Gates", "#Modules", "Metrics"]
    metrics = ["Comm Costs", "#RGate", "#RSWAP", "Exec Time"]
    for dis in distributors:
        headers.append(dis.name)
    data = {}
    for head in headers:
        data[head] = []
    gate_counts = circ.count_ops()
    total_gates = sum(gate_counts.values())

    for m in metrics:
        data["Circuit"].append(cname)
        data["#Qubits"].append(circ.num_qubits)
        data["#Depths"].append(circ.depth())
        data["#Gates"].append(total_gates)
        data["#Modules"].append(len(qpus))
        data["Metrics"].append(m)

    # 对每个distributor，写入四行
    for distributor in distributors:
        data[distributor.name].append(distributor.num_comms)
        data[distributor.name].append(distributor.num_gates)
        data[distributor.name].append(distributor.num_swaps)
        data[distributor.name].append(distributor.exec_time)

    print(data)

    filename = f"./outputs/data.csv"
    df = pd.DataFrame(data)
    df.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)
    return

if __name__ == '__main__':
    get_args()
