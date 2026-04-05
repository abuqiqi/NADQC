import argparse
from typing import List
import pandas as pd
from typing import Any
import numpy as np
import os
import json
from pathlib import Path

_cached_config = None

def parse_int_list(input_str: str) -> List[int]:
    """Convert a comma-separated string to a list of integers (e.g., '4,6,8' -> [4,6,8])"""
    try:
        return [int(item.strip()) for item in input_str.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list format: '{input_str}'. Please use comma-separated integers, e.g., '4,6,8'")

def parse_str_list(input_str: str) -> List[str]:
    """Convert a comma-separated string to a list of strings (e.g., 'cu1,u3' -> ['cu1', 'u3'])"""
    return [item.strip() for item in input_str.split(",")]

def get_args():
    parser = argparse.ArgumentParser(description='Distributed quantum circuit mapping parameter configuration')

    # Required arguments
    # Global Information
    parser.add_argument('--global_config_path', '-gconf', type=str, required=False, default='config.json',
                        help='Path to the global configuration JSON file (string)')

    # Circuit Information
    parser.add_argument('--circuit_name', '-cname', type=str, required=True,
                        help='Name of the quantum circuit (string)')
    parser.add_argument('--qubit_count', '-nq', type=int, required=True,
                        help='Number of qubits in the quantum circuit (integer)')

    # Network Information
    parser.add_argument('--core_count', '-core', type=int, required=True,
                        help='Number of QPUs (integer)')
    parser.add_argument('--core_capacity', '-cap', type=parse_int_list, required=True,
                        help='Capacity of each QPU (comma-separated integers, e.g., "4" or "4,6,8")')
    parser.add_argument('--backend_name', '-bname', type=parse_str_list, required=False, default=['ibm_torino'],
                        help='Name of the backend (comma-separated strings, e.g., "ibm_torino" or "ibm_torino,ibm_osaka")')
    parser.add_argument('--date', '-date', type=parse_str_list, required=False, default=['2026-03-01'],
                        help='Date of the backend properties (comma-separated strings in YYYY-MM-DD format, e.g., "2025-11-09" or "2025-11-09,2025-11-10")')
    
    parser.add_argument('--network', '-net', type=str, required=False, default='all_to_all',
                        help='Name of the network (string)')
    parser.add_argument('--gate_set', '-gset', type=str, required=False, default='cu1,u3',
                        help='Comma-separated list of basis gates (string)')

    args = parser.parse_args()

    # Unify the capacity list for each QPU
    if len(args.core_capacity) == 1:
        args.core_capacity = args.core_capacity * args.core_count

    # Unify the backend name list
    if len(args.backend_name) == 1:
        args.backend_name = args.backend_name * args.core_count

    # Unify the backend information date list
    if len(args.date) == 1:
        args.date = args.date * args.core_count

    assert len(args.core_capacity) == args.core_count, \
        f"The number of QPU capacities {args.core_capacity} must match the number of QPUs {args.core_count}."
    assert len(args.backend_name) == args.core_count, \
        f"The number of backend names {args.backend_name} must match the number of QPUs {args.core_count}."
    assert len(args.date) == args.core_count, \
        f"The number of backend property dates {args.date} must match the number of QPUs {args.core_count}."

    # Validate the basis gate set
    if not args.gate_set:
        raise ValueError("The basis gate set must not be empty.")
    args.gate_set = args.gate_set.split(",")

    print("[INFO] Configuration parameters:")
    print(f"[INFO] Global config path: {args.global_config_path}")

    print(f"[INFO] Name of the quantum circuit: {args.circuit_name}")
    print(f"[INFO] Number of qubits in the quantum circuit: {args.qubit_count}")

    print(f"[INFO] Number of QPUs: {args.core_count}")
    print(f"[INFO] Capacity of each QPU: {args.core_capacity}")
    print(f"[INFO] Backend names: {args.backend_name}")
    print(f"[INFO] Backend properties dates: {args.date}")

    print(f"[INFO] Name of the network: {args.network}")
    # print(f"[INFO] Basis gate set: {args.gate_set}")
    return args


def get_config(config_filename: str = "config.json"):
    """
    从项目根目录加载 JSON 配置文件。
    
    Args:
        config_filename (str): 配置文件名，默认为 'config.json'
    
    Returns:
        dict: 解析后的配置字典
    
    Raises:
        FileNotFoundError: 如果配置文件不存在
        json.JSONDecodeError: 如果 JSON 格式无效
    """
    global _cached_config

    if _cached_config is not None:
        return _cached_config.copy()

    # 定位项目根目录：
    # get_config.py → src/utils/ → 上两级 = 项目根
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / config_filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {config_path}\n"
            f"请确保在项目根目录 ({project_root}) 中存在 '{config_filename}' 文件。\n"
            f"你可以复制 'config.example.json' 并重命名为 '{config_filename}'。"
        )

    with open(config_path, encoding="utf-8") as f:
        _cached_config = json.load(f)

    return _cached_config.copy()


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
    print(result_info[first_compiler])
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


if __name__ == '__main__':
    get_args()
