import sys
import time
import datetime
from pprint import pprint

from src.utils import get_args, get_config, select_circuit, write_compiler_results_to_csv, Backend, Network
from src.compiler import Compiler, CompilerFactory

def main(args):
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

    # 调用不同的compiler
    result_info = {}

    compiler_ids = CompilerFactory.register_compilers(global_config.get("compiler_modules"))
    print(f"Registered compiler IDs: {compiler_ids}")
    compilers: list[Compiler] = []
    for compiler_id in compiler_ids:
        compilers.append(CompilerFactory.get_compiler(compiler_id)())

    for compiler in compilers:
        print(f"Compiler: [{compiler.name}]")
        result = compiler.compile(trans_circ, network, {"circuit_name": f"{args.circuit_name}{args.qubit_count}"})
        pprint(result.total_costs)
        result_info[compiler.name] = result.total_costs.to_dict()

    # Write results to CSV
    write_compiler_results_to_csv(task_info, result_info, f"{global_config.get('output_folder')}compiler_results.csv")

    return

if __name__ == "__main__":
    # 获取全局配置
    args = get_args()
    filename = f"{args.circuit_name}_{args.qubit_count}_{args.core_count}"
    original_stdout = sys.stdout
    with open(f'outputs/{filename}.txt', 'w') as f:
        sys.stdout = f
        start_time = time.time()
        main(args)
        end_time = time.time()
        print(f"[Total Runtime] {end_time - start_time} seconds\n\n")
        sys.stdout = original_stdout
