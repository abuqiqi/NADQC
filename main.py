# """
# 主程序：运行所有测试
# """

# import sys
# import time
# from pathlib import Path

# import src.tests as my_tests

# # 注册 src 为模块根目录
# sys.path.insert(0, str(Path(__file__).parent / "src"))

# def run_test(test_func, test_name):
#     """运行单个测试并打印结果"""
#     print(f"=== Running {test_name}... ===")
#     start_time = time.time()
    
#     try:
#         passed, message = test_func()
#         duration = time.time() - start_time
        
#         if passed:
#             print(f"\033[92mPASS\033[0m ({duration:.4f}s) - {message}")
#             return True
#         else:
#             print(f"\033[91mFAIL\033[0m ({duration:.4f}s) - {message}")
#             return False
#     except Exception as e:
#         duration = time.time() - start_time
#         print(f"\033[91mERROR\033[0m ({duration:.4f}s) - Unexpected error: {str(e)}")
#         return False

# def main():
#     """主函数"""
#     print("=" * 50)
#     print("Running all tests...")
#     print("=" * 50)
    
#     # 定义要运行的测试
#     tests = [
#         # (my_tests.test_network_initialization, "Network Initialization"),
#         # (my_tests.test_optimal_path, "Optimal Path Calculation"),
#         (my_tests.test_partition, "Circuit Partitioning")
#     ]
    
#     # 运行所有测试
#     results = []
#     for test_func, test_name in tests:
#         result = run_test(test_func, test_name)
#         results.append((test_name, result))
    
#     # 汇总结果
#     print("\n" + "=" * 50)
#     print("TEST SUMMARY")
#     print("=" * 50)
    
#     total = len(results)
#     passed = sum(1 for _, r in results if r)
    
#     for test_name, result in results:
#         status = "\033[92mPASSED\033[0m" if result else "\033[91mFAILED\033[0m"
#         print(f"{test_name}: {status}")
    
#     print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    
#     if passed == total:
#         print("\n\033[92m✓ All tests passed!\033[0m")
#         sys.exit(0)
#     else:
#         print(f"\n\033[91m✗ {total - passed} tests failed!\033[0m")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

from pprint import pprint
from src.nadqc import Backend, Network, NADQC

# 自定义网络配置
network_config = {
    'type': 'self_defined',
    'network_coupling': {
        (0, 1): 0.99,
        (1, 2): 0.98,
        (0, 2): 0.97
    }
}

backend = Backend(config={"num_qubits": 10})
backend_config = [backend for _ in range(3)]

net = Network(network_config, backend_config)

# 创建一个简单的量子电路
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume, QFT
qc = QuantumVolume(30, seed=26).decompose()
# qc = QFT(15).decompose()
qc = transpile(qc, basis_gates=["cu1", "u3"], optimization_level=0)
# print(qc)

# 分配
nadqc = NADQC(circ=qc, network=net)
nadqc.distribute()
partition_plan = nadqc.get_partition_plan()
total_comm_cost, mapping_sequence = nadqc.calculate_comm_cost_dynamic(partition_plan)
print("Partition Plan:")
pprint(partition_plan)
print(f"Total Communication Cost: {total_comm_cost:.4f}")
print("Mapping Sequence:")
pprint(mapping_sequence)