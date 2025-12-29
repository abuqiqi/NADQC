"""
主程序：运行所有测试
"""

import sys
import time
from pathlib import Path

import src.tests as my_tests

# 注册 src 为模块根目录
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_test(test_func, test_name):
    """运行单个测试并打印结果"""
    print(f"=== Running {test_name}... ===")
    start_time = time.time()
    
    try:
        passed, message = test_func()
        duration = time.time() - start_time
        
        if passed:
            print(f"\033[92mPASS\033[0m ({duration:.4f}s) - {message}")
            return True
        else:
            print(f"\033[91mFAIL\033[0m ({duration:.4f}s) - {message}")
            return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"\033[91mERROR\033[0m ({duration:.4f}s) - Unexpected error: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("Running all tests...")
    print("=" * 50)
    
    # 定义要运行的测试
    tests = [
        (my_tests.test_network_initialization, "Network Initialization"),
        (my_tests.test_optimal_path, "Optimal Path Calculation"),
    ]
    
    # 运行所有测试
    results = []
    for test_func, test_name in tests:
        result = run_test(test_func, test_name)
        results.append((test_name, result))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    for test_name, result in results:
        status = "\033[92mPASSED\033[0m" if result else "\033[91mFAILED\033[0m"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    
    if passed == total:
        print("\n\033[92m✓ All tests passed!\033[0m")
        sys.exit(0)
    else:
        print(f"\n\033[91m✗ {total - passed} tests failed!\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()