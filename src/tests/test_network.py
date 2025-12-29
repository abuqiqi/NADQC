import datetime
from pprint import pprint

# 简单的测试函数，返回 (是否通过, 错误信息)
def test_network_initialization():
    """测试 Network 类初始化"""
    try:
        from nadqc import Backend, Network
        from utils import get_config

        config = get_config()
        backend_name = "ibm_torino"
        date = datetime.datetime(2025, 12, 9)

        # 加载噪声数据
        backend = Backend()
        backend.load_properties(config, backend_name, date)
        backend.print()

        # 测试配置
        network_config = {
            'type': 'mesh_grid',
            'size': (2, 3),
            'fidelity_range': (0.92, 0.97)
        }
        backend_config = [backend for _ in range(6)]

        # 创建网络
        net = Network(network_config, backend_config)

        # 检查基本属性
        assert net.num_backends == 6, f"Expected 6 backends, got {net.num_backends}"
        assert len(net.network_coupling) > 0, "No coupling edges found"
        
        # 可视化网络
        print(f"Hop weight: {net.hop_weight}")
        net.draw_network_graph()

        return True, "Network initialization passed"
    except Exception as e:
        return False, f"Network initialization failed: {str(e)}"

def test_optimal_path():
    """测试最优路径计算"""
    try:
        from nadqc.backends import Network
        
        # 自定义网络配置
        network_config = {
            'type': 'self_defined',
            'network_coupling': {
                (0, 1): 0.99,
                (1, 2): 0.98,
                (0, 2): 0.93
            }
        }
        backend_config = ["QPU_0", "QPU_1", "QPU_2"]
        
        net = Network(network_config, backend_config)

        # 可视化网络
        net.print_info()
        net.draw_network_graph("optimal_path_test")

        # 获取0到1的路径
        path = net.get_optimal_path(0, 2)
        pprint(f"Optimal path from 0 to 2: {path}")

        assert len(path) > 0, "No path found between 0 and 2"
        assert path[0] == 0 and path[-1] == 2, f"Path {path} does not start/end correctly"

        return True, "Optimal path test passed"
    except Exception as e:
        return False, f"Optimal path test failed: {str(e)}"