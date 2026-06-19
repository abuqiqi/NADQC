import datetime

from src.utils import Backend, Network, get_config


class _MockBackend:
    num_qubits = 10
    name = "mock"
    basis_gates = ["x", "cx"]
    two_qubit_gates = {"cx"}
    gate_dict = {}


def test_mesh_grid_infers_square_size_for_four_qpus():
    net = Network(
        {"type": "mesh_grid", "fidelity_range": (0.92, 0.97)},
        [_MockBackend() for _ in range(4)],
        build_all_to_all_copy=False,
    )

    assert net.size == (2, 2)
    assert set(net.network_coupling) == {(0, 1), (2, 3), (0, 2), (1, 3)}


def test_network_comm_slot_reserve_defaults_to_two_and_allows_zero_override():
    default_net = Network(
        {"type": "all_to_all", "fidelity_range": (0.92, 0.97)},
        [_MockBackend(), _MockBackend()],
        build_all_to_all_copy=False,
    )
    zero_reserve_net = Network(
        {
            "type": "all_to_all",
            "fidelity_range": (0.92, 0.97),
            "comm_slot_reserve": 0,
        },
        [_MockBackend(), _MockBackend()],
        build_all_to_all_copy=False,
    )

    assert Network.DEFAULT_COMM_SLOT_RESERVE == 2
    assert default_net.comm_slot_reserve == 2
    assert zero_reserve_net.comm_slot_reserve == 0


def test_network_comm_slot_reserve_resolver_preserves_legacy_priority():
    assert Network.resolve_comm_slot_reserve({}, {}) == 2
    assert Network.resolve_comm_slot_reserve(
        {},
        {"comm_slot_reserve": 1},
    ) == 1
    assert Network.resolve_comm_slot_reserve(
        {"comm_slot_reserve": 3},
        {"comm_slot_reserve": 1},
    ) == 3
    assert Network.resolve_comm_slot_reserve(
        {"comm_slot_reserve": 0},
        {"comm_slot_reserve": 1},
    ) == 0


def test_network_fidelity_range_defaults_to_config_value_and_allows_override():
    default_net = Network(
        {"type": "all_to_all"},
        [_MockBackend(), _MockBackend()],
        build_all_to_all_copy=False,
    )
    override_net = Network(
        {"type": "all_to_all", "fidelity_range": (0.92, 0.97)},
        [_MockBackend(), _MockBackend()],
        build_all_to_all_copy=False,
    )

    assert Network.DEFAULT_FIDELITY_RANGE == (0.90, 0.93)
    assert tuple(default_net.fidelity_range) == (0.90, 0.93)
    assert override_net.fidelity_range == (0.92, 0.97)


def test_mesh_grid_infers_rectangular_size_for_eight_qpus():
    net = Network(
        {"type": "mesh_grid", "fidelity_range": (0.92, 0.97)},
        [_MockBackend() for _ in range(8)],
        build_all_to_all_copy=False,
    )

    assert net.size == (2, 4)
    assert set(net.network_coupling) == {
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    }


def test_network_initialization():
    """测试 Network 类初始化"""
    config = get_config()
    backend_config = {
        'backend_name': 'ibm_torino_sampled_10q',
        'date': datetime.datetime(2025, 11, 9),
    }
    backend = Backend(config, backend_config)
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

# def test_optimal_path():
#     """测试最优路径计算"""
#     try:
#         from utils import Network
        
#         # 自定义网络配置
#         network_config = {
#             'type': 'self_defined',
#             'network_coupling': {
#                 (0, 1): 0.99,
#                 (1, 2): 0.98,
#                 (0, 2): 0.93
#             }
#         }
#         backend_config = ["QPU_0", "QPU_1", "QPU_2"]
        
#         net = Network(network_config, backend_config)

#         # 获取0到1的路径
#         path = net.get_optimal_path(0, 2)
#         pprint(f"Optimal path from 0 to 2: {path}")

#         # 可视化网络
#         net.print_info()
#         net.draw_network_graph("test_optimal_path", highlight_path=path)

#         assert len(path) > 0, "No path found between 0 and 2"
#         assert path[0] == 0 and path[-1] == 2, f"Path {path} does not start/end correctly"

#         return True, "Optimal path test passed"
#     except Exception as e:
#         return False, f"Optimal path test failed: {str(e)}"
