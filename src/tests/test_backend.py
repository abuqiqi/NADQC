import datetime

from src.utils import Backend, get_config

def test_backend_sample():
    """测试 Backend 类初始化"""
    global_config = get_config()
    backend_config = {
        'backend_name': 'ibm_torino_sampled_10q',
        'date': datetime.datetime(2025, 11, 9)
    }

    backend = Backend(global_config, backend_config)
    backend.print()
