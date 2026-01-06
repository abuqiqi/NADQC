import datetime
from pprint import pprint

def test_backend_sample():
    """测试 Backend 类初始化"""
    try:
        from utils import get_config
        from nadqc import Backend

        global_config = get_config()
        backend_config = {
            'backend_name': 'ibm_torino_sampled_10q',
            'date': datetime.datetime(2025, 11, 9)
        }

        backend = Backend(global_config, backend_config)
        backend.print()

        return True, "Backend initialization passed"
    except Exception as e:
        return False, f"Backend initialization failed: {str(e)}"
