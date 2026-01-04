import datetime
from pprint import pprint

def test_backend_sample():
    """测试 Backend 类初始化"""
    try:
        from utils import get_config
        from nadqc import Backend

        config = get_config()

        backend = Backend()
        date = datetime.datetime(2025, 11, 9)
        backend.load_properties(config, "ibm_torino", date)

        backend.sample_and_export(10, config["output_folder"])

        return True, "Backend initialization passed"
    except Exception as e:
        return False, f"Backend initialization failed: {str(e)}"