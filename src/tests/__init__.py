import importlib
from pathlib import Path

# 获取当前目录（即 src/test/）
test_dir = Path(__file__).parent

# 遍历所有 test_*.py 文件（排除 __init__.py 和 __pycache__）
for file in test_dir.glob("test_*.py"):
    module_name = file.stem  # 例如 "test_network"
    module = importlib.import_module(f".{module_name}", package=__name__)

    # 遍历模块中的所有属性
    for attr_name in dir(module):
        if attr_name.startswith("test_"):
            # 将函数导入到当前命名空间
            attr = getattr(module, attr_name)
            if callable(attr):  # 确保是函数或可调用对象
                globals()[attr_name] = attr

# 可选：定义 __all__，支持 from test import *
__all__ = [name for name in globals() if name.startswith("test_")]