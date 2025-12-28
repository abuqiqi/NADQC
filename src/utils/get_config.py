# src/utils/get_config.py
import json
from pathlib import Path

_cached_config = None


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