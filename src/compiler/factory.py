import os
import importlib
import inspect
import sys
from typing import Type

from .base import Compiler

class CompilerFactory:
    """自动发现编译器的工厂类"""
    # 注册表：key=compiler_id, value=编译器类
    _registry: dict[str, Type[Compiler]] = {}

    @classmethod
    def register_compilers(cls, compiler_paths: list[str]) -> list[str]:
        """
        创建编译器注册表
        :param compiler_paths: 编译器包的绝对路径
        """
        # for compiler_path in compiler_paths:
        #     # 验证目录是否存在
        #     if not os.path.exists(compiler_path):
        #         print(f"[WARNING] 编译器路径不存在: {compiler_path}, 跳过")
        #         continue

        #     # 获取 baselines 的父目录
        #     parent_dir = os.path.dirname(compiler_path)

        #     # 将编译器路径转换为模块路径并导入
        #     module_path = os.path.relpath(compiler_path, parent_dir).replace(os.sep, ".")
        #     print(f"[DEBUG] Importing compiler module: {module_path} from {compiler_path}")
        #     module = importlib.import_module(module_path)

        # # 工程根目录
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # # 程序执行的工作目录
        # current_script_dir = os.path.abspath(os.getcwd())

        # for rel_compiler_path in compiler_paths: # 相对于工程根目录
        #     abs_compiler_path = os.path.abspath(os.path.join(project_root, rel_compiler_path))
        #     print(f"[DEBUG] compiler_path: {abs_compiler_path}")
            
        #     if not os.path.exists(abs_compiler_path):
        #         print(f"[WARNING] 扫描目录不存在: {abs_compiler_path}, 跳过")
        #         continue

        #     # 将文件路径转换为Python模块路径（核心步骤）
        #     # 示例：/project_root/compilers/xxx → compilers.xxx
        #     rel_path = os.path.relpath(abs_compiler_path, current_script_dir)
        #     print(f"[DEBUG] relative path to cwd: {rel_path}")
        #     module_path = rel_path.replace(os.sep, ".")
        #     print(f"[DEBUG] module_path: {module_path}")

        for module_path in compiler_paths:
            # 导入模块
            module = importlib.import_module(module_path)
        
            # 遍历模块中暴露的所有成员
            for name, obj in inspect.getmembers(module):
                # 筛选条件：
                # 1. 是类
                # 2. 是Compiler的子类（且不是Compiler本身）
                # 3. 包含compiler_id类属性
                if (inspect.isclass(obj) and 
                    issubclass(obj, Compiler) and 
                    obj is not Compiler and 
                    hasattr(obj, "compiler_id") and 
                    isinstance(obj.compiler_id, str) and 
                    obj.compiler_id.strip()):
                    
                    # 注册到注册表（去重，后注册的覆盖先注册的）
                    compiler_id = obj.compiler_id.strip()
                    cls._registry[compiler_id] = obj
                    print(f"[INFO] 成功注册编译器: {compiler_id} -> {obj.__name__}")
        return list(cls._registry.keys())

    @classmethod
    def get_compiler(cls, compiler_id: str) -> Type[Compiler]:
        """
        根据ID获取编译器类
        :param compiler_id: 编译器唯一标识
        :return: 对应的编译器类
        :raises ValueError: 当compiler_id不存在时抛出异常
        """
        # 先清理ID的首尾空格，和注册时保持一致
        clean_id = compiler_id.strip()
        compiler = cls._registry.get(clean_id)
        
        if compiler is None:
            # 抛出异常，包含清晰的提示信息（列出所有可用ID）
            available_ids = ", ".join(cls._registry.keys()) or "无"
            raise ValueError(
                f"找不到ID为 '{clean_id}' 的编译器！"
                f"已注册的编译器ID：[{available_ids}]"
            )

        return compiler

    @classmethod
    def get_available_compiler_ids(cls) -> list[str]:
        """获取所有已注册的编译器ID"""
        return list(cls._registry.keys())
