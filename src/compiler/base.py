from abc import ABC, abstractmethod
from typing import Any, Optional

from .compiler_utils import MappingRecordList

class Compiler(ABC):
    """
    编译器接口
    """
    compiler_id: str

    def __init_subclass__(cls, **kwargs):
        """子类初始化时校验：必须设置 compiler_id"""
        super().__init_subclass__(** kwargs)
        if cls.compiler_id is None:
            raise NotImplementedError(f"子类 {cls.__name__} 必须定义 compiler_id 属性")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取编译器名称
        """
        pass

    @abstractmethod
    def compile(self, circuit: Any, 
                network: Any, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        编译电路
        :param circuit: 电路对象
        :param network: 网络对象
        :return: 编译后的电路对象
        """
        pass

