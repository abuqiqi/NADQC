from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Optional

from .compiler_utils import CompilerUtils, MappingRecordList

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

    def _evaluate_raw_mapping_records_with_timing(
        self,
        mapping_record_list: MappingRecordList,
        network: Any,
        policy_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        start_time: Optional[float] = None,
        pre_eval_time: Optional[float] = None,
    ) -> tuple[MappingRecordList, float, float]:
        if pre_eval_time is None:
            if start_time is None:
                raise ValueError("start_time or pre_eval_time is required")
            pre_eval_time = time.time() - start_time

        print(
            f"[TIMING] [{self.name}] pre-evaluate raw mapping record time: "
            f"{pre_eval_time:.6f} sec"
        )
        eval_start_time = time.time()
        evaluated = CompilerUtils.evaluate_raw_mapping_records(
            mapping_record_list,
            network,
            policy_name=policy_name,
            config=config,
        )
        eval_time = time.time() - eval_start_time
        print(
            f"[TIMING] [{self.name}] evaluate raw mapping record time: "
            f"{eval_time:.6f} sec"
        )
        return evaluated, pre_eval_time, eval_time
