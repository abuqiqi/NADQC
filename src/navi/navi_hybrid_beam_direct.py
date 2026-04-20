from typing import Any, Optional

from qiskit import QuantumCircuit

from ..compiler import MappingRecordList
from ..utils import Network
from .navi_hybrid import NAVIHybrid


class NAVIHybridBeamDirect(NAVIHybrid):
    """
    旧 hybrid beam records 构造路径，但最终固定使用 Direct Mapper。
    """

    compiler_id = "navihybriddirectmapper"

    @property
    def name(self) -> str:
        return "NAVI Hybrid+direct mapper"

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> MappingRecordList:
        return self._compile_impl(
            circuit,
            network,
            config,
            use_direct_noise_aware=False,
            override_mapper_id="direct",
        )
