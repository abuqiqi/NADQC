from typing import Optional

from qiskit import QuantumCircuit

from ..compiler import MappingRecordList
from ..utils import Network
from .navi_hybrid import NAVIHybrid


class NAVIHybridTeledataDirect(NAVIHybrid):
    """
    消融路径：仅使用 teledata-only records，再交给 Direct Mapper 重算真实成本。
    """

    compiler_id = "navihybridtd"

    @property
    def name(self) -> str:
        return "NAVI Hybrid TD Direct"

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, object]] = None,
    ) -> MappingRecordList:
        print(f"Compiling with [{self.name}]...")
        shared_ctx, shared_prefix_time = self.build_shared_prefix_context(circuit, network, config)
        return self.compile_teledata_only_direct_from_shared_prefix(
            shared_ctx,
            config,
            shared_prefix_time=shared_prefix_time,
        )
