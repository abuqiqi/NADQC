import dataclasses
from dataclasses import dataclass
import networkx as nx
import json
import numpy as np
import copy
from collections import defaultdict
from typing import Any, Optional
import math
import os
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate

from ..utils import Network


@dataclass
class ExecCosts:
    remote_hops: int = 0
    remote_swaps: int = 0
    cat_ents: int = 0
    epairs: int = 0
    remote_fidelity_loss: float = 0.0
    remote_fidelity_log_sum: float = 0.0
    remote_fidelity: float = 1.0
    local_fidelity_loss: float = 0.0
    local_fidelity_log_sum: float = 0.0
    local_fidelity: float = 1.0
    execution_time: float = 0.0

    local_gate_num: int = 0    # 本地量子门总个数
    flush_calls: int = 0       # _flush_local_subcircuits 调用次数（含空flush）
    nonempty_flushes: int = 0  # 实际触发了至少一个非空子线路结算的flush次数
    local_transpile_calls: int = 0  # 本地子线路 transpile 总次数（按QPU逐个累计）
    comm_block_events: int = 0  # CommOp 事件数量
    comm_block_local_gate_num: int = 0  # CommOp gate_list 中的本地执行门数（未含routing）
    comm_block_local_fidelity_loss: float = 0.0  # CommOp gate_list 对应本地执行损失
    comm_block_local_fidelity_log_sum: float = 0.0  # CommOp gate_list 对应本地执行log保真度和
    comm_block_remote_fidelity_loss: float = 0.0  # CommOp 对应远程链路损失
    comm_block_remote_fidelity_log_sum: float = 0.0  # CommOp 对应远程链路log保真度和
    telegate_exec_events: int = 0  # 普通跨QPU门按synthetic CommOp(cat)执行的事件数
    telegate_exec_local_gate_num: int = 0  # telegate目标端本地执行门数（未含routing）
    telegate_exec_local_fidelity_loss: float = 0.0  # telegate目标端本地执行损失
    telegate_exec_local_fidelity_log_sum: float = 0.0  # telegate目标端本地执行log保真度和
    telegate_exec_remote_fidelity_loss: float = 0.0  # telegate对应远程链路损失
    telegate_exec_remote_fidelity_log_sum: float = 0.0  # telegate对应远程链路log保真度和

    @property
    def num_comms(self) -> int:
        return self.remote_hops + self.remote_swaps

    @property
    def total_fidelity_loss(self) -> float:
        return self.remote_fidelity_loss + self.local_fidelity_loss

    @property
    def total_fidelity_log_sum(self) -> float:
        return self.local_fidelity_log_sum + self.remote_fidelity_log_sum

    @property
    def total_fidelity(self) -> float:
        return self.remote_fidelity * self.local_fidelity

    @property
    def remote_geometric_mean_fidelity(self) -> float:
        if self.num_comms == 0:
            return 1.0
        rgeo_mean_fid = np.exp(self.remote_fidelity_log_sum / self.num_comms)
        return rgeo_mean_fid

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"ExecCosts 没有属性 {key}")

    def __str__(self) -> str:
        return (
            f"ExecCosts("
            f"comms={self.num_comms}, "
            f"cat_ents={self.cat_ents}, "
            f"epairs={self.epairs}, "
            f"rgeo_mean_fid={self.remote_geometric_mean_fidelity}, "
            f"fidelity={self.total_fidelity:.4f}, "
            f"loss={self.total_fidelity_loss:.4f}, "
            f"rhops={self.remote_hops}, "
            f"rswaps={self.remote_swaps}, "
            f"remote_fidelity_loss={self.remote_fidelity_loss}, "
            f"remote_fidelity_log_sum={self.remote_fidelity_log_sum}, "
            f"remote_fidelity={self.remote_fidelity}, "
            f"local_fidelity_loss={self.local_fidelity_loss}, "
            f"local_fidelity_log_sum={self.local_fidelity_log_sum}, "
            f"local_fidelity={self.local_fidelity}, "
            f"time={self.execution_time:.2f}), "
            f"local_gate_num={self.local_gate_num}"
            # Debug fields kept for temporary diagnostics; hide from default pprint/terminal output.
            # f", flush_calls={self.flush_calls}"
            # f", nonempty_flushes={self.nonempty_flushes}"
            # f", local_transpile_calls={self.local_transpile_calls}"
            # f", comm_block_events={self.comm_block_events}"
            # f", comm_block_local_gate_num={self.comm_block_local_gate_num}"
            # f", comm_block_local_fidelity_loss={self.comm_block_local_fidelity_loss}"
            # f", comm_block_local_fidelity_log_sum={self.comm_block_local_fidelity_log_sum}"
            # f", comm_block_remote_fidelity_loss={self.comm_block_remote_fidelity_loss}"
            # f", comm_block_remote_fidelity_log_sum={self.comm_block_remote_fidelity_log_sum}"
            # f", telegate_exec_events={self.telegate_exec_events}"
            # f", telegate_exec_local_gate_num={self.telegate_exec_local_gate_num}"
            # f", telegate_exec_local_fidelity_loss={self.telegate_exec_local_fidelity_loss}"
            # f", telegate_exec_local_fidelity_log_sum={self.telegate_exec_local_fidelity_log_sum}"
            # f", telegate_exec_remote_fidelity_loss={self.telegate_exec_remote_fidelity_loss}"
            # f", telegate_exec_remote_fidelity_log_sum={self.telegate_exec_remote_fidelity_log_sum}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other: "ExecCosts") -> "ExecCosts":
        if not isinstance(other, ExecCosts):
            raise TypeError(f"不能与 {type(other)} 累加")

        self.remote_hops += other.remote_hops
        self.remote_swaps += other.remote_swaps
        self.cat_ents += other.cat_ents
        self.epairs += other.epairs
        self.remote_fidelity_loss += other.remote_fidelity_loss
        self.remote_fidelity_log_sum += other.remote_fidelity_log_sum
        self.local_fidelity_loss += other.local_fidelity_loss
        self.local_fidelity_log_sum += other.local_fidelity_log_sum
        self.execution_time += other.execution_time
        self.remote_fidelity *= other.remote_fidelity
        self.local_fidelity *= other.local_fidelity
        self.local_gate_num += other.local_gate_num
        self.flush_calls += other.flush_calls
        self.nonempty_flushes += other.nonempty_flushes
        self.local_transpile_calls += other.local_transpile_calls
        self.comm_block_events += other.comm_block_events
        self.comm_block_local_gate_num += other.comm_block_local_gate_num
        self.comm_block_local_fidelity_loss += other.comm_block_local_fidelity_loss
        self.comm_block_local_fidelity_log_sum += other.comm_block_local_fidelity_log_sum
        self.comm_block_remote_fidelity_loss += other.comm_block_remote_fidelity_loss
        self.comm_block_remote_fidelity_log_sum += other.comm_block_remote_fidelity_log_sum
        self.telegate_exec_events += other.telegate_exec_events
        self.telegate_exec_local_gate_num += other.telegate_exec_local_gate_num
        self.telegate_exec_local_fidelity_loss += other.telegate_exec_local_fidelity_loss
        self.telegate_exec_local_fidelity_log_sum += other.telegate_exec_local_fidelity_log_sum
        self.telegate_exec_remote_fidelity_loss += other.telegate_exec_remote_fidelity_loss
        self.telegate_exec_remote_fidelity_log_sum += other.telegate_exec_remote_fidelity_log_sum
        return self

    def to_dict(self) -> dict:
        base_dict = dataclasses.asdict(self)
        base_dict.update({
            "num_comms": self.num_comms,
            "total_fidelity_log_sum": self.total_fidelity_log_sum,
            "total_fidelity_loss": self.total_fidelity_loss,
            "rgeo_mean_fid": self.remote_geometric_mean_fidelity,
            "total_fidelity": self.total_fidelity,
        })
        sorted_keys = [
            "total_fidelity_loss",
            "epairs",
            "execution_time",
            "num_comms",
            "rgeo_mean_fid",
            "total_fidelity_log_sum",
            "total_fidelity",
            "remote_hops",
            "remote_swaps",
            "cat_ents",
            "local_fidelity_loss",
            "remote_fidelity_loss",
            "local_fidelity_log_sum",
            "remote_fidelity_log_sum",
            "local_fidelity",
            "remote_fidelity",
            "local_gate_num",
            # Debug fields kept for temporary diagnostics; hide from default CSV/JSON summaries.
            # "flush_calls",
            # "nonempty_flushes",
            # "local_transpile_calls",
            # "comm_block_events",
            # "comm_block_local_gate_num",
            # "comm_block_local_fidelity_loss",
            # "comm_block_local_fidelity_log_sum",
            # "comm_block_remote_fidelity_loss",
            # "comm_block_remote_fidelity_log_sum",
            # "telegate_exec_events",
            # "telegate_exec_local_gate_num",
            # "telegate_exec_local_fidelity_loss",
            # "telegate_exec_local_fidelity_log_sum",
            # "telegate_exec_remote_fidelity_loss",
            # "telegate_exec_remote_fidelity_log_sum",
        ]

        # 生成有序字典
        return {key: base_dict[key] for key in sorted_keys if key in base_dict}


class CommOp(Gate):
    """
    通信操作封装为Qiskit自定义Gate，便于直接插入QuantumCircuit。
    gate_list中保存通信块的Qiskit门（每个门额外携带_global_lqids元数据）。
    """

    def __init__(
        self,
        comm_type: str,
        source_qubit: int,
        src_qpu: int,
        dst_qpu: int,
        involved_qubits: list[int],
        gate_list: Optional[list[Gate]] = None,
    ):
        if comm_type not in {"cat", "rtp", "tp"}:
            raise ValueError(f"Unsupported comm_type: {comm_type}")
        if len(involved_qubits) == 0:
            raise ValueError("involved_qubits cannot be empty")

        self.comm_type = comm_type
        self.source_qubit = int(source_qubit)
        self.src_qpu = int(src_qpu)
        self.dst_qpu = int(dst_qpu)
        self.involved_qubits = [int(q) for q in involved_qubits]

        normalized: list[Gate] = []
        for g in gate_list or []:
            if not isinstance(g, Gate):
                raise TypeError(f"gate_list items must be qiskit Gate, got {type(g)}")
            normalized.append(g)
        self.gate_list = normalized

        super().__init__(
            name=f"comm_{comm_type}",
            num_qubits=len(self.involved_qubits),
            params=[],
        )

    def __repr__(self) -> str:
        return (
            f"CommOp(type={self.comm_type}, source={self.source_qubit}, "
            f"src_qpu={self.src_qpu}, dst_qpu={self.dst_qpu}, "
            f"involved_qubits={self.involved_qubits}, gates={len(self.gate_list)})"
        )


@dataclass
class MappingRecord:
    """
    映射记录类：记录线路层级范围、映射类型、开销及时间
    """
    # 必选字段：线路层级范围
    layer_start: int = -1          # 起始层级（第几层）
    layer_end: int = -1            # 结束层级（第几层）
    # 必选字段：量子比特划分
    partition: list[list[int]] = dataclasses.field(default_factory=list) # 划分结果，格式为 list of lists，每个子列表代表一个分区的量子比特索引
    # 必选字段：映射信息
    mapping_type: str = ""         # 映射类型（如 "teledata"、"telegate"）
    costs: ExecCosts = ExecCosts() # 执行成本，包含保真度损失、通信开销、执行时间等指标
    logical_phy_map: dict[int, tuple[int, int | None]] = dataclasses.field(default_factory=dict) # 量子比特映射信息，记录每个全局逻辑比特在物理上的位置（QPU编号和物理比特编号）
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 冻结模式下修改字段需用 object.__setattr__
        object.__setattr__(self, "partition", copy.deepcopy(self.partition))
        object.__setattr__(self, "costs", copy.deepcopy(self.costs))
        object.__setattr__(self, "logical_phy_map", copy.deepcopy(self.logical_phy_map))
        if self.extra_info is not None:
            object.__setattr__(self, "extra_info", copy.deepcopy(self.extra_info))

    def to_dict(self) -> dict:
        """将 MappingRecord 转为字典，包含嵌套的 ExecCosts 字典"""
        return {
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "partition": self.partition,
            "mapping_type": self.mapping_type,
            "costs": self.costs.to_dict(),  # 直接用自定义 to_dict
            "logical_phy_map": self.logical_phy_map,
            "extra_info": self.extra_info
        }

    def update(self, **kwargs):
        """批量更新字段"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"MappingRecord 没有属性 {key}")


# 辅助类：管理多条记录
@dataclass
class MappingRecordList:
    """
    映射记录管理器：批量存储、查询记录
    """
    total_costs: ExecCosts = dataclasses.field(default_factory=ExecCosts)
    num_records: int = 0
    records: list[MappingRecord] = dataclasses.field(default_factory=list)

    def add_record(self, record: MappingRecord):
        """添加一条记录"""
        self.records.append(record)

    def summarize_total_costs(self):
        """汇总所有记录的成本信息"""
        total_costs = ExecCosts()
        for record in self.records:
            total_costs += record.costs
        self.total_costs = total_costs
        self.num_records = len(self.records)
        return

    def update_total_costs(self, **kwargs):
        """批量更新total_costs指标"""
        self.total_costs.update(**kwargs)
        return

    def get_records_by_layer_range(self, layer_start: int, layer_end: int) -> list[MappingRecord]:
        """按层级范围查询记录（包含交集）"""
        return [
            r for r in self.records
            if not (r.layer_end < layer_start or r.layer_start > layer_end)
        ]

    def save_records(self, filename: str):
        """
        将记录保存到文件，支持 JSON/CSV 格式
        Args:
            filename: 保存路径
        """
        if not self.records:
            print("⚠️ 无映射记录可保存")
            return

        # 统一序列化：将 dataclass 转为字典（兼容可选字段 extra_info）
        # 将total_costs转为字典
        total_costs_dict = self.total_costs.to_dict()
        # 将每条记录转为字典
        records_dict = [record.to_dict() for record in self.records]
        data_dict = {
            "total_costs": total_costs_dict,
            "num_records": self.num_records,
            "records": records_dict
        }
        # data_dict = dataclasses.asdict(self)
        data_dict = self._convert_numpy_types(data_dict)
        _, data_dict = self._prune_unserializable(data_dict)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 按格式保存
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                data_dict,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=False  # 保持字段顺序，更易读
            )
        print(f"✅ 成功保存 {len(data_dict['records'])} 条映射记录到 JSON 文件：{filename}")
        return

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """
        递归转换所有NumPy类型为原生Python类型
        支持：字典、列表、元组、np.int64/np.float64等
        """
        # 处理NumPy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # 处理列表/元组：递归转换每个元素
        elif isinstance(obj, (list, tuple)):
            return [MappingRecordList._convert_numpy_types(item) for item in obj]
        # 处理字典：递归转换每个键值对
        elif isinstance(obj, dict):
            return {k: MappingRecordList._convert_numpy_types(v) for k, v in obj.items()}
        # 其他类型直接返回（如str、int、float、None等）
        else:
            return obj

    @staticmethod
    def _prune_unserializable(obj: Any) -> tuple[bool, Any]:
        """
        递归过滤不可JSON序列化的字段。
        返回 (是否保留, 过滤后的对象)。
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return True, obj

        if isinstance(obj, dict):
            kept: dict[Any, Any] = {}
            for k, v in obj.items():
                ok, new_v = MappingRecordList._prune_unserializable(v)
                if ok:
                    kept[k] = new_v
            return True, kept

        if isinstance(obj, (list, tuple)):
            kept_list: list[Any] = []
            for item in obj:
                ok, new_item = MappingRecordList._prune_unserializable(item)
                if ok:
                    kept_list.append(new_item)
            return True, kept_list

        try:
            json.dumps(obj)
            return True, obj
        except TypeError:
            return False, None


class CompilerUtils:

    @staticmethod
    def _check_logical_map_partition_consistency(
        partition: list[list[int]],
        logical_phy_map: dict[int, tuple[int, int | None]],
    ) -> tuple[bool, str]:
        """
        检查 logical_phy_map 与 partition 的一致性：
        1) 每个 partition 中的逻辑比特必须出现在 logical_phy_map 中
        2) 每个逻辑比特在 logical_phy_map 中的 qpu_id 必须等于其 partition 所在 qpu
        3) 非 None 的物理槽位 (qpu_id, phy_id) 不能被多个逻辑比特复用
        4) logical_phy_map 不应包含 partition 外的逻辑比特
        """
        expected_qpu: dict[int, int] = {} # 每个逻辑比特期望对应的qpu
        for qpu_id, group in enumerate(partition):
            for global_lqid in group:
                expected_qpu[global_lqid] = qpu_id

        # 1) + 2)
        for global_lqid, qpu_id in expected_qpu.items():
            if global_lqid not in logical_phy_map:
                return False, f"缺少逻辑比特映射: lqid={global_lqid}, expected_qpu={qpu_id}"
            mapped_qpu = logical_phy_map[global_lqid][0]
            if mapped_qpu != qpu_id:
                return False, (
                    f"QPU归属不一致: lqid={global_lqid}, expected_qpu={qpu_id}, "
                    f"mapped_qpu={mapped_qpu}"
                )

        # 4)
        for mapped_lqid in logical_phy_map:
            if mapped_lqid not in expected_qpu:
                return False, f"发现 partition 外逻辑比特: lqid={mapped_lqid}"

        # 3)
        slot_owner: dict[tuple[int, int], int] = {}
        for lqid, (qpu_id, phy_id) in logical_phy_map.items():
            if phy_id is None:
                continue
            slot = (qpu_id, phy_id)
            if slot in slot_owner and slot_owner[slot] != lqid:
                return False, (
                    f"物理槽位冲突: slot={slot}, owners=({slot_owner[slot]}, {lqid})"
                )
            slot_owner[slot] = lqid

        return True, "ok"

    @staticmethod
    def _resolve_comm_qpu_endpoints_from_logical_map(
        logical_phy_map: dict[int, tuple[int, int | None]],
        op: CommOp,
    ) -> tuple[int, int]:
        """
        基于 logical_phy_map 和 CommOp 的逻辑比特信息解析通信端点。
        优先使用逻辑比特当前位置（物理QPU索引），解析失败时回退到 CommOp 自带 src/dst。
        """
        def _lookup_qpu(lqid: int) -> Optional[int]:
            if lqid in logical_phy_map:
                return logical_phy_map[lqid][0]
            key_str = str(lqid)
            if key_str in logical_phy_map:
                return logical_phy_map[key_str][0]
            return None

        src_qpu = _lookup_qpu(op.source_qubit)
        if src_qpu is None:
            src_qpu = op.src_qpu

        dst_qpu = None
        dst_candidates: set[int] = set()
        for lqid in op.involved_qubits:
            if lqid == op.source_qubit:
                continue
            cand = _lookup_qpu(lqid)
            if cand is None:
                continue
            dst_candidates.add(cand)

        if len(dst_candidates) > 1:
            # AutoComm may pack source-side single-qubit gates into CommOp.
            # Keep endpoint metadata as authority when involved_qubits are mixed.
            dst_qpu = op.dst_qpu
            return src_qpu, dst_qpu

        if len(dst_candidates) == 1:
            dst_qpu = next(iter(dst_candidates))
            if int(op.dst_qpu) != int(dst_qpu):
                raise RuntimeError(
                    "[COMM_ENDPOINT] dst_qpu metadata inconsistent with involved_qubits-derived destination: "
                    f"source_qubit={op.source_qubit}, involved={op.involved_qubits}, "
                    f"src_qpu={src_qpu}, op.dst_qpu={op.dst_qpu}, derived_dst_qpu={dst_qpu}"
                )

        if dst_qpu is None:
            dst_qpu = op.dst_qpu

        return src_qpu, dst_qpu

    # @staticmethod
    # def diagnose_commop_endpoint_consistency(
    #     circuit: QuantumCircuit,
    #     logical_phy_map: dict[int, tuple[int, int | None]],
    #     stage: str,
    #     strict: bool = True,
    # ) -> list[dict[str, Any]]:
    #     """
    #     诊断子线路中CommOp端点是否与logical_phy_map一致。
    #     主要检查：
    #     1) （已停用）非source involved_qubits是否落在唯一dst_qpu
    #     2) op.dst_qpu是否与involved_qubits推断dst一致
    #     3) op.src_qpu是否与source当前位置一致
    #     """
    #     issues: list[dict[str, Any]] = []

    #     def _lookup_qpu(lqid: int) -> Optional[int]:
    #         if lqid in logical_phy_map:
    #             return int(logical_phy_map[lqid][0])
    #         key_str = str(lqid)
    #         if key_str in logical_phy_map:
    #             return int(logical_phy_map[key_str][0])
    #         return None

    #     for inst_idx, instruction in enumerate(circuit.data):
    #         op = instruction.operation
    #         if not isinstance(op, CommOp):
    #             continue

    #         src_qpu_mapped = _lookup_qpu(op.source_qubit)

    #         dst_candidates: set[int] = set()
    #         for lqid in op.involved_qubits:
    #             if int(lqid) == int(op.source_qubit):
    #                 continue
    #             cand = _lookup_qpu(int(lqid))
    #             if cand is not None:
    #                 dst_candidates.add(int(cand))

    #         issue_reason = None
    #         # NOTE:
    #         # AutoComm may pack source-side single-qubit gates into the same CommOp,
    #         # which can make non-source involved_qubits span multiple QPUs.
    #         # Therefore we intentionally skip "multiple_dst_candidates" as an error.
    #         if len(dst_candidates) == 1 and int(op.dst_qpu) != int(next(iter(dst_candidates))):
    #             issue_reason = "dst_qpu_mismatch"
    #         elif src_qpu_mapped is not None and int(op.src_qpu) != int(src_qpu_mapped):
    #             issue_reason = "src_qpu_mismatch"

    #         if issue_reason is not None:
    #             issues.append(
    #                 {
    #                     "stage": stage,
    #                     "inst_idx": inst_idx,
    #                     "reason": issue_reason,
    #                     "comm_type": op.comm_type,
    #                     "source_qubit": int(op.source_qubit),
    #                     "involved_qubits": [int(q) for q in op.involved_qubits],
    #                     "op_src_qpu": int(op.src_qpu),
    #                     "op_dst_qpu": int(op.dst_qpu),
    #                     "mapped_src_qpu": src_qpu_mapped,
    #                     "dst_candidates": sorted(dst_candidates),
    #                     "gate_count": len(op.gate_list),
    #                 }
    #             )

    #     if strict and len(issues) > 0:
    #         first = issues[0]
    #         raise RuntimeError(
    #             "[COMM_DIAG] CommOp endpoint inconsistency detected. "
    #             f"stage={first['stage']}, inst_idx={first['inst_idx']}, reason={first['reason']}, "
    #             f"comm_type={first['comm_type']}, source_qubit={first['source_qubit']}, "
    #             f"involved_qubits={first['involved_qubits']}, op_src_qpu={first['op_src_qpu']}, "
    #             f"op_dst_qpu={first['op_dst_qpu']}, mapped_src_qpu={first['mapped_src_qpu']}, "
    #             f"dst_candidates={first['dst_candidates']}, gate_count={first['gate_count']}, "
    #             f"total_issues={len(issues)}"
    #         )

    #     return issues

    """
    编译工具类
    """
    @staticmethod
    def allocate_qubits(num_qubits: int, network: Network) -> list[list[int]]:
        """
        Initialize the partition
        """
        partition = []
        cnt_qubits = 0
        for qpu_size in network.backend_sizes:
            remain = num_qubits - cnt_qubits
            if remain == 0:
                break
            end_index = min(cnt_qubits + qpu_size, num_qubits)
            part = list(range(cnt_qubits, end_index))
            partition.append(part)
            cnt_qubits = end_index
        assert(cnt_qubits == num_qubits)
        for _ in range(len(partition), network.num_backends):
            partition.append([])
        return partition

    @staticmethod
    def build_qubit_interaction_graph(circuit: QuantumCircuit) -> nx.Graph:
        """
        Construct the qubit interaction graph from the circuit
        """
        qig = nx.Graph()
        for qubit in range(circuit.num_qubits):
            qig.add_node(qubit)
        for instruction in circuit:
            # gate = instruction.operation
            qubits = [qubit._index for qubit in instruction.qubits]
            if qubits[0] == None:
                qubits = [circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) > 1:
                if instruction.name == "barrier":
                    continue
                assert len(qubits) == 2, f"instruction: {instruction}"
                if qig.has_edge(qubits[0], qubits[1]):
                    qig[qubits[0]][qubits[1]]['weight'] += 1
                else:
                    qig.add_edge(qubits[0], qubits[1], weight=1)
        return qig

    @staticmethod
    def get_subcircuit_by_level(num_qubits: int, 
                                circuit: QuantumCircuit, 
                                circuit_layers: list[list[Any]], 
                                layer_start: int, 
                                layer_end: int) -> QuantumCircuit:
        """
        从DAGOpNode分层中提取子线路
        """
        subcircuit = QuantumCircuit(num_qubits)

        # 遍历指定层级的DAGOpNode
        for layer in circuit_layers[layer_start:layer_end + 1]:
            # print(f"[DEBUG] Processing layer with {len(layer)} nodes, layer_start: {layer_start}, layer_end: {layer_end}")
            for node in layer: # node 是 DAGOpNode 对象
                if node.op.name == "barrier":
                    continue
                gate_instruction = node.op  # 获取门指令（Instruction对象）
                # qubit_indices = [q._index for q in node.qargs]  # 提取量子比特索引
                # if qubit_indices[0] == None:
                qubit_indices = [circuit.qubits.index(q) for q in node.qargs]
                # print(f"[DEBUG] [{qubit_indices}] gate: {gate_instruction}")
                assert qubit_indices[0] is not None, f"无法找到量子比特索引，node.qargs: {node.qargs}"
                # 将门添加到子线路
                subcircuit.append(gate_instruction, qubit_indices)
        
        return subcircuit

    @staticmethod
    def evaluate_remote_hops(qig: nx.Graph, 
                           partition: list[list[int]], 
                           network: Any) -> int:
        """
        计算qubit interaction graph在partitions下的割
        """
        node_to_partition = {} # 构建节点到划分编号的映射
        for i, part in enumerate(partition):
            for node in part:
                node_to_partition[node] = i
        remote_hops = 0
        for u, v in qig.edges(): # 遍历图中的每一条边，也就是双量子门
            qpu_u = node_to_partition[u]
            qpu_v = node_to_partition[v]
            if qpu_u != qpu_v:
                remote_hops += network.Hops[qpu_u][qpu_v] * qig[u][v]['weight']
                # fidelity_loss += (1 - network.move_fidelity[qpu_u][qpu_v]) * qig[u][v]['weight']
                # fidelity *= network.move_fidelity[qpu_u][qpu_v] ** qig[u][v]['weight']
        return remote_hops

    @staticmethod
    def evaluate_local_and_telegate_with_cat(
        arg: MappingRecord | list[list[int]],
        circuit: QuantumCircuit,
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {},
        optimization_level: int = 0,
        strict_flush_on_remote: bool = True,
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]]]:
        """
        按CommOp + 门序列统一评估通信与本地门成本
        """
        partition = None

        if isinstance(arg, MappingRecord):
            partition = arg.partition
            logical_phy_map = arg.logical_phy_map
        else:
            partition = arg

        if len(logical_phy_map) == 0:
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        ok, reason = CompilerUtils._check_logical_map_partition_consistency(partition, logical_phy_map)
        if not ok:
            print(
                f"[MAP_CHECK][evaluate_local_and_telegate_with_cat][pre] {reason}",
                file=sys.stderr,
            )
            print(f"[MAP_CHECK][partition] {partition}", file=sys.stderr)
            print(f"[MAP_CHECK][logical_phy_map] {logical_phy_map}", file=sys.stderr)
            assert ok, reason

        # 构建局部索引（local_lqid 直接使用槽位sid），并维护运行时位置。
        global_to_local_lqid: dict[int, int] = {}
        runtime_pos: dict[int, tuple[int, int]] = {}

        # 子线路容量使用“当前分组大小 + comm_slot_reserve”，避免直接按后端总量建超大子线路。
        reserve = int(getattr(network, "comm_slot_reserve", 1) or 1)
        backend_caps_full = network.get_backend_qubit_counts(include_comm_slot=True)
        capacity_by_qpu: list[int] = [] # |partition[j]| + reserve
        comm_slot_start: dict[int, int] = {}
        for qpu_id, group in enumerate(partition):
            # 记录每个QPU的容量（含通信槽位）
            target_cap = len(group) + reserve
            full_cap = backend_caps_full[qpu_id]
            assert target_cap <= full_cap, f"QPU {qpu_id} 的目标容量 {target_cap} 超过了后端容量 {full_cap}"
            capacity_by_qpu.append(target_cap)
            comm_slot_start[qpu_id] = len(group)

        # slot_owner[qpu_id][local_lqid] = 当前占用该槽位的global_lqid，None表示空闲。
        slot_owner: list[list[Optional[int]]] = [
            [None for _ in range(capacity_by_qpu[qpu_id])] for qpu_id in range(len(partition))
        ]

        # 先为每个逻辑比特分配“计算槽”（通信槽预留给CommOp临时使用）。
        for qpu_id, group in enumerate(partition):
            for sid, global_lqid in enumerate(group):
                slot_owner[qpu_id][sid] = global_lqid
                global_to_local_lqid[global_lqid] = sid
                runtime_pos[global_lqid] = (qpu_id, sid)

        # --- 建立每个QPU对应的子线路（按容量建模，支持通信缓冲槽位）---
        subcircuits = [QuantumCircuit(capacity_by_qpu[qpu_id]) for qpu_id in range(len(partition))]

        # 初始化噪声
        costs = ExecCosts()

        def _find_free_slot(qpu_id: int, owner: Optional[int] = None) -> int:
            for sid in range(comm_slot_start[qpu_id], capacity_by_qpu[qpu_id]):
                if slot_owner[qpu_id][sid] is None:
                    slot_owner[qpu_id][sid] = owner
                    return sid
            raise RuntimeError(
                "[CAPACITY] 目标QPU无可用通信槽位。"
                f" qpu={qpu_id}, comm_range=[{comm_slot_start[qpu_id]}, {capacity_by_qpu[qpu_id]}), "
                f"slot_owner={slot_owner[qpu_id]}"
            )

        def _release_slot(qpu_id: int, sid: int, expected_owner: Optional[int] = None) -> None:
            if not (comm_slot_start[qpu_id] <= sid < capacity_by_qpu[qpu_id]):
                raise RuntimeError(
                    f"[COMM_SLOT] 尝试释放非通信槽。qpu={qpu_id}, sid={sid}, "
                    f"comm_start={comm_slot_start[qpu_id]}"
                )
            if expected_owner is not None and slot_owner[qpu_id][sid] != expected_owner:
                raise RuntimeError(
                    f"[COMM_SLOT] 通信槽占用者不匹配。qpu={qpu_id}, sid={sid}, "
                    f"expected={expected_owner}, actual={slot_owner[qpu_id][sid]}"
                )
            slot_owner[qpu_id][sid] = None

        def _build_tmp_local_map_for_comm(source_qubit: int, source_temp_slot: int) -> dict[int, int]:
            tmp_map = dict(global_to_local_lqid)
            tmp_map[source_qubit] = source_temp_slot
            return tmp_map

        def _append_comm_gate_block(
            dst_qpu: int,
            gate_list: list[Gate],
            tmp_global_to_local_lqid: dict[int, int],
        ) -> None:
            """
            将通信块里的门绑定到目标QPU的具体槽位，并并入目标子线路。
            """
            if len(gate_list) == 0:
                return

            for gate_op in gate_list:
                global_lqids = getattr(gate_op, "_global_lqids", None)
                if global_lqids is None:
                    raise RuntimeError(
                        f"[COMM_FLOW] gate_list门缺少 _global_lqids/_autocomm_qids 元数据: gate={gate_op}"
                    )

                mapped_qubits: list[int] = [tmp_global_to_local_lqid[q] for q in global_lqids]
                subcircuits[dst_qpu].append(gate_op, mapped_qubits)

        def _estimate_gate_error(backend: Any, gate_name: str, qubits: list[int]) -> float:
            gate_key = f"{gate_name}{'_'.join(map(str, qubits))}"
            gate_error = backend.gate_dict.get(gate_key, {}).get("gate_error_value", None)
            if gate_error == 1:
                gate_error = 0.99
                print(f"[WARNING] {gate_key}: {gate_error}")

            if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
                gate_error = backend.gate_dict[gate_name]["gate_error_value"]

            assert gate_error is not None, f"Gate error not found for gate_key: {gate_key} in backend.gate_dict"
            gate_error = float(gate_error)
            return min(max(gate_error, 0.0), 0.99)

        def _accumulate_comm_block_local_stats(
            dst_qpu: int,
            gate_list: list[Gate],
            tmp_global_to_local_lqid: dict[int, int],
            stats_prefix: str,
        ) -> None:
            if len(gate_list) == 0:
                return

            backend = network.backends[dst_qpu]
            for gate_op in gate_list:
                global_lqids = getattr(gate_op, "_global_lqids", None)
                if global_lqids is None:
                    raise RuntimeError(
                        f"[COMM_FLOW] gate_list门缺少 _global_lqids/_autocomm_qids 元数据: gate={gate_op}"
                    )
                mapped_qubits = [tmp_global_to_local_lqid[q] for q in global_lqids]
                gate_error = _estimate_gate_error(backend, gate_op.name, mapped_qubits)
                if stats_prefix == "comm_block":
                    costs.comm_block_local_gate_num += 1
                    costs.comm_block_local_fidelity_loss += gate_error
                    costs.comm_block_local_fidelity_log_sum += np.log(1 - gate_error)
                else:
                    costs.telegate_exec_local_gate_num += 1
                    costs.telegate_exec_local_fidelity_loss += gate_error
                    costs.telegate_exec_local_fidelity_log_sum += np.log(1 - gate_error)

        def _build_synthetic_telegate_commop(
            op: Any,
            global_qids: list[int],
        ) -> Optional[CommOp]:
            if len(global_qids) != 2 or not isinstance(op, Gate):
                return None

            gate_name = str(op.name).lower()
            if gate_name in {"barrier", "measure"}:
                return None

            src_global = int(global_qids[0])
            dst_global = int(global_qids[1])
            src_qpu = int(runtime_pos[src_global][0])
            dst_qpu = int(runtime_pos[dst_global][0])
            if src_qpu == dst_qpu:
                return None

            gate_copy = op.to_mutable() if hasattr(op, "to_mutable") else copy.deepcopy(op)
            setattr(gate_copy, "_global_lqids", [src_global, dst_global])
            return CommOp(
                comm_type="cat",
                source_qubit=src_global,
                src_qpu=src_qpu,
                dst_qpu=dst_qpu,
                involved_qubits=[src_global, dst_global],
                gate_list=[gate_copy],
            )

        def _process_comm_like_op(comm_op: CommOp, stats_prefix: str) -> None:
            nonlocal costs
            src_qpu, dst_qpu = CompilerUtils._resolve_comm_qpu_endpoints_from_logical_map(
                logical_phy_map=logical_phy_map,
                op=comm_op,
            )

            if strict_flush_on_remote:
                _flush_local_subcircuits()

            remote_loss_before = costs.remote_fidelity_loss
            remote_log_before = costs.remote_fidelity_log_sum

            if stats_prefix == "comm_block":
                costs.comm_block_events += 1
            else:
                costs.telegate_exec_events += 1

            if comm_op.comm_type == "cat":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 1, network
                )
                costs.cat_ents += 1
                dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                try:
                    tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                    _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                    _append_comm_gate_block(dst_qpu, comm_op.gate_list, tmp_map)
                finally:
                    _release_slot(dst_qpu, dst_slot, expected_owner=comm_op.source_qubit)

            elif comm_op.comm_type == "rtp":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 2, network
                )
                dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                try:
                    tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                    _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                    _append_comm_gate_block(dst_qpu, comm_op.gate_list, tmp_map)
                finally:
                    _release_slot(dst_qpu, dst_slot, expected_owner=comm_op.source_qubit)

            elif comm_op.comm_type == "tp":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 1, network
                )
                dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                try:
                    tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                    _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                    _append_comm_gate_block(dst_qpu, comm_op.gate_list, tmp_map)
                finally:
                    _release_slot(dst_qpu, dst_slot, expected_owner=comm_op.source_qubit)

            remote_loss_delta = costs.remote_fidelity_loss - remote_loss_before
            remote_log_delta = costs.remote_fidelity_log_sum - remote_log_before
            if stats_prefix == "comm_block":
                costs.comm_block_remote_fidelity_loss += remote_loss_delta
                costs.comm_block_remote_fidelity_log_sum += remote_log_delta
            else:
                costs.telegate_exec_remote_fidelity_loss += remote_loss_delta
                costs.telegate_exec_remote_fidelity_log_sum += remote_log_delta

            if strict_flush_on_remote:
                _flush_local_subcircuits()

        def _flush_local_subcircuits() -> None:
            """
            将当前缓存的本地子线路统一结算到local fidelity，并更新logical_phy_map。
            strict_flush_on_remote=True时由远程事件触发分段结算，避免跨远程边界的过度合并。
            """
            nonlocal logical_phy_map, subcircuits, costs
            costs.flush_calls += 1
            flushed_any = False

            for qpu_id in range(len(subcircuits)):
                subcircuit = subcircuits[qpu_id]
                backend = network.backends[qpu_id]

                if subcircuit.size() == 0:
                    continue

                flushed_any = True
                costs.local_transpile_calls += 1

                initial_layout = CompilerUtils.get_initial_layout(
                    subcircuit,
                    partition[qpu_id],
                    global_to_local_lqid,
                    logical_phy_map,
                )

                transpiled_circuit = transpile(
                    subcircuit,
                    coupling_map=backend.coupling_map,
                    basis_gates=backend.basis_gates,
                    initial_layout=initial_layout,
                    optimization_level=optimization_level,
                    seed_transpiler=42,
                )

                logical_phy_map = CompilerUtils.get_logical_to_physical_map(
                    transpiled_circuit, partition[qpu_id], global_to_local_lqid, logical_phy_map
                )

                for instruction in transpiled_circuit:
                    gate_name = instruction.operation.name
                    qubits = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
                    assert qubits[0] is not None, f"Qubit index is None for instruction: {instruction}"
                    gate_error = _estimate_gate_error(backend, gate_name, qubits)
                    costs.local_gate_num += 1
                    costs.local_fidelity_loss += gate_error
                    costs.local_fidelity *= (1 - gate_error)
                    costs.local_fidelity_log_sum += np.log(1 - gate_error)

                # 清空该QPU缓冲，下一段继续累计。
                subcircuits[qpu_id] = QuantumCircuit(capacity_by_qpu[qpu_id])

            if flushed_any:
                costs.nonempty_flushes += 1

        for instruction in circuit:
            op = instruction.operation
            global_qids = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            # print(f"\n\n[DEBUG] Processing instruction: {instruction}")
            # print(f"[DEBUG] qids: {global_qids}")

            if isinstance(op, CommOp):
                _process_comm_like_op(op, "comm_block")

            else:
                # print(f"[DEBUG] Processing normal gate: {op}")
                
                # 检查操作涉及的量子比特属于哪个分区
                involved_qpus = set(runtime_pos[q][0] for q in global_qids)

                # 本地操作，加入对应子线路（需要转换为局部编号）
                if len(involved_qpus) == 1:
                    qpu_id = involved_qpus.pop()
                    mapped_qubits = [runtime_pos[q][1] for q in global_qids]
                    subcircuits[qpu_id].append(instruction.operation, mapped_qubits)
                # 操作跨越多个分区，为telegate操作
                else:
                    synthetic_comm = _build_synthetic_telegate_commop(op, global_qids)
                    if synthetic_comm is not None:
                        _process_comm_like_op(synthetic_comm, "telegate")
                    else:
                        if strict_flush_on_remote:
                            _flush_local_subcircuits()

                        for i in range(len(global_qids) - 1):
                            q1, q2 = global_qids[i], global_qids[i + 1]
                            p1, p2 = runtime_pos[q1][0], runtime_pos[q2][0]
                            if p1 != p2:
                                costs = CompilerUtils.update_remote_move_costs(
                                    costs, p1, p2, 1, network
                                )

                        if strict_flush_on_remote:
                            _flush_local_subcircuits()

        # 检查所有通信槽是否都已释放。
        for qpu_id in range(len(partition)):
            for sid in range(comm_slot_start[qpu_id], capacity_by_qpu[qpu_id]):
                if slot_owner[qpu_id][sid] is not None:
                    raise RuntimeError(
                        f"[COMM_SLOT] 片段结束后发现未释放通信槽: qpu={qpu_id}, sid={sid}, "
                        f"owner={slot_owner[qpu_id][sid]}"
                    )

        # 片段结束后结算剩余本地门。
        _flush_local_subcircuits()

        if isinstance(arg, MappingRecord):
            arg.costs += costs
            arg.logical_phy_map = logical_phy_map
            return arg.costs, logical_phy_map

        return costs, logical_phy_map

    @staticmethod
    def evaluate_telegate_with_cat(
        arg: MappingRecord | list[list[int]],  # 兼容两种类型：record / partition
        circuit: QuantumCircuit, 
        network: Network
    ) -> ExecCosts:
        partition = None
        record: Optional[MappingRecord] = None
        
        if isinstance(arg, MappingRecord):
            record = arg
            partition = record.partition
            logical_phy_map = copy.deepcopy(record.logical_phy_map)
        else:
            # arg is a list of lists representing the partition
            partition = arg
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        if len(logical_phy_map) == 0:
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        # 建立一个反向索引，用于快速查询每个量子比特属于哪个分区
        qubit_to_partition = {}
        for idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_partition[qubit] = idx

        # 初始化噪声
        costs = ExecCosts()

        # 遍历circuit上的每个操作，如果操作完全属于某个group，则添加到对应的subcircuit；
        # 如果操作跨越多个group，则记录为telegate操作（支持可选CAT-aware计费）。
        for instruction in circuit:
            # 获取量子操作
            op = instruction.operation
            # 操作属于原始的全局量子比特编号，需要转换为子线路的局部编号
            global_qids = [circuit.qubits.index(qubit) for qubit in instruction.qubits]

            if isinstance(op, CommOp):
                src_qpu, dst_qpu = CompilerUtils._resolve_comm_qpu_endpoints_from_logical_map(
                    logical_phy_map=logical_phy_map,
                    op=op,
                )

                if op.comm_type == "cat":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, src_qpu, dst_qpu, 1, network
                    )
                    costs.cat_ents += 1
                elif op.comm_type == "rtp":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, src_qpu, dst_qpu, 2, network
                    )
                elif op.comm_type == "tp":
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, src_qpu, dst_qpu, 1, network
                    )

            else:
                # 检查操作涉及的量子比特属于哪个分区
                involved_qpus = set(qubit_to_partition[q] for q in global_qids)
                # 操作跨越多个分区，为telegate操作
                if len(involved_qpus) > 1:
                    for i in range(len(global_qids) - 1):
                        q1, q2 = global_qids[i], global_qids[i + 1]
                        p1, p2 = qubit_to_partition[q1], qubit_to_partition[q2]
                        if p1 != p2:
                            costs = CompilerUtils.update_remote_move_costs(
                                costs, p1, p2, 1, network
                            )

        if isinstance(arg, MappingRecord):
            # 更新record的costs
            arg.costs += costs

        return costs

    @staticmethod
    def _extract_cat_controls_for_circuit(
        circuit: QuantumCircuit,
        support: Optional[set[str]] = None,
    ) -> list[int]:
        """
        提取具有CAT复用潜力的控制位：
        同一控制位在连续片段内作用到>=2个不同目标位；
        控制位被其他门触碰时会截断片段。
        """
        gate_support = support if support is not None else {"cx", "cz", "rzz"}
        symmetric_support = {"cz", "rzz"}
        active: dict[int, set[int]] = {}
        controls: set[int] = set()

        def _flush(ctrl: int) -> None:
            tgts = active.get(ctrl)
            if tgts is None:
                return
            if len(tgts) >= 2:
                controls.add(ctrl)
            active.pop(ctrl, None)

        for instruction in circuit:
            qids = [circuit.qubits.index(q) for q in instruction.qubits]
            if not qids:
                continue

            gate_name = instruction.operation.name
            is_supported_remote = (
                gate_name in gate_support and len(qids) == 2 and qids[0] != qids[1]
            )

            anchor_candidates: set[int] = set()
            if is_supported_remote:
                q1, q2 = qids
                if gate_name in symmetric_support:
                    anchor_candidates = {q1, q2}
                else:
                    anchor_candidates = {q1}

            touched = [ctrl for ctrl in list(active.keys()) if ctrl in qids]
            for ctrl in touched:
                if ctrl not in anchor_candidates:
                    _flush(ctrl)

            if is_supported_remote:
                q1, q2 = qids
                for ctrl in anchor_candidates:
                    tgt = q2 if ctrl == q1 else q1
                    if ctrl not in active:
                        active[ctrl] = set()
                    active[ctrl].add(tgt)

        for ctrl in list(active.keys()):
            _flush(ctrl)

        return sorted(controls)

    @staticmethod
    def evaluate_teledata(
        arg1: MappingRecord | list[list[int]],  # 兼容两种类型：prev_record / prev_partition
        arg2: MappingRecord | list[list[int]],  # 兼容两种类型：curr_record / curr_partition
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {}
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]]]:
        """
        计算切换划分的通信开销，支持两种输入格式：
        格式1：arg1=prev_record(MappingRecord), arg2=curr_record(MappingRecord), network
        格式2：arg1=prev_partition(list[list[int]]), arg2=curr_partition(list[list[int]]), network
        """
        # print(f"[DEBUG] evaluate_teledata")
        prev_record, curr_record = None, None
        prev_partition, curr_partition = None, None

        # ========== 第一步：类型判断 + 参数校验 ==========
        # 场景1：输入是 MappingRecord
        if isinstance(arg1, MappingRecord) and isinstance(arg2, MappingRecord):
            prev_record, curr_record = arg1, arg2
            # 提取 partition
            prev_partition = prev_record.partition
            curr_partition = curr_record.partition
            # 初始化logical_phy_map
            logical_phy_map = curr_record.logical_phy_map

            # 进入teledata前，logical_phy_map理论上应与上一片段分区一致
            ok, reason = CompilerUtils._check_logical_map_partition_consistency(prev_partition, logical_phy_map)
            if not ok:
                print(f"[MAP_CHECK][evaluate_teledata][pre-prev] {reason}", file=sys.stderr)
                print(f"[MAP_CHECK][prev_partition] {prev_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][logical_phy_map] {logical_phy_map}", file=sys.stderr)
                assert ok, reason

        # 场景2：输入是 list[list[int]]
        elif isinstance(arg1, list) and isinstance(arg2, list):
            prev_partition, curr_partition = arg1, arg2
            # # 检查logical_phy_map不为空
            # if len(logical_phy_map) == 0:
            #     raise ValueError("当输入为partition时，必须提供非空的logical_phy_map")

        # 场景3：类型不匹配（抛错提示）
        else:
            raise TypeError(
                "输入参数类型错误！仅支持两种格式：\n"
                "1. arg1=MappingRecord, arg2=MappingRecord\n"
                "2. arg1=list[list[int]], arg2=list[list[int]]"
            )

        G = nx.DiGraph() # 初始化有向图
        G.add_nodes_from(range(len(prev_partition))) # 每个partition对应一个节点

        costs = ExecCosts()

        # 记录每个qubit在prev和curr的分区号
        qubit_mapping = {}
        for pno, part in enumerate(prev_partition):
            # print(f"{pno}: {partition}")
            for qubit in part:
                qubit_mapping[qubit] = [pno, -1]
        for pno, part in enumerate(curr_partition):
            # print(f"{pno}: {partition}")
            for qubit in part:
                qubit_mapping[qubit][1] = pno

        # ---------- 第三步：构建流量图 (记录具体 Qubit) ----------
        # 我们不再只记录 weight，而是把具体的 qubit 塞进边的属性里
        # 边属性结构: {'qubits': [], 'weight': int}
        
        for qubit, (p_part, c_part) in qubit_mapping.items():
            if p_part == c_part:
                continue
            
            u, v = p_part, c_part
            
            # 检查是否存在反向边 (v, u)，如果有，则可以配对做 Swap
            if G.has_edge(v, u):
                # 取出一个反向移动的 qubit 作为交换对象
                # 注意：这里我们从图的边属性里 pop 一个出来
                if len(G[v][u]['qubits']) > 0:
                    swap_partner = G[v][u]['qubits'].pop(0)
                    G[v][u]['weight'] -= 1
                    
                    if G[v][u]['weight'] == 0:
                        G.remove_edge(v, u)
                    
                    # 1. 计算开销
                    costs = CompilerUtils.update_remote_swap_costs(costs, u, v, 1, network)
                    
                    # 2. [核心] 更新 logical_phy_map：交换这两个 qubit 的物理位置
                    if logical_phy_map:
                        logical_phy_map[qubit], logical_phy_map[swap_partner] = \
                        logical_phy_map[swap_partner], logical_phy_map[qubit]

                    continue # 处理完了，不用加边了

            # 如果不能抵消，添加正向边 (u, v)
            if G.has_edge(u, v):
                G[u][v]['qubits'].append(qubit)
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1, qubits=[qubit])

        # ---------- 第四步：处理大环 (Length >= 3) ----------
        all_cycles = nx.simple_cycles(G)
        cycles_by_length = defaultdict(list)
        # 收集长度大于2的环
        for cycle in all_cycles:
            length = len(cycle)
            assert(3 <= length <= network.num_backends)
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            for cycle in cycles_by_length[length]:
                # 检查环是否还存在，并找出最小权重
                min_weight = float('inf')
                valid = True
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i+1) % length]
                    if not G.has_edge(u, v):
                        valid = False
                        break
                    min_weight = min(min_weight, G[u][v]['weight']) # 记录环的个数
                
                if not valid: # 当前环不存在了
                    continue

                # 执行 min_weight 次循环交换
                for _ in range(int(min_weight)):
                    # 从环的每条边取出一个 qubit
                    cycle_qubits = []
                    for i in range(length):
                        u, v = cycle[i], cycle[(i + 1) % length]
                        q = G[u][v]['qubits'].pop(0)
                        cycle_qubits.append(q)
                    
                    # 循环移动
                    # 先保存第一个的位置
                    first_q = cycle_qubits[0]
                    first_pos = (-1, -1)

                    if logical_phy_map:
                        first_pos = logical_phy_map[first_q]
                    
                    # 依次后移
                    for i in range(length - 1):
                        curr_q = cycle_qubits[i]
                        next_q = cycle_qubits[i + 1]
                        # 把next的位置给curr
                        if logical_phy_map:
                            logical_phy_map[curr_q] = logical_phy_map[next_q]
                    
                    # 把最初的位置给最后一个
                    last_q = cycle_qubits[-1]
                    if logical_phy_map:
                        logical_phy_map[last_q] = first_pos

                # 更新图权重
                for i in range(length): # 从G中移除这些环
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    G[u][v]['weight'] -= min_weight
                    if G[u][v]['weight'] == 0:
                        G.remove_edge(u, v)
                    # 对环中的每一条边，计算通信开销
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, u, v, int(min_weight), network
                    )

        # 获取剩余的边
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            # print(f"[DEBUG] QPU容量满，不应该有单向边，但发现了：{u} -> {v}，weight={data['weight']}")
            # exit(1)

            qubits_to_move = data['qubits']

            if logical_phy_map:
                for qubit in qubits_to_move:
                    logical_phy_map[qubit] = (v, None) # TODO: 移到了一个新的QPU上，物理位不固定

            costs = CompilerUtils.update_remote_move_costs(
                costs, u, v, data['weight'], network
            )

        if isinstance(arg2, MappingRecord):
            # teledata结束后，logical_phy_map应与当前分区一致
            ok, reason = CompilerUtils._check_logical_map_partition_consistency(curr_partition, logical_phy_map)
            if not ok:
                print(f"[MAP_CHECK][evaluate_teledata][post-curr] {reason}", file=sys.stderr)
                print(f"[MAP_CHECK][prev_partition] {prev_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][curr_partition] {curr_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][logical_phy_map] {logical_phy_map}", file=sys.stderr)
                assert ok, reason

            # 更新costs
            arg2.costs += costs
            arg2.logical_phy_map = logical_phy_map
            return arg2.costs, logical_phy_map

        return costs, logical_phy_map

    @staticmethod
    def update_remote_move_costs(costs: ExecCosts, src: int, dst: int, weight: int, network: Network):
        if src == dst:
            return costs

        hops = network.Hops[src][dst]
        
        costs.remote_hops += hops * weight
        costs.epairs += hops * weight
        costs.remote_fidelity_loss += network.move_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.move_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.move_fidelity[src][dst]) * weight

        return costs

    @staticmethod
    def update_remote_swap_costs(costs: ExecCosts, src: int, dst: int, weight: int, network: Network):
        if src == dst:
            return costs
        
        hops = network.Hops[src][dst]
        rswaps = 2 * hops - 1
        
        costs.remote_swaps += rswaps * weight
        costs.epairs += 2 * rswaps * weight
        costs.remote_fidelity_loss += network.swap_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.swap_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.swap_fidelity[src][dst]) * weight

        return costs

    # @staticmethod
    # def update_local_gate_costs_by_name(costs: ExecCosts, backend: Any, gate_name: str, weight: int):
    #     """
    #     按门类型（不区分具体物理位）累计本地门误差。
    #     用于CAT将跨分区门转成本地门时的保真度统计。
    #     """
    #     if weight <= 0:
    #         return costs

    #     gname = gate_name.lower()
    #     gate_entry = backend.gate_dict.get(gname, {})
    #     gate_error = gate_entry.get("gate_error_value", None)

    #     if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
    #         raise ValueError(f"Gate error for '{gate_name}' is not available.")

    #     gate_error = float(gate_error)
    #     gate_error = min(max(gate_error, 0.0), 0.99)

    #     costs.local_gate_num += weight
    #     costs.local_fidelity_loss += gate_error * weight
    #     costs.local_fidelity *= (1 - gate_error) ** weight
    #     costs.local_fidelity_log_sum += np.log(1 - gate_error) * weight
    #     return costs


    # 
    # 维护逻辑量子比特->QPU物理量子比特的稳定映射关系
    # 
    @staticmethod
    def init_logical_phy_map(partition: list[list[int]]) -> dict[int, tuple[int, int | None]]:
        """从初始分区初始化唯一字典，第一次transpile前用"""
        # TODO: 是否要对comm_slot也预留logical_phy_map
        logical_phy_map = {}
        for qpu_id, qubits in enumerate(partition):
            for qubit in qubits:
                # 初始只赋值tuple[0]，也就是QPU id
                # tuple[1]会在第一次transpile后会更新为真实物理位
                logical_phy_map[qubit] = (qpu_id, None)
        return logical_phy_map
    
    @staticmethod
    def get_logical_to_physical_map(
        transpiled_circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]]
    ) -> dict[int, tuple[int, int | None]]:
        """
        从transpiled电路中提取稳定的逻辑→物理比特映射
        """
        # 初始化：按当前partition实际使用的本地索引建表。
        # 注意：启用通信预留槽位后，本地索引可能是稀疏的（如 0,2,5）。
        local_slots = sorted({global_to_local_lqid[q] for q in partition_qubits})
        local_lqid_to_pqid = {slot: None for slot in local_slots}

        if hasattr(transpiled_circuit, "layout") and transpiled_circuit.layout is not None:
            layout = transpiled_circuit.layout.final_layout
            # print(f"\n[DEBUG] layout: {layout}\n", file=sys.stderr)
            if layout is None:
                layout = transpiled_circuit.layout.initial_layout
            if layout is None:
                raise RuntimeError("[ERROR] Layout is None")
            # [示例] layout: Layout({ 物理: 逻辑
            # 0: Qubit(QuantumRegister(2, 'q'), 0),
            # 1: Qubit(QuantumRegister(2, 'q'), 1),
            # 2: Qubit(QuantumRegister(1, 'ancilla'), 0)
            # })

            # print(f"[DEBUG] layout (phy->log): {layout}")

            phy_to_logic_dict = layout.get_physical_bits()
            # print(f"[phy_to_log] {phy_to_logic_dict}")

            # 遍历 Layout 字典
            # phy_qid: 物理比特编号 (int, 例如 0, 1, 2)
            # logic_qubit: 逻辑比特对象 (Qubit)
            for phy_qid, logic_qubit in phy_to_logic_dict.items():
                # # 获取逻辑比特所在的寄存器名字
                reg = getattr(logic_qubit, 'register', None)
                if reg is None:
                    reg = getattr(logic_qubit, '_register', None)
                
                # # 获取逻辑比特在原始寄存器里的索引
                # 尝试获取 .index，如果没有则获取 ._index
                logic_idx = getattr(logic_qubit, 'index', None)
                if logic_idx is None:
                    logic_idx = getattr(logic_qubit, '_index', None)
                
                # 安全检查：如果都获取失败，跳过
                if reg is None or logic_idx is None:
                    raise RuntimeError(f"[ERROR] Could not extract register info from qubit: {logic_qubit}")
                
                reg_name = reg.name
                
                # print(f"[DEBUG] Checking: Phy {phy_qid} <-> Logic ({reg_name}, {logic_idx})")

                # [关键筛选]
                # 只要寄存器名字是 'q' 的，我们就要；ancilla 直接忽略
                if reg_name == 'q':
                    # 这里的 logic_idx (例如 0 或 1) 正好对应子电路里的“本地索引”
                    # 因为我们的子电路原来就是 2 个比特，名字叫 'q'
                    
                    # 安全检查：确保这个索引在我们预期的索引集合内
                    if logic_idx in local_lqid_to_pqid:
                        local_lqid_to_pqid[logic_idx] = phy_qid
                #         print(f"  [OK] Mapped Local Logic {logic_idx} -> Physical {phy_qid}")
                #     else:
                #         print(f"  [WARN] Logic index {logic_idx} out of range for current partition")
                # else:
                #     print(f"  [SKIP] Ignoring ancilla/other register: {reg_name}")
        else: # Layout为None，返回1:1平凡映射
            num_qubits = transpiled_circuit.num_qubits
            local_lqid_to_pqid = {i: i for i in range(num_qubits)}
            raise ValueError("Layout is None")

        for global_q in partition_qubits:
            # 获取每个global逻辑比特对应的子线路本地逻辑比特索引
            local_idx = global_to_local_lqid[global_q]
            # 更新logical_phy_map，把物理比特索引填上
            phy_idx = local_lqid_to_pqid.get(local_idx)
            if phy_idx is None and 0 <= local_idx < transpiled_circuit.num_qubits:
                # 回退：如果layout未给出该位，保守使用本地索引自身。
                phy_idx = local_idx
            logical_phy_map[global_q] = (logical_phy_map[global_q][0], phy_idx)

        return logical_phy_map

    @staticmethod
    def get_initial_layout(
        circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]],
    ) -> dict:
        """
        构建QuantumCircuit Qubit Register到物理比特的初始布局。
        规则：
        1) 对应global_lqid已知preferred物理位的local_lqid，固定到该物理位；
        2) 其余local_lqid按顺序分配剩余物理位。
        """
        initial_layout: dict[Any, int] = {}

        anchored_local_to_phy: dict[int, int] = {}
        preferred_phys = {
            phy
            for q in partition_qubits
            for phy in [logical_phy_map[q][1]]
            if phy is not None and phy >= 0
        }
        max_phy = max(preferred_phys) if len(preferred_phys) > 0 else -1
        pool_upper = max(circuit.num_qubits - 1, max_phy)
        unused_phy_ids: set[int] = set(range(pool_upper + 1))

        # 先固定有preferred物理位的local_lqid。
        for q in partition_qubits:
            local_lqid = global_to_local_lqid[q]
            if not (0 <= local_lqid < circuit.num_qubits):
                raise RuntimeError(
                    f"[LAYOUT] local_lqid越界: local_lqid={local_lqid}, circuit_qubits={circuit.num_qubits}, q={q}"
                )

            preferred_phy = logical_phy_map[q][1]
            if preferred_phy is None:
                continue
            if preferred_phy < 0:
                raise RuntimeError(f"[LAYOUT] 非法preferred_phy: q={q}, phy={preferred_phy}")

            if local_lqid in anchored_local_to_phy and anchored_local_to_phy[local_lqid] != preferred_phy:
                raise RuntimeError(
                    f"[LAYOUT] 同一local_lqid出现冲突锚点: local_lqid={local_lqid}, "
                    f"phy1={anchored_local_to_phy[local_lqid]}, phy2={preferred_phy}"
                )

            anchored_local_to_phy[local_lqid] = preferred_phy

        used_anchored_phys: dict[int, int] = {}
        for local_lqid in sorted(anchored_local_to_phy.keys()):
            phy_id = anchored_local_to_phy[local_lqid]
            if phy_id in used_anchored_phys and used_anchored_phys[phy_id] != local_lqid:
                raise RuntimeError(
                    f"[LAYOUT] 锚点物理位冲突: phy={phy_id}, "
                    f"locals=({used_anchored_phys[phy_id]}, {local_lqid})"
                )

            initial_layout[circuit.qubits[local_lqid]] = phy_id
            used_anchored_phys[phy_id] = local_lqid
            unused_phy_ids.discard(phy_id)

        # 其余local_lqid按顺序分配剩余物理位。
        for local_lqid in range(circuit.num_qubits):
            qobj = circuit.qubits[local_lqid]
            if qobj in initial_layout:
                continue

            if len(unused_phy_ids) == 0:
                raise RuntimeError(
                    f"[LAYOUT] 无可用物理位给未分配local_lqid={local_lqid}"
                )

            phy_id = min(unused_phy_ids)
            initial_layout[qobj] = phy_id
            unused_phy_ids.discard(phy_id)

        return initial_layout
