from __future__ import annotations

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
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Gate

from ..utils import Network, log


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

    local_gate_num: int = 0    # transpile 后本地物理量子门总个数
    local_pre_transpile_gate_num: int = 0  # transpile 前进入本地子线路的量子门总数（不含barrier/delay）
    local_transpile_added_gate_num: int = 0  # transpile 后相对输入额外增加的本地门数
    local_payload_gate_num: int = 0  # payload/body buffer transpile 后本地物理门数
    local_comm_protocol_gate_num: int = 0  # CAT/CommOp协议 buffer transpile 后本地物理门数
    local_teledata_gate_num: int = 0  # teledata/compaction/landing buffer transpile 后本地物理门数
    local_uncategorized_gate_num: int = 0  # 旧评估路径或无法归因的本地物理门数
    payload_gate_num: int = 0  # transpile 前进入本地执行流程的 payload/body 门数
    flush_calls: int = 0       # _flush_local_subcircuits 调用次数（含空flush）
    nonempty_flushes: int = 0  # 实际触发了至少一个非空子线路结算的flush次数
    local_transpile_calls: int = 0  # 本地子线路 transpile 总次数（按QPU逐个累计）
    comm_block_events: int = 0  # CommOp 事件数量
    comm_block_remote_fidelity_loss: float = 0.0  # CommOp 对应远程链路损失
    comm_block_remote_fidelity_log_sum: float = 0.0  # CommOp 对应远程链路log保真度和
    telegate_exec_events: int = 0  # 普通跨QPU门按synthetic CommOp(cat)执行的事件数
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
    def local_gate_breakdown_num(self) -> int:
        return (
            self.local_payload_gate_num
            + self.local_comm_protocol_gate_num
            + self.local_teledata_gate_num
            + self.local_uncategorized_gate_num
        )

    @property
    def local_gate_breakdown_gap(self) -> int:
        return self.local_gate_num - self.local_gate_breakdown_num

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
            f"local_gate_num={self.local_gate_num}, "
            f"local_pre_transpile_gate_num={self.local_pre_transpile_gate_num}, "
            f"local_transpile_added_gate_num={self.local_transpile_added_gate_num}, "
            f"local_payload_gate_num={self.local_payload_gate_num}, "
            f"local_comm_protocol_gate_num={self.local_comm_protocol_gate_num}, "
            f"local_teledata_gate_num={self.local_teledata_gate_num}, "
            f"local_uncategorized_gate_num={self.local_uncategorized_gate_num}, "
            f"payload_gate_num={self.payload_gate_num}"
            # Debug fields kept for temporary diagnostics; hide from default pprint/terminal output.
            # f", flush_calls={self.flush_calls}"
            # f", nonempty_flushes={self.nonempty_flushes}"
            # f", local_transpile_calls={self.local_transpile_calls}"
            # f", comm_block_events={self.comm_block_events}"
            # f", comm_block_remote_fidelity_loss={self.comm_block_remote_fidelity_loss}"
            # f", comm_block_remote_fidelity_log_sum={self.comm_block_remote_fidelity_log_sum}"
            # f", telegate_exec_events={self.telegate_exec_events}"
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
        self.local_pre_transpile_gate_num += other.local_pre_transpile_gate_num
        self.local_transpile_added_gate_num += other.local_transpile_added_gate_num
        self.local_payload_gate_num += other.local_payload_gate_num
        self.local_comm_protocol_gate_num += other.local_comm_protocol_gate_num
        self.local_teledata_gate_num += other.local_teledata_gate_num
        self.local_uncategorized_gate_num += other.local_uncategorized_gate_num
        self.payload_gate_num += other.payload_gate_num
        self.flush_calls += other.flush_calls
        self.nonempty_flushes += other.nonempty_flushes
        self.local_transpile_calls += other.local_transpile_calls
        self.comm_block_events += other.comm_block_events
        self.comm_block_remote_fidelity_loss += other.comm_block_remote_fidelity_loss
        self.comm_block_remote_fidelity_log_sum += other.comm_block_remote_fidelity_log_sum
        self.telegate_exec_events += other.telegate_exec_events
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
            "local_gate_breakdown_num": self.local_gate_breakdown_num,
            "local_gate_breakdown_gap": self.local_gate_breakdown_gap,
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
            "local_pre_transpile_gate_num",
            "local_transpile_added_gate_num",
            "local_payload_gate_num",
            "local_comm_protocol_gate_num",
            "local_teledata_gate_num",
            "local_uncategorized_gate_num",
            "local_gate_breakdown_num",
            "local_gate_breakdown_gap",
            "payload_gate_num",
            # Debug fields kept for temporary diagnostics; hide from default CSV/JSON summaries.
            # "flush_calls",
            # "nonempty_flushes",
            # "local_transpile_calls",
            # "comm_block_events",
            # "comm_block_remote_fidelity_loss",
            # "comm_block_remote_fidelity_log_sum",
            # "telegate_exec_events",
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
    comm_phy_map: dict[int, list[int | None]] = dataclasses.field(default_factory=dict) # 每个QPU长期通信logical ancilla的物理位置
    # 可选字段：扩展信息（如额外配置、备注）
    extra_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 冻结模式下修改字段需用 object.__setattr__
        object.__setattr__(self, "partition", copy.deepcopy(self.partition))
        object.__setattr__(self, "costs", copy.deepcopy(self.costs))
        object.__setattr__(self, "logical_phy_map", copy.deepcopy(self.logical_phy_map))
        object.__setattr__(self, "comm_phy_map", copy.deepcopy(self.comm_phy_map))
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
            "comm_phy_map": self.comm_phy_map,
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

    def ensure_record_ops(self, circuit: QuantumCircuit, circuit_layers: list[list[Any]]) -> None:
        """
        Populate extra_info["ops"] for records that do not have it.
        """
        for record in self.records:
            extra_info = record.extra_info
            if extra_info is None:
                extra_info = {}
            if not isinstance(extra_info, dict):
                raise TypeError(f"record.extra_info must be dict or None, got {type(extra_info)}")
            if "ops" in extra_info:
                continue
            if record.layer_start < 0 or record.layer_end < 0:
                raise ValueError(
                    "record.layer_start/layer_end must be set to build ops when missing"
                )
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end,
            )
            extra_info["ops"] = subcircuit
            record.extra_info = extra_info
        return

    def save_records(self, filename: str, dump_type: str = "evaluated"):
        """
        将记录保存到文件，支持 JSON/CSV 格式
        Args:
            filename: 保存路径
            dump_type: "raw" 或 "evaluated"，raw 不写 costs/logical_phy_map/comm_phy_map
        """
        if not self.records:
            print("⚠️ 无映射记录可保存")
            return

        if dump_type not in {"raw", "evaluated"}:
            raise ValueError(f"Unsupported dump_type: {dump_type}")

        # 统一序列化：将 dataclass 转为字典（兼容可选字段 extra_info）
        # 将total_costs转为字典
        total_costs_dict = self.total_costs.to_dict()
        # 将每条记录转为字典
        records_dict = []
        for record in self.records:
            record_dict = record.to_dict()
            record_dict["extra_info"] = self._serialize_extra_info(record_dict.get("extra_info"))
            records_dict.append(record_dict)
        if dump_type == "raw":
            for record_dict in records_dict:
                record_dict.pop("costs", None)
                record_dict.pop("logical_phy_map", None)
                record_dict.pop("comm_phy_map", None)
        data_dict = {
            "dump_type": dump_type,
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

    @staticmethod
    def _serialize_extra_info(extra_info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if extra_info is None:
            return None
        if not isinstance(extra_info, dict):
            return extra_info
        copied = copy.deepcopy(extra_info)
        if "ops" in copied:
            copied["ops"] = MappingRecordList._encode_quantum_circuit(copied["ops"])
        return copied

    @staticmethod
    def _deserialize_extra_info(extra_info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if extra_info is None:
            return None
        if not isinstance(extra_info, dict):
            return extra_info
        copied = copy.deepcopy(extra_info)
        if "ops" in copied:
            copied["ops"] = MappingRecordList._decode_quantum_circuit(copied["ops"])
        return copied

    @staticmethod
    def _encode_quantum_circuit(circuit: Any) -> Any:
        if circuit is None:
            return None
        if isinstance(circuit, dict) and circuit.get("__type__") == "quantum_circuit":
            return circuit
        if not isinstance(circuit, QuantumCircuit):
            return circuit

        payload: dict[str, Any] = {
            "__type__": "quantum_circuit",
            "num_qubits": int(circuit.num_qubits),
            "num_clbits": int(circuit.num_clbits),
            "global_phase": MappingRecordList._serialize_param(getattr(circuit, "global_phase", 0.0)),
            "data": [],
        }
        for instruction in circuit.data:
            op = instruction.operation
            qargs = [int(circuit.find_bit(q).index) for q in instruction.qubits]
            cargs = [int(circuit.find_bit(c).index) for c in instruction.clbits]
            payload["data"].append(
                {
                    "op": MappingRecordList._encode_op(op),
                    "qargs": qargs,
                    "cargs": cargs,
                }
            )
        return payload

    @staticmethod
    def _decode_quantum_circuit(payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, QuantumCircuit):
            return payload
        if not isinstance(payload, dict) or payload.get("__type__") != "quantum_circuit":
            return payload

        num_qubits = int(payload.get("num_qubits", 0))
        num_clbits = int(payload.get("num_clbits", 0))
        circuit = QuantumCircuit(num_qubits, num_clbits)
        circuit.global_phase = MappingRecordList._deserialize_param(payload.get("global_phase", 0.0))
        for item in payload.get("data", []):
            op = MappingRecordList._decode_op(item.get("op"))
            qargs = [int(q) for q in item.get("qargs", [])]
            cargs = [int(c) for c in item.get("cargs", [])]
            if op is None:
                continue
            circuit.append(op, qargs, cargs)
        return circuit

    @staticmethod
    def _encode_op(op: Any) -> Any:
        if isinstance(op, CommOp):
            return {
                "__type__": "commop",
                "comm_type": str(op.comm_type),
                "source_qubit": int(op.source_qubit),
                "src_qpu": int(op.src_qpu),
                "dst_qpu": int(op.dst_qpu),
                "involved_qubits": [int(q) for q in op.involved_qubits],
                "gate_list": [MappingRecordList._encode_gate(g) for g in op.gate_list],
            }

        if isinstance(op, Gate):
            return MappingRecordList._encode_gate(op)

        return {
            "__type__": "opaque",
            "repr": str(op),
        }

    @staticmethod
    def _decode_op(data: Any) -> Any:
        if data is None:
            return None
        if isinstance(data, Gate):
            return data
        if not isinstance(data, dict):
            return None

        if data.get("__type__") == "commop":
            gate_list = [
                MappingRecordList._decode_gate(g)
                for g in data.get("gate_list", [])
                if g is not None
            ]
            return CommOp(
                comm_type=str(data.get("comm_type", "cat")),
                source_qubit=int(data.get("source_qubit", 0)),
                src_qpu=int(data.get("src_qpu", 0)),
                dst_qpu=int(data.get("dst_qpu", 0)),
                involved_qubits=[int(q) for q in data.get("involved_qubits", [])],
                gate_list=gate_list,
            )

        if data.get("__type__") == "gate":
            return MappingRecordList._decode_gate(data)

        return None

    @staticmethod
    def _encode_gate(op: Gate) -> dict[str, Any]:
        payload = {
            "__type__": "gate",
            "name": str(op.name),
            "num_qubits": int(op.num_qubits),
            "params": [MappingRecordList._serialize_param(p) for p in list(op.params or [])],
        }
        label = getattr(op, "label", None)
        if label is not None:
            payload["label"] = str(label)
        global_lqids = getattr(op, "_global_lqids", None)
        if global_lqids is None:
            global_lqids = getattr(op, "_autocomm_qids", None)
        if global_lqids is not None:
            payload["global_lqids"] = [int(q) for q in list(global_lqids)]
        return payload

    @staticmethod
    def _decode_gate(data: Any) -> Optional[Gate]:
        if data is None:
            return None
        if isinstance(data, Gate):
            return data
        if not isinstance(data, dict):
            return None
        name = str(data.get("name", ""))
        num_qubits = int(data.get("num_qubits", 0))
        params = [MappingRecordList._deserialize_param(p) for p in data.get("params", [])]
        if num_qubits <= 0:
            return None
        gate = Gate(name, num_qubits, params)
        label = data.get("label")
        if label is not None:
            gate.label = str(label)
        global_lqids = data.get("global_lqids")
        if global_lqids is not None:
            setattr(gate, "_global_lqids", [int(q) for q in list(global_lqids)])
        return gate

    @staticmethod
    def _serialize_param(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, complex):
            return {"__type__": "complex", "re": float(value.real), "im": float(value.imag)}
        if isinstance(value, (list, tuple)):
            return [MappingRecordList._serialize_param(v) for v in value]
        if isinstance(value, dict):
            return {k: MappingRecordList._serialize_param(v) for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"__type__": "repr", "value": str(value)}

    @staticmethod
    def _deserialize_param(value: Any) -> Any:
        if isinstance(value, dict) and value.get("__type__") == "complex":
            return complex(value.get("re", 0.0), value.get("im", 0.0))
        if isinstance(value, dict) and value.get("__type__") == "repr":
            return value.get("value")
        if isinstance(value, list):
            return [MappingRecordList._deserialize_param(v) for v in value]
        if isinstance(value, dict):
            return {k: MappingRecordList._deserialize_param(v) for k, v in value.items()}
        return value


class CompilerUtils:

    @staticmethod
    def build_circuit_layers(circuit: QuantumCircuit) -> list[list[Any]]:
        """
        Build circuit layers as lists of DAG op nodes.
        """
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())
        circuit_layers: list[list[Any]] = []
        for layer in layers:
            circuit_layers.append(list(layer["graph"].op_nodes()))
        return circuit_layers

    @staticmethod
    def resolve_evaluation_policy(policy_name: Optional[str] = None):
        from .evaluator import EvaluationPolicy

        name = str(policy_name or "full_realistic").lower()
        if name in {"local_all_to_all", "local"}:
            return EvaluationPolicy.local_all_to_all()
        if name in {"comm_to_all", "comm"}:
            return EvaluationPolicy.comm_to_all()
        return EvaluationPolicy.full_realistic()

    @staticmethod
    def evaluate_raw_mapping_records(
        mapping_record_list: MappingRecordList,
        network: Network,
        policy_name: Optional[str] = None,
    ) -> MappingRecordList:
        from .evaluator import MappingEvaluator

        evaluator = MappingEvaluator()
        policy = CompilerUtils.resolve_evaluation_policy(policy_name)
        local_eval_mode = str(getattr(network, "local_eval_mode", "immediate") or "immediate").lower()
        if local_eval_mode not in {"immediate", "deferred"}:
            raise ValueError(f"unknown local_eval_mode: {local_eval_mode}")
        policy = dataclasses.replace(
            policy,
            local_eval_mode=local_eval_mode,
            deferred_route_local_gates=bool(getattr(network, "deferred_route_local_gates", True)),
            deferred_initial_layout=str(getattr(network, "deferred_initial_layout", "fixed") or "fixed").lower(),
        )
        return evaluator.evaluate(mapping_record_list, network, policy)

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
        optimization_level: int = 3,
        strict_flush_on_remote: bool = True,
        flush_each_comm_gate: Optional[bool] = None,
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]], dict[int, list[int | None]]]:
        """
        按CommOp + 门序列统一评估通信与本地门成本
        """
        partition = None

        if isinstance(arg, MappingRecord):
            partition = arg.partition
            logical_phy_map = arg.logical_phy_map
            comm_phy_map = copy.deepcopy(getattr(arg, "comm_phy_map", {}) or {})
        else:
            partition = arg
            comm_phy_map = {}

        if len(logical_phy_map) == 0:
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)

        if bool(getattr(network, "sequential_initial_layout", False)):
            logical_phy_map = CompilerUtils.apply_sequential_initial_layout_if_unset(
                partition,
                logical_phy_map,
            )

        if flush_each_comm_gate is None:
            flush_each_comm_gate = bool(getattr(network, "flush_each_comm_gate", False))
        else:
            flush_each_comm_gate = bool(flush_each_comm_gate)

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
        debug_layout_tracking = bool(getattr(network, "debug_layout_tracking", False))
        skip_comm_payload_local_routing = bool(getattr(network, "skip_comm_payload_local_routing", False))
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

        normalized_comm_phy_map: dict[int, list[int | None]] = {}
        for qpu_id in range(len(partition)):
            saved = comm_phy_map.get(qpu_id, comm_phy_map.get(str(qpu_id), []))
            saved = list(saved) if saved is not None else []
            slots: list[int | None] = []
            for i in range(reserve):
                value = saved[i] if i < len(saved) else None
                slots.append(int(value) if value is not None else None)
            normalized_comm_phy_map[qpu_id] = slots
        comm_phy_map = normalized_comm_phy_map

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
            flush_each_gate: bool = False,
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
                if flush_each_gate:
                    _flush_local_subcircuits()

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
            costs.payload_gate_num += len(comm_op.gate_list)

            if comm_op.comm_type == "cat":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 1, network
                )
                costs.cat_ents += 1
                if not skip_comm_payload_local_routing:
                    src_slot = _find_free_slot(src_qpu, owner=comm_op.source_qubit)
                    dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                    try:
                        src_payload_slot = runtime_pos[comm_op.source_qubit][1]
                        subcircuits[src_qpu].cx(src_payload_slot, src_slot)

                        tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                        # Debug-only pre-transpile CommOp payload stats are disabled.
                        # _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                        _append_comm_gate_block(
                            dst_qpu,
                            comm_op.gate_list,
                            tmp_map,
                            flush_each_gate=flush_each_comm_gate,
                        )
                        subcircuits[dst_qpu].h(dst_slot)
                    finally:
                        _release_slot(src_qpu, src_slot, expected_owner=comm_op.source_qubit)
                        _release_slot(dst_qpu, dst_slot, expected_owner=comm_op.source_qubit)

            elif comm_op.comm_type == "rtp":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 2, network
                )
                if not skip_comm_payload_local_routing:
                    src_slot = _find_free_slot(src_qpu, owner=comm_op.source_qubit)
                    dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                    try:
                        src_payload_slot = runtime_pos[comm_op.source_qubit][1]
                        subcircuits[src_qpu].cx(src_payload_slot, src_slot)
                        subcircuits[src_qpu].h(src_payload_slot)

                        tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                        # Debug-only pre-transpile CommOp payload stats are disabled.
                        # _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                        _append_comm_gate_block(
                            dst_qpu,
                            comm_op.gate_list,
                            tmp_map,
                            flush_each_gate=flush_each_comm_gate,
                        )
                        dst_return_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                        subcircuits[dst_qpu].cx(dst_slot, dst_return_slot)
                        subcircuits[dst_qpu].h(dst_slot)
                        _release_slot(dst_qpu, dst_return_slot, expected_owner=comm_op.source_qubit)
                    finally:
                        _release_slot(src_qpu, src_slot, expected_owner=comm_op.source_qubit)
                        _release_slot(dst_qpu, dst_slot, expected_owner=comm_op.source_qubit)

            elif comm_op.comm_type == "tp":
                costs = CompilerUtils.update_remote_move_costs(
                    costs, src_qpu, dst_qpu, 1, network
                )
                if not skip_comm_payload_local_routing:
                    src_slot = _find_free_slot(src_qpu, owner=comm_op.source_qubit)
                    dst_slot = _find_free_slot(dst_qpu, owner=comm_op.source_qubit)
                    try:
                        src_payload_slot = runtime_pos[comm_op.source_qubit][1]
                        subcircuits[src_qpu].cx(src_payload_slot, src_slot)
                        subcircuits[src_qpu].h(src_payload_slot)

                        tmp_map = _build_tmp_local_map_for_comm(comm_op.source_qubit, dst_slot)
                        # Debug-only pre-transpile CommOp payload stats are disabled.
                        # _accumulate_comm_block_local_stats(dst_qpu, comm_op.gate_list, tmp_map, stats_prefix)
                        _append_comm_gate_block(
                            dst_qpu,
                            comm_op.gate_list,
                            tmp_map,
                            flush_each_gate=flush_each_comm_gate,
                        )
                    finally:
                        _release_slot(src_qpu, src_slot, expected_owner=comm_op.source_qubit)
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
            nonlocal logical_phy_map, comm_phy_map, subcircuits, costs
            costs.flush_calls += 1
            flushed_any = False

            for qpu_id in range(len(subcircuits)):
                subcircuit = subcircuits[qpu_id]
                backend = network.backends[qpu_id]

                if subcircuit.size() == 0:
                    continue

                flushed_any = True
                costs.local_transpile_calls += 1

                comm_initial_layout = {
                    comm_slot_start[qpu_id] + slot_offset: phy_id
                    for slot_offset, phy_id in enumerate(comm_phy_map[qpu_id])
                    if phy_id is not None
                }
                prev_comm_phy_snapshot = list(comm_phy_map[qpu_id])
                if debug_layout_tracking and any(p is not None for p in prev_comm_phy_snapshot) and len(comm_initial_layout) == 0:
                    print(
                        f"[LAYOUT_TRACK][WARN] qpu={qpu_id} comm initial layout unexpectedly empty; "
                        f"prev_comm_phy_map={prev_comm_phy_snapshot}",
                        # file=sys.stderr,
                    )
                if debug_layout_tracking:
                    print(
                        f"[LAYOUT_TRACK][PRE] qpu={qpu_id} flush={costs.flush_calls} "
                        f"subc_size={subcircuit.size()} comm_initial_layout={comm_initial_layout} "
                        f"prev_comm_phy_map={prev_comm_phy_snapshot}",
                        # file=sys.stderr,
                    )

                initial_layout = CompilerUtils.get_initial_layout(
                    subcircuit,
                    partition[qpu_id],
                    global_to_local_lqid,
                    logical_phy_map,
                    fixed_local_layout=comm_initial_layout,
                    physical_qubit_count=backend.num_qubits,
                    fill_unassigned=bool(getattr(network, "sequential_initial_layout", False)),
                )
                transpile_initial_layout = initial_layout if len(initial_layout) > 0 else None
                initial_layout_local_to_phy = (
                    None
                    if transpile_initial_layout is None
                    else {
                        int(subcircuit.qubits.index(qobj)): int(phy_id)
                        for qobj, phy_id in transpile_initial_layout.items()
                    }
                )

                transpiled_circuit = transpile(
                    subcircuit,
                    coupling_map=backend.coupling_map,
                    basis_gates=backend.basis_gates,
                    initial_layout=transpile_initial_layout,
                    optimization_level=optimization_level,
                    seed_transpiler=42,
                )

                logical_phy_map_before_update = copy.deepcopy(logical_phy_map)
                local_phy_map_after_transpile = CompilerUtils.get_local_to_physical_map(transpiled_circuit)
                bad_local_phy = {
                    int(local_lqid): int(phy_id)
                    for local_lqid, phy_id in local_phy_map_after_transpile.items()
                    if phy_id is not None and int(phy_id) >= int(backend.num_qubits)
                }
                if bad_local_phy:
                    log(
                        f"[LAYOUT_BAD_MAP] qpu={qpu_id} flush={costs.flush_calls} "
                        f"backend_num_qubits={backend.num_qubits} "
                        f"subcircuit_num_qubits={subcircuit.num_qubits} "
                        f"transpiled_num_qubits={transpiled_circuit.num_qubits} "
                        f"bad_local_phy={bad_local_phy} "
                        f"local_phy_map={local_phy_map_after_transpile}"
                    )
                    log(f"[LAYOUT_BAD_MAP][initial_layout_arg] {transpile_initial_layout}")
                    log(f"[LAYOUT_BAD_MAP][qiskit_initial_layout] {transpiled_circuit.layout.initial_layout}")
                    log(f"[LAYOUT_BAD_MAP][qiskit_final_layout] {transpiled_circuit.layout.final_layout}")
                    raise RuntimeError(
                        f"[LAYOUT] transpile layout越界: qpu={qpu_id}, "
                        f"bad_local_phy={bad_local_phy}, physical_qubits={backend.num_qubits}"
                    )

                logical_phy_map = CompilerUtils.get_logical_to_physical_map(
                    transpiled_circuit,
                    partition[qpu_id],
                    global_to_local_lqid,
                    logical_phy_map,
                )
                local_phy_map = local_phy_map_after_transpile
                for slot_offset in range(reserve):
                    local_slot = comm_slot_start[qpu_id] + slot_offset
                    phy_idx = local_phy_map.get(local_slot)
                    if phy_idx is None and 0 <= local_slot < transpiled_circuit.num_qubits:
                        phy_idx = local_slot
                    comm_phy_map[qpu_id][slot_offset] = phy_idx

                CompilerUtils._assert_no_resident_comm_phy_overlap(
                    "evaluate_local_and_telegate_with_cat:post-flush",
                    logical_phy_map,
                    comm_phy_map,
                    details={
                        "qpu_id": qpu_id,
                        "flush_call": costs.flush_calls,
                        "partition": partition[qpu_id],
                        "comm_slot_start": comm_slot_start[qpu_id],
                        "comm_initial_layout": comm_initial_layout,
                        "prev_comm_phy_map": prev_comm_phy_snapshot,
                        "initial_local_to_phy": initial_layout_local_to_phy,
                        "final_local_to_phy": local_phy_map_after_transpile,
                        "resident_after": {
                            q: logical_phy_map.get(q)
                            for q in partition[qpu_id]
                        },
                        "comm_after": comm_phy_map[qpu_id],
                    },
                )

                layout_trace_records = getattr(network, "layout_trace_records", None)
                if isinstance(layout_trace_records, list):
                    initial_local_to_phy: dict[int, int] = {}
                    for local_lqid, qobj in enumerate(subcircuit.qubits):
                        phy_id = initial_layout.get(qobj)
                        if phy_id is not None:
                            initial_local_to_phy[int(local_lqid)] = int(phy_id)
                    resident_before = {
                        int(q): list(logical_phy_map_before_update[q])
                        for q in partition[qpu_id]
                        if q in logical_phy_map_before_update
                    }
                    resident_after = {
                        int(q): list(logical_phy_map[q])
                        for q in partition[qpu_id]
                        if q in logical_phy_map
                    }
                    layout_trace_records.append({
                        "qpu_id": int(qpu_id),
                        "flush_call": int(costs.flush_calls),
                        "subcircuit_size": int(subcircuit.size()),
                        "subcircuit_depth": int(subcircuit.depth() or 0),
                        "subcircuit_ops": dict(subcircuit.count_ops()),
                        "transpiled_size": int(transpiled_circuit.size()),
                        "transpiled_depth": int(transpiled_circuit.depth() or 0),
                        "transpiled_ops": dict(transpiled_circuit.count_ops()),
                        "initial_local_to_phy": initial_local_to_phy,
                        "final_local_to_phy": {
                            int(k): (int(v) if v is not None else None)
                            for k, v in local_phy_map_after_transpile.items()
                        },
                        "comm_initial_layout": {
                            int(k): int(v)
                            for k, v in comm_initial_layout.items()
                        },
                        "comm_phy_map_before": [
                            int(v) if v is not None else None
                            for v in prev_comm_phy_snapshot
                        ],
                        "comm_phy_map_after": [
                            int(v) if v is not None else None
                            for v in comm_phy_map[qpu_id]
                        ],
                        "resident_logical_phy_before": resident_before,
                        "resident_logical_phy_after": resident_after,
                    })
                if debug_layout_tracking:
                    log(
                        f"[LAYOUT_TRACK][POST] qpu={qpu_id} flush={costs.flush_calls} "
                        f"updated_comm_phy_map={comm_phy_map[qpu_id]}",
                        # file=sys.stderr,
                    )

                for instruction in transpiled_circuit:
                    gate_name = instruction.operation.name
                    qubits = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
                    assert qubits[0] is not None, f"Qubit index is None for instruction: {instruction}"
                    gate_error = CompilerUtils._get_sampled_backend_gate_error(backend, gate_name, qubits)
                    costs.local_gate_num += 1
                    costs.local_uncategorized_gate_num += 1
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
                    costs.payload_gate_num += 1
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
            arg.comm_phy_map = comm_phy_map
            return arg.costs, logical_phy_map, comm_phy_map

        return costs, logical_phy_map, comm_phy_map

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

    # @staticmethod
    # def _extract_cat_controls_for_circuit(
    #     circuit: QuantumCircuit,
    #     support: Optional[set[str]] = None,
    # ) -> list[int]:
    #     """
    #     提取具有CAT复用潜力的控制位：
    #     同一控制位在连续片段内作用到>=2个不同目标位；
    #     控制位被其他门触碰时会截断片段。
    #     """
    #     gate_support = support if support is not None else {"cx", "cz", "rzz"}
    #     symmetric_support = {"cz", "rzz"}
    #     active: dict[int, set[int]] = {}
    #     controls: set[int] = set()

    #     def _flush(ctrl: int) -> None:
    #         tgts = active.get(ctrl)
    #         if tgts is None:
    #             return
    #         if len(tgts) >= 2:
    #             controls.add(ctrl)
    #         active.pop(ctrl, None)

    #     for instruction in circuit:
    #         qids = [circuit.qubits.index(q) for q in instruction.qubits]
    #         if not qids:
    #             continue

    #         gate_name = instruction.operation.name
    #         is_supported_remote = (
    #             gate_name in gate_support and len(qids) == 2 and qids[0] != qids[1]
    #         )

    #         anchor_candidates: set[int] = set()
    #         if is_supported_remote:
    #             q1, q2 = qids
    #             if gate_name in symmetric_support:
    #                 anchor_candidates = {q1, q2}
    #             else:
    #                 anchor_candidates = {q1}

    #         touched = [ctrl for ctrl in list(active.keys()) if ctrl in qids]
    #         for ctrl in touched:
    #             if ctrl not in anchor_candidates:
    #                 _flush(ctrl)

    #         if is_supported_remote:
    #             q1, q2 = qids
    #             for ctrl in anchor_candidates:
    #                 tgt = q2 if ctrl == q1 else q1
    #                 if ctrl not in active:
    #                     active[ctrl] = set()
    #                 active[ctrl].add(tgt)

    #     for ctrl in list(active.keys()):
    #         _flush(ctrl)

    #     return sorted(controls)

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
            if not getattr(curr_record, "comm_phy_map", None):
                curr_record.comm_phy_map = copy.deepcopy(getattr(prev_record, "comm_phy_map", {}) or {})

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
    def evaluate_teledata_with_local(
        arg1: MappingRecord | list[list[int]],
        arg2: MappingRecord | list[list[int]],
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]] = {},
        comm_phy_map: Optional[dict[int, list[int | None]]] = None,
    ) -> tuple[ExecCosts, dict[int, tuple[int, int | None]], dict[int, list[int | None]]]:
        """
        终态评估用的 teledata 成本模型。

        remote swap 使用 comp-to-comp remote SWAP primitive，不补 landing。
        remote move 使用 one-way teleportation primitive，teleport 后状态在目标
        QPU comm slot，因此补 comm slot -> 目标 comp physical qubit 的本地
        landing SWAP 成本。
        """
        if isinstance(arg1, MappingRecord) and isinstance(arg2, MappingRecord):
            prev_record, curr_record = arg1, arg2
            prev_partition = prev_record.partition
            curr_partition = curr_record.partition
            logical_phy_map = curr_record.logical_phy_map
            if not getattr(curr_record, "comm_phy_map", None):
                curr_record.comm_phy_map = copy.deepcopy(getattr(prev_record, "comm_phy_map", {}) or {})
            working_comm_phy_map: dict[int, list[int | None]] = copy.deepcopy(
                comm_phy_map
                if comm_phy_map is not None
                else (getattr(curr_record, "comm_phy_map", {}) or {})
            )

            ok, reason = CompilerUtils._check_logical_map_partition_consistency(prev_partition, logical_phy_map)
            if not ok:
                print(f"[MAP_CHECK][evaluate_teledata_with_local][pre-prev] {reason}", file=sys.stderr)
                print(f"[MAP_CHECK][prev_partition] {prev_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][logical_phy_map] {logical_phy_map}", file=sys.stderr)
                assert ok, reason
        elif isinstance(arg1, list) and isinstance(arg2, list):
            prev_partition, curr_partition = arg1, arg2
            working_comm_phy_map = copy.deepcopy(comm_phy_map or {})
        else:
            raise TypeError(
                "输入参数类型错误！仅支持两种格式：\n"
                "1. arg1=MappingRecord, arg2=MappingRecord\n"
                "2. arg1=list[list[int]], arg2=list[list[int]]"
            )

        if len(logical_phy_map) == 0:
            logical_phy_map = CompilerUtils.init_logical_phy_map(prev_partition)

        CompilerUtils._log_resident_comm_phy_overlap(
            "evaluate_teledata_with_local:entry",
            logical_phy_map,
            working_comm_phy_map,
        )
        CompilerUtils._assert_no_resident_comm_phy_overlap(
            "evaluate_teledata_with_local:entry",
            logical_phy_map,
            working_comm_phy_map,
            details={
                "prev_partition": prev_partition,
                "curr_partition": curr_partition,
            },
        )

        costs = ExecCosts()
        G = nx.DiGraph()
        G.add_nodes_from(range(len(prev_partition)))

        qubit_mapping: dict[int, list[int]] = {}
        for pno, part in enumerate(prev_partition):
            for qubit in part:
                qubit_mapping[qubit] = [pno, -1]
        for pno, part in enumerate(curr_partition):
            for qubit in part:
                qubit_mapping[qubit][1] = pno

        for qubit, (p_part, c_part) in qubit_mapping.items():
            if p_part == c_part:
                continue

            u, v = p_part, c_part
            if G.has_edge(v, u) and len(G[v][u]["qubits"]) > 0:
                swap_partner = G[v][u]["qubits"].pop(0)
                G[v][u]["weight"] -= 1
                if G[v][u]["weight"] == 0:
                    G.remove_edge(v, u)

                costs = CompilerUtils.update_remote_swap_costs(costs, u, v, 1, network)
                costs = CompilerUtils._add_remote_swap_local_protocol_cost(
                    costs,
                    network,
                    logical_phy_map,
                    working_comm_phy_map,
                    u,
                    v,
                    qubit,
                    swap_partner,
                )
                logical_phy_map[qubit], logical_phy_map[swap_partner] = \
                    logical_phy_map[swap_partner], logical_phy_map[qubit]
                CompilerUtils._assert_no_resident_comm_phy_overlap(
                    "evaluate_teledata_with_local:remote-swap",
                    logical_phy_map,
                    working_comm_phy_map,
                    details={
                        "qubit": qubit,
                        "swap_partner": swap_partner,
                        "edge": (u, v),
                        "prev_partition": prev_partition,
                        "curr_partition": curr_partition,
                    },
                )
                continue

            if G.has_edge(u, v):
                G[u][v]["qubits"].append(qubit)
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1, qubits=[qubit])

        cycles_by_length = defaultdict(list)
        for cycle in nx.simple_cycles(G):
            length = len(cycle)
            assert 3 <= length <= network.num_backends
            cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            for cycle in cycles_by_length[length]:
                min_weight = float("inf")
                valid = True
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    if not G.has_edge(u, v):
                        valid = False
                        break
                    min_weight = min(min_weight, G[u][v]["weight"])

                if not valid:
                    continue

                for _ in range(int(min_weight)):
                    cycle_qubits = []
                    for i in range(length):
                        u = cycle[i]
                        v = cycle[(i + 1) % length]
                        q = G[u][v]["qubits"].pop(0)
                        cycle_qubits.append(q)

                    old_positions = {q: logical_phy_map.get(q) for q in cycle_qubits}
                    stale_comm_overlaps = []
                    for q, pos in old_positions.items():
                        if pos is None:
                            continue
                        old_qpu, old_phy = pos
                        if old_phy is None:
                            continue
                        comm_phys = {
                            int(phy)
                            for phy in (working_comm_phy_map.get(int(old_qpu), []) or [])
                            if phy is not None
                        }
                        if int(old_phy) in comm_phys:
                            stale_comm_overlaps.append({
                                "q": q,
                                "qpu": int(old_qpu),
                                "phy": int(old_phy),
                                "comm_slots": list(working_comm_phy_map.get(int(old_qpu), []) or []),
                            })
                    if stale_comm_overlaps:
                        details = {
                            "cycle": cycle,
                            "cycle_qubits": cycle_qubits,
                            "old_positions": old_positions,
                            "comm_phy_map": working_comm_phy_map,
                            "overlaps": stale_comm_overlaps,
                        }
                        log(
                            f"[COMM_OVERLAP_PREEXISTING][evaluate_teledata_with_local:cycle-entry] "
                            f"{details}"
                        )
                        raise RuntimeError(
                            "[COMM_OVERLAP_PREEXISTING] cycle old comp position already overlaps "
                            f"comm slot before cycle execution: {details}"
                        )
                    new_positions: dict[int, tuple[int, int | None]] = {}
                    tentative_logical_phy_map = copy.deepcopy(logical_phy_map)

                    for i, q in enumerate(cycle_qubits):
                        dst_qpu = cycle[(i + 1) % length]
                        replaced_q = cycle_qubits[(i + 1) % length]
                        target_pos = old_positions.get(replaced_q)
                        # 寻找一个可用的comm qubit
                        dst_comm_phy = CompilerUtils._ensure_landing_comm_phy(
                            dst_qpu,
                            logical_phy_map,
                            working_comm_phy_map,
                            network,
                        )
                        # 寻找目标comp qubit
                        if target_pos is not None and target_pos[1] is not None:
                            dst_payload_phy = int(target_pos[1])
                        else:
                            dst_payload_phy = CompilerUtils._ensure_landing_comp_phy(
                                dst_qpu,
                                logical_phy_map=tentative_logical_phy_map,
                                comm_phy_map=working_comm_phy_map,
                                network=network,
                            )
                        if dst_payload_phy == dst_comm_phy:
                            details = (
                                f"[TELEDATA_ALLOC_DEBUG][cycle] payload phy equals comm phy: "
                                f"q={q}, dst_qpu={dst_qpu}, dst_payload_phy={dst_payload_phy}, "
                                f"dst_comm_phy={dst_comm_phy}, target_pos={target_pos}, "
                                f"comm_slots={working_comm_phy_map.get(dst_qpu)}, "
                                f"old_positions={old_positions}"
                            )
                            log(details)
                            raise RuntimeError(details)
                        costs = CompilerUtils._add_remote_move_local_protocol_cost(
                            costs,
                            network,
                            logical_phy_map,
                            working_comm_phy_map,
                            old_qpu=int(old_positions[q][0]),
                            qubit=q,
                        )
                        costs = CompilerUtils._add_landing_swap_local_cost(
                            costs,
                            network.backends[dst_qpu],
                            dst_comm_phy,
                            dst_payload_phy,
                        )
                        new_positions[q] = (dst_qpu, dst_payload_phy)
                        tentative_logical_phy_map[q] = (dst_qpu, dst_payload_phy)
                        CompilerUtils._assert_no_resident_comm_phy_overlap(
                            "evaluate_teledata_with_local:cycle-tentative",
                            tentative_logical_phy_map,
                            working_comm_phy_map,
                            details={
                                "q": q,
                                "cycle": cycle,
                                "cycle_qubits": cycle_qubits,
                                "dst_qpu": dst_qpu,
                                "replaced_q": replaced_q,
                                "target_pos": target_pos,
                                "dst_payload_phy": dst_payload_phy,
                                "dst_comm_phy": dst_comm_phy,
                                "old_positions": old_positions,
                                "new_positions": new_positions,
                            },
                        )

                    for q, new_pos in new_positions.items():
                        logical_phy_map[q] = new_pos
                    CompilerUtils._assert_no_resident_comm_phy_overlap(
                        "evaluate_teledata_with_local:cycle-commit",
                        logical_phy_map,
                        working_comm_phy_map,
                        details={
                            "cycle": cycle,
                            "cycle_qubits": cycle_qubits,
                            "old_positions": old_positions,
                            "new_positions": new_positions,
                        },
                    )

                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    G[u][v]["weight"] -= min_weight
                    if G[u][v]["weight"] == 0:
                        G.remove_edge(u, v)
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, u, v, int(min_weight), network
                    )

        for u, v, data in list(G.edges(data=True)):
            qubits_to_move = data["qubits"]

            if logical_phy_map:
                for qubit in qubits_to_move:
                    dst_payload_phy = CompilerUtils._ensure_landing_comp_phy(
                        v,
                        logical_phy_map=logical_phy_map,
                        comm_phy_map=working_comm_phy_map,
                        network=network,
                    )
                    dst_comm_phy = CompilerUtils._ensure_landing_comm_phy(
                        v,
                        logical_phy_map,
                        working_comm_phy_map,
                        network,
                    )
                    if dst_payload_phy == dst_comm_phy:
                        details = (
                            f"[TELEDATA_ALLOC_DEBUG][single_move] payload phy equals comm phy: "
                            f"q={qubit}, edge=({u}->{v}), dst_payload_phy={dst_payload_phy}, "
                            f"dst_comm_phy={dst_comm_phy}, comm_slots={working_comm_phy_map.get(v)}, "
                            f"logical_pos_before={logical_phy_map.get(qubit)}"
                        )
                        log(details)
                        raise RuntimeError(details)
                    costs = CompilerUtils._add_remote_move_local_protocol_cost(
                        costs,
                        network,
                        logical_phy_map,
                        working_comm_phy_map,
                        old_qpu=u,
                        qubit=qubit,
                    )
                    costs = CompilerUtils._add_landing_swap_local_cost(
                        costs,
                        network.backends[v],
                        dst_comm_phy,
                        dst_payload_phy,
                    )
                    logical_phy_map[qubit] = (v, dst_payload_phy)
                    CompilerUtils._assert_no_resident_comm_phy_overlap(
                        "evaluate_teledata_with_local:single-move",
                        logical_phy_map,
                        working_comm_phy_map,
                        details={
                            "qubit": qubit,
                            "edge": (u, v),
                            "dst_payload_phy": dst_payload_phy,
                            "dst_comm_phy": dst_comm_phy,
                            "comm_slots": working_comm_phy_map.get(v),
                        },
                    )

            costs = CompilerUtils.update_remote_move_costs(
                costs, u, v, data["weight"], network
            )

        CompilerUtils._log_resident_comm_phy_overlap(
            "evaluate_teledata_with_local:return",
            logical_phy_map,
            working_comm_phy_map,
        )

        if isinstance(arg2, MappingRecord):
            ok, reason = CompilerUtils._check_logical_map_partition_consistency(curr_partition, logical_phy_map)
            if not ok:
                print(f"[MAP_CHECK][evaluate_teledata_with_local][post-curr] {reason}", file=sys.stderr)
                print(f"[MAP_CHECK][prev_partition] {prev_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][curr_partition] {curr_partition}", file=sys.stderr)
                print(f"[MAP_CHECK][logical_phy_map] {logical_phy_map}", file=sys.stderr)
                assert ok, reason

            arg2.costs += costs
            arg2.logical_phy_map = logical_phy_map
            arg2.comm_phy_map = copy.deepcopy(working_comm_phy_map)
            return arg2.costs, logical_phy_map, working_comm_phy_map

        return costs, logical_phy_map, working_comm_phy_map
    
    @staticmethod
    def _log_resident_comm_phy_overlap(
        where: str,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: Optional[dict[int, list[int | None]]],
    ) -> None:
        for qpu_id, slots in (comm_phy_map or {}).items():
            qpu_id_int = int(qpu_id)
            comm_phys = {
                int(phy_id)
                for phy_id in (slots or [])
                if phy_id is not None
            }
            if len(comm_phys) == 0:
                continue

            for logical_q, pos in logical_phy_map.items():
                if pos is None:
                    continue
                mapped_qpu, phy_id = pos
                if phy_id is None:
                    continue
                if int(mapped_qpu) == qpu_id_int and int(phy_id) in comm_phys:
                    log(
                        f"[COMM_OVERLAP_DEBUG][{where}] qpu={qpu_id_int}, "
                        f"phy={int(phy_id)}, resident_global={logical_q}, "
                        f"resident_pos={pos}, comm_slots={slots}"
                    )

    @staticmethod
    def _find_resident_comm_phy_overlaps(
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: Optional[dict[int, list[int | None]]],
    ) -> list[dict[str, Any]]:
        overlaps: list[dict[str, Any]] = []
        for qpu_id, slots in (comm_phy_map or {}).items():
            qpu_id_int = int(qpu_id)
            comm_phys = {
                int(phy_id)
                for phy_id in (slots or [])
                if phy_id is not None
            }
            if len(comm_phys) == 0:
                continue

            for logical_q, pos in logical_phy_map.items():
                if pos is None:
                    continue
                mapped_qpu, phy_id = pos
                if phy_id is None:
                    continue
                if int(mapped_qpu) == qpu_id_int and int(phy_id) in comm_phys:
                    overlaps.append({
                        "qpu": qpu_id_int,
                        "phy": int(phy_id),
                        "resident_global": logical_q,
                        "resident_pos": pos,
                        "comm_slots": list(slots or []),
                    })
        return overlaps

    @staticmethod
    def _assert_no_resident_comm_phy_overlap(
        where: str,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: Optional[dict[int, list[int | None]]],
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        overlaps = CompilerUtils._find_resident_comm_phy_overlaps(
            logical_phy_map,
            comm_phy_map,
        )
        if len(overlaps) == 0:
            return

        log(
            f"[COMM_OVERLAP_ASSERT][{where}] overlaps={overlaps}, "
            f"details={details or {}}"
        )
        raise RuntimeError(
            f"[COMM_OVERLAP] resident physical slot overlaps comm slot at {where}: "
            f"{overlaps}"
        )

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
        # rswaps = 2 * hops - 1
        
        costs.remote_swaps += 2 * hops * weight # rswaps * weight
        costs.epairs += 2 * hops * weight # 2 * rswaps * weight
        costs.remote_fidelity_loss += network.swap_fidelity_loss[src][dst] * weight
        costs.remote_fidelity *= network.swap_fidelity[src][dst] ** weight
        costs.remote_fidelity_log_sum += np.log(network.swap_fidelity[src][dst]) * weight

        return costs

    @staticmethod
    def _get_sampled_backend_gate_error(backend: Any, gate_name: str, qubits: list[int]) -> float:
        gate_key = f"{gate_name}{'_'.join(map(str, qubits))}"
        gate_error = backend.gate_dict.get(gate_key, {}).get("gate_error_value", None)
        if gate_error == 1:
            gate_error = 0.99
            print(f"[WARNING] {gate_key}: {gate_error}")

        if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
            gate_error = backend.gate_dict.get(gate_name, {}).get("gate_error_value", None)

        # Virtual phase gates can appear in transpiled/synthesized local blocks even
        # when the backend calibration table only contains rz-like primitives.
        if gate_error is None or (isinstance(gate_error, float) and math.isnan(gate_error)):
            if len(qubits) == 1 and gate_name in {"z", "s", "sdg", "t", "tdg", "p", "u1"}:
                print(f"[WARNING] Missing calibrated error for virtual gate {gate_name}; assuming 0.0")
                gate_error = 0.0

        assert gate_error is not None, f"Gate error not found for gate_key: {gate_key} in backend.gate_dict"
        gate_error = float(gate_error)
        return min(max(gate_error, 0.0), 0.99)

    @staticmethod
    def _accumulate_local_transpiled_gate_costs(
        costs: ExecCosts,
        transpiled_circuit: QuantumCircuit,
        backend: Any,
    ) -> ExecCosts:
        for instruction in transpiled_circuit:
            gate_name = instruction.operation.name
            qubits = [transpiled_circuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qubits) == 0:
                continue
            gate_error = CompilerUtils._get_sampled_backend_gate_error(backend, gate_name, qubits)
            costs.local_gate_num += 1
            costs.local_uncategorized_gate_num += 1
            costs.local_fidelity_loss += gate_error
            costs.local_fidelity *= (1 - gate_error)
            costs.local_fidelity_log_sum += np.log(1 - gate_error)
        return costs

    @staticmethod
    def _occupied_phys_on_qpu(
        qpu_id: int,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
    ) -> set[int]:

        occupied: set[int] = CompilerUtils._resident_phys_on_qpu(qpu_id, logical_phy_map)

        for phy_id in comm_phy_map.get(qpu_id, []) or []:
            if phy_id is not None:
                occupied.add(int(phy_id))
        return occupied

    @staticmethod
    def _resident_phys_on_qpu(
        qpu_id: int,
        logical_phy_map: dict[int, tuple[int, int | None]],
    ) -> set[int]:
        occupied: set[int] = set()
        for _, pos in logical_phy_map.items():
            if pos is None:
                continue
            mapped_qpu, phy_id = pos
            if mapped_qpu == qpu_id and phy_id is not None:
                occupied.add(int(phy_id))
        return occupied

    @staticmethod
    def _choose_smallest_free_phy(
        qpu_id: int,
        occupied: set[int],
        network: Network,
    ) -> int:
        for phy_id in range(network.backends[qpu_id].num_qubits):
            if phy_id not in occupied:
                return int(phy_id)

        raise RuntimeError(
            f"[TELEDATA_LOCAL] QPU {qpu_id} has no available physical qubit; "
            f"occupied={sorted(occupied)}, backend_qubits={network.backends[qpu_id].num_qubits}"
        )

    @staticmethod
    def _ensure_landing_comm_phy(
        qpu_id: int,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        network: Network,
    ) -> int | None:

        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        if reserve <= 0:
            raise ValueError(
                f"network.comm_slot_reserve must be positive for teledata landing evaluation, got {reserve}"
            )

        saved = list(comm_phy_map.get(qpu_id, []))
        if len(saved) < reserve:
            saved.extend([None] * (reserve - len(saved)))
        comm_phy_map[qpu_id] = saved

        if saved and saved[0] is not None:
            return int(saved[0])

        occupied = CompilerUtils._occupied_phys_on_qpu(qpu_id, logical_phy_map, comm_phy_map)
        phy_id = CompilerUtils._choose_smallest_free_phy(qpu_id, occupied, network)
        saved[0] = int(phy_id)
        comm_phy_map[qpu_id] = saved
        return int(phy_id)

    @staticmethod
    def _ensure_landing_comm_phys(
        qpu_id: int,
        count: int,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        network: Network,
    ) -> list[int]:
        reserve = int(getattr(network, "comm_slot_reserve", 0) or 0)
        if reserve < count:
            raise ValueError(
                f"network.comm_slot_reserve must be at least {count} for complete remote swap, got {reserve}"
            )

        saved = list(comm_phy_map.get(qpu_id, []))
        if len(saved) < reserve:
            saved.extend([None] * (reserve - len(saved)))
        comm_phy_map[qpu_id] = saved

        for idx in range(count):
            if saved[idx] is not None:
                continue
            occupied = CompilerUtils._occupied_phys_on_qpu(qpu_id, logical_phy_map, comm_phy_map)
            phy_id = CompilerUtils._choose_smallest_free_phy(qpu_id, occupied, network)
            saved[idx] = int(phy_id)

        result: list[int] = []
        for idx in range(count):
            value = saved[idx]
            if value is None:
                raise RuntimeError(
                    f"[TELEDATA_LOCAL] comm physical slot was not allocated: qpu={qpu_id}, idx={idx}"
                )
            result.append(int(value))
        return result

    @staticmethod
    def _ensure_landing_comp_phy(
        qpu_id: int,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        network: Network,
    ) -> int:
        occupied = CompilerUtils._occupied_phys_on_qpu(qpu_id, logical_phy_map, comm_phy_map)
        return CompilerUtils._choose_smallest_free_phy(qpu_id, occupied, network)

    @staticmethod
    def _add_remote_swap_endpoint_local_cost(
        costs: ExecCosts,
        backend: Any,
        payload_phy: int,
        comm1_phy: int,
        comm2_phy: int,
        *,
        is_alice: bool,
    ) -> ExecCosts:
        qc = QuantumCircuit(3)
        comm1 = 0
        payload = 1
        comm2 = 2

        if is_alice:
            qc.cx(comm1, payload)
            qc.cx(payload, comm2)
        else:
            qc.cx(payload, comm2)
            qc.cx(comm1, payload)

        initial_layout = {
            qc.qubits[payload]: int(payload_phy),
            qc.qubits[comm1]: int(comm1_phy),
            qc.qubits[comm2]: int(comm2_phy),
        }
        transpiled_circuit = transpile(
            qc,
            coupling_map=backend.coupling_map,
            basis_gates=backend.basis_gates,
            initial_layout=initial_layout,
            optimization_level=0,
            seed_transpiler=42,
        )
        return CompilerUtils._accumulate_local_transpiled_gate_costs(
            costs,
            transpiled_circuit,
            backend,
        )

    @staticmethod
    def _add_complete_remote_swap_edge_local_cost(
        costs: ExecCosts,
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        left_qpu: int,
        right_qpu: int,
        left_payload_phy: int,
        right_payload_phy: int,
    ) -> ExecCosts:
        left_comm1, left_comm2 = CompilerUtils._ensure_landing_comm_phys(
            left_qpu,
            2,
            logical_phy_map,
            comm_phy_map,
            network,
        )
        right_comm1, right_comm2 = CompilerUtils._ensure_landing_comm_phys(
            right_qpu,
            2,
            logical_phy_map,
            comm_phy_map,
            network,
        )

        costs = CompilerUtils._add_remote_swap_endpoint_local_cost(
            costs,
            network.backends[left_qpu],
            left_payload_phy,
            left_comm1,
            left_comm2,
            is_alice=True,
        )
        costs = CompilerUtils._add_remote_swap_endpoint_local_cost(
            costs,
            network.backends[right_qpu],
            right_payload_phy,
            right_comm1,
            right_comm2,
            is_alice=False,
        )
        return costs

    @staticmethod
    def _add_remote_swap_local_protocol_cost(
        costs: ExecCosts,
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        src_qpu: int,
        dst_qpu: int,
        src_qubit: int,
        dst_qubit: int,
    ) -> ExecCosts:
        if src_qpu == dst_qpu:
            return costs

        src_pos = logical_phy_map.get(src_qubit)
        dst_pos = logical_phy_map.get(dst_qubit)
        if src_pos is None or dst_pos is None or src_pos[1] is None or dst_pos[1] is None:
            return costs

        return CompilerUtils._add_complete_remote_swap_edge_local_cost(
            costs,
            network,
            logical_phy_map,
            comm_phy_map,
            src_qpu,
            dst_qpu,
            int(src_pos[1]),
            int(dst_pos[1]),
        )

    @staticmethod
    def _add_remote_move_local_protocol_cost(
        costs: ExecCosts,
        network: Network,
        logical_phy_map: dict[int, tuple[int, int | None]],
        comm_phy_map: dict[int, list[int | None]],
        old_qpu: int,
        qubit: int,
    ) -> ExecCosts:
        src_pos = logical_phy_map.get(qubit)
        if src_pos is None or src_pos[1] is None or int(src_pos[0]) != int(old_qpu):
            return costs

        src_comm_phy = CompilerUtils._ensure_landing_comm_phy(
            old_qpu,
            logical_phy_map,
            comm_phy_map,
            network,
        )
        if src_comm_phy is None:
            return costs

        qc = QuantumCircuit(2)
        payload = 0
        comm = 1
        qc.cx(payload, comm)
        qc.h(payload)

        initial_layout = {
            qc.qubits[payload]: int(src_pos[1]),
            qc.qubits[comm]: int(src_comm_phy),
        }
        transpiled_circuit = transpile(
            qc,
            coupling_map=network.backends[old_qpu].coupling_map,
            basis_gates=network.backends[old_qpu].basis_gates,
            initial_layout=initial_layout,
            optimization_level=0,
            seed_transpiler=42,
        )
        return CompilerUtils._accumulate_local_transpiled_gate_costs(
            costs,
            transpiled_circuit,
            network.backends[old_qpu],
        )

    @staticmethod
    def _add_landing_swap_local_cost(
        costs: ExecCosts,
        backend: Any,
        comm_phy: int | None,
        payload_phy: int | None,
    ) -> ExecCosts:
        if comm_phy is None or payload_phy is None or comm_phy == payload_phy:
            return costs

        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        initial_layout = {
            qc.qubits[0]: int(comm_phy),
            qc.qubits[1]: int(payload_phy),
        }
        transpiled_circuit = transpile(
            qc,
            coupling_map=backend.coupling_map,
            basis_gates=backend.basis_gates,
            initial_layout=initial_layout,
            optimization_level=0,
            seed_transpiler=42,
        )
        return CompilerUtils._accumulate_local_transpiled_gate_costs(
            costs,
            transpiled_circuit,
            backend,
        )

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
        logical_phy_map = {}
        for qpu_id, qubits in enumerate(partition):
            for qubit in qubits:
                # 初始只赋值tuple[0]，也就是QPU id
                # tuple[1]会在第一次transpile后会更新为真实物理位
                logical_phy_map[qubit] = (qpu_id, None)
        return logical_phy_map

    @staticmethod
    def apply_sequential_initial_layout_if_unset(
        partition: list[list[int]],
        logical_phy_map: dict[int, tuple[int, int | None]],
    ) -> dict[int, tuple[int, int | None]]:
        """
        若当前分区尚未建立任何resident物理位映射，则按partition内顺序显式初始化为0..n-1。
        已有部分物理映射时不改动，避免覆盖teledata或前一record继承下来的布局。
        """
        resident_qubits = [qubit for group in partition for qubit in group]
        if any(logical_phy_map[qubit][1] is not None for qubit in resident_qubits):
            return logical_phy_map

        for qpu_id, qubits in enumerate(partition):
            for local_idx, qubit in enumerate(qubits):
                logical_phy_map[qubit] = (qpu_id, local_idx)
        return logical_phy_map
    
    @staticmethod
    def get_logical_to_physical_map(
        transpiled_circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]],
    ) -> dict[int, tuple[int, int | None]]:
        """
        从transpiled电路中提取稳定的逻辑→物理比特映射
        """
        # 初始化：按当前partition实际使用的本地索引建表。
        # 注意：启用通信预留槽位后，本地索引可能是稀疏的（如 0,2,5）。
        local_slots = sorted({global_to_local_lqid[q] for q in partition_qubits})
        all_local_lqid_to_pqid = CompilerUtils.get_local_to_physical_map(transpiled_circuit)
        local_lqid_to_pqid = {
            slot: all_local_lqid_to_pqid.get(slot)
            for slot in local_slots
        }

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
    def get_local_to_physical_map(
        transpiled_circuit: QuantumCircuit,
    ) -> dict[int, int | None]:
        """
        从transpiled电路中提取子线路local logical slot到物理位的映射。
        """
        local_lqid_to_pqid = {i: None for i in range(transpiled_circuit.num_qubits)}

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

        return local_lqid_to_pqid

    @staticmethod
    def get_initial_layout(
        circuit: QuantumCircuit,
        partition_qubits: list[int],
        global_to_local_lqid: dict[int, int],
        logical_phy_map: dict[int, tuple[int, int | None]],
        fixed_local_layout: Optional[dict[int, int]] = None,
        physical_qubit_count: Optional[int] = None,
        fill_unassigned: bool = True,
    ) -> dict:
        """
        构建QuantumCircuit Qubit Register到物理比特的初始布局。
        规则：
        1) 对应global_lqid已知preferred物理位的local_lqid，固定到该物理位；
        2) fixed_local_layout中的local_lqid固定到指定物理位（用于通信槽）；
        3) fill_unassigned=True时，其余local_lqid按顺序分配剩余物理位。
        """
        initial_layout: dict[Any, int] = {}
        fixed_local_layout = fixed_local_layout or {}
        physical_qubit_count = physical_qubit_count or circuit.num_qubits

        anchored_local_to_phy: dict[int, int] = {}
        fixed_phys = set(fixed_local_layout.values())
        out_of_range_fixed = [
            (local_lqid, phy_id)
            for local_lqid, phy_id in fixed_local_layout.items()
            if not (0 <= local_lqid < circuit.num_qubits and 0 <= phy_id < physical_qubit_count)
        ]
        if out_of_range_fixed:
            raise RuntimeError(
                f"[LAYOUT] fixed_local_layout越界: {out_of_range_fixed}, "
                f"circuit_qubits={circuit.num_qubits}, physical_qubits={physical_qubit_count}"
            )

        unused_phy_ids: set[int] = set(range(physical_qubit_count))
        unused_phy_ids -= fixed_phys

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
            if preferred_phy >= physical_qubit_count:
                raise RuntimeError(
                    f"[LAYOUT] preferred_phy越界: q={q}, phy={preferred_phy}, "
                    f"physical_qubits={physical_qubit_count}"
                )
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

        # 固定通信槽等非resident local slot。
        for local_lqid, phy_id in sorted(fixed_local_layout.items()):
            qobj = circuit.qubits[local_lqid]
            if qobj in initial_layout and initial_layout[qobj] != phy_id:
                raise RuntimeError(
                    f"[LAYOUT] fixed local slot与已有layout冲突: local_lqid={local_lqid}, "
                    f"existing_phy={initial_layout[qobj]}, fixed_phy={phy_id}"
                )
            if phy_id in used_anchored_phys and used_anchored_phys[phy_id] != local_lqid:
                resident_local = used_anchored_phys[phy_id]
                resident_global = None
                for global_lqid, mapped_local_lqid in global_to_local_lqid.items():
                    if mapped_local_lqid == resident_local:
                        resident_global = global_lqid
                        break
                log(
                    f"[LAYOUT_CONFLICT] fixed physical slot冲突: phy={phy_id}, "
                    f"resident_local={resident_local}, resident_global={resident_global}, "
                    f"resident_pos={logical_phy_map.get(resident_global)}, "
                    f"fixed_local={local_lqid}, "
                    f"fixed_local_layout={fixed_local_layout}, anchored_local_to_phy={anchored_local_to_phy}"
                )
                raise RuntimeError(
                    f"[LAYOUT] fixed physical slot与resident锚点冲突: phy={phy_id}, "
                    f"resident_local={resident_local}, fixed_local={local_lqid}"
                )
            initial_layout[qobj] = phy_id
            used_anchored_phys[phy_id] = local_lqid
            unused_phy_ids.discard(phy_id)

        if fill_unassigned:
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
