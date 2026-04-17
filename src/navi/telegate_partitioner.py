from abc import ABC, abstractmethod
from typing import Any, Optional
import time
import copy
import sys
from qiskit import QuantumCircuit
from qiskit import transpile

from ..utils import Network, log
from ..compiler import MappingRecord, CompilerUtils, MappingRecordList
from ..compiler.compiler_utils import CommOp
from ..baselines import OEE

class TelegatePartitioner(ABC):
    """
    基于telegate的划分器接口
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """获取划分器名称"""
        pass

    @abstractmethod
    def partition(self,
                  circuit: QuantumCircuit,
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord | MappingRecordList:
        """
        执行图划分
        :param circuit: 待划分的电路
        :param network: 网络拓扑信息
        :param config: 划分配置参数
        :return: 包含划分结果和代价的 MappingRecord
        """
        pass


class DirectTelegatePartitioner(TelegatePartitioner):
    """
    直接使用telegate进行划分的实现类
    """

    @property
    def name(self) -> str:
        return "DirectTelegatePartitioner"

    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:
        
        # 直接沿用config里面的partition
        partition = config.get("partition", []) if config else []
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1
        
        # 如果config里没有partition，就按顺序分配
        if not partition:
            partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        # 计算telegate代价
        costs = CompilerUtils.evaluate_telegate_with_cat(partition, circuit, network)

        return MappingRecord(
            layer_start  = layer_start,
            layer_end    = layer_end,
            partition    = partition,
            mapping_type = "telegate",
            costs        = costs
        )


class OEEPartitioner(TelegatePartitioner):
    _CAT_BLOCK_SUPPORTED_GATES = {"cx", "cz", "crz", "cu1", "rzz"}

    @property
    def name(self) -> str:
        return "OEETelegatePartitioner"

    def _build_cat_comm_ops(
        self,
        circuit: QuantumCircuit,
        partition: list[list[int]],
    ) -> tuple[QuantumCircuit, dict[str, int]]:
        """
        三阶段：
        1) 识别telegate候选门；
        2) 在语义安全约束下重排，尽量按控制位(source)聚集；
        3) 将连续同(source,dst_qpu)候选块聚合为CommOp(cat)。
        """
        qubit_to_qpu: dict[int, int] = {}
        for qpu_id, group in enumerate(partition):
            for q in group:
                qubit_to_qpu[q] = qpu_id

        qindex = {q: i for i, q in enumerate(circuit.qubits)}

        # ---------- Stage 1: 识别telegate候选 ----------
        def _is_telegate_candidate(inst: Any) -> bool:
            if len(inst.qubits) != 2 or len(inst.clbits) != 0:
                return False
            if inst.operation.name not in self._CAT_BLOCK_SUPPORTED_GATES:
                return False
            q0 = qindex[inst.qubits[0]]
            q1 = qindex[inst.qubits[1]]
            return qubit_to_qpu[q0] != qubit_to_qpu[q1]

        def _select_source_and_dst(inst: Any, endpoint_freq: dict[int, int]) -> tuple[int, int, int]:
            q0 = qindex[inst.qubits[0]]
            q1 = qindex[inst.qubits[1]]
            gate_name = inst.operation.name

            # 有向门默认第1个量子位作为控制位。
            if gate_name in {"cx", "crz", "cu1"}:
                source = q0
                target = q1
            else:
                # 对称门用“出现频次更高”的端点作为锚点，帮助后续按source聚集。
                f0 = endpoint_freq.get(q0, 0)
                f1 = endpoint_freq.get(q1, 0)
                if (f0 > f1) or (f0 == f1 and q0 <= q1):
                    source, target = q0, q1
                else:
                    source, target = q1, q0

            dst_qpu = qubit_to_qpu[target]
            return source, target, dst_qpu

        def _reorder_candidate_segment(segment: list[Any]) -> list[dict[str, Any]]:
            """
            对连续候选段做依赖安全重排：
            - 保持每个量子比特上的原始先后约束；
            - 在可选范围内尽量连续选择同source(再同dst)的门。
            """
            if len(segment) <= 1:
                if not segment:
                    return []
                inst = segment[0]
                source, target, dst_qpu = _select_source_and_dst(inst, {})
                return [{
                    "inst": inst,
                    "is_candidate": True,
                    "source": source,
                    "target": target,
                    "dst_qpu": dst_qpu,
                }]

            endpoint_freq: dict[int, int] = {}
            for inst in segment:
                q0 = qindex[inst.qubits[0]]
                q1 = qindex[inst.qubits[1]]
                endpoint_freq[q0] = endpoint_freq.get(q0, 0) + 1
                endpoint_freq[q1] = endpoint_freq.get(q1, 0) + 1

            nodes: list[dict[str, Any]] = []
            for inst in segment:
                source, target, dst_qpu = _select_source_and_dst(inst, endpoint_freq)
                nodes.append({
                    "inst": inst,
                    "is_candidate": True,
                    "source": source,
                    "target": target,
                    "dst_qpu": dst_qpu,
                })

            n = len(nodes)
            succ: list[set[int]] = [set() for _ in range(n)]
            indeg = [0 for _ in range(n)]

            # 每个量子位上的原始顺序约束（依赖安全）。
            last_on_qubit: dict[int, int] = {}
            for i, node in enumerate(nodes):
                q0 = qindex[node["inst"].qubits[0]]
                q1 = qindex[node["inst"].qubits[1]]
                for q in (q0, q1):
                    prev = last_on_qubit.get(q)
                    if prev is not None and i not in succ[prev]:
                        succ[prev].add(i)
                        indeg[i] += 1
                    last_on_qubit[q] = i

            ready: set[int] = {i for i in range(n) if indeg[i] == 0}
            ordered_idx: list[int] = []
            current_source: Optional[int] = None
            current_dst: Optional[int] = None

            while ready:
                src_ready_cnt: dict[int, int] = {}
                for i in ready:
                    s = nodes[i]["source"]
                    src_ready_cnt[s] = src_ready_cnt.get(s, 0) + 1

                chosen = max(
                    ready,
                    key=lambda i: (
                        1 if nodes[i]["source"] == current_source else 0,
                        1 if nodes[i]["dst_qpu"] == current_dst else 0,
                        src_ready_cnt[nodes[i]["source"]],
                        -i,
                    ),
                )

                ready.remove(chosen)
                ordered_idx.append(chosen)
                current_source = nodes[chosen]["source"]
                current_dst = nodes[chosen]["dst_qpu"]

                for nxt in succ[chosen]:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        ready.add(nxt)

            return [nodes[i] for i in ordered_idx]

        # ---------- Stage 2: 分段重排（候选段内） ----------
        reordered_plan: list[dict[str, Any]] = []
        run_buffer: list[Any] = []
        reorder_runs = 0
        reorder_gates = 0
        reorder_moved = 0

        def _flush_reorder_run() -> None:
            nonlocal run_buffer, reorder_runs, reorder_gates, reorder_moved
            if not run_buffer:
                return
            reorder_runs += 1
            reorder_gates += len(run_buffer)
            reordered_nodes = _reorder_candidate_segment(run_buffer)
            reorder_moved += sum(
                1 for i, node in enumerate(reordered_nodes)
                if node["inst"] is not run_buffer[i]
            )
            reordered_plan.extend(reordered_nodes)
            run_buffer = []

        for inst in circuit.data:
            if _is_telegate_candidate(inst):
                run_buffer.append(inst)
            else:
                _flush_reorder_run()
                reordered_plan.append({
                    "inst": inst,
                    "is_candidate": False,
                    "source": None,
                    "target": None,
                    "dst_qpu": None,
                })

        _flush_reorder_run()

        # ---------- Stage 3: 按(source,dst_qpu)聚合CAT块 ----------
        ops = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        instructions = reordered_plan
        idx = 0

        folded_blocks = 0
        folded_gates = 0
        telegate_candidates = sum(1 for node in reordered_plan if node["is_candidate"])

        while idx < len(instructions):
            node = instructions[idx]
            inst = node["inst"]

            if not node["is_candidate"]:
                qargs = [ops.qubits[qindex[q]] for q in inst.qubits]
                if len(inst.clbits) > 0:
                    cargs = [ops.clbits[circuit.clbits.index(c)] for c in inst.clbits]
                    ops.append(inst.operation, qargs, cargs)
                else:
                    ops.append(inst.operation, qargs)
                idx += 1
                continue

            source = int(node["source"])
            dst_qpu = int(node["dst_qpu"])
            src_qpu = qubit_to_qpu[source]
            block_nodes: list[dict[str, Any]] = []
            targets: set[int] = set()
            j = idx

            while j < len(instructions):
                cur = instructions[j]
                if not cur["is_candidate"]:
                    break
                if int(cur["source"]) != source or int(cur["dst_qpu"]) != dst_qpu:
                    break
                block_nodes.append(cur)
                targets.add(int(cur["target"]))
                j += 1

            # 只有足够“可复用”的块才聚成cat，否则逐门保留。
            if len(block_nodes) < 2 or len(targets) < 2:
                qargs = [ops.qubits[qindex[q]] for q in inst.qubits]
                ops.append(inst.operation, qargs)
                idx += 1
                continue

            involved_qubits = [source] + sorted(targets)

            gate_list = []
            for block_node in block_nodes:
                block_inst = block_node["inst"]
                # Qiskit部分基础门（如CZGate）是不可变单例，需先转为可变实例。
                gate_op = block_inst.operation
                if hasattr(gate_op, "to_mutable"):
                    gate_op = gate_op.to_mutable()
                else:
                    gate_op = copy.deepcopy(gate_op)
                gate_qids = [qindex[q] for q in block_inst.qubits]
                setattr(gate_op, "_global_lqids", gate_qids)
                gate_list.append(gate_op)

            comm_op = CommOp(
                comm_type="cat",
                source_qubit=source,
                src_qpu=src_qpu,
                dst_qpu=dst_qpu,
                involved_qubits=involved_qubits,
                gate_list=gate_list,
            )
            ops.append(comm_op, [ops.qubits[q] for q in involved_qubits])

            folded_blocks += 1
            folded_gates += len(block_nodes)
            idx = j

        return ops, {
            "telegate_candidates": telegate_candidates,
            "reorder_runs": reorder_runs,
            "reorder_gates": reorder_gates,
            "reorder_moved": reorder_moved,
            "folded_blocks": folded_blocks,
            "folded_gates": folded_gates,
        }
    
    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:
        # 将config里面的prev_partition作为OEE的起点
        prev_partition = config.get("partition", []) if config else []
        partition = copy.deepcopy(prev_partition) if prev_partition else []
        iteration_count = config.get("iteration", 50) if config else 50
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1

        add_teledata_costs = config.get("add_teledata_costs", False) if config else False
        enable_cat_telegate = config.get("enable_cat_telegate", True) if config else True
        cat_controls = config.get("cat_controls", []) if config else []
        debug_cat_telegate = config.get("debug_cat_telegate", True) if config else True

        # 如果config里没有partition，就按顺序分配
        if not partition:
            partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        qig = CompilerUtils.build_qubit_interaction_graph(circuit)
        
        # oee_start_time = time.time()
        partition = OEE.partition(partition, qig, network, iteration_count)
        # print(f"[DEBUG] OEE Time: {time.time() - oee_start_time}")

        record = MappingRecord(
            layer_start = layer_start,
            layer_end = layer_end,
            partition = partition,
            mapping_type = "telegate",
            extra_info = {"ops": circuit},
        )

        # 评估当前线路的telegate代价
        if enable_cat_telegate:
            cat_ops, cat_scan_stats = self._build_cat_comm_ops(circuit, partition)
            record.extra_info = {
                **(record.extra_info or {}),
                "ops": cat_ops,
                "cat_scan_stats": cat_scan_stats,
            }
            costs = CompilerUtils.evaluate_telegate_with_cat(record, cat_ops, network)
        else:
            costs = CompilerUtils.evaluate_telegate_with_cat(partition, circuit, network)

        if enable_cat_telegate and debug_cat_telegate:
            # 主流程已评估过cat_ops，这里直接复用，避免重复计算。
            cat_costs = costs
            plain_costs = CompilerUtils.evaluate_telegate_with_cat(partition, circuit, network)
            log(
                f"[cat_debug][telegate_partitioner] layer=[{layer_start},{layer_end}] "
                f"controls={len(cat_controls)} plain_epairs={plain_costs.epairs} cat_epairs={cat_costs.epairs}"
            )

        # 评估和前一段线路的切分代价（如果prev_partition存在）
        if prev_partition and add_teledata_costs:
            teledata_costs, _ = CompilerUtils.evaluate_teledata(
                prev_partition,
                partition,
                network)

            costs += teledata_costs

        record.costs = costs
        return record


class CatEntPartitioner(TelegatePartitioner):
    """
    返回一个用CatEnt方式切分的量子线路
    """
    @property
    def name(self) -> str:
        return "CatEntPartitioner"
    
    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecordList:
        cfg = config or {}
        layer_start = cfg.get("layer_start", 0)
        layer_end = cfg.get("layer_end", circuit.depth() - 1)
        use_oee_init = bool(cfg.get("use_oee_init", True))

        # 先准备初始划分；若未指定则按容量顺序分配。
        base_partition = copy.deepcopy(cfg.get("partition", []))
        if not base_partition:
            base_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)

        if use_oee_init:
            qig = CompilerUtils.build_qubit_interaction_graph(circuit)
            oee_iteration = int(cfg.get("iteration", 50))
            init_partition = OEE.partition(base_partition, qig, network, oee_iteration)
        else:
            init_partition = base_partition

        # 在初始划分基础上调用QAutoComm，拿到cat-ent优化记录。
        from ..baselines.autocomm import QAutoComm

        autocomm_cfg = dict(cfg)
        autocomm_cfg["partition"] = init_partition
        autocomm_cfg["save_records"] = False
        autocomm_cfg["comm_only_costs"] = True

        cat_result = QAutoComm().compile(
            circuit=circuit,
            network=network,
            config=autocomm_cfg,
        )

        for record in cat_result.records:
            record.layer_start = layer_start
            record.layer_end = layer_end

        return cat_result


class PytketDQCPartitioner(TelegatePartitioner):
    @property
    def name(self) -> str:
        return "PytketDQCPartitioner"

    def partition(self, 
                  circuit: QuantumCircuit, 
                  network: Network,
                  config: Optional[dict[str, Any]]) -> MappingRecord:

        from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
        from pytket_dqc.utils import DQCPass
        from pytket_dqc.networks import NISQNetwork
        from pytket_dqc.distributors import CoverEmbeddingSteinerDetached, PartitioningAnnealing

        # 预处理电路
        circuit = transpile(circuit, basis_gates=["cu1", "rz", "h"], optimization_level=0)
        tk_circ = qiskit_to_tk(circuit)
        DQCPass().apply(tk_circ)

        # 构建NISQ网络
        server_coupling, server_qubits = network.get_network_coupling_and_qubits()
        nisq_network = NISQNetwork(server_coupling, server_qubits)
        
        # 获取算法
        workflow = config.get("workflow", "CE") if config else "CE"
        seed = config.get("seed", 26) if config else 26
        layer_start = config.get("layer_start", 0) if config else 0
        layer_end = config.get("layer_end", circuit.depth() - 1) if config else circuit.depth() - 1

        distribution = None
        if workflow == "CE":
            print("[Pytket-DQC: CoverEmbeddingSteinerDetached]")
            distribution = CoverEmbeddingSteinerDetached().distribute(tk_circ, nisq_network, seed=seed)
        elif workflow == "PA":
            print("[Pytket-DQC: PartitioningAnnealing]")
            distribution = PartitioningAnnealing().distribute(tk_circ, nisq_network, seed=seed)
        else:
            raise ValueError("Unsupported workflow. Use 'CE' for CoverEmbeddingSteinerDetached or 'PA' for PartitioningAnnealing.")

        # 获取划分
        partition = [[] for _ in range(network.num_backends)]
        for i in range(circuit.num_qubits):
            target = distribution.placement.placement[i]
            partition[target].append(i)

        # TODO: 计算cat-ent的telegate代价
        costs, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(partition, circuit, network)
        
        return MappingRecord(
            layer_start=layer_start,
            layer_end=layer_end,
            partition=partition,
            mapping_type="telegate",
            costs=costs
        )


class TelegatePartitionerFactory:
    """划分分配器工厂类"""
    _registry = {
        "direct": "DirectTelegatePartitioner",
        "oee": "OEEPartitioner",
        "cat": "CatEntPartitioner",
        "catent": "CatEntPartitioner",
        "cat_ent": "CatEntPartitioner",
        "pytket_dqc": "PytketDQCPartitioner"
    }

    @classmethod
    def create_telegate_partitioner(cls, partitioner_type: str) -> TelegatePartitioner:
        """
        创建指定类型的划分器

        :param partitioner_type: 划分配器类型字符串
        :return: 对应的划分配器实例
        """
        partitioner_type = partitioner_type.lower()
        if partitioner_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown partitioner: '{partitioner_type}', available: {available}")
        
        # 从注册表获取类名，然后创建实例
        partitioner_class_name = cls._registry[partitioner_type]
        partitioner_class = globals()[partitioner_class_name]
        return partitioner_class()

    @classmethod
    def register_telegate_partitioner(cls, name: str, class_name: str):
        """
        动态注册新的划分分配器类型
        
        :param name: 分配器类型名称
        :param class_name: 对应的类名字符串
        """
        cls._registry[name] = class_name
    
    @classmethod
    def unregister_telegate_partitioner(cls, name: str):
        """
        移除注册的划分分配器类型
        
        :param name: 要移除的分配器类型名称
        """
        if name in cls._registry:
            del cls._registry[name]
    
    @classmethod
    def get_available_telegate_partitioners(cls):
        """
        获取所有可用的划分分配器类型
        
        :return: 可用分配器类型列表
        """
        return list(cls._registry.keys())

