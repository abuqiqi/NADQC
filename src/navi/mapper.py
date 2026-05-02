from abc import ABC, abstractmethod
from collections import defaultdict
import time
import numpy as np
import itertools
from typing import Any, Optional
import copy
from math import inf
from pprint import pprint
import networkx as nx
from dataclasses import replace
import sys

from qiskit import QuantumCircuit

from ..compiler import MappingRecord, MappingRecordList, CompilerUtils, ExecCosts
from ..compiler.compiler_utils import CommOp
from ..utils import Network

class Mapper(ABC):
    """
    量子线路映射器接口，定义所有映射算法必须实现的方法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取映射器名称，用于标识和比较
        """
        pass

    @abstractmethod
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果
        """
        pass

    def _reevaluate_mapping_record_list(
            self, 
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None,
        ) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件（基线实现）
        """
        # print(f"\n\n\n[DEBUG] _reevaluate_mapping_record_list\n\n\n")
        # ---------- 更新映射记录 ----------
        k = len(mapping_record_list.records)  # 时间片数量

        logical_phy_map = {}
        comm_phy_map = {}

        for t in range(k):
            # print(f"\n\n\n[DEBUG] {t} / {k}")

            record = mapping_record_list.records[t]

            if t == 0:
                logical_phy_map = CompilerUtils.init_logical_phy_map(record.partition)
                comm_phy_map = {}
            else:
                # 沿用上一个record的logical_phy_map作为初始状态
                logical_phy_map = mapping_record_list.records[t-1].logical_phy_map
                comm_phy_map = copy.deepcopy(getattr(mapping_record_list.records[t - 1], "comm_phy_map", {}) or {})

            record.costs = ExecCosts()  # 初始化成本对象
            record.logical_phy_map = logical_phy_map.copy() # 初始化当前时间片的logical_phy_map
            record.comm_phy_map = copy.deepcopy(comm_phy_map)

            # print(f"[DEBUG] After new init: \n{record}")
            # print(f"[DEBUG] partition: {record.partition}")
            # print(f"[DEBUG] logical_phy_map: {logical_phy_map}")

            # telegate partition的时候会构建子线路
            # teledata的记录不需要考虑子线路，所以没有
            if record.extra_info is not None and "ops" in record.extra_info:
                subcircuit = record.extra_info["ops"]
            else:
                if record.mapping_type == "cat":
                    raise ValueError(f"[ERROR] For cat-type record, extra_info should have ops.")
                # 获取子线路
                subcircuit = CompilerUtils.get_subcircuit_by_level(
                    num_qubits=circuit.num_qubits,
                    circuit=circuit,
                    circuit_layers=circuit_layers,
                    layer_start=record.layer_start,
                    layer_end=record.layer_end
                )

            # print(f"[DEBUG] original subcircuit:")
            # print(subcircuit)

            # 计算teledata损失
            if t != 0:
                prev_record = mapping_record_list.records[t - 1]
                _ = CompilerUtils.evaluate_teledata(
                    prev_record,
                    record,
                    network
                )
                # print(f"[DEBUG] record (teledata): \n{record}")
                # print(f"[DEBUG] after teledata: {record.logical_phy_map}")

            tg_costs, curr_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                record,
                subcircuit,
                network,
            )

        return mapping_record_list

    def _prepare_records_for_reevaluate(self, mapping_record_list: MappingRecordList) -> None:
        """
        确保调用 _reevaluate_mapping_record_list 前，每条record都已处于最终物理QPU顺序，
        且 CommOp 端点元数据已更新。
        """
        for record in mapping_record_list.records:
            if record.extra_info is None:
                continue
            perm = record.extra_info.get("perm")
            if not isinstance(perm, (list, tuple)):
                continue

            perm = tuple(int(x) for x in perm)
            n_physical = len(record.partition)
            if len(perm) != n_physical:
                raise ValueError(
                    f"[REMAP_PRECHECK] perm length mismatch: len(perm)={len(perm)}, "
                    f"partition_size={n_physical}"
                )

            original_partition = record.partition
            record.partition = [original_partition[i] for i in perm]

            if "ops" in record.extra_info:
                record.extra_info["ops"] = self._remap_commop_endpoints_by_perm(
                    copy.deepcopy(record.extra_info["ops"]),
                    perm,
                )

            del record.extra_info["perm"]

    def _remap_commop_endpoints_by_perm(
            self,
            subcircuit: QuantumCircuit,
            perm: tuple[int, ...],
        ) -> QuantumCircuit:
        """
        按 perm 将 CommOp 的 src_qpu/dst_qpu 从 old_qpu 重写到 new_qpu。
        约定：new_partition[new_qpu] = old_partition[perm[new_qpu]]。
        """
        if len(perm) == 0:
            return subcircuit

        old_to_new_qpu: dict[int, int] = {}
        for new_qpu, old_qpu in enumerate(perm):
            old_to_new_qpu[int(old_qpu)] = int(new_qpu)

        if len(old_to_new_qpu) != len(perm):
            raise ValueError(f"invalid perm (duplicate old_qpu): perm={perm}")

        for instruction in subcircuit:
            op = instruction.operation
            if not isinstance(op, CommOp):
                continue

            if op.src_qpu not in old_to_new_qpu or op.dst_qpu not in old_to_new_qpu:
                raise RuntimeError(
                    "[COMMOP_ENDPOINT] src/dst qpu not in remap table. "
                    f"src_qpu={op.src_qpu}, dst_qpu={op.dst_qpu}, remap_keys={sorted(old_to_new_qpu.keys())}"
                )

            op.src_qpu = old_to_new_qpu[op.src_qpu]
            op.dst_qpu = old_to_new_qpu[op.dst_qpu]

        return subcircuit

    def _estimate_partition_comm_matrix_for_single_record(
            self,
            subcircuit: QuantumCircuit,
            partition: list[list[int]],
        ) -> list[list[float]]:
        """
        估计单record场景下，各partition之间的通信强度矩阵。
        """
        n = len(partition)
        comm = [[0.0 for _ in range(n)] for __ in range(n)]

        qubit_to_block: dict[int, int] = {}
        for bidx, group in enumerate(partition):
            for q in group:
                qubit_to_block[int(q)] = bidx

        for instruction in subcircuit:
            op = instruction.operation

            if isinstance(op, CommOp):
                src = int(op.source_qubit)
                src_b = qubit_to_block.get(src)
                if src_b is None:
                    continue

                if op.comm_type == "rtp":
                    w = 2.0
                else:
                    w = 1.0

                # 单source的收尾TP不形成块间关系，跳过即可
                for q in op.involved_qubits:
                    q = int(q)
                    if q == src:
                        continue
                    dst_b = qubit_to_block.get(q)
                    if dst_b is None or dst_b == src_b:
                        continue
                    comm[src_b][dst_b] += w
                    comm[dst_b][src_b] += w
                continue

            qids = [subcircuit.qubits.index(qubit) for qubit in instruction.qubits]
            if len(qids) <= 1:
                continue

            # 与evaluate_local_and_telegate_with_cat一致，按相邻qids计跨分区通信关系
            for i in range(len(qids) - 1):
                q1, q2 = int(qids[i]), int(qids[i + 1])
                b1 = qubit_to_block.get(q1)
                b2 = qubit_to_block.get(q2)
                if b1 is None or b2 is None or b1 == b2:
                    continue
                comm[b1][b2] += 1.0
                comm[b2][b1] += 1.0

        return comm

    def _greedy_perm_by_comm_matrix(
            self,
            comm_matrix: list[list[float]],
            hops: list[list[int]],
        ) -> tuple[int, ...]:
        """
        基于通信强度与网络跳数，使用交换改进的贪心策略寻找perm。
        返回perm，满足 new_partition[new_qpu] = old_partition[perm[new_qpu]].
        """
        n = len(comm_matrix)
        assign = list(range(n))  # logical_idx -> physical_idx

        def _score(curr_assign: list[int]) -> float:
            s = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    w = comm_matrix[i][j]
                    if w == 0:
                        continue
                    s += w * hops[curr_assign[i]][curr_assign[j]]
            return s

        curr_score = _score(assign)
        while True:
            best_pair: tuple[int, int] | None = None
            best_score = curr_score

            for i in range(n):
                for j in range(i + 1, n):
                    trial = assign.copy()
                    trial[i], trial[j] = trial[j], trial[i]
                    s = _score(trial)
                    if s < best_score:
                        best_score = s
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            assign[i], assign[j] = assign[j], assign[i]
            curr_score = best_score

        # assign: logical->physical, 转成 perm: new_physical->old_logical
        inv = [0] * n
        for logical_idx, physical_idx in enumerate(assign):
            inv[physical_idx] = logical_idx

        return tuple(inv)

    def _infer_perm_from_partitions(
            self,
            original_partition: list[list[int]],
            mapped_partition: list[list[int]],
        ) -> tuple[int, ...]:
        """
        推断满足 mapped_partition[i] = original_partition[perm[i]] 的 perm。
        """
        index_by_block: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, block in enumerate(original_partition):
            index_by_block[tuple(block)].append(idx)

        perm: list[int] = []
        for block in mapped_partition:
            key = tuple(block)
            if key not in index_by_block or not index_by_block[key]:
                raise ValueError(f"Failed to infer perm, block {block} not found in original partition.")
            perm.append(index_by_block[key].pop(0))

        return tuple(perm)

    def _print_final_perm_sequence(
            self,
            original_partitions: list[list[list[int]]],
            final_record_list: MappingRecordList,
        ) -> None:
        """
        打印最终被选中的每个时间片对应的 perm 序列。
        """
        perms: list[tuple[int, ...] | str] = []
        records = final_record_list.records
        n = min(len(original_partitions), len(records))

        for t in range(n):
            try:
                perm = self._infer_perm_from_partitions(original_partitions[t], records[t].partition)
                perms.append(perm)
            except Exception as e:
                perms.append(f"<infer_failed:{e}>")

        if len(original_partitions) != len(records):
            print(
                f"[WARN] [{self.name}] partition count mismatch when printing perms: "
                f"original={len(original_partitions)}, final={len(records)}"
            )

        print(f"[INFO] [{self.name}] final mapping perm sequence: {perms}")


class DirectMapper(Mapper):
    """
    基线映射器：直接使用输入的逻辑QPU顺序（逻辑QPU i -> 物理QPU i）
    作为比较基准，不进行任何优化
    """

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Direct Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件（基线实现）
        """
        mapping_record_list = self._reevaluate_mapping_record_list(
            mapping_record_list,
            circuit,
            circuit_layers,
            network,
            config=config,
        )
        return mapping_record_list


class SingleRecordExhaustiveMapper(Mapper):
    """
    调试专用映射器：当且仅当record数量为1时，枚举m!个映射并输出每个映射的ExecCosts，
    最后选择(total_fidelity_loss, epairs)最小的映射返回。
    """

    @property
    def name(self) -> str:
        return "Single Record Exhaustive Debug Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        start_time = time.time()
        k = len(mapping_record_list.records)

        if k != 1:
            print(
                f"[WARNING] [{self.name}] expects exactly 1 record, got {k}. "
                "Fallback to Direct Mapper."
            )
            return self._reevaluate_mapping_record_list(
                mapping_record_list,
                circuit,
                circuit_layers,
                network,
                config=config,
            )

        record = mapping_record_list.records[0]
        original_partition = record.partition
        n_physical = network.num_backends
        all_perms = list(itertools.permutations(range(n_physical)))

        if record.extra_info is not None and "ops" in record.extra_info:
            subcircuit = record.extra_info["ops"]
        else:
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end,
            )

        print(
            f"[INFO] [{self.name}] record_count=1, enumerate {len(all_perms)} permutations "
            f"(m={n_physical})."
        )

        best_record: Optional[MappingRecord] = None

        for idx, perm in enumerate(all_perms):
            partition = [original_partition[i] for i in perm]
            trial_record = MappingRecord(
                layer_start=record.layer_start,
                layer_end=record.layer_end,
                partition=partition,
                mapping_type=record.mapping_type,
                logical_phy_map=CompilerUtils.init_logical_phy_map(partition),
                extra_info=record.extra_info,
            )

            # 更新CommOp里面的src_qpu和dst_qpu
            trial_subcircuit = self._remap_commop_endpoints_by_perm(
                copy.deepcopy(subcircuit),
                perm,
            )
            if trial_record.extra_info is None:
                trial_record.extra_info = {}
            trial_record.extra_info["ops"] = trial_subcircuit
            trial_record.extra_info["perm"] = list(perm)

            costs, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(
                trial_record, # logical_phy_map会在record中更新
                trial_subcircuit,
                network,
            )

            print(
                f"[DEBUG] [{self.name}] perm#{idx:02d}={perm}, "
                f"epairs={costs.epairs}, total_fidelity_loss={costs.total_fidelity_loss:.12f}, "
                f"num_comms={costs.num_comms}, remote_hops={costs.remote_hops}, remote_swaps={costs.remote_swaps}"
            )

            score = (costs.total_fidelity_loss, costs.epairs)
            if best_record is None:
                best_record = copy.deepcopy(trial_record)
            else:
                best_score = (best_record.costs.total_fidelity_loss, best_record.costs.epairs)
                if score < best_score:
                    best_record = copy.deepcopy(trial_record)

        assert best_record is not None
        assert best_record.extra_info is not None

        print(
            f"[INFO] [{self.name}] best_perm={tuple(best_record.extra_info['perm'])}, "
            f"epairs={best_record.costs.epairs}, "
            f"total_fidelity_loss={best_record.costs.total_fidelity_loss:.12f}"
        )
        print(f"[INFO] [{self.name}] completed in {time.time() - start_time:.2f} seconds.")

        mapping_record_list.records[0] = best_record

        return mapping_record_list


class SingleRecordGreedyMapper(Mapper):
    """
    单record贪心映射器：
    先估计partition间通信强度，再通过交换改进贪心选择perm，
    最后用真实成本评估函数计算一次并返回。
    """

    @property
    def name(self) -> str:
        return "Single Record Greedy Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        start_time = time.time()
        k = len(mapping_record_list.records)

        if k != 1:
            print(
                f"[WARNING] [{self.name}] expects exactly 1 record, got {k}. "
                "Fallback to baseline reevaluation without greedy search."
            )
            return self._reevaluate_mapping_record_list(
                mapping_record_list,
                circuit,
                circuit_layers,
                network,
                config=config,
            )

        record = mapping_record_list.records[0]
        original_partition = record.partition
        n_physical = network.num_backends

        if record.extra_info is not None and "ops" in record.extra_info:
            subcircuit = record.extra_info["ops"]
        else:
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end,
            )

        comm_matrix = self._estimate_partition_comm_matrix_for_single_record(subcircuit, original_partition)
        greedy_perm = self._greedy_perm_by_comm_matrix(comm_matrix, network.Hops)
        best_partition = [original_partition[i] for i in greedy_perm]
        
        trial_record = MappingRecord(
            layer_start=record.layer_start,
            layer_end=record.layer_end,
            partition=best_partition,
            mapping_type=record.mapping_type,
            logical_phy_map=CompilerUtils.init_logical_phy_map(best_partition),
            extra_info=record.extra_info,
        )

        # 更新CommOp里面的src_qpu和dst_qpu
        trial_subcircuit = self._remap_commop_endpoints_by_perm(
            copy.deepcopy(subcircuit),
            greedy_perm,
        )
        if trial_record.extra_info is None:
            trial_record.extra_info = {}
        trial_record.extra_info["ops"] = trial_subcircuit
        trial_record.extra_info["perm"] = list(greedy_perm)
        
        costs, _ = CompilerUtils.evaluate_local_and_telegate_with_cat(
            trial_record,
            trial_subcircuit,
            network,
        )

        mapping_record_list.records[0] = trial_record

        print(
            f"[INFO] [{self.name}] greedy_perm={greedy_perm}, epairs={costs.epairs}, "
            f"total_fidelity_loss={costs.total_fidelity_loss:.12f}"
        )
        print(f"[INFO] [{self.name}] completed in {time.time() - start_time:.2f} seconds.")
        return mapping_record_list


class GreedyMapper(Mapper):

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Greedy Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None,
        ) -> MappingRecordList:

        start_time = time.time()
        original_partitions = [copy.deepcopy(record.partition) for record in mapping_record_list.records]
        # 对于每一段子线路，我们尝试找到一个局部最优的映射
        k = len(mapping_record_list.records)  # 时间片数量
        n_physical = network.num_backends  # 物理QPU数量

        all_perms = list(itertools.permutations(range(n_physical)))

        logical_phy_map = {}

        for t in range(k): # 对于每一段线路

            # print(f"\n\n\n[DEBUG] ========== {t} / {k} ==========")

            # 使用贪心算法为下一时间片找到映射
            curr_record = mapping_record_list.records[t]
            # print(f"[DEBUG] Current record before mapping:\n{curr_record}")
            original_partition = curr_record.partition

            # print(f"[DEBUG] original_partition: {original_partition}")

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=curr_record.layer_start,
                layer_end=curr_record.layer_end
            )
            # print(f"[DEBUG] subcircuit:\n{subcircuit}")

            # 记录最佳排列
            best_perm = None
            min_costs = None
            # min_num_comms = float('inf')
            min_fidelity_loss = float('inf')
            best_logical_phy_map = {}

            # perm所有可能的排列
            for perm in all_perms:
                # 获取映射顺序
                order = list(perm)

                # 根据order的顺序构建新的partition
                partition = []
                for idx in order:
                    partition.append(original_partition[idx])
                # print(f"[DEBUG] Trying permutation {perm}, partition: {partition}")

                costs = ExecCosts()

                # 根据partition构造logical_phy_map
                # 如果是第一段线路，logical_phy_map直接根据partition构造；否则沿用上一个时间片的logical_phy_map并根据新的partition调整
                if t == 0:
                    logical_phy_map = CompilerUtils.init_logical_phy_map(partition)
                    # print(f"[DEBUG] initial logical_phy_map: {logical_phy_map}")
                else:
                    # 沿用上一个record的logical_phy_map作为初始状态
                    logical_phy_map = mapping_record_list.records[t-1].logical_phy_map.copy()
                    # print(f"[DEBUG] initial logical_phy_map: {logical_phy_map}")
                
                    # 评估当前线路的teledata开销并更新logical_phy_map
                    costs, logical_phy_map = CompilerUtils.evaluate_teledata(
                        mapping_record_list.records[t-1].partition,
                        partition,
                        network,
                        logical_phy_map
                    )

                # 评估当前排列的local_and_telegate_cost
                telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                    partition,
                    subcircuit,
                    network,
                    logical_phy_map,
                    optimization_level=0,
                )
                costs += telegate_costs

                curr_fidelity_loss = costs.total_fidelity_loss
                # curr_epairs = costs.epairs

                if curr_fidelity_loss < min_fidelity_loss:
                    # 比较最小num_comms和最小fidelity_loss的排列是否一致
                    # if curr_epairs <= min_num_comms:
                    #     min_num_comms = curr_epairs
                    # else:
                        # print(f"[NOTE] Found a permutation with lower fidelity loss but higher communication cost: {curr_epairs} vs {min_num_comms}")
                    min_fidelity_loss = curr_fidelity_loss
                    best_perm = perm
                    min_costs = costs
                    best_logical_phy_map = logical_phy_map.copy()

            # 调整curr_record成最佳排列
            assert best_perm is not None and min_costs is not None, "未找到最佳排列，可能存在问题"
            best_partition = []
            for idx in best_perm:
                best_partition.append(original_partition[idx])
            curr_record.partition = copy.deepcopy(best_partition)
            curr_record.costs = replace(min_costs)
            curr_record.logical_phy_map = best_logical_phy_map.copy()

            # print(f"\n[DEBUG] Best permutation for time slice {t}: {best_perm}\npartition: {best_partition}\ncosts: {min_costs}\nlogical_phy_map: {best_logical_phy_map}")

        end_time = time.time()
        print(f"[INFO] [Time] Greedy Mapper completed in {end_time - start_time:.2f} seconds.")
        self._print_final_perm_sequence(original_partitions, mapping_record_list)
        return mapping_record_list


class DPMapper(Mapper):

    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "DP Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None,
        ) -> MappingRecordList:
        """
        使用动态规划为每个时间片选择最优的物理QPU映射（排列），
        使得所有时间片的总成本（保真度损失, 通信成本）最小。
        """

        # 
        # dp[t][perm] := 在t段量子线路采用perm排列的partition得到的全程最优的(总保真度损失, 总eparis)
        # % dp[t][curr_perm] = min_{prev_perm ∈ all_perm} [
        # %     dp[t-1][prev_perm]  // 前一阶段的累计开销
        # %     + teledata_cost(prev_perm对应的partition, curr_perm, network, prev_perm对应的logical_phy_map)  // 分区切换开销（对应evaluate_teledata）
        # %     + telegate_cost(curr_perm, subcircuit_t, network, prev_perm对应的logical_phy_map被teledata更新到curr_perm的)  // 当前子线路开销（对应evaluate_local_and_telegate）
        # % ]
        start_time = time.time()
        k = len(mapping_record_list.records)
        n_physical = network.num_backends

        # 所有可能的物理QPU排列（即映射状态）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # 只需记录上一和当前时间片的最佳映射状态，DP过程中不需要完整记录路径
        # 包括partition, costs, logical_phy_map
        prev_bests: list[MappingRecord] = []
        curr_bests: list[MappingRecord] = []

        # back[t][idx] 存储达到该状态的最优前一状态索引
        back: list[list[int]] = [[-1 for _ in range(num_states)] for _ in range(k)]

        # ----- t=0 的初始化 -----
        # 第一个时间片只需要记录telegate开销，logical_phy_map由原始分区直接构建
        record = mapping_record_list.records[0]
        original_partition = record.partition
        subcircuit = CompilerUtils.get_subcircuit_by_level(
            num_qubits=circuit.num_qubits,
            circuit=circuit,
            circuit_layers=circuit_layers,
            layer_start=record.layer_start,
            layer_end=record.layer_end
        )
        for perm in all_perms:
            partition = [original_partition[i] for i in perm]
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)
            costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                partition, subcircuit, network, logical_phy_map=logical_phy_map, optimization_level=0)

            prev_bests.append(MappingRecord(
                partition = partition,
                costs = costs,
                logical_phy_map = logical_phy_map
            ))
            # print(f"[DEBUG] t=0, perm: {perm}, partition: {partition}, costs: {costs}, logical_phy_map: {logical_phy_map})

        # ----- t=1~k-1 的迭代更新 -----
        for t in range(1, k):
            record = mapping_record_list.records[t]
            original_partition = record.partition
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )

            # 更新dp[t][0~num_states-1] / curr_bests[0~num_states-1]
            for curr_idx, curr_perm in enumerate(all_perms):
                # 更新curr_bests[curr_idx]选取perm的时候，全局最优映射路径

                # 获取当前的partition
                partition = [original_partition[i] for i in curr_perm]

                # 最佳累计开销
                best_record: MappingRecord | None = None
                best_prev = -1

                # 遍历每一个可能的前序record
                for prev_idx in range(num_states):
                    prev_record = prev_bests[prev_idx]
                    curr_total_costs = copy.deepcopy(prev_record.costs)

                    # 衡量前序节点和当前节点的costs
                    # 获取logical_phy_map
                    logical_phy_map = prev_record.logical_phy_map.copy()

                    # 计算teledata costs
                    curr_costs, logical_phy_map = CompilerUtils.evaluate_teledata(
                        prev_record.partition,
                        partition,
                        network,
                        logical_phy_map
                    )

                    # 计算telegate costs
                    telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                        partition,
                        subcircuit,
                        network,
                        logical_phy_map = logical_phy_map,
                        optimization_level = 0
                    )

                    curr_costs += telegate_costs
                    curr_total_costs += curr_costs

                    if best_record is None or \
                        (curr_total_costs.total_fidelity_loss, curr_total_costs.epairs) < \
                        (best_record.costs.total_fidelity_loss, best_record.costs.epairs):

                        best_record = MappingRecord(
                            partition = partition, # 记录最后一个partition
                            costs = curr_total_costs, # 记录全路径的costs
                            logical_phy_map = logical_phy_map # 记录最后一个logical_phy_map
                        )
                        best_prev = prev_idx

                # 更新curr_bests[curr_idx]
                assert best_record is not None
                curr_bests.append(best_record)
                back[t][curr_idx] = best_prev

            # 交换prev_bests和curr_bests
            prev_bests = curr_bests
            curr_bests = []

        # ----- 回溯找到最优路径 -----
        # 从最后一个时间片里找出最小总开销的项
        best_last = 0
        for idx, record in enumerate(prev_bests):
            if (record.costs.total_fidelity_loss, record.costs.epairs) < \
               (prev_bests[best_last].costs.total_fidelity_loss, prev_bests[best_last].costs.epairs):
                best_last = idx
        
        perm_indices = [0] * k
        perm_indices[k-1] = best_last
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]

        final_perm_sequence = [all_perms[idx] for idx in perm_indices]

        # ----- 更新映射记录 -----
        final_record_list = MappingRecordList()

        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]

            original_partition = record.partition

            best_partition = [original_partition[idx] for idx in perm]
            new_extra_info = copy.deepcopy(record.extra_info) if record.extra_info is not None else {}
            if "ops" in new_extra_info:
                new_extra_info["ops"] = self._remap_commop_endpoints_by_perm(
                    copy.deepcopy(new_extra_info["ops"]),
                    perm,
                )
            new_extra_info["perm"] = list(perm)
            
            new_record = MappingRecord(
                layer_start = record.layer_start,
                layer_end = record.layer_end,
                partition = best_partition,
                mapping_type = record.mapping_type,
                logical_phy_map = CompilerUtils.init_logical_phy_map(best_partition),
                extra_info = new_extra_info
            )

            final_record_list.add_record(new_record)

        final_record_list = self._reevaluate_mapping_record_list(
            final_record_list,
            circuit,
            circuit_layers,
            network,
            config=config,
        )

        end_time = time.time()
        print(f"[INFO] [Time] DP Mapper completed in {end_time - start_time:.2f} seconds.")
        print(f"[INFO] [DP Mapper] final mapping perm sequence (from DP backtrace): {final_perm_sequence}")

        return final_record_list


class NeighborhoodBoundedDPMapper(Mapper):
    # O\big(K \cdot C \cdot \text{beam_width} \cdot C_{\text{eval}}\big)

    @property
    def name(self) -> str:
        return "Neighborhood Bounded DP Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None,
        ) -> MappingRecordList:
        """
        带束搜索优化的动态规划映射器
        解决 QPU 数量增多时 m! 状态爆炸问题
        """
        start_time = time.time()
        k = len(mapping_record_list.records)
        n_physical = network.num_backends
        if k == 0:
            return mapping_record_list

        mapper_cfg = config or {}
        beam_width = int(mapper_cfg.get("beam_width", 3))

        # 根据网络类型自适应设置mapping预算：
        # - all-to-all（或等效max_hop=1）默认较小预算
        # - 非all-to-all默认较大预算
        net_type = getattr(network, "net_type", "unknown")
        is_all_to_all_like = (net_type == "all_to_all")

        # 将K段压缩为不超过mapping_budget个“决策组”，每组共享同一个mapping。
        # 例如 K=40, mapping_budget=20 -> group_size=2，即每相邻两段共享mapping。
        if "mapping_budget" in mapper_cfg:
            mapping_budget = int(mapper_cfg.get("mapping_budget", k))
        else:
            if is_all_to_all_like:
                mapping_budget = int(mapper_cfg.get("mapping_budget_all_to_all", min(k, 1)))
            else:
                mapping_budget = int(mapper_cfg.get("mapping_budget_non_all_to_all", min(k, 40)))
        mapping_budget = max(1, mapping_budget)
        group_size = max(1, (k + mapping_budget - 1) // mapping_budget)
        group_starts = list(range(0, k, group_size))
        num_groups = len(group_starts)

        print(f"[INFO] [NeighborhoodBoundedDPMapper] beam_width: {beam_width}, n_subcs: {k}, n_QPUs: {n_physical}")
        print(f"[INFO] [NeighborhoodBoundedDPMapper] mapping_budget: {mapping_budget}, group_size: {group_size}, num_groups: {num_groups}")
        print(f"[INFO] [NeighborhoodBoundedDPMapper] beam_width: {beam_width}, n_subcs: {k}, n_QPUs: {n_physical}", file=sys.stderr)
        print(f"[INFO] [NeighborhoodBoundedDPMapper] mapping_budget: {mapping_budget}, group_size: {group_size}, num_groups: {num_groups}", file=sys.stderr)
        
        # 所有物理QPU排列 + 固定索引（idx = state_key）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)
        perm_to_idx = {perm: idx for idx, perm in enumerate(all_perms)}

        # 预计算“一次交换”邻域，避免每次扩展都重复 list/tuple 构造与字典查找。
        one_swap_neighbors: list[list[int]] = [[] for _ in range(num_states)]
        for idx, perm in enumerate(all_perms):
            neighbors: list[int] = []
            for i in range(n_physical):
                for j in range(i + 1, n_physical):
                    p = list(perm)
                    p[i], p[j] = p[j], p[i]
                    neighbors.append(perm_to_idx[tuple(p)])
            one_swap_neighbors[idx] = neighbors

        # 候选状态控制参数：避免每层都遍历 n_physical! 全排列
        # 对 n<=4 采用更激进的收缩策略；对 n>=5 保持较宽搜索覆盖。
        # if n_physical <= 4:
        #     # 例如 n=4 时 num_states=24，默认束宽5 -> 候选上限约10
        #     max_candidate_states_cfg = int(beam_width * 2)
        #     max_neighbor_swaps = 1
        # else:
        #     max_candidate_states_cfg = int(beam_width * max(4, n_physical * (n_physical - 1) // 2))
        #     max_neighbor_swaps = max(1, min(2, n_physical - 1))

        # max_candidate_states = int(min(num_states, max(beam_width, max_candidate_states_cfg)))
        default_max_candidate_states = max(beam_width, 2 * beam_width)
        max_candidate_states = int(mapper_cfg.get("max_candidate_states", default_max_candidate_states))
        max_candidate_states = int(min(num_states, max(beam_width, max_candidate_states)))
        max_neighbor_swaps = int(mapper_cfg.get("max_neighbor_swaps", 1))
        max_neighbor_swaps = max(1, max_neighbor_swaps)
        print(
            f"[INFO] [NeighborhoodBoundedDPMapper] num_states={num_states}, max_candidate_states={max_candidate_states}, max_neighbor_swaps={max_neighbor_swaps}"
        )
        print(
            f"[INFO] [NeighborhoodBoundedDPMapper] num_states={num_states}, max_candidate_states={max_candidate_states}, max_neighbor_swaps={max_neighbor_swaps}", file=sys.stderr
        )

        def _add_state(cands: list[int], seen: set[int], idx: int) -> bool:
            if idx in seen:
                return False
            cands.append(idx)
            seen.add(idx)
            return True

        def _build_candidate_indices(seed_indices: list[int], cap: int) -> list[int]:
            """
            从上一层保留状态出发，生成局部邻域候选。
            顺序优先：seed 本身 -> 单次交换邻域 -> 双次交换邻域（可选）。
            """
            cands: list[int] = []
            seen: set[int] = set()

            # 1) 保留原状态
            for idx in seed_indices:
                _add_state(cands, seen, idx)
                if len(cands) >= cap:
                    return cands

            # 2) 邻域扩展
            frontier = list(seed_indices)
            for _ in range(max_neighbor_swaps):
                next_frontier: list[int] = []
                for idx in frontier:
                    for nidx in one_swap_neighbors[idx]:
                        if _add_state(cands, seen, nidx):
                            next_frontier.append(nidx)
                            if len(cands) >= cap:
                                return cands
                frontier = next_frontier
                if not frontier:
                    break

            return cands

        def _apply_perm(partition: list[list[int]], perm: tuple[int, ...]) -> list[list[int]]:
            return [partition[i] for i in perm]

        def _evaluate_group_with_perm(
            perm: tuple[int, ...],
            seg_start: int,
            seg_end: int,
            prev_partition: Optional[list[list[int]]],
            logical_phy_map: dict[Any, Any],
            network: Network,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
        ) -> tuple[ExecCosts, dict[Any, Any], list[list[int]]]:
            """
            在固定perm下顺序评估一个组内所有子段：
            - 组间/组内切换的 teledata
            - 每段的 local+telegate
            返回该组总增量成本、更新后的logical_phy_map、组末partition。
            """
            group_costs = ExecCosts()
            curr_prev_partition = prev_partition
            curr_map = copy.deepcopy(logical_phy_map)

            for seg_idx in range(seg_start, seg_end):
                seg_record = mapping_record_list.records[seg_idx]
                partition = _apply_perm(seg_record.partition, perm)
                seg_cat_controls = None
                if seg_record.extra_info is not None:
                    seg_cat_controls = seg_record.extra_info.get("cat_controls")

                if curr_prev_partition is not None:
                    td_costs, curr_map = CompilerUtils.evaluate_teledata(
                        curr_prev_partition, partition, network, curr_map
                    )
                    group_costs += td_costs

                if seg_record.extra_info is not None and "ops" in seg_record.extra_info:
                    subcircuit = copy.deepcopy(seg_record.extra_info["ops"])
                    subcircuit = self._remap_commop_endpoints_by_perm(subcircuit, perm)
                else:
                    if seg_record.mapping_type == "cat":
                        raise ValueError("[ERROR] For cat-type record, extra_info should have ops.")
                    subcircuit = CompilerUtils.get_subcircuit_by_level(
                        num_qubits=circuit.num_qubits,
                        circuit=circuit,
                        circuit_layers=circuit_layers,
                        layer_start=seg_record.layer_start,
                        layer_end=seg_record.layer_end,
                    )
                tg_costs, curr_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                    partition,
                    subcircuit,
                    network,
                    logical_phy_map=curr_map,
                    optimization_level=0,
                )
                group_costs += tg_costs
                curr_prev_partition = partition

            assert curr_prev_partition is not None
            return group_costs, curr_map, curr_prev_partition

        # --------------------------
        # 滚动DP数组（仅保留上一层）
        # --------------------------
        prev_bests: list[MappingRecord] = []

        # --------------------------
        # 束搜索核心数据结构
        # --------------------------
        # back[t][kept_idx] = 上一层的kept_idx
        back: list[list[int]] = []
        # 每层保留的【原始perm idx】列表：map_kept[t][kept_idx] = original_perm_idx
        map_kept_to_original: list[list[int]] = []

        # ==========================
        # t=0 初始化（第一层）
        # ==========================
        group0_start = group_starts[0]
        group0_end = min(k, group0_start + group_size)

        # t=0 候选状态：状态空间小时全枚举，否则从恒等排列邻域扩展
        if num_states <= max_candidate_states:
            t0_candidate_indices = list(range(num_states))
        else:
            identity_idx = perm_to_idx[tuple(range(n_physical))]
            t0_candidate_indices = _build_candidate_indices([identity_idx], max_candidate_states)

        # 计算 t=0 候选状态
        t0_full = []
        for state_idx in t0_candidate_indices:
            perm = all_perms[state_idx]
            init_partition = _apply_perm(mapping_record_list.records[group0_start].partition, perm)
            logical_phy_map = CompilerUtils.init_logical_phy_map(init_partition)
            costs, logical_phy_map, last_partition = _evaluate_group_with_perm(
                perm=perm,
                seg_start=group0_start,
                seg_end=group0_end,
                prev_partition=None,
                logical_phy_map=logical_phy_map,
                network=network,
                circuit=circuit,
                circuit_layers=circuit_layers,
            )
            t0_full.append(MappingRecord(
                partition=last_partition,
                costs=costs,
                logical_phy_map=logical_phy_map
            ))

        # --------------------------
        # t0 束搜索：保留 top-K
        # --------------------------
        t0_sorted = sorted(
            enumerate(t0_full),
            key=lambda x: (x[1].costs.total_fidelity_loss, x[1].costs.epairs)
        )
        t0_sorted = t0_sorted[:beam_width]
        kept_indices_t0 = [t0_candidate_indices[i] for i, _ in t0_sorted]
        kept_records_t0 = [r for _, r in t0_sorted]

        prev_bests = kept_records_t0 # 记录前序最好的MappingRecord
        map_kept_to_original.append(kept_indices_t0)
        back.append([])  # t0 无前驱

        # ==========================
        # t = 1 ~ num_groups-1 迭代
        # ==========================
        for t in range(1, num_groups):
            subc_start_time = time.time()
            group_start = group_starts[t]
            group_end = min(k, group_start + group_size)

            prev_original_indices = map_kept_to_original[t - 1]
            candidate_indices = _build_candidate_indices(prev_original_indices, max_candidate_states)

            curr_bests : list[MappingRecord] = []
            back_t_full: list[int] = []

            # 仅计算候选状态，避免每层全枚举 n_physical! 排列
            for curr_idx in candidate_indices:
                curr_perm = all_perms[curr_idx]

                best_record = None
                best_prev_kept_idx = -1

                # 只遍历上一层保留的K个状态
                for prev_kept_idx in range(len(prev_bests)):
                    prev_record = prev_bests[prev_kept_idx]
                    curr_total_costs = copy.deepcopy(prev_record.costs)
                    logical_phy_map = copy.deepcopy(prev_record.logical_phy_map)

                    group_costs, logical_phy_map, last_partition = _evaluate_group_with_perm(
                        perm=curr_perm,
                        seg_start=group_start,
                        seg_end=group_end,
                        prev_partition=prev_record.partition,
                        logical_phy_map=logical_phy_map,
                        network=network,
                        circuit=circuit,
                        circuit_layers=circuit_layers,
                    )
                    curr_total_costs += group_costs

                    # 更新最优
                    if best_record is None or \
                        (curr_total_costs.total_fidelity_loss, curr_total_costs.epairs) < \
                        (best_record.costs.total_fidelity_loss, best_record.costs.epairs):
                        best_record = MappingRecord(
                            partition=last_partition,
                            costs=curr_total_costs,
                            logical_phy_map=logical_phy_map
                        )
                        best_prev_kept_idx = prev_kept_idx

                assert best_record is not None
                curr_bests.append(best_record)
                back_t_full.append(best_prev_kept_idx)

            # --------------------------
            # 束搜索：当前层排序 + 裁剪 top-K
            # --------------------------
            curr_sorted = sorted(
                enumerate(curr_bests),
                key=lambda x: (x[1].costs.total_fidelity_loss, x[1].costs.epairs)
            )
            curr_sorted = curr_sorted[:beam_width]

            kept_original_indices = [candidate_indices[i] for i, _ in curr_sorted]
            kept_records = [r for _, r in curr_sorted]

            # 裁剪 back 指针
            back_t_kept = [back_t_full[i] for i, _ in curr_sorted]
            back.append(back_t_kept)

            # 更新滚动数组
            prev_bests = kept_records
            map_kept_to_original.append(kept_original_indices)
            curr_bests = []

            subc_end_time = time.time()
            print(
                f"[DEBUG] subc: {t}, candidates={len(candidate_indices)}, "
                f"kept={len(kept_original_indices)}, time: {subc_end_time-subc_start_time}"
            )


        # ==========================
        # 回溯最优路径
        # ==========================
        # 最后一层最优 kept_idx
        best_last_kept = 0
        for idx, r in enumerate(prev_bests):
            if (r.costs.total_fidelity_loss, r.costs.epairs) < \
               (prev_bests[best_last_kept].costs.total_fidelity_loss, prev_bests[best_last_kept].costs.epairs):
                best_last_kept = idx

        # 存储每层的原始 perm idx
        perm_original_indices_group = [0] * num_groups
        perm_original_indices_group[num_groups-1] = map_kept_to_original[num_groups-1][best_last_kept]

        # 反向回溯
        current_kept_idx = best_last_kept # 在保留项当中的最优解的编号
        for t in range(num_groups-2, -1, -1):
            # 从 back 表找上一层 kept idx
            current_kept_idx = back[t+1][current_kept_idx]
            # 转为原始 perm idx
            perm_original_indices_group[t] = map_kept_to_original[t][current_kept_idx]

        # 将组级决策展开到每个时间片：同组子段共享同一个perm。
        perm_original_indices = [0] * k
        for gid, start_idx in enumerate(group_starts):
            end_idx = min(k, start_idx + group_size)
            for t in range(start_idx, end_idx):
                perm_original_indices[t] = perm_original_indices_group[gid]

        final_perm_sequence = [all_perms[idx] for idx in perm_original_indices]

        # ==========================
        # 生成最终映射结果
        # ==========================
        final_record_list = MappingRecordList()
        for t in range(k):
            record = mapping_record_list.records[t]
            orig_idx = perm_original_indices[t]
            perm = all_perms[orig_idx]
            best_partition = [record.partition[i] for i in perm]
            new_extra_info = copy.deepcopy(record.extra_info) if record.extra_info is not None else {}
            if "ops" in new_extra_info:
                new_extra_info["ops"] = self._remap_commop_endpoints_by_perm(
                    copy.deepcopy(new_extra_info["ops"]),
                    perm,
                )
            new_extra_info["perm"] = list(perm)

            new_record = MappingRecord(
                layer_start=record.layer_start,
                layer_end=record.layer_end,
                partition=best_partition,
                mapping_type=record.mapping_type,
                logical_phy_map=CompilerUtils.init_logical_phy_map(best_partition),
                extra_info=new_extra_info
            )
            final_record_list.add_record(new_record)

        # 最终精确重评估（保持你原来的逻辑）
        final_record_list = self._reevaluate_mapping_record_list(
            final_record_list, circuit, circuit_layers, network, config=config
        )

        end_time = time.time()
        print(f"[INFO] Neighborhood Bounded DP 运行完成 | 耗时={end_time - start_time:.2f}s")
        print(f"[INFO] Neighborhood Bounded DP 运行完成 | 耗时={end_time - start_time:.2f}s", file=sys.stderr)
        print(f"[INFO] [Neighborhood Bounded DP Mapper] final mapping perm sequence (from DP backtrace): {final_perm_sequence}")

        return final_record_list


class BoundedDPMapper(Mapper):
    @property
    def name(self) -> str:
        return "Bounded DP Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network,
            config: Optional[dict[str, Any]] = None,
        ) -> MappingRecordList:
        """
        原始版本：带束搜索的动态规划映射器。
        保留全状态枚举当前层（num_states = n_physical!），仅在前驱层做束裁剪。
        """
        start_time = time.time()
        mapper_cfg = config or {}
        beam_width = int(mapper_cfg.get("beam_width", 5))
        k = len(mapping_record_list.records)
        n_physical = network.num_backends
        print(f"[INFO] [BoundedDPMapper] beam_width: {beam_width}, n_subcs: {k}, n_QPUs: {n_physical}")

        # 所有物理QPU排列 + 固定索引（idx = state_key）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # --------------------------
        # 滚动DP数组（仅保留上一层）
        # --------------------------
        prev_bests: list[MappingRecord] = []

        # --------------------------
        # 束搜索核心数据结构
        # --------------------------
        # back[t][kept_idx] = 上一层的kept_idx
        back: list[list[int]] = []
        # 每层保留的【原始perm idx】列表：map_kept[t][kept_idx] = original_perm_idx
        map_kept_to_original: list[list[int]] = []

        # ==========================
        # t=0 初始化（第一层）
        # ==========================
        record = mapping_record_list.records[0]
        original_partition = record.partition
        subcircuit = CompilerUtils.get_subcircuit_by_level(
            num_qubits=circuit.num_qubits,
            circuit=circuit,
            circuit_layers=circuit_layers,
            layer_start=record.layer_start,
            layer_end=record.layer_end
        )

        # 计算所有初始状态
        t0_full = []
        for perm in all_perms:
            partition = [original_partition[i] for i in perm]
            logical_phy_map = CompilerUtils.init_logical_phy_map(partition)
            costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                partition, subcircuit, network,
                logical_phy_map=logical_phy_map,
                optimization_level=0
            )
            t0_full.append(MappingRecord(
                partition=partition,
                costs=costs,
                logical_phy_map=logical_phy_map
            ))

        # --------------------------
        # t0 束搜索：保留 top-K
        # --------------------------
        t0_sorted = sorted(
            enumerate(t0_full),
            key=lambda x: (x[1].costs.total_fidelity_loss, x[1].costs.epairs)
        )
        t0_sorted = t0_sorted[:beam_width]
        kept_indices_t0 = [i for i, _ in t0_sorted]
        kept_records_t0 = [r for _, r in t0_sorted]

        prev_bests = kept_records_t0
        map_kept_to_original.append(kept_indices_t0)
        back.append([])  # t0 无前驱

        # ==========================
        # t = 1 ~ k-1 迭代
        # ==========================
        for t in range(1, k):
            subc_start_time = time.time()
            record = mapping_record_list.records[t]
            original_partition = record.partition
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )

            curr_bests: list[MappingRecord] = []
            back_t_full = [-1] * num_states

            # 计算当前层所有状态
            for curr_idx in range(num_states):
                curr_perm = all_perms[curr_idx]
                partition = [original_partition[i] for i in curr_perm]

                best_record = None
                best_prev_kept_idx = -1

                # 只遍历上一层保留的K个状态
                for prev_kept_idx in range(len(prev_bests)):
                    prev_record = prev_bests[prev_kept_idx]
                    curr_total_costs = copy.deepcopy(prev_record.costs)
                    logical_phy_map = copy.deepcopy(prev_record.logical_phy_map)

                    # 计算切换开销 + 执行开销
                    teledata_costs, logical_phy_map = CompilerUtils.evaluate_teledata(
                        prev_record.partition, partition, network, logical_phy_map
                    )
                    telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate_with_cat(
                        partition, subcircuit, network,
                        logical_phy_map=logical_phy_map,
                        optimization_level=0
                    )

                    curr_total_costs += teledata_costs
                    curr_total_costs += telegate_costs

                    # 更新最优
                    if best_record is None or \
                        (curr_total_costs.total_fidelity_loss, curr_total_costs.epairs) < \
                        (best_record.costs.total_fidelity_loss, best_record.costs.epairs):
                        best_record = MappingRecord(
                            partition=partition,
                            costs=curr_total_costs,
                            logical_phy_map=logical_phy_map
                        )
                        best_prev_kept_idx = prev_kept_idx

                assert best_record is not None
                curr_bests.append(best_record)
                back_t_full[curr_idx] = best_prev_kept_idx

            # --------------------------
            # 束搜索：当前层排序 + 裁剪 top-K
            # --------------------------
            curr_sorted = sorted(
                enumerate(curr_bests),
                key=lambda x: (x[1].costs.total_fidelity_loss, x[1].costs.epairs)
            )
            curr_sorted = curr_sorted[:beam_width]

            kept_original_indices = [i for i, _ in curr_sorted]
            kept_records = [r for _, r in curr_sorted]

            # 裁剪 back 指针
            back_t_kept = [back_t_full[i] for i in kept_original_indices]
            back.append(back_t_kept)

            # 更新滚动数组
            prev_bests = kept_records
            map_kept_to_original.append(kept_original_indices)
            curr_bests = []

            subc_end_time = time.time()
            print(f"[DEBUG] subc: {t}, time: {subc_end_time-subc_start_time}")


        # ==========================
        # 回溯最优路径
        # ==========================
        # 最后一层最优 kept_idx
        best_last_kept = 0
        for idx, r in enumerate(prev_bests):
            if (r.costs.total_fidelity_loss, r.costs.epairs) < \
               (prev_bests[best_last_kept].costs.total_fidelity_loss, prev_bests[best_last_kept].costs.epairs):
                best_last_kept = idx

        # 存储每层的原始 perm idx
        perm_original_indices = [0] * k
        perm_original_indices[k-1] = map_kept_to_original[k-1][best_last_kept]

        # 反向回溯
        current_kept_idx = best_last_kept # 在保留项当中的最优解的编号
        for t in range(k-2, -1, -1):
            # 从 back 表找上一层 kept idx
            current_kept_idx = back[t+1][current_kept_idx]
            # 转为原始 perm idx
            perm_original_indices[t] = map_kept_to_original[t][current_kept_idx]

        final_perm_sequence = [all_perms[idx] for idx in perm_original_indices]

        # ==========================
        # 生成最终映射结果
        # ==========================
        final_record_list = MappingRecordList()
        for t in range(k):
            record = mapping_record_list.records[t]
            orig_idx = perm_original_indices[t]
            perm = all_perms[orig_idx]
            best_partition = [record.partition[i] for i in perm]
            new_extra_info = copy.deepcopy(record.extra_info) if record.extra_info is not None else {}
            if "ops" in new_extra_info:
                new_extra_info["ops"] = self._remap_commop_endpoints_by_perm(
                    copy.deepcopy(new_extra_info["ops"]),
                    perm,
                )
            new_extra_info["perm"] = list(perm)

            new_record = MappingRecord(
                layer_start=record.layer_start,
                layer_end=record.layer_end,
                partition=best_partition,
                mapping_type=record.mapping_type,
                logical_phy_map=CompilerUtils.init_logical_phy_map(best_partition),
                extra_info=new_extra_info
            )
            final_record_list.add_record(new_record)

        # 最终精确重评估
        final_record_list = self._reevaluate_mapping_record_list(
            final_record_list, circuit, circuit_layers, network, config=config
        )

        end_time = time.time()
        print(f"[INFO] Bounded DP 运行完成 | 束宽={beam_width} | 耗时={end_time - start_time:.2f}s")
        print(f"[INFO] [Bounded DP Mapper] final mapping perm sequence (from DP backtrace): {final_perm_sequence}")

        return final_record_list


class MapperFactory:
    """映射器工厂类"""
    _registry = {
        "direct": "DirectMapper",
        "single_record_debug": "SingleRecordExhaustiveMapper",
        "single_record_greedy": "SingleRecordGreedyMapper",
        # "single_record_refined": "SingleRecordGreedyRefinedMapper",
        "greedy": "GreedyMapper",
        "dp": "DPMapper",
        "boundeddp": "BoundedDPMapper",
        "boundeddp_neighbor": "NeighborhoodBoundedDPMapper"
    }
    
    @classmethod
    def create_mapper(cls, mapper_type: str) -> Mapper:
        """
        创建指定类型的映射器
        
        :param mapper_type: 映射器类型字符串
        :return: 对应的映射器实例
        """
        mapper_type = mapper_type.lower()
        if mapper_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown mapper: '{mapper_type}'. Available mappers: {available}")
        
        # 从注册表获取类名，然后创建实例
        mapper_class_name = cls._registry[mapper_type]
        mapper_class = globals()[mapper_class_name]
        return mapper_class()
    
    @classmethod
    def register_mapper(cls, name: str, class_name: str):
        """
        动态注册新的映射器类型
        
        :param name: 映射器类型名称
        :param class_name: 对应的类名字符串
        """
        cls._registry[name.lower()] = class_name
    
    @classmethod
    def unregister_mapper(cls, name: str):
        """
        移除注册的映射器类型
        
        :param name: 要移除的映射器类型名称
        """
        if name.lower() in cls._registry:
            del cls._registry[name.lower()]
    
    @classmethod
    def get_available_mappers(cls):
        """
        获取所有可用的映射器类型
        
        :return: 可用映射器类型列表
        """
        return list(cls._registry.keys())
