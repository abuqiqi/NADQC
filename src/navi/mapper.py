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

from qiskit import QuantumCircuit

from ..compiler import MappingRecord, MappingRecordList, CompilerUtils, ExecCosts
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
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件
        :param partition_plan: 量子比特划分
        :param network: 目标量子硬件
        :return: 映射结果
        """
        pass

    def _compute_hop_demand(self,
                            partition: list[list[int]],
                            circuit: QuantumCircuit,
                            circuit_layers: list,
                            layer_start: int,
                            layer_end: int
        ) -> np.ndarray:
        """
        计算在原始排列下，每两个QPU之间的remote hops数量
        """
        m_logical = len(partition)
        D_hop = np.zeros((m_logical, m_logical), dtype=int)

        # 创建量子比特到逻辑 QPU 索引的快速查找表
        qubit_to_logical = {}
        for logical_idx, group in enumerate(partition):
            for qubit in group:
                qubit_to_logical[qubit] = logical_idx

        # 遍历指定范围内的所有电路层
        for layer_idx in range(layer_start, layer_end):
            layer = circuit_layers[layer_idx]

            # 遍历该层的所有门操作
            for node in layer:
                qubits = [circuit.qubits.index(node.qargs[i]) for i in range(len(node.qargs))]
                assert all(q is not None for q in qubits), "量子比特索引获取失败"  # 增强断言
                
                # 只处理两量子比特门（单量子比特门无跨QPU通信需求）
                if len(qubits) < 2:
                    continue  # 跳过单比特门

                # 生成相邻的量子比特对（按qargs顺序）
                # 例如qubits = [0,1,2] → 生成(0,1)、(1,2)；qubits=[a,b,c,d] → (a,b)、(b,c)、(c,d)
                for i in range(len(qubits) - 1):
                    q1 = qubits[i]
                    q2 = qubits[i + 1]
                    
                    # 获取相邻对所属的逻辑QPU并统计跨QPU hops
                    qpu_a = qubit_to_logical[q1]
                    qpu_b = qubit_to_logical[q2]
                    if qpu_a != qpu_b:
                        D_hop[qpu_a][qpu_b] += 1
                        D_hop[qpu_b][qpu_a] += 1

        return D_hop

    def _compute_move_demand(self, 
                             current_partition: list[list[int]], 
                             next_partition: list[list[int]]
        ) -> np.ndarray:
        """
        计算单次切换的通信需求
        :param current_partition: 当前时间片的分区
        :param next_partition: 下一时间片的分区
        :return: D_move
        """
        # 创建量子比特到逻辑QPU的映射
        current_qubit_to_logical = {}
        for logical_idx, group in enumerate(current_partition):
            for qubit in group:
                current_qubit_to_logical[qubit] = logical_idx
        
        next_qubit_to_logical = {}
        for logical_idx, group in enumerate(next_partition):
            for qubit in group:
                next_qubit_to_logical[qubit] = logical_idx
        
        # 初始化需求矩阵
        m_logical_current = len(current_partition)
        m_logical_next = len(next_partition)
        D_move = np.zeros((m_logical_current, m_logical_next), dtype=int)
        
        # 记录每个量子比特的移动
        # qubit_movements = {}
        
        # 计算需求
        num_qubits = len(current_qubit_to_logical)
        for qubit in range(num_qubits):
            curr_logical = current_qubit_to_logical[qubit]
            next_logical = next_qubit_to_logical[qubit]
            # qubit_movements[qubit] = (curr_logical, next_logical)
            D_move[curr_logical][next_logical] += 1

        return D_move #, qubit_movements

    def _evaluate_hop_cost(self, 
                           network: Network,
                           D_hops: np.ndarray,
                           perm: tuple[int, ...]) -> ExecCosts:
        """
        计算单个时间片内 remote hops 导致的通信成本（telegate成本）
        :param D_hops: 逻辑QPU间的hop需求矩阵
        :param perm: 逻辑QPU到物理QPU的排列（映射）
        :return: 封装后的成本对象
        """
        move_fidelity = network.move_fidelity  # 物理QPU间的有效保真度/权重矩阵
        move_fidelity_loss = network.move_fidelity_loss
        Hops  = network.Hops

        costs = ExecCosts()

        # 遍历所有逻辑QPU对
        m_logical = D_hops.shape[0]

        for i in range(m_logical):
            for j in range(i + 1, m_logical): # j从i+1开始
                hop_count = D_hops[i][j]
                if hop_count == 0:
                    continue

                # 映射到物理QPU
                physical_i = perm[i]
                physical_j = perm[j]
                
                # 计算成本
                # w_loss = move_fidelity_loss[physical_i][physical_j]
                # w = move_fidelity[physical_i][physical_j]
                # d = Hops[physical_i][physical_j]
                
                # costs.remote_fidelity_loss += w_loss * hop_count
                # costs.remote_fidelity_log_sum += hop_count * np.log(w)
                # costs.remote_fidelity *= (w ** hop_count)
                # costs.remote_hops += d * hop_count
                # costs.epairs += d * hop_count
                costs = CompilerUtils.update_remote_move_costs(
                    costs, physical_i, physical_j, hop_count, network
                )

        return costs

    def _evaluate_move_cost(self, 
                            network: Network,
                            D_move: np.ndarray, 
                            mapping_current: tuple[int, ...], 
                            mapping_next: tuple[int, ...]) -> ExecCosts:
        """
        计算切换成本
        :param D_move: 通信需求矩阵
        :param mapping_current: 当前映射
        :param mapping_next: 下一映射
        :return: ExecCosts
        """
        move_fidelity = network.move_fidelity
        Hops  = network.Hops
        
        costs = ExecCosts()

        # 基于需求矩阵计算（可选）
        for i in range(D_move.shape[0]):
            for j in range(D_move.shape[1]):
                demand = D_move[i][j]
                if demand > 0:
                    from_physical = mapping_current[i]
                    to_physical = mapping_next[j]

                    # 计算切换成本
                    # w = move_fidelity[from_physical][to_physical]
                    # d = Hops[from_physical][to_physical]

                    # costs.remote_fidelity_loss += (1 - w) * demand
                    # costs.remote_fidelity_log_sum += demand * np.log(w)
                    # costs.remote_fidelity *= (w ** demand)
                    # costs.remote_swaps += d * demand
                    # costs.epairs += d * demand
                    costs = CompilerUtils.update_remote_move_costs(
                        costs, from_physical, to_physical, demand, network
                    )

        return costs

    def _fast_evaluate_teledata(
        self,
        logical_move_count: np.ndarray,
        perm_prev: tuple[int, ...],  # 物理QPU→逻辑prev QPU的排列
        perm_curr: tuple[int, ...],  # 物理QPU→逻辑curr QPU的排列
        network: Network
    ) -> ExecCosts:
        """
        基于预计算的逻辑移动计数，快速计算teledata成本
        完全复刻原函数的图处理、抵消、环计算逻辑，结果100%一致
        """
        costs = ExecCosts()

        # 1. 构建逆映射：逻辑QPU→物理QPU
        # perm_prev[p] = A → 物理QPU p对应逻辑prev QPU A → 逻辑A对应物理p
        inv_prev = {A: p for p, A in enumerate(perm_prev)}
        inv_curr = {B: p for p, B in enumerate(perm_curr)}

        # 2. 生成物理QPU间的移动计数（无需遍历量子比特，仅矩阵映射）
        n_phys = network.num_backends
        phys_count = np.zeros((n_phys, n_phys), dtype=int)
        m_prev, m_curr = logical_move_count.shape
        for A in range(m_prev):
            for B in range(m_curr):
                cnt = logical_move_count[A][B]
                if cnt == 0:
                    continue
                u = inv_prev[A]  # 逻辑A→物理u
                v = inv_curr[B]  # 逻辑B→物理v
                phys_count[u][v] += cnt

        # 3. 1:1复刻原函数的图处理逻辑
        G = nx.DiGraph()
        G.add_nodes_from(range(n_phys))

        # 3.1 处理边抵消（2节点环）
        for u in range(n_phys):
            for v in range(n_phys):
                cnt = phys_count[u][v]
                if cnt == 0 or u == v:
                    continue
                
                # 检查反向边是否存在
                if G.has_edge(v, u):
                    # 可抵消的数量
                    cancel_num = min(cnt, G[v][u]['weight'])
                    if cancel_num == 0:
                        continue
                    
                    # 1:1复刻原函数的抵消成本计算
                    # num_rswaps = (2 * network.Hops[v][u] - 1) * cancel_num
                    # w = network.move_fidelity[u][v]
                    # costs.remote_swaps += num_rswaps
                    # costs.epairs += 2 * num_rswaps
                    # costs.remote_fidelity_loss += (1 - w) * num_rswaps
                    # costs.remote_fidelity *= w ** num_rswaps
                    # costs.remote_fidelity_log_sum += num_rswaps * np.log(w)
                    costs = CompilerUtils.update_remote_swap_costs(
                        costs, u, v, cancel_num, network
                    )

                    # 更新图的边权重
                    if G[v][u]['weight'] > cancel_num:
                        G[v][u]['weight'] -= cancel_num
                    else:
                        G.remove_edge(v, u)
                    
                    # 剩余未抵消的数量
                    cnt -= cancel_num
                    if cnt == 0:
                        continue
                
                # 添加剩余的边到图中
                if G.has_edge(u, v):
                    G[u][v]['weight'] += cnt
                else:
                    G.add_edge(u, v, weight=cnt)

        # 3.2 处理长度≥3的环（1:1复刻原函数）
        all_cycles = nx.simple_cycles(G)
        cycles_by_length = defaultdict(list)
        for cycle in all_cycles:
            length = len(cycle)
            if 3 <= length <= network.num_backends:
                cycles_by_length[length].append(cycle)

        for length in sorted(cycles_by_length.keys()):
            for cycle in cycles_by_length[length]:
                exist = True
                min_weight = float('inf')
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    if not G.has_edge(u, v):
                        exist = False
                        break
                    min_weight = min(min_weight, G[u][v]['weight'])
                if not exist:
                    continue

                # 计算环的成本
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    # num_rswaps = (2 * network.Hops[u][v] - 1) * min_weight
                    # w = network.move_fidelity[u][v]
                    # costs.remote_swaps += num_rswaps
                    # costs.epairs += 2 * num_rswaps
                    # costs.remote_fidelity_loss += (1 - w) * num_rswaps
                    # costs.remote_fidelity *= w ** num_rswaps
                    # costs.remote_fidelity_log_sum += num_rswaps * np.log(w)
                    costs = CompilerUtils.update_remote_swap_costs(
                        costs, u, v, int(min_weight), network
                    )

                # 更新图的边权重
                for i in range(length):
                    u = cycle[i]
                    v = cycle[(i + 1) % length]
                    if G[u][v]['weight'] > min_weight:
                        G[u][v]['weight'] -= min_weight
                    else:
                        G.remove_edge(u, v)

        # 3.3 处理剩余边（1:1复刻原函数）
        remaining_edges = G.edges(data=True)
        for u, v, data in remaining_edges:
            # path_len = network.Hops[u][v]
            # num_rswaps = path_len * data['weight']
            # w = network.move_fidelity[u][v]
            # costs.remote_swaps += num_rswaps
            # costs.epairs += num_rswaps
            # costs.remote_fidelity_loss += (1 - w) * num_rswaps
            # costs.remote_fidelity *= w ** num_rswaps
            # costs.remote_fidelity_log_sum += num_rswaps * np.log(w)
            costs = CompilerUtils.update_remote_move_costs(
                costs, u, v, data['weight'], network
            )

        return costs

    def _reevaluate_mapping_record_list(
            self, 
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network
        ) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件（基线实现）
        """
        # print(f"\n\n\n[DEBUG] _reevaluate_mapping_record_list\n\n\n")
        # ---------- 更新映射记录 ----------
        k = len(mapping_record_list.records)  # 时间片数量

        logical_phy_map = {}
        for t in range(k):
            # print(f"\n\n\n[DEBUG] {t} / {k}")

            record = mapping_record_list.records[t]

            if t == 0:
                logical_phy_map = CompilerUtils.init_logical_phy_map(record.partition)
            else:
                # 沿用上一个record的logical_phy_map作为初始状态
                logical_phy_map = mapping_record_list.records[t-1].logical_phy_map

            record.costs = ExecCosts()  # 初始化成本对象
            record.logical_phy_map = logical_phy_map.copy() # 初始化当前时间片的logical_phy_map

            # 
            # print(f"[DEBUG] After new init: \n{record}")
            # print(f"[DEBUG] partition: {record.partition}")
            # print(f"[DEBUG] logical_phy_map: {logical_phy_map}")

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

            # 计算local和telegate损失
            _ = CompilerUtils.evaluate_local_and_telegate(record, subcircuit, network)
            # print(f"[DEBUG] record (local and telegate): \n{record}\n\n")

            # print(f"[DEBUG] logical_phy_map: {record.logical_phy_map}")
            # print(f"[DEBUG] After mapping - Time slice {t}: \n{record} \n\n")

        return mapping_record_list

    # def _evaluate_initial_mapping(self, network: Network, mapping: list[int], D_total: np.ndarray) -> float:
    #     """
    #     评估初始映射的质量
    #     :param mapping: 映射
    #     :param D_total: 总通信需求矩阵
    #     :return: 映射得分
    #     """
    #     move_fidelity = network.move_fidelity
    #     # pprint(move_fidelity)
    #     mapping_score = 0.0
    #     for i in range(D_total.shape[0]):
    #         for j in range(D_total.shape[1]):
    #             from_physical = mapping[i]
    #             to_physical = mapping[j]
    #             mapping_score += move_fidelity[from_physical][to_physical] * D_total[i][j]
    #     return mapping_score

    # def _validate_network_attributes(self, network):
    #     """验证网络对象是否具有必要属性"""
    #     if not hasattr(network, 'num_backends') or not hasattr(network, 'move_fidelity'):
    #         raise AttributeError("Network must have 'num_backends' and 'move_fidelity' attributes")

    # def _build_partition_plan(self, partition_plan, mapping_sequence):
        
    #     # 根据映射序列更新partition_plan
    #     adjusted_partition_plan = []

    #     for t, mapping in enumerate(mapping_sequence):
    #         current_partition = partition_plan[t]
            
    #         # 构建新的分区
    #         new_partition = [[] for _ in range(len(current_partition))]

    #         for logical_idx, group in enumerate(current_partition):
    #             physical_idx = mapping[logical_idx]
    #             new_partition[physical_idx].extend(group)
            
    #         adjusted_partition_plan.append(new_partition)

    #     return adjusted_partition_plan


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
            network: Network) -> MappingRecordList:
        """
        将量子线路映射到特定量子硬件（基线实现）
        """
        mapping_record_list = self._reevaluate_mapping_record_list(
            mapping_record_list,
            circuit,
            circuit_layers,
            network
        )
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
            network: Network
        ) -> MappingRecordList:

        start_time = time.time()
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
                telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate(
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
            network: Network
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
            costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate(
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
                    telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate(
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

        # ----- 更新映射记录 -----
        final_record_list = MappingRecordList()

        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]

            original_partition = record.partition

            best_partition = [original_partition[idx] for idx in perm]
            
            new_record = MappingRecord(
                layer_start = record.layer_start,
                layer_end = record.layer_end,
                partition = best_partition,
                mapping_type = record.mapping_type
            )

            final_record_list.add_record(new_record)

        final_record_list = self._reevaluate_mapping_record_list(
            final_record_list,
            circuit,
            circuit_layers,
            network
        )

        end_time = time.time()
        print(f"[INFO] [Time] DP Mapper completed in {end_time - start_time:.2f} seconds.")

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
            beam_width: int = 5  # 束宽
        ) -> MappingRecordList:
        """
        带束搜索优化的动态规划映射器
        解决 QPU 数量增多时 m! 状态爆炸问题
        """
        start_time = time.time()
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
            costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate(
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

        prev_bests = kept_records_t0 # 记录前序最好的MappingRecord
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

            curr_bests : list[MappingRecord] = []
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
                    telegate_costs, logical_phy_map = CompilerUtils.evaluate_local_and_telegate(
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

        # ==========================
        # 生成最终映射结果
        # ==========================
        final_record_list = MappingRecordList()
        for t in range(k):
            record = mapping_record_list.records[t]
            orig_idx = perm_original_indices[t]
            perm = all_perms[orig_idx]
            best_partition = [record.partition[i] for i in perm]

            new_record = MappingRecord(
                layer_start=record.layer_start,
                layer_end=record.layer_end,
                partition=best_partition,
                mapping_type=record.mapping_type
            )
            final_record_list.add_record(new_record)

        # 最终精确重评估（保持你原来的逻辑）
        final_record_list = self._reevaluate_mapping_record_list(
            final_record_list, circuit, circuit_layers, network
        )

        end_time = time.time()
        print(f"[INFO] Bounded DP 运行完成 | 束宽={beam_width} | 耗时={end_time - start_time:.2f}s")

        return final_record_list

# class NewDPMapper(Mapper):
#     @property
#     def name(self) -> str:
#         """获取映射器名称"""
#         return "New DP Mapper"

#     def map(self,
#             mapping_record_list: MappingRecordList,
#             circuit: QuantumCircuit,
#             circuit_layers: list[Any],
#             network: Network
#         ) -> MappingRecordList:
#         start_time = time.time()
#         k = len(mapping_record_list.records)
#         n_physical = network.num_backends

#         # 所有可能的物理QPU排列（即映射状态）
#         all_perms = list(itertools.permutations(range(n_physical)))
#         num_states = len(all_perms)

#         # ---------- 预计算 local and telegate 成本（原有逻辑保持不变） ----------
#         telegate_costs: list[list[ExecCosts]] = []
#         for t in range(k):
#             record = mapping_record_list.records[t]
#             original_partition = record.partition
#             subcircuit = CompilerUtils.get_subcircuit_by_level(
#                 num_qubits=circuit.num_qubits,
#                 circuit=circuit,
#                 circuit_layers=circuit_layers,
#                 layer_start=record.layer_start,
#                 layer_end=record.layer_end
#             )
#             cost_list = []
#             for perm in all_perms:
#                 partition = [original_partition[idx] for idx in perm]
#                 costs = CompilerUtils.evaluate_local_and_telegate(partition, subcircuit, network, optimization_level=0)
#                 cost_list.append(costs)
#             telegate_costs.append(cost_list)

#         # ---------- 优化后的：预计算 teledata 成本 ----------
#         teledata_costs: list[list[list[ExecCosts]]] = []
#         for t in range(k - 1):
#             record_prev = mapping_record_list.records[t]
#             record_curr = mapping_record_list.records[t+1]
#             orig_part_prev = record_prev.partition
#             orig_part_curr = record_curr.partition

#             # 【核心优化1】仅对原始分区运行1次图结构预计算
#             D_move = self._compute_move_demand(orig_part_prev, orig_part_curr)

#             # 【核心优化2】对每个排列，仅通过映射快速计算开销
#             matrix: list[list[ExecCosts]] = [[ExecCosts()] * num_states for _ in range(num_states)]
#             for prev_idx, perm_prev in enumerate(all_perms):
#                 for curr_idx, perm_curr in enumerate(all_perms):
#                     # 快速计算：图结构 + 物理映射 → 开销
#                     costs = self._fast_evaluate_teledata(D_move, perm_prev, perm_curr, network)
#                     matrix[prev_idx][curr_idx] = costs
#             # print(f"[DEBUG] t: {t}, matrix: ")
#             # pprint(matrix)
#             teledata_costs.append(matrix)

#         # ---------- 动态规划 ----------
#         # dp[t][idx] 存储到时间片 t 状态 idx 的最小累计成本（元组：(fidelity_loss, num_comms)）
#         # dp: list[list[tuple[float, int]]] = [[(float(inf), 9999999)] * num_states for _ in range(k)]
#         dp: list[list[tuple[float, int]]] = [
#             [(inf, -1) for _ in range(num_states)] for _ in range(k)
#         ]
#         # back[t][idx] 存储达到该状态的最优前一状态索引
#         back: list[list[int]] = [[-1 for _ in range(num_states)] for _ in range(k)]

#         # 初始化第一个时间片
#         for idx in range(num_states):
#             cost = telegate_costs[0][idx]
#             dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)
#             back[0][idx] = -1

#         # 递推后续时间片
#         for t in range(1, k):
#             for curr_idx in range(num_states):
#                 curr_telegate = telegate_costs[t][curr_idx]
#                 curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
#                 best_cost = (inf, 0)
#                 best_prev = -1
#                 for prev_idx in range(num_states):
#                     prev_cost = dp[t-1][prev_idx]
#                     teledata_cost = teledata_costs[t-1][prev_idx][curr_idx]   # 从 t-1 到 t 的转移成本
#                     # assert isinstance(prev_cost, tuple), f"Expected tuple, got {type(prev_cost)}"
#                     # assert isinstance(tel, ExecCosts), f"Expected ExecCosts, got {type(tel)}"
#                     tel_tuple = (teledata_cost.total_fidelity_loss, teledata_cost.num_comms)
#                     total = (prev_cost[0] + curr_telegate_tuple[0] + tel_tuple[0],
#                              prev_cost[1] + curr_telegate_tuple[1] + tel_tuple[1])
#                     if total < best_cost:
#                         best_cost = total
#                         best_prev = prev_idx
#                 dp[t][curr_idx] = best_cost
#                 back[t][curr_idx] = best_prev

#         # ---------- 回溯找到最优路径 ----------
#         best_last = min(range(num_states), key=lambda i: dp[k-1][i])
#         perm_indices = [0] * k
#         perm_indices[k-1] = best_last
#         for t in range(k-2, -1, -1):
#             perm_indices[t] = back[t+1][perm_indices[t+1]]

#         # ---------- 更新映射记录 ----------
#         for t in range(k):
#             record = mapping_record_list.records[t]
#             perm = all_perms[perm_indices[t]]
#             original_partition = record.partition
            
#             best_partition = [original_partition[idx] for idx in perm]
#             record.partition = copy.deepcopy(best_partition)

#             # 计算该时间片对应的成本对象

#             # 获取子线路
#             subcircuit = CompilerUtils.get_subcircuit_by_level(
#                 num_qubits=circuit.num_qubits,
#                 circuit=circuit,
#                 circuit_layers=circuit_layers,
#                 layer_start=record.layer_start,
#                 layer_end=record.layer_end
#             )
#             # print(subcircuit)

#             # 计算local和telegate损失
#             record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)

#         # for t in range(k):
#         #     record = mapping_record_list.records[t]
#         #     perm = all_perms[perm_indices[t]]
#         #     original_partition = record.partition
#         #     best_partition = [original_partition[idx] for idx in perm]
#         #     record.partition = copy.deepcopy(best_partition)

#         #     # 计算该时间片对应的成本对象
#         #     record.costs = copy.deepcopy(telegate_costs[t][perm_indices[t]])

#             if t != 0:
#                 record.costs += teledata_costs[t-1][perm_indices[t-1]][perm_indices[t]]
#                 # prev_record = mapping_record_list.records[t - 1]
#                 # record.costs += CompilerUtils.evaluate_teledata(
#                 #     prev_record.partition,
#                 #     record.partition,
#                 #     network
#                 # )

#         end_time = time.time()
#         print(f"[INFO] [Time] New DP Mapper completed in {end_time - start_time:.2f} seconds.")
#         return mapping_record_list


# class LinkOrientedDPMapper(Mapper):
#     @property
#     def name(self) -> str:
#         """获取映射器名称"""
#         return "Link-Ori DP Mapper"
    
#     def map(self,
#             mapping_record_list: MappingRecordList,
#             circuit: QuantumCircuit,
#             circuit_layers: list[Any],
#             network: Network
#         ) -> MappingRecordList:
#         """
#         使用动态规划为每个时间片选择最优的物理QPU映射（排列），
#         使得所有时间片的通信相关成本最小。
#         """
#         start_time = time.time()

#         k = len(mapping_record_list.records)          # 时间片数量
#         n_physical = network.num_backends             # 物理QPU数量

#         # 所有可能的物理QPU排列（即映射状态）
#         all_perms = list(itertools.permutations(range(n_physical)))
#         num_states = len(all_perms)

#         # ---------- 预计算每个时间片各状态下的 telegate 成本 ----------
#         telegate_costs: list[list[ExecCosts]] = [] # telegate_costs[t][idx] -> Costs 对象

#         for t in range(k):
#             record = mapping_record_list.records[t]
#             original_partition = record.partition

#             # 1. 统计每一对QPU间的远程操作数量
#             D_hops = self._compute_hop_demand(
#                 original_partition,
#                 circuit,
#                 circuit_layers,
#                 record.layer_start,
#                 record.layer_end
#             )
            
#             # 2. 为每个排列计算 telegate 成本
#             time_slice_costs = []
#             for perm in all_perms:
#                 telegate_cost = self._evaluate_hop_cost(network, D_hops, perm)
#                 time_slice_costs.append(telegate_cost)
            
#             telegate_costs.append(time_slice_costs)

#         # ---------- 预计算相邻时间片之间的 teledata 成本 ----------
#         # 使用D_move来估算，不要调用CompilerUtils.evaluate_teledata
#         teledata_costs: list[list[list[ExecCosts]]] = [] # teledata_costs[t][prev_idx][curr_idx] 对应从 t 到 t+1 的转移成本
        
#         for t in range(k - 1):  # 从 t 到 t+1 的转移，共 k-1 个转移
#             record_current = mapping_record_list.records[t]
#             record_next = mapping_record_list.records[t + 1]
            
#             # 计算D_move
#             D_move = self._compute_move_demand(
#                 record_current.partition,  # 原始分区（未排列）
#                 record_next.partition     # 原始分区（未排列）
#             )

#             m_logical = D_move.shape[0]
            
#             transition_costs = []
#             for prev_idx, prev_perm in enumerate(all_perms):
#                 row_costs = []
#                 for curr_idx, curr_perm in enumerate(all_perms):
#                     # 无需重复计算D_move！只需将原始D_move映射到物理QPU
#                     # 构建“原始逻辑QPU→物理QPU”的映射
#                     # 计算该排列对的切换成本（直接基于原始D_move映射）
#                     move_cost = self._evaluate_move_cost(
#                         network,
#                         D_move,
#                         prev_perm,
#                         curr_perm
#                     )
#                     row_costs.append(move_cost)
#                 transition_costs.append(row_costs)
#             teledata_costs.append(transition_costs)

#         # ---------- 动态规划 ----------
#         # dp[t][idx]：时间片t处于状态idx的最小累计成本（fidelity_loss, num_comms）
#         dp = [[(inf, inf)] * num_states for _ in range(k)]
#         # back[t][idx]：记录最优路径的前驱状态索引
#         back = [[-1 for _ in range(num_states)] for _ in range(k)]

#         # 初始化第一个时间片
#         for idx in range(num_states):
#             cost = telegate_costs[0][idx]
#             dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)

#         # 递推后续时间片
#         for t in range(1, k):
#             for curr_idx in range(num_states):
#                 curr_telegate = telegate_costs[t][curr_idx]
#                 curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                
#                 # 寻找最优前驱状态
#                 best_cost = (inf, inf)
#                 best_prev_idx = -1

#                 for prev_idx in range(num_states):
#                     # 前驱累计成本 + 当前telegate成本 + 转移成本
#                     prev_total = dp[t-1][prev_idx]
#                     transfer_cost = teledata_costs[t-1][prev_idx][curr_idx]
#                     transfer_tuple = (transfer_cost.total_fidelity_loss, transfer_cost.num_comms)
                    
#                     # 计算总累计成本
#                     total_loss = prev_total[0] + curr_telegate_tuple[0] + transfer_tuple[0]
#                     total_comms = prev_total[1] + curr_telegate_tuple[1] + transfer_tuple[1]
#                     total_cost = (total_loss, total_comms)
                    
#                     # 更新最优解（优先按保真度损失排序，其次通信次数）
#                     if total_cost < best_cost:
#                         best_cost = total_cost
#                         best_prev_idx = prev_idx
    
#                 dp[t][curr_idx] = best_cost
#                 back[t][curr_idx] = best_prev_idx

#         # ---------- 回溯找到最优路径 ----------
#         best_last = min(range(num_states), key=lambda i: dp[k-1][i])
#         perm_indices = [0] * k
#         perm_indices[k-1] = best_last

#         # 反向推导最优路径
#         for t in range(k-2, -1, -1):
#             perm_indices[t] = back[t+1][perm_indices[t+1]]

#         # ---------- 更新映射记录 ----------
#         for t in range(k):
#             record = mapping_record_list.records[t]
#             perm = all_perms[perm_indices[t]]
#             original_partition = record.partition
            
#             best_partition = [original_partition[idx] for idx in perm]
#             record.partition = copy.deepcopy(best_partition)

#             # 计算该时间片对应的成本对象

#             # 获取子线路
#             subcircuit = CompilerUtils.get_subcircuit_by_level(
#                 num_qubits=circuit.num_qubits,
#                 circuit=circuit,
#                 circuit_layers=circuit_layers,
#                 layer_start=record.layer_start,
#                 layer_end=record.layer_end
#             )
#             # print(subcircuit)

#             # 计算local和telegate损失
#             record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)
#             # print(f"[DEBUG] record.costs (local and telegate): {record.costs}")

#             # 计算teledata损失
#             if t != 0:
#                 prev_record = mapping_record_list.records[t - 1]
#                 record.costs += CompilerUtils.evaluate_teledata(
#                     prev_record.partition,
#                     record.partition,
#                     network
#                 )
#                 # print(f"[DEBUG] record.costs (teledata): {record.costs}")

#         # 输出耗时信息
#         end_time = time.time()
#         print(f"[INFO] [Time] Link-Oriented DP Mapper 完成，耗时 {end_time - start_time:.2f} 秒")

#         # print("===================================================")
#         # for record in mapping_record_list.records:
#         #     print(f"[DEBUG] record.costs: {record.costs}")

#         return mapping_record_list


class MapperFactory:
    """映射器工厂类"""
    _registry = {
        "direct": "DirectMapper",
        "greedy": "GreedyMapper",
        "dp": "DPMapper",
        "boundeddp": "BoundedDPMapper"
        # "newdp": "NewDPMapper",
        # "linkdp": "LinkOrientedDPMapper"
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

