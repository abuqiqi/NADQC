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

from qiskit import QuantumCircuit

from ..compiler import MappingRecordList, CompilerUtils, ExecCosts
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
        # ---------- 更新映射记录 ----------
        k = len(mapping_record_list.records)  # 时间片数量
        for t in range(k):
            record = mapping_record_list.records[t]

            # 计算该时间片对应的成本对象

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            # print(subcircuit)

            # 计算local和telegate损失
            record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)
            # print(f"[DEBUG] record.costs (local and telegate): {record.costs}")

            # 计算teledata损失
            if t != 0:
                prev_record = mapping_record_list.records[t - 1]
                record.costs += CompilerUtils.evaluate_teledata(
                    prev_record.partition,
                    record.partition,
                    network
                )
                # print(f"[DEBUG] record.costs (teledata): {record.costs}")
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

        for t in range(k): # 对于每一段线路
            # 使用贪心算法为下一时间片找到映射
            curr_record = mapping_record_list.records[t]
            original_partition = curr_record.partition

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=curr_record.layer_start,
                layer_end=curr_record.layer_end
            )

            # 记录最佳排列
            best_perm = None
            min_costs = None
            min_num_comms = float('inf')
            min_fidelity_loss = float('inf')

            # perm所有可能的排列
            for perm in all_perms:
                # 获取映射顺序
                order = list(perm)

                # 根据order的顺序构建新的partition
                partition = []
                for idx in order:
                    partition.append(original_partition[idx])

                # 评估当前排列的local_and_telegate_cost
                costs = CompilerUtils.evaluate_local_and_telegate(
                    partition,
                    subcircuit,
                    network
                )

                # 评估当前排列和前一个排列（如果有）的teledata_cost
                if t > 0:
                    costs += CompilerUtils.evaluate_teledata(
                        mapping_record_list.records[t-1].partition,
                        partition,
                        network
                    )

                curr_fidelity_loss = costs.total_fidelity_loss
                curr_num_comms = costs.num_comms

                if curr_fidelity_loss < min_fidelity_loss:
                    # 比较最小num_comms和最小fidelity_loss的排列是否一致
                    if curr_num_comms <= min_num_comms:
                        min_num_comms = curr_num_comms
                    # else:
                        # print(f"[NOTE] Found a permutation with lower fidelity loss but higher communication cost: {curr_num_comms} vs {min_num_comms}")
                    min_fidelity_loss = curr_fidelity_loss
                    best_perm = perm
                    min_costs = costs

            # 调整curr_record.partition成最佳排列
            assert best_perm is not None and min_costs is not None, "未找到最佳排列，可能存在问题"
            best_partition = []
            for idx in best_perm:
                best_partition.append(original_partition[idx])
            curr_record.partition = copy.deepcopy(best_partition)
            curr_record.costs = copy.deepcopy(min_costs)

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
        使得所有时间片的总成本（保真度损失+通信成本）最小。
        """
        start_time = time.time()
        k = len(mapping_record_list.records)          # 时间片数量
        n_physical = network.num_backends             # 物理QPU数量

        # 所有可能的物理QPU排列（即映射状态）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # ---------- 预计算每个时间片各状态下的 local_and_telegate 成本 ----------
        telegate_costs: list[list[ExecCosts]] = [] # telegate_costs[t][idx] -> Costs 对象
        for t in range(k):
            record = mapping_record_list.records[t]
            original_partition = record.partition
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            cost_list = []
            for perm in all_perms:
                partition = [original_partition[idx] for idx in perm]
                costs = CompilerUtils.evaluate_local_and_telegate(partition, subcircuit, network, optimization_level=0)
                cost_list.append(costs)
            telegate_costs.append(cost_list)

        # ---------- 预计算相邻时间片之间的 teledata 成本 ----------
        teledata_costs: list[list[list[ExecCosts]]] = []       # teledata_costs[t][prev_idx][curr_idx] 对应从 t 到 t+1 的转移成本
        for t in range(k - 1):
            record_prev = mapping_record_list.records[t]
            record_curr = mapping_record_list.records[t+1]
            orig_part_prev = record_prev.partition
            orig_part_curr = record_curr.partition
            # 构建转移矩阵
            matrix: list[list[ExecCosts]] = [[ExecCosts()] * num_states for _ in range(num_states)]
            for prev_idx, perm_prev in enumerate(all_perms):
                partition_prev = [orig_part_prev[idx] for idx in perm_prev]
                for curr_idx, perm_curr in enumerate(all_perms):
                    partition_curr = [orig_part_curr[idx] for idx in perm_curr]
                    costs = CompilerUtils.evaluate_teledata(partition_prev, partition_curr, network)
                    matrix[prev_idx][curr_idx] = costs
            # print(f"[DEBUG] t: {t}, matrix: ")
            # pprint(matrix)
            teledata_costs.append(matrix)

        # ---------- 动态规划 ----------
        # dp[t][idx] 存储到时间片 t 状态 idx 的最小累计成本（元组：(fidelity_loss, num_comms)）
        # dp: list[list[tuple[float, int]]] = [[(float(inf), 9999999)] * num_states for _ in range(k)]
        dp: list[list[tuple[float, int]]] = [
            [(inf, -1) for _ in range(num_states)] for _ in range(k)
        ]
        # back[t][idx] 存储达到该状态的最优前一状态索引
        back: list[list[int]] = [[-1 for _ in range(num_states)] for _ in range(k)]

        # 初始化第一个时间片
        for idx in range(num_states):
            cost = telegate_costs[0][idx]
            dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)
            back[0][idx] = -1

        # 递推后续时间片
        for t in range(1, k):
            for curr_idx in range(num_states):
                curr_telegate = telegate_costs[t][curr_idx]
                curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                best_cost = (inf, 0)
                best_prev = -1
                for prev_idx in range(num_states):
                    prev_cost = dp[t-1][prev_idx]
                    teledata_cost = teledata_costs[t-1][prev_idx][curr_idx]   # 从 t-1 到 t 的转移成本
                    # assert isinstance(prev_cost, tuple), f"Expected tuple, got {type(prev_cost)}"
                    # assert isinstance(tel, ExecCosts), f"Expected ExecCosts, got {type(tel)}"
                    tel_tuple = (teledata_cost.total_fidelity_loss, teledata_cost.num_comms)
                    total = (prev_cost[0] + curr_telegate_tuple[0] + tel_tuple[0],
                             prev_cost[1] + curr_telegate_tuple[1] + tel_tuple[1])
                    if total < best_cost:
                        best_cost = total
                        best_prev = prev_idx
                dp[t][curr_idx] = best_cost
                back[t][curr_idx] = best_prev

        # ---------- 回溯找到最优路径 ----------
        best_last = min(range(num_states), key=lambda i: dp[k-1][i])
        perm_indices = [0] * k
        perm_indices[k-1] = best_last
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]

        # ---------- 更新映射记录 ----------
        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]
            original_partition = record.partition
            
            best_partition = [original_partition[idx] for idx in perm]
            record.partition = copy.deepcopy(best_partition)

            # 计算该时间片对应的成本对象

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            # print(subcircuit)

            # 计算local和telegate损失
            record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)

        # for t in range(k):
        #     record = mapping_record_list.records[t]
        #     perm = all_perms[perm_indices[t]]
        #     original_partition = record.partition
        #     best_partition = [original_partition[idx] for idx in perm]
        #     record.partition = copy.deepcopy(best_partition)

        #     # 计算该时间片对应的成本对象
        #     record.costs = copy.deepcopy(telegate_costs[t][perm_indices[t]])

            if t != 0:
                record.costs += teledata_costs[t-1][perm_indices[t-1]][perm_indices[t]]

        end_time = time.time()
        print(f"[INFO] [Time] DP Mapper completed in {end_time - start_time:.2f} seconds.")
        return mapping_record_list


class NewDPMapper(Mapper):
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "New DP Mapper"

    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network
        ) -> MappingRecordList:
        start_time = time.time()
        k = len(mapping_record_list.records)
        n_physical = network.num_backends

        # 所有可能的物理QPU排列（即映射状态）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # ---------- 预计算 local and telegate 成本（原有逻辑保持不变） ----------
        telegate_costs: list[list[ExecCosts]] = []
        for t in range(k):
            record = mapping_record_list.records[t]
            original_partition = record.partition
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            cost_list = []
            for perm in all_perms:
                partition = [original_partition[idx] for idx in perm]
                costs = CompilerUtils.evaluate_local_and_telegate(partition, subcircuit, network, optimization_level=0)
                cost_list.append(costs)
            telegate_costs.append(cost_list)

        # ---------- 优化后的：预计算 teledata 成本 ----------
        teledata_costs: list[list[list[ExecCosts]]] = []
        for t in range(k - 1):
            record_prev = mapping_record_list.records[t]
            record_curr = mapping_record_list.records[t+1]
            orig_part_prev = record_prev.partition
            orig_part_curr = record_curr.partition

            # 【核心优化1】仅对原始分区运行1次图结构预计算
            D_move = self._compute_move_demand(orig_part_prev, orig_part_curr)

            # 【核心优化2】对每个排列，仅通过映射快速计算开销
            matrix: list[list[ExecCosts]] = [[ExecCosts()] * num_states for _ in range(num_states)]
            for prev_idx, perm_prev in enumerate(all_perms):
                for curr_idx, perm_curr in enumerate(all_perms):
                    # 快速计算：图结构 + 物理映射 → 开销
                    costs = self._fast_evaluate_teledata(D_move, perm_prev, perm_curr, network)
                    matrix[prev_idx][curr_idx] = costs
            # print(f"[DEBUG] t: {t}, matrix: ")
            # pprint(matrix)
            teledata_costs.append(matrix)

        # ---------- 动态规划 ----------
        # dp[t][idx] 存储到时间片 t 状态 idx 的最小累计成本（元组：(fidelity_loss, num_comms)）
        # dp: list[list[tuple[float, int]]] = [[(float(inf), 9999999)] * num_states for _ in range(k)]
        dp: list[list[tuple[float, int]]] = [
            [(inf, -1) for _ in range(num_states)] for _ in range(k)
        ]
        # back[t][idx] 存储达到该状态的最优前一状态索引
        back: list[list[int]] = [[-1 for _ in range(num_states)] for _ in range(k)]

        # 初始化第一个时间片
        for idx in range(num_states):
            cost = telegate_costs[0][idx]
            dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)
            back[0][idx] = -1

        # 递推后续时间片
        for t in range(1, k):
            for curr_idx in range(num_states):
                curr_telegate = telegate_costs[t][curr_idx]
                curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                best_cost = (inf, 0)
                best_prev = -1
                for prev_idx in range(num_states):
                    prev_cost = dp[t-1][prev_idx]
                    teledata_cost = teledata_costs[t-1][prev_idx][curr_idx]   # 从 t-1 到 t 的转移成本
                    # assert isinstance(prev_cost, tuple), f"Expected tuple, got {type(prev_cost)}"
                    # assert isinstance(tel, ExecCosts), f"Expected ExecCosts, got {type(tel)}"
                    tel_tuple = (teledata_cost.total_fidelity_loss, teledata_cost.num_comms)
                    total = (prev_cost[0] + curr_telegate_tuple[0] + tel_tuple[0],
                             prev_cost[1] + curr_telegate_tuple[1] + tel_tuple[1])
                    if total < best_cost:
                        best_cost = total
                        best_prev = prev_idx
                dp[t][curr_idx] = best_cost
                back[t][curr_idx] = best_prev

        # ---------- 回溯找到最优路径 ----------
        best_last = min(range(num_states), key=lambda i: dp[k-1][i])
        perm_indices = [0] * k
        perm_indices[k-1] = best_last
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]

        # ---------- 更新映射记录 ----------
        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]
            original_partition = record.partition
            
            best_partition = [original_partition[idx] for idx in perm]
            record.partition = copy.deepcopy(best_partition)

            # 计算该时间片对应的成本对象

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            # print(subcircuit)

            # 计算local和telegate损失
            record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)

        # for t in range(k):
        #     record = mapping_record_list.records[t]
        #     perm = all_perms[perm_indices[t]]
        #     original_partition = record.partition
        #     best_partition = [original_partition[idx] for idx in perm]
        #     record.partition = copy.deepcopy(best_partition)

        #     # 计算该时间片对应的成本对象
        #     record.costs = copy.deepcopy(telegate_costs[t][perm_indices[t]])

            if t != 0:
                record.costs += teledata_costs[t-1][perm_indices[t-1]][perm_indices[t]]
                # prev_record = mapping_record_list.records[t - 1]
                # record.costs += CompilerUtils.evaluate_teledata(
                #     prev_record.partition,
                #     record.partition,
                #     network
                # )

        end_time = time.time()
        print(f"[INFO] [Time] New DP Mapper completed in {end_time - start_time:.2f} seconds.")
        return mapping_record_list


class LinkOrientedDPMapper(Mapper):
    @property
    def name(self) -> str:
        """获取映射器名称"""
        return "Link-Ori DP Mapper"
    
    def map(self,
            mapping_record_list: MappingRecordList,
            circuit: QuantumCircuit,
            circuit_layers: list[Any],
            network: Network
        ) -> MappingRecordList:
        """
        使用动态规划为每个时间片选择最优的物理QPU映射（排列），
        使得所有时间片的通信相关成本最小。
        """
        start_time = time.time()

        k = len(mapping_record_list.records)          # 时间片数量
        n_physical = network.num_backends             # 物理QPU数量

        # 所有可能的物理QPU排列（即映射状态）
        all_perms = list(itertools.permutations(range(n_physical)))
        num_states = len(all_perms)

        # ---------- 预计算每个时间片各状态下的 telegate 成本 ----------
        telegate_costs: list[list[ExecCosts]] = [] # telegate_costs[t][idx] -> Costs 对象

        for t in range(k):
            record = mapping_record_list.records[t]
            original_partition = record.partition

            # 1. 统计每一对QPU间的远程操作数量
            D_hops = self._compute_hop_demand(
                original_partition,
                circuit,
                circuit_layers,
                record.layer_start,
                record.layer_end
            )
            
            # 2. 为每个排列计算 telegate 成本
            time_slice_costs = []
            for perm in all_perms:
                telegate_cost = self._evaluate_hop_cost(network, D_hops, perm)
                time_slice_costs.append(telegate_cost)
            
            telegate_costs.append(time_slice_costs)

        # ---------- 预计算相邻时间片之间的 teledata 成本 ----------
        # 使用D_move来估算，不要调用CompilerUtils.evaluate_teledata
        teledata_costs: list[list[list[ExecCosts]]] = [] # teledata_costs[t][prev_idx][curr_idx] 对应从 t 到 t+1 的转移成本
        
        for t in range(k - 1):  # 从 t 到 t+1 的转移，共 k-1 个转移
            record_current = mapping_record_list.records[t]
            record_next = mapping_record_list.records[t + 1]
            
            # 计算D_move
            D_move = self._compute_move_demand(
                record_current.partition,  # 原始分区（未排列）
                record_next.partition     # 原始分区（未排列）
            )

            m_logical = D_move.shape[0]
            
            transition_costs = []
            for prev_idx, prev_perm in enumerate(all_perms):
                row_costs = []
                for curr_idx, curr_perm in enumerate(all_perms):
                    # 无需重复计算D_move！只需将原始D_move映射到物理QPU
                    # 构建“原始逻辑QPU→物理QPU”的映射
                    # 计算该排列对的切换成本（直接基于原始D_move映射）
                    move_cost = self._evaluate_move_cost(
                        network,
                        D_move,
                        prev_perm,
                        curr_perm
                    )
                    row_costs.append(move_cost)
                transition_costs.append(row_costs)
            teledata_costs.append(transition_costs)

        # ---------- 动态规划 ----------
        # dp[t][idx]：时间片t处于状态idx的最小累计成本（fidelity_loss, num_comms）
        dp = [[(inf, inf)] * num_states for _ in range(k)]
        # back[t][idx]：记录最优路径的前驱状态索引
        back = [[-1 for _ in range(num_states)] for _ in range(k)]

        # 初始化第一个时间片
        for idx in range(num_states):
            cost = telegate_costs[0][idx]
            dp[0][idx] = (cost.total_fidelity_loss, cost.num_comms)

        # 递推后续时间片
        for t in range(1, k):
            for curr_idx in range(num_states):
                curr_telegate = telegate_costs[t][curr_idx]
                curr_telegate_tuple = (curr_telegate.total_fidelity_loss, curr_telegate.num_comms)
                
                # 寻找最优前驱状态
                best_cost = (inf, inf)
                best_prev_idx = -1

                for prev_idx in range(num_states):
                    # 前驱累计成本 + 当前telegate成本 + 转移成本
                    prev_total = dp[t-1][prev_idx]
                    transfer_cost = teledata_costs[t-1][prev_idx][curr_idx]
                    transfer_tuple = (transfer_cost.total_fidelity_loss, transfer_cost.num_comms)
                    
                    # 计算总累计成本
                    total_loss = prev_total[0] + curr_telegate_tuple[0] + transfer_tuple[0]
                    total_comms = prev_total[1] + curr_telegate_tuple[1] + transfer_tuple[1]
                    total_cost = (total_loss, total_comms)
                    
                    # 更新最优解（优先按保真度损失排序，其次通信次数）
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev_idx = prev_idx
    
                dp[t][curr_idx] = best_cost
                back[t][curr_idx] = best_prev_idx

        # ---------- 回溯找到最优路径 ----------
        best_last = min(range(num_states), key=lambda i: dp[k-1][i])
        perm_indices = [0] * k
        perm_indices[k-1] = best_last

        # 反向推导最优路径
        for t in range(k-2, -1, -1):
            perm_indices[t] = back[t+1][perm_indices[t+1]]

        # ---------- 更新映射记录 ----------
        for t in range(k):
            record = mapping_record_list.records[t]
            perm = all_perms[perm_indices[t]]
            original_partition = record.partition
            
            best_partition = [original_partition[idx] for idx in perm]
            record.partition = copy.deepcopy(best_partition)

            # 计算该时间片对应的成本对象

            # 获取子线路
            subcircuit = CompilerUtils.get_subcircuit_by_level(
                num_qubits=circuit.num_qubits,
                circuit=circuit,
                circuit_layers=circuit_layers,
                layer_start=record.layer_start,
                layer_end=record.layer_end
            )
            # print(subcircuit)

            # 计算local和telegate损失
            record.costs = CompilerUtils.evaluate_local_and_telegate(record.partition, subcircuit, network)
            # print(f"[DEBUG] record.costs (local and telegate): {record.costs}")

            # 计算teledata损失
            if t != 0:
                prev_record = mapping_record_list.records[t - 1]
                record.costs += CompilerUtils.evaluate_teledata(
                    prev_record.partition,
                    record.partition,
                    network
                )
                # print(f"[DEBUG] record.costs (teledata): {record.costs}")

        # 输出耗时信息
        end_time = time.time()
        print(f"[INFO] [Time] Link-Oriented DP Mapper 完成，耗时 {end_time - start_time:.2f} 秒")

        # print("===================================================")
        # for record in mapping_record_list.records:
        #     print(f"[DEBUG] record.costs: {record.costs}")

        return mapping_record_list

# class HybridMapper(Mapper):
#     """
#     混合方法映射器：根据问题规模动态选择
#     """
    
#     @property
#     def name(self) -> str:
#         """获取映射器名称"""
#         return "HybridMapper"

#     def map(self,
#             mapping_record_list: MappingRecordList,
#             circuit_layers: list[Any],
#             network: Network) -> MappingRecordList:
#         start_time = time.time()
        
#         # 验证网络属性
#         self._validate_network_attributes(network)

#         k = len(partition_plan)  # 时间片数量
#         m = len(partition_plan[0]) # 第一个时间步的逻辑QPU数量

#         # TODO: 如果m较大，num_perms会非常大，需要限制或使用启发式方法

        
#         # 生成所有可能的排列 (m!)
#         all_perms = list(itertools.permutations(range(m)))
#         num_perms = len(all_perms)

#         # dp[t][perm_idx] = 从时间步0到时间步t-1的最大保真度和
#         dp = [[-float('inf')] * num_perms for _ in range(k)]
#         # 路径记录: path[t][perm_idx] = 前一个时间步的排列索引
#         path = [[-1] * num_perms for _ in range(k)]
        
#         # 初始化dp[0]：第一个时间步的fidelity loss
#         # TODO: 假设
#         for perm_idx, perm in enumerate(all_perms):
#             dp[0][perm_idx] = evaluate_mapping(perm, partition_plan[0], network)

#         end_time = time.time()
#         return {
#             "partition_plan": partition_plan,
#             "execution_time (sec)": end_time - start_time
#         }

#     # 这里的evaluate mapping和真实的量子线路有关。
#     # 实际上是要评估一段量子线路，在特定映射下的保真度损失。
#     # 所以map函数的输入不能只是partition_plan，还需要量子线路的描述。


class MapperFactory:
    """映射器工厂类"""
    _registry = {
        "direct": "DirectMapper",
        # "link_oriented": "LinkOrientedMapper",
        # "exact": "ExactOptimizationMapper",
        "greedy": "GreedyMapper",
        "dp": "DPMapper",
        "newdp": "NewDPMapper",
        "linkdp": "LinkOrientedDPMapper"
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

