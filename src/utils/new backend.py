    def _sample_subsystem(self, num_qubits: int) -> dict[str, Any]:
        """内部方法：采样子系统，不重映射，返回原始 qubit 编号的数据。
        优先从 general_qlists 中按长度降序提取连通分量，直到满足 n 个 qubits。
        确保最终采样的 n 个 qubit 是连通的，且优先选择 read_out_error 小的 qubit。
        """
        if num_qubits > self.num_qubits:
            raise ValueError(f"Requested {num_qubits} qubits, but backend only has {self.num_qubits}.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")

        # 1. 提取所有 qubit 的 read_out_error（核心：噪声数据）
        qubit_noise = self._get_qubit_readout_error()
        # 2. 获取所有 qubit 编号和连通分量
        all_qubit_ids = [q['qubit_id'] for q in self.qubit_info]
        connected_components = self._get_connected_components()
        if not connected_components:
            raise RuntimeError("No connected components found in backend coupling map.")

        selected_qubits = set()
        general_qlists = self.basic_info.get('general_qlists', [])

        # Step 1: 优先从 general_qlists 中找符合条件的连通 qubit 列表（选总噪声最小的）
        candidate_lists = []
        for item in general_qlists:
            qlist = item.get('qubits', [])
            # 筛选：长度等于 num_qubits 且连通
            if len(qlist) == num_qubits and self._is_connected(qlist):
                # 计算该列表的总 read_out_error（越小越好）
                total_error = sum(qubit_noise.get(q, float('inf')) for q in qlist)
                candidate_lists.append((total_error, qlist))
        
        # 选总噪声最小的候选列表
        if candidate_lists:
            candidate_lists.sort(key=lambda x: x[0])  # 按总噪声升序
            selected_qubits = set(candidate_lists[0][1])

        # Step 2: 如果 Step1 没找到，从最大连通分量中采样低噪声的连通 qubit
        if len(selected_qubits) < num_qubits:
            # 找第一个能容纳 num_qubits 的连通分量
            target_component = None
            for component in connected_components:
                if len(component) >= num_qubits:
                    target_component = component
                    break
            
            if not target_component:
                max_size = len(connected_components[0]) if connected_components else 0
                raise RuntimeError(
                    f"Cannot sample {num_qubits} connected qubits. "
                    f"Largest connected component has only {max_size} qubits."
                )

            # 从目标连通分量中采样：优先选 read_out_error 小的连通 qubit
            selected_qubits = self._sample_low_noise_connected_subset(
                target_component, num_qubits, qubit_noise
            )

        selected_qubits = sorted(selected_qubits)

        # Step 3: 过滤 qubit_info/gate_info/general_qlists/coupling_map（逻辑不变）
        new_qubit_info = [q.copy() for q in self.qubit_info if q['qubit_id'] in selected_qubits]
        new_gate_info = []
        for gate in self.gate_info:
            gate_qubits = gate['qubits']
            if all(q in selected_qubits for q in gate_qubits):
                new_gate_info.append(gate.copy())
        new_general_qlists = [{'name': f'lf_{len(selected_qubits)}', 'qubits': selected_qubits}]
        coupling_map = self.coupling_map
        new_coupling_map = []
        for u, v in coupling_map:
            if u in selected_qubits and v in selected_qubits:
                new_coupling_map.append([u, v])

        return {
            'selected_qubits': selected_qubits,
            'basic_info': {
                **self.basic_info,
                'general_qlists': new_general_qlists,
                'sampled_qubits_original': selected_qubits,
                'coupling_map': new_coupling_map
            },
            'gate_info': new_gate_info,
            'qubit_info': new_qubit_info,
        }

    # ------------------------------ 新增/修改的辅助函数 ------------------------------
    def _get_qubit_readout_error(self) -> dict[int, float]:
        """提取每个 qubit 的 read_out_error 数值，返回 {qubit_id: read_out_error} 字典。
        如果某个 qubit 没有 read_out_error 数据，默认设为无穷大（最后选）。
        """
        qubit_noise = {}
        for q_info in self.qubit_info:
            qid = q_info['qubit_id']
            # 从 qubit_info 中读取 read_out_error 的值（注意字段名是 read_out_error_value）
            readout_error = q_info.get('read_out_error_value', float('inf'))
            # 处理空值/异常值
            if pd.isna(readout_error) or not isinstance(readout_error, (int, float)):
                readout_error = float('inf')
            qubit_noise[qid] = readout_error
        return qubit_noise

    def _sample_low_noise_connected_subset(
        self, 
        component: list[int], 
        num_qubits: int, 
        qubit_noise: dict[int, float]
    ) -> set[int]:
        """从连通分量中采样 num_qubits 个连通的 qubit，优先选 read_out_error 小的。
        步骤：
        1. 选分量内 read_out_error 最小的 qubit 作为起始点；
        2. BFS 扩展时，优先遍历噪声更小的邻居，保证整体噪声最低。
        """
        if len(component) == num_qubits:
            return set(component)
        
        # 1. 构建分量内的邻接表（只保留分量内的耦合）
        adj = {}
        for q in component:
            adj[q] = set()
        for u, v in self.coupling_map:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)
        
        # 2. 选分量内 read_out_error 最小的 qubit 作为起始点（核心：低噪声优先）
        component_sorted = sorted(component, key=lambda q: qubit_noise.get(q, float('inf')))
        start_qubit = component_sorted[0]  # 噪声最小的 qubit 作为起始点
        
        # 3. BFS 扩展：优先访问噪声更小的邻居（保证选到的 qubit 整体噪声低）
        visited = set()
        # 用优先队列（按噪声升序）替代普通队列，优先处理低噪声邻居
        import heapq
        heap = []
        heapq.heappush(heap, (qubit_noise[start_qubit], start_qubit))
        visited.add(start_qubit)
        
        while heap and len(visited) < num_qubits:
            # 取出当前噪声最小的 qubit
            current_noise, current_q = heapq.heappop(heap)
            # 遍历邻居：按噪声升序排序后加入优先队列
            neighbors = list(adj[current_q])
            # 邻居按 read_out_error 升序排序（低噪声优先）
            neighbors_sorted = sorted(neighbors, key=lambda q: qubit_noise.get(q, float('inf')))
            
            for neighbor in neighbors_sorted:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(heap, (qubit_noise[neighbor], neighbor))
                    # 达到目标数量就停止
                    if len(visited) == num_qubits:
                        break
        
        if len(visited) < num_qubits:
            raise RuntimeError(f"Failed to sample {num_qubits} connected qubits from component of size {len(component)}.")
        
        return visited

    # ------------------------------ 原有辅助函数（不变） ------------------------------
    def _get_connected_components(self) -> list[list[int]]:
        """基于 coupling_map 计算所有连通分量，按大小降序排列。"""
        parent = {qid: qid for qid in [q['qubit_id'] for q in self.qubit_info]}
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        for u, v in self.coupling_map:
            union(u, v)
        components = {}
        for qid in parent:
            root = find(qid)
            if root not in components:
                components[root] = []
            components[root].append(qid)
        sorted_components = sorted(components.values(), key=lambda x: len(x), reverse=True)
        sorted_components = [sorted(comp) for comp in sorted_components]
        return sorted_components

    def _is_connected(self, qubit_list: list[int]) -> bool:
        """验证给定的 qubit 列表是否构成连通子图"""
        if len(qubit_list) <= 1:
            return True
        sub_coupling = {}
        for q in qubit_list:
            sub_coupling[q] = set()
        for u, v in self.coupling_map:
            if u in sub_coupling and v in sub_coupling:
                sub_coupling[u].add(v)
                sub_coupling[v].add(u)
        visited = set()
        start = qubit_list[0]
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop(0)
            for neighbor in sub_coupling[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(qubit_list)