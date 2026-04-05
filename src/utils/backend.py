from qiskit_ibm_runtime import QiskitRuntimeService
from pprint import pprint
import pickle as pkl
import datetime
import os
import pandas as pd
from typing import Any, Optional
import json
import ast
import glob
from collections import defaultdict
import math

class QiskitBackendImporter:
    def __init__(self, token=None, instance=None, proxies=None):
        # 初始化时 token 变为可选，如果只是为了转 pkl，不需要登录
        if token:
            QiskitRuntimeService.save_account(token, instance=instance, overwrite=True, proxies=proxies)
            self.service = QiskitRuntimeService()
        else:
            self.service = None
        return

    def download_backend_info(self, backend_name, start_date, end_date, folder):
        if not self.service:
            raise RuntimeError("Service not initialized. Please provide token when creating the instance.")
        
        backend = self.service.backend(backend_name)
        self._print_backend(backend)
        config = backend.configuration() # type: ignore

        folder = f'{folder}{backend_name}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        date = start_date
        while date <= end_date:
            properties = backend.properties(datetime=date) # type: ignore
            pkl.dump(
                (properties, config),
                open(
                    os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.data'),
                    'wb'
                )
            )
            print('Device properties saved for date:', date.strftime("%Y-%m-%d"))

            # 将properties和coupling_map写入xlsx文件
            data_dict = {
                **properties.to_dict(),
                'coupling_map': config.coupling_map
            }
            self._to_xlsx(data_dict, os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.xlsx'))

            date += datetime.timedelta(days=1)
        return

    def download_all_backend_info(self, start_date, end_date, folder):
        if not self.service:
            raise RuntimeError("Service not initialized. Please provide token when creating the instance.")
        for backend in self.service.backends():
            self.download_backend_info(backend.name, start_date, end_date, folder)
        return

    def convert_pkl_to_xlsx(self, input_path: str, output_folder: Optional[str] = None):
        """
        新增：将本地 pkl(.data) 文件批量转换为 Excel。
        
        参数:
            input_path: 可以是单个 .data 文件路径，也可以是包含 .data 文件的文件夹路径
            output_folder: 输出文件夹，默认为 None (与输入文件同目录)
        """
        if os.path.isfile(input_path):
            # 单个文件模式
            print(f"Processing single file: {input_path}")
            self._convert_single_pkl(input_path, output_folder)
        elif os.path.isdir(input_path):
            # 文件夹模式：递归查找所有 .data 文件
            print(f"Processing folder: {input_path}")
            # 也可以查找子文件夹，用 recursive=True
            pkl_files = glob.glob(os.path.join(input_path, "*.data")) 
            
            if not pkl_files:
                # 尝试在子文件夹中找 (比如 folder/backend_name/file.data)
                pkl_files = glob.glob(os.path.join(input_path, "**", "*.data"), recursive=True)

            if not pkl_files:
                print(f"No .data files found in {input_path}.")
                return

            print(f"Found {len(pkl_files)} files.")
            for f in pkl_files:
                try:
                    # 保持目录结构：如果 input_path 是根目录，output_folder 也需要对应
                    # 这里简单处理：如果指定了 output_folder，都放那里；否则放在原文件旁边
                    specific_out = output_folder
                    if not specific_out:
                        specific_out = os.path.dirname(f)
                    
                    self._convert_single_pkl(f, specific_out)
                except Exception as e:
                    print(f"Failed to process {f}: {e}")
        else:
            print(f"Path not found: {input_path}")
        return

    def _convert_single_pkl(self, pkl_path: str, output_folder: str | None = None):
        """内部方法：转换单个 pkl 文件"""
        print(f"Reading: {pkl_path} ...")
        
        # 1. 读取 Pickle
        with open(pkl_path, 'rb') as f:
            properties, config = pkl.load(f)

        # 2. 构造字典 (复用下载时的逻辑)
        data_dict = {
            **properties.to_dict(),
            'coupling_map': config.coupling_map
        }

        # 3. 确定输出路径
        if not output_folder:
            output_folder = os.path.dirname(pkl_path)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成输出文件名：替换后缀为 .xlsx
        filename = os.path.basename(pkl_path)
        if filename.endswith('.data'):
            filename = filename[:-5] + '.xlsx'
        else:
            filename = filename + '.xlsx'
            
        out_path = os.path.join(output_folder, filename)

        # 4. 写入 Excel (复用现有的 _to_xlsx 方法)
        self._to_xlsx(data_dict, out_path)
        return

    def _print_backend(self, backend):
        print(
            f"Name: {backend.name}\n"
            f"Version: {backend.version}\n"
            f"#Qubits: {backend.num_qubits}\n"
        )
        return

    def _to_iso(self, dt):
        if isinstance(dt, datetime.datetime):
            return dt.isoformat()
        return dt

    def _build_basic_info(self, data: dict):
        row = {
            'backend_name': data.get('backend_name'),
            'backend_version': data.get('backend_version'),
            'general_qlists': json.dumps(data.get('general_qlists')),
            'last_update_date': self._to_iso(data.get('last_update_date')),
            'coupling_map': json.dumps(data.get('coupling_map'))
        }
        return pd.DataFrame([row])

    def _build_gate_info(self, data: dict):
        gates = data.get('gates', [])
        rows = []
        for g in gates:
            row = {
                'name': g.get('name'),
                'gate': g.get('gate'),
            }
            # qubits 列：保存为逗号分隔字符串（也可改为列表）
            qubits = g.get('qubits', [])
            row['qubits'] = json.dumps(qubits)

            # 展开 parameters 列表 -> {param_name}_value/unit/date
            params = g.get('parameters', [])
            for p in params or []:
                p_name = p.get('name')
                if not p_name:
                    continue
                row[f'{p_name}_value'] = p.get('value')
                row[f'{p_name}_unit'] = p.get('unit')
                row[f'{p_name}_date'] = self._to_iso(p.get('date'))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_qubit_info(self, data: dict):
        qubits = data.get('qubits', [])
        rows = []
        for qid, qlist in enumerate(qubits):
            row: dict[str, Any] = {'qubit_id': qid}
            if isinstance(qlist, list):
                for item in qlist:
                    name = item.get('name')
                    if not name:
                        continue
                    row[f'{name}_value'] = item.get('value')
                    row[f'{name}_unit'] = item.get('unit')
                    row[f'{name}_date'] = self._to_iso(item.get('date'))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_general_info(self, data: dict):
        gens = data.get('general', [])
        rows = []
        for item in gens:
            name = item.get('name')
            row = {
                'name': name,
                'value': item.get('value'),
                'unit': item.get('unit'),
                'date': self._to_iso(item.get('date')),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _save_to_excel(self, basic_df, gate_df, qubit_df, general_df, out_path: str):
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            basic_df.to_excel(writer, index=False, sheet_name='basic_info')
            gate_df.to_excel(writer, index=False, sheet_name='gate_info')
            qubit_df.to_excel(writer, index=False, sheet_name='qubit_info')
            # general_df.to_excel(writer, index=False, sheet_name='general_info')

    def _to_xlsx(self, data: dict, output_path: str = 'noise_parsed.xlsx'):
        basic_df = self._build_basic_info(data)
        gate_df = self._build_gate_info(data)
        qubit_df = self._build_qubit_info(data)
        general_df = self._build_general_info(data)

        # 可选：按列名排序，让 *_value, *_unit, *_date 更整齐
        def sort_cols(df):
            cols = list(df.columns)
            # 将核心列放前面
            front = [c for c in ['backend_name','backend_version','general_qlists','last_update_date',
                                'name','gate','qubits','qubit_id','value','unit','date'] if c in cols]
            rest = [c for c in cols if c not in front]
            return df[front + sorted(rest)]
        try:
            basic_df = sort_cols(basic_df)
            gate_df = sort_cols(gate_df)
            qubit_df = sort_cols(qubit_df)
            general_df = sort_cols(general_df)
        except Exception:
            pass

        self._save_to_excel(basic_df, gate_df, qubit_df, general_df, output_path)
        print(f'已生成 Excel：{output_path}')


class Backend:
    def __init__(self, global_config: dict = {}, backend_config: dict = {}):
        # 初始化QPU上的噪声信息
        if backend_config is not None:
            self._init_backend(global_config, backend_config)
        return

    def _init_backend(self, global_config: dict = {}, backend_config: dict = {}):
        # pprint(backend_config)
        if 'backend_name' in backend_config:
            self.name = backend_config['backend_name']
            self.load_properties(global_config, self.name, backend_config.get('date', datetime.datetime.today()))
        elif 'num_qubits' in backend_config:
            self.num_qubits = backend_config.get('num_qubits', 0)
        return

    def load_properties(self, global_config: dict, backend_name: str, date: datetime.datetime):
        filepath = f"{global_config['device_properties_folder']}{backend_name}/{backend_name}_{date.strftime('%Y-%m-%d')}.xlsx"
        
        # 下载数据（如果文件不存在）
        if 'sampled' in backend_name and not os.path.exists(filepath):
            original_backend_name = backend_name.split('_sampled_')[0]
            original_filepath = f"{global_config['device_properties_folder']}{original_backend_name}/{original_backend_name}_{date.strftime('%Y-%m-%d')}.xlsx"
            if not os.path.exists(original_filepath):
                # 调用QiskitBackendImporter下载数据
                self._download_backend_data(global_config, original_backend_name, date)
            print(f"Sampling backend from {original_backend_name} for {backend_name} ...")
            # 读取原始backend
            original_backend = Backend()
            original_backend.load_properties(global_config, original_backend_name, date)
            # 解析采样数量
            try:
                sampled_part = backend_name.split('_sampled_')[1]
                num_qubits = int(sampled_part[:-1])  # 去掉最后的 'q'
            except Exception as e:
                raise ValueError(f"Invalid sampled backend name: {backend_name}") from e
            # 采样并保存
            original_backend.sample_and_export(num_qubits, global_config['device_properties_folder'])
        elif not os.path.exists(filepath):
            print(f"Downloading the device properties for {backend_name} on {date.strftime('%Y-%m-%d')} ...")
            self._download_backend_data(global_config, backend_name, date)

        # 从文件加载噪声数据
        # 读取basic_info, gate_info, qubit_info, general_info表
        basic_df = pd.read_excel(filepath, sheet_name='basic_info')
        gate_df = pd.read_excel(filepath, sheet_name='gate_info')
        qubit_df = pd.read_excel(filepath, sheet_name='qubit_info')

        # --- 核心修复：类型强制清洗与转换 ---
        
        # 1. 初始化 basic_info
        self.basic_info = basic_df.to_dict(orient='records')[0]
        self.name = self.basic_info.get('backend_name', 'unknown')
        self.date = date
        
        # 清洗 coupling_map: JSON str -> list[list[int]]
        self.coupling_map = self._safe_parse_json(self.basic_info.get('coupling_map', '[]'))
        
        # 清洗 general_qlists: JSON str -> list
        self.basic_info['general_qlists'] = self._safe_parse_json(self.basic_info.get('general_qlists', '[]'))

        # 2. 初始化 gate_info
        self.gate_info = gate_df.to_dict(orient='records')
        for gate in self.gate_info:
            # 清洗 qubits: JSON str -> list[int]
            # 无论 Excel 里存的是 "[0,1]" 还是 "0" 还是 0，这里统一变成 list[int]
            gate['qubits'] = self._parse_qubits_list(gate.get('qubits', '[]'))
        
        # 2.1. 先初始化原始的单门数据 (id0, id1, rx0...)
        self.gate_dict = {gate['name']: gate for gate in self.gate_info}

        # 2.2. 计算平均值，并将其合并到上面的字典中 (id, rx, cz...)
        self.gate_dict.update(self._calculate_average_gate_errors(self.gate_info))

        # 3. 初始化 qubit_info
        self.qubit_info = qubit_df.to_dict(orient='records')
        # 确保 qubit_id 是 int
        for q in self.qubit_info:
            if 'qubit_id' in q and not pd.isna(q['qubit_id']):
                q['qubit_id'] = int(q['qubit_id'])
        
        self.num_qubits = len(self.qubit_info)

        # 4. 初始化 basis gates
        self.basis_gates, self.two_qubit_gates = self._get_basis_gates()
        return

    def _safe_parse_json(self, data_str):
        """安全解析 JSON，处理 NaN 和 Excel 读取的奇怪类型"""
        if pd.isna(data_str):
            return []
        if isinstance(data_str, list):
            return data_str
        if isinstance(data_str, str):
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                # 备用：如果是 "[1, 2]" 格式 json 失败，试试 ast.literal_eval
                try:
                    return ast.literal_eval(data_str)
                except:
                    return []
        return []

    def _parse_qubits_list(self, val) -> list[int]:
        """专门用于解析 gate['qubits']，确保返回 list[int]"""
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, int):
            return [val]
        if pd.isna(val):
            return []
        
        val_str = str(val).strip()
        # 情况 A: JSON 格式 "[0, 1]"
        if val_str.startswith('['):
            parsed = self._safe_parse_json(val_str)
            return [int(x) for x in parsed]
        # 情况 B: 逗号分隔 "0,1" (兼容旧数据)
        elif ',' in val_str:
            parts = val_str.split(',')
            return [int(x.strip()) for x in parts if x.strip().isdigit()]
        # 情况 C: 单个数字字符串 "0"
        elif val_str.isdigit():
            return [int(val_str)]
        return []

    def _download_backend_data(self, config: dict, backend_name: str, date: datetime.datetime):
        backend_importer = QiskitBackendImporter(token=config["ibm_quantum_token"],
                                                instance=config["ibm_quantum_instance"],
                                                proxies=config.get("proxies", None)
                                                )
        backend_importer.download_backend_info(backend_name=backend_name,
                                              start_date=date,
                                              end_date=date,
                                              folder=config["device_properties_folder"])
        return

    def print(self):
        print(f"Backend Name: {self.name}, #Qubits: {self.num_qubits}, Date: {self.date.strftime('%Y-%m-%d')}")
        # pprint(self.basic_info)
        # pprint(self.gate_info)
        # pprint(self.qubit_info)
        # pprint(self.general_info)
        return

    def sample_and_export(
        self,
        num_qubits: int,
        output_folder: str,
        remap_qubits: bool = True
    ):
        """
        从当前 backend 中采样 n 个量子比特的子系统，并导出为新的 Excel 文件。

        参数:
            n (int): 要采样的量子比特数量。
            output_folder (str): 输出文件夹路径。
            date (datetime): 对应的日期（用于文件名）。
            remap_qubits (bool): 是否将物理 qubit 编号重映射为 0~n-1。
        """

        # Step 1: 采样子系统（保留原始 qubit 编号）
        sampled = self._sample_subsystem(num_qubits)

        selected_qubits = sampled['selected_qubits']  # e.g., [32, 33, 37, ...]
        new_qubit_info = sampled['qubit_info']
        new_gate_info = sampled['gate_info']
        new_general_qlists = sampled['basic_info']['general_qlists']
        new_coupling_map = sampled['basic_info']['coupling_map']

        # Step 2: 重映射 qubit 编号（如果需要）
        if remap_qubits:
            qubit_map = {old: new for new, old in enumerate(selected_qubits)}  # {32:0, 33:1, ...}
            # print(f"Qubit mapping: {qubit_map}")

            # 重映射 qubit_info 的索引（可选：添加 original_qubit 字段）
            for i, qinfo in enumerate(new_qubit_info):
                qinfo['original_qubit'] = selected_qubits[i]
                qinfo['qubit_id'] = i  # 新编号

            # print("Remapped qubit_info:")
            # pprint(new_qubit_info)

            # 重映射 gate_info 中的 qubits
            for gate in new_gate_info:
                # 这里现在直接是 list[int] 了，不需要 split
                orig_qubits = gate['qubits'] 
                new_qubits = [qubit_map[q] for q in orig_qubits]
                # 更新gate_name
                gate['name'] = f"{gate['gate']}{'_'.join(map(str, new_qubits))}"
                gate['qubits'] = new_qubits # 保持 list[int]

            # 重映射 general_qlists
            remapped_general_qlists = []
            for item in new_general_qlists:
                remapped_q = [qubit_map[q] for q in item['qubits'] if q in qubit_map]
                if remapped_q:
                    remapped_general_qlists.append({'name': item['name'], 'qubits': remapped_q})
            new_general_qlists = remapped_general_qlists

            # 重映射 coupling_map
            remapped_coupling_map = [[qubit_map[u], qubit_map[v]] for u, v in new_coupling_map]

            # 更新 basic_info 中的 num_qubits
            sampled['basic_info']['general_qlists'] = new_general_qlists
            sampled['basic_info']['coupling_map'] = remapped_coupling_map

        # Step 3: 更新数据并构造 DataFrames
        self.basic_info = sampled['basic_info']
        self.name = f"{self.name}_sampled_{num_qubits}q"
        self.gate_info = new_gate_info
        # self.gate_dict = {gate['name']: gate for gate in new_gate_info}
        self.qubit_info = new_qubit_info
        self.num_qubits = num_qubits

        # --- 保存前准备：把 list 转回 JSON 字符串以便存入 Excel ---
        basic_df = pd.DataFrame([sampled['basic_info']])
        # 转回 JSON 字符串
        basic_df['coupling_map'] = basic_df['coupling_map'].apply(json.dumps)
        basic_df['general_qlists'] = basic_df['general_qlists'].apply(json.dumps)

        gate_df = pd.DataFrame(new_gate_info)
        # 把 qubits list 转回 JSON 字符串
        gate_df['qubits'] = gate_df['qubits'].apply(json.dumps)

        qubit_df = pd.DataFrame(new_qubit_info)
        # general_df = pd.DataFrame(sampled['general_info'])  # 可选：也可过滤或留空

        # Step 4: 保存到 Excel
        os.makedirs(f"{output_folder}{self.name}/", exist_ok=True)
        filename = f"{self.name}_{self.date.strftime('%Y-%m-%d')}.xlsx"
        filepath = os.path.join(f"{output_folder}{self.name}/", filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            basic_df.to_excel(writer, sheet_name='basic_info', index=False)
            gate_df.to_excel(writer, sheet_name='gate_info', index=False)
            qubit_df.to_excel(writer, sheet_name='qubit_info', index=False)
            # general_df.to_excel(writer, sheet_name='general_info', index=False)

        print(f"Sampled subsystem saved to: {filepath}")
        return filepath

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
        # 从耦合图获取所有连通分量（按大小降序排列）
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
            # 'general_info': self.general_info
        }

    def _get_basis_gates(self) -> tuple[list[str], set[str]]:
        basis_gates = set()
        two_qubit_gates = set()
        for gate in self.gate_info:
            # 修复：直接访问 list，不需要 split
            qubits_list = gate['qubits'] 
            n_qubits = len(qubits_list)
            
            if n_qubits > 2:
                print(f"[WARNING] Found gate with more than 2 qubits: {gate['name']} with qubits {qubits_list}, skipping it for basis gate set.")
                continue
            # 跳过measure
            if gate['gate'] == 'measure':
                continue
            basis_gates.add(gate['gate'])
            if n_qubits == 2:
                two_qubit_gates.add(gate['gate'])
        return list(basis_gates), two_qubit_gates

    def _calculate_average_gate_errors(self, gate_info) -> dict:
        """
        统计每种量子门的平均错误率。

        Args:
            gate_info (list): 包含量子门信息的字典列表。

        Returns:
            dict: 键为门名称，值为包含平均错误率的字典。
        """
        # 1. 用于临时存储每种门的所有错误率
        error_collector = defaultdict(list)

        # 2. 遍历数据，收集错误率
        for gate in gate_info:
            gate_type = gate['gate']
            error_val = gate.get('gate_error_value')
            
            # 修改逻辑：如果是空值(None)或不是数字，视为0.0
            if not isinstance(error_val, (int, float)) or math.isnan(error_val):
                error_val = 0.0
                
            error_collector[gate_type].append(error_val)

        # 3. 构建最终的结果字典
        avg_gate_dict = {}
        for gate_type, errors in error_collector.items():
            # 计算平均值，如果该组没有数据则为 None
            avg_error = sum(errors) / len(errors) if errors else None
            
            # 构建新的条目，其他字段留空
            avg_gate_dict[gate_type] = {
                'name': gate_type,
                'gate': gate_type,
                'qubits': None,
                'gate_error_date': None,
                'gate_error_unit': None,
                'gate_error_value': avg_error,
                'gate_length_date': None,
                'gate_length_unit': None,
                'gate_length_value': None
            }

        return avg_gate_dict

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
        """基于 coupling_map 计算所有连通分量，按分量大小降序排列。
        使用并查集（Union-Find）算法，高效求解图的连通分量。
        """
        # 初始化并查集
        parent = {qid: qid for qid in [q['qubit_id'] for q in self.qubit_info]}
        
        def find(x):
            """查找根节点（带路径压缩）"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """合并两个节点"""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # 遍历耦合图，合并相连的 qubit
        for u, v in self.coupling_map:
            union(u, v)

        # 按根节点分组，得到连通分量
        components = {}
        for qid in parent:
            root = find(qid)
            if root not in components:
                components[root] = []
            components[root].append(qid)

        # 按分量大小降序排列，且每个分量内的 qubit 排序
        sorted_components = sorted(components.values(), key=lambda x: len(x), reverse=True)
        # 对每个分量内的 qubit 编号排序（可选，提升可读性）
        sorted_components = [sorted(comp) for comp in sorted_components]
        return sorted_components

    def _is_connected(self, qubit_list: list[int]) -> bool:
        """验证给定的 qubit 列表是否构成连通子图"""
        if len(qubit_list) <= 1:
            return True  # 单个/空 qubit 天然连通
        
        # 构建子图的耦合关系
        sub_coupling = {}
        for q in qubit_list:
            sub_coupling[q] = set()
        # 从全局耦合图中提取子图的边
        for u, v in self.coupling_map:
            if u in sub_coupling and v in sub_coupling:
                sub_coupling[u].add(v)
                sub_coupling[v].add(u)
        
        # 广度优先搜索（BFS）验证连通性
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
        
        # 所有 qubit 都被访问到 = 连通
        return len(visited) == len(qubit_list)