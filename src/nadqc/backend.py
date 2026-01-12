from qiskit_ibm_runtime import QiskitRuntimeService
from pprint import pprint
import pickle as pkl
import datetime
import os

import pandas as pd
import random
from typing import Any
import json

class QiskitBackendImporter:
    def __init__(self, token, instance, proxies):
        QiskitRuntimeService.save_account(token, instance=instance, overwrite=True, proxies=proxies)
        self.service = QiskitRuntimeService()
        return

    def download_backend_info(self, backend_name, start_date, end_date, folder):
        backend = self.service.backend(backend_name)
        self._print_backend(backend)
        config = backend.configuration()

        folder = f'{folder}{backend_name}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        date = start_date
        while date <= end_date:
            properties = backend.properties(datetime=date)
            pkl.dump(
                (properties, config),
                open(
                    os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.data'),
                    'wb'
                )
            )
            print('Device properties saved for date:', date.strftime("%Y-%m-%d"))

            # 将properties写入xlsx文件
            self._to_xlsx(properties.to_dict(), os.path.join(folder, f'{backend.name}_{date.strftime("%Y-%m-%d")}.xlsx'))

            date += datetime.timedelta(days=1)
        return

    def download_all_backend_info(self, start_date, end_date, folder):
        for backend in self.service.backends():
            self.download_backend_info(backend.name, start_date, end_date, folder)
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
            'general_qlists': data.get('general_qlists'),
            'last_update_date': self._to_iso(data.get('last_update_date')),
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
            row['qubits'] = ','.join(map(str, qubits)) if isinstance(qubits, list) else qubits

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
            row = {'qubit_id': qid}
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
    def __init__(self, global_config: dict = None, backend_config: dict = None):
        # 初始化QPU上的噪声信息
        if backend_config is not None:
            self._init_backend(global_config, backend_config)
        return
    
    def _init_backend(self, global_config: dict = None, backend_config: dict = None):
        pprint(backend_config)
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
        # general_df = pd.read_excel(filepath, sheet_name='general_info')

        # 初始化basic_info
        self.basic_info = basic_df.to_dict(orient='records')[0]
        self.name = self.basic_info.get('backend_name', 'unknown')
        self.date = date
        # 初始化gate_info
        self.gate_info = gate_df.to_dict(orient='records')
        # 初始化qubit_info
        self.qubit_info = qubit_df.to_dict(orient='records')
        self.num_qubits = len(self.qubit_info)
        # 初始化general_info
        # self.general_info = general_df.to_dict(orient='records')
        return

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
                orig_qubits = gate['qubits']
                new_qubits = sorted([qubit_map[q] for q in orig_qubits])
                # 更新gate_name
                gate['name'] = f"{gate['gate']}{'_'.join(map(str, new_qubits))}"
                gate['qubits'] = ','.join(map(str, new_qubits))

            # 重映射 general_qlists
            remapped_general_qlists = []
            for item in new_general_qlists:
                remapped_q = [qubit_map[q] for q in item['qubits'] if q in qubit_map]
                if remapped_q:
                    remapped_general_qlists.append({'name': item['name'], 'qubits': remapped_q})
            new_general_qlists = remapped_general_qlists

            # 更新 basic_info 中的 num_qubits
            sampled['basic_info']['general_qlists'] = new_general_qlists

        # Step 3: 更新数据并构造 DataFrames
        self.basic_info = sampled['basic_info']
        self.name = f"{self.name}_sampled_{num_qubits}q"
        self.gate_info = new_gate_info
        self.qubit_info = new_qubit_info
        self.num_qubits = num_qubits

        basic_df = pd.DataFrame([sampled['basic_info']])
        gate_df = pd.DataFrame(new_gate_info)
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
        """
        if num_qubits > self.num_qubits:
            raise ValueError(f"Requested {num_qubits} qubits, but backend only has {self.num_qubits}.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")

        # 获取真实的量子比特编号（避免假设 0~num_qubits-1 连续）
        # 假设 self.qubit_info 的索引即为物理 qubit 编号（常见于 Qiskit 导出格式）
        all_qubit_ids = list(range(self.num_qubits))  # ⚠️ 若编号非连续，请替换为此：
        # all_qubit_ids = [q['qubit_id'] for q in self.qubit_info]  # 如果有 'qubit' 字段

        selected_qubits: set[int] = set()
        general_qlists = self.basic_info.get('general_qlists', [])
        # 把general_qlists从json字符串转成list
        if isinstance(general_qlists, str):
            general_qlists = json.loads(general_qlists.replace("'", '"'))

        # Step 1: 尝试直接从general_qlists里找到符合要求的一组量子比特
        for item in general_qlists:
            qlist = item.get('qubits', [])
            if len(qlist) == num_qubits:
                selected_qubits = set(qlist)
                break

        # Step 2: 如果还不够，从剩余 qubit 中随机补充
        if len(selected_qubits) < num_qubits:
            remaining = [q for q in all_qubit_ids if q not in selected_qubits]
            need = num_qubits - len(selected_qubits)
            if len(remaining) < need:
                raise RuntimeError(f"Not enough qubits available to sample {num_qubits}. "
                                f"Available: {len(remaining) + len(selected_qubits)}")
            selected_qubits.update(random.sample(remaining, need))

        selected_qubits = sorted(selected_qubits)

        # Step 3: Filter qubit_info
        new_qubit_info = [self.qubit_info[q] for q in selected_qubits]

        # Step 4: Filter gate_info
        new_gate_info = []
        for gate in self.gate_info:
            gate_qubits = [int(q) for q in gate['qubits'].split(',')]
            gate['qubits'] = gate_qubits
            if all(q in selected_qubits for q in gate_qubits):
                new_gate_info.append(gate)
        # print("Filtered gate_info:")
        # pprint(new_gate_info)

        # Step 5: Update general_qlists (保留与 selected_qubits 有交集的路径)
        # print(f"selected_qubits: {selected_qubits}")
        new_general_qlists = [{'name': f'lf_{len(selected_qubits)}', 'qubits': selected_qubits}]
        # for item in general_qlists:
        #     filtered_q = [q for q in item.get('qubits', []) if q in selected_qubits]
        #     print(f"Original qlist: {item.get('qubits', [])}, Filtered: {filtered_q}")
        #     if filtered_q:
        #         new_general_qlists.append({'name': item.get('name', ''), 'qubits': filtered_q})
        # print("Filtered general_qlists:")
        # pprint(new_general_qlists)

        return {
            'selected_qubits': selected_qubits,
            'basic_info': {
                **self.basic_info,
                'general_qlists': new_general_qlists,
                'sampled_qubits_original': selected_qubits
            },
            'gate_info': new_gate_info,
            'qubit_info': new_qubit_info,
            # 'general_info': self.general_info
        }
