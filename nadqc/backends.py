from qiskit_ibm_runtime import QiskitRuntimeService
from pprint import pprint
import pickle as pkl
import datetime
import os

import datetime
import pandas as pd

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
            general_df.to_excel(writer, index=False, sheet_name='general_info')

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
    def __init__(self):
        # 初始化QPU上的噪声信息
        return

    def load_properties(self, filepath: str):
        # 从文件加载噪声数据
        # 读取basic_info, gate_info, qubit_info, general_info表
        basic_df = pd.read_excel(filepath, sheet_name='basic_info')
        gate_df = pd.read_excel(filepath, sheet_name='gate_info')
        qubit_df = pd.read_excel(filepath, sheet_name='qubit_info')
        general_df = pd.read_excel(filepath, sheet_name='general_info')

        # 初始化basic_info
        self.basic_info = basic_df.to_dict(orient='records')[0]
        self.name = self.basic_info.get('backend_name', 'unknown')
        # 初始化gate_info
        self.gate_info = gate_df.to_dict(orient='records')
        # 初始化qubit_info
        self.qubit_info = qubit_df.to_dict(orient='records')
        self.num_qubits = len(self.qubit_info)
        # 初始化general_info
        self.general_info = general_df.to_dict(orient='records')
        return
    
    def print(self):
        print(f"Backend Name: {self.name}, #Qubits: {self.num_qubits}")
        # pprint(self.basic_info)
        # pprint(self.gate_info)
        # pprint(self.qubit_info)
        # pprint(self.general_info)
        return

class Network:
    def __init__(self, network_config: dict, backend_config: list):
        self.num_backends = len(backend_config)
        self.connectivity = self._build_network(network_config)
        self.backends = backend_config
        return

    def _build_network(self, network_config: dict):
        net_type = network_config.get('type', 'all_to_all')
        size = network_config.get('size', (self.num_backends, 1))

        if net_type == 'all_to_all':
            self.network_coupling = [
                [i, j]
                for i in range(self.num_backends)
                for j in range(self.num_backends)
                if i != j
            ]
        elif net_type == 'mesh_grid':
            n_rows, n_cols = size
            assert self.num_backends == n_rows * n_cols, "Size does not match number of backends"
            self.network_coupling = []
            for row in range(n_rows):
                for col in range(n_cols - 1):
                    self.network_coupling.append([row * n_cols + col, row * n_cols + col + 1])
            for row in range(n_rows - 1):
                for col in range(n_cols):
                    self.network_coupling.append([row * n_cols + col, (row + 1) * n_cols + col])
        else:
            raise ValueError(f"Unsupported network type: {net_type}")
        return