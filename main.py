import os
import json
import datetime

from nadqc import QiskitBackendImporter, Backend, Network

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    backend = Backend()
    backend_name = "ibm_torino"
    date = datetime.datetime(2025, 12, 9)

    # 检查文件是否存在并加载噪声数据
    filepath = f"{config['device_properties_folder']}{backend_name}/{backend_name}_{date.strftime('%Y-%m-%d')}.xlsx"
    if os.path.exists(filepath):
        backend.load_properties(filepath)
        backend.print()
    else:
        print(f"Downloading the device properties for {backend_name} on {date.strftime('%Y-%m-%d')} ...")
        backend_importer = QiskitBackendImporter(token=config["ibm_quantum_token"],
                                                 instance=config["ibm_quantum_instance"],
                                                 proxies=config.get("proxies", None)
                                                )
        
        backend_importer.download_backend_info(backend_name=backend_name,
                                               start_date=date,
                                               end_date=date,
                                               folder=config["device_properties_folder"])
        backend.load_properties(filepath)
        backend.print()

    # 构建网络
    network_config = {
        "type": "all_to_all",
        "size": (2, 1),
    }
    networks = Network(network_config=network_config,
                       backend_config=[backend, backend])
