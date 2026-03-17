import datetime

from src.utils import get_config, QiskitBackendImporter

def main():
    global_config = get_config()
    backend_importer = QiskitBackendImporter(token=global_config["ibm_quantum_token"],
                                             instance=global_config["ibm_quantum_instance"],
                                             proxies=global_config.get("proxies", None)
                                            )
    
    # 每间隔10天拉一次
    for i in range(0, 70, 10):
        date = datetime.datetime(2025, 11, 10) + datetime.timedelta(days=i)
        backend_importer.download_all_backend_info(date, date, global_config["device_properties_folder"])

if __name__ == "__main__":
    main()