import json
import datetime

from nadqc import QiskitBackendImporter

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    qbi = QiskitBackendImporter(token=config["ibm_quantum_token"],
                                instance=config["ibm_quantum_instance"],
                                proxies=config.get("proxies", None)
                                )

    # qbi.download_all_backend_info(start_date=datetime.datetime(2025, 11, 9),
    #                               end_date=datetime.datetime(2025, 12, 9),
    #                               folder=config["device_properties_folder"])

    qbi.download_backend_info(backend_name="ibm_torino",
                              start_date=datetime.datetime(2025, 10, 20),
                              end_date=datetime.datetime(2025, 10, 20),
                              folder=config["device_properties_folder"])
