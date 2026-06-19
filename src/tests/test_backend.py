import datetime

import pytest

from src.utils import Backend, QiskitBackendImporter, get_config

def test_backend_sample():
    """测试 Backend 类初始化"""
    global_config = get_config()
    backend_config = {
        'backend_name': 'ibm_torino_sampled_10q',
        'date': datetime.datetime(2025, 11, 9)
    }

    backend = Backend(global_config, backend_config)
    backend.print()


def test_backend_download_requires_environment_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    backend = Backend(backend_config=None)

    with pytest.raises(RuntimeError, match="IBM_QUANTUM_TOKEN"):
        backend._download_backend_data(
            {"device_properties_folder": "unused"},
            "ibm_test",
            datetime.datetime(2026, 1, 1),
        )


def test_backend_download_reads_credentials_from_environment(monkeypatch):
    captured = {}

    def fake_init(self, token=None, instance=None, proxies=None):
        captured.update(token=token, instance=instance, proxies=proxies)

    def fake_download(self, backend_name, start_date, end_date, folder):
        captured.update(
            backend_name=backend_name,
            start_date=start_date,
            end_date=end_date,
            folder=folder,
        )

    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")
    monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "test-instance")
    monkeypatch.setattr(QiskitBackendImporter, "__init__", fake_init)
    monkeypatch.setattr(QiskitBackendImporter, "download_backend_info", fake_download)

    date = datetime.datetime(2026, 1, 1)
    backend = Backend(backend_config=None)
    backend._download_backend_data(
        {
            "device_properties_folder": "/tmp/device-properties/",
            "proxies": {"urls": {"https": "http://localhost:1234"}},
        },
        "ibm_test",
        date,
    )

    assert captured == {
        "token": "test-token",
        "instance": "test-instance",
        "proxies": {"urls": {"https": "http://localhost:1234"}},
        "backend_name": "ibm_test",
        "start_date": date,
        "end_date": date,
        "folder": "/tmp/device-properties/",
    }
