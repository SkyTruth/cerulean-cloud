import contextlib
import os
import sys
import threading
import time
from unittest.mock import patch

import pytest
import rasterio
import uvicorn

# add stack path to enable relative imports from stack
sys.path.append(os.path.join(os.path.abspath("."), "cerulean_cloud"))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

with open(os.path.join(FIXTURES_DIR, "productInfo.json")) as f:
    S1_METADATA = f.read()


class Server(uvicorn.Server):
    """Uvicorn Server."""

    def install_signal_handlers(self):
        """install handlers."""
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        """run in thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def mock_rasterio_open(asset):
    """Mock rasterio Open."""
    assert asset.startswith("s3://sentinel-s1-l1c")
    asset = os.path.join(FIXTURES_DIR, os.path.basename(asset))
    return rasterio.open(asset)


@pytest.fixture(autouse=True)
def testing_env_var(monkeypatch):
    """Set fake env to make sure we don't hit AWS services."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ryan")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "rde")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/noconfigheere")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/tmp/noconfighereeither")
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    monkeypatch.setenv("API_KEY", "mykey")


@pytest.fixture(autouse=True)
async def titiler_application():
    """Run app in Thread."""
    with patch("rio_tiler.io.cogeo.rasterio") as rio:
        with patch("rio_tiler_pds.sentinel.aws.sentinel1.get_object") as s1meta:
            s1meta.return_value = S1_METADATA
            rio.open = mock_rasterio_open

            from cerulean_cloud.titiler_sentinel.handler import app

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=5000,
                log_level="info",
                loop="asyncio",
            )
            server = Server(config=config)
            with server.run_in_thread():
                yield "http://127.0.0.1:5000"
