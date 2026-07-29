import os
import tempfile
import pytest

# Ensure a test secret key is available before app.py reads .env
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-not-for-production")

import storage


@pytest.fixture(autouse=True)
def temp_data_dir():
    """Use an isolated temporary directory for moofile data in every test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage.DATA_DIR = tmpdir
        yield


@pytest.fixture(autouse=True)
def fresh_http_client():
    """Drop the proxy's pooled httpx client around every test.

    The client is a process-wide singleton built lazily from httpx.Client, so
    without this a test that monkeypatches httpx.Client would either miss the
    already-built client or leak its fake into later tests.
    """
    import proxy

    proxy.reset_http_client()
    yield
    proxy.reset_http_client()


@pytest.fixture
def client():
    """Flask test client."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
