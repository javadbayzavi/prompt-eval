import pytest
from fastapi.testclient import TestClient

from prompt_eval.config import Settings
from prompt_eval.main import create_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'test.db'}")
    with TestClient(create_app(settings)) as client:
        yield client
