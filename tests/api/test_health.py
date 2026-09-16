from fastapi.testclient import TestClient

from aquaops.api.app import create_app
from aquaops.config import Settings


def test_health_returns_service_name_and_environment() -> None:
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "service": "aquaops",
        "status": "ok",
        "environment": "test",
    }


def test_default_qdrant_url_targets_the_compose_service() -> None:
    assert Settings().qdrant_url == "http://qdrant:6333"
