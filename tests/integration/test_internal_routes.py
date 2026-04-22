"""Integration tests for POST /internal/ai/execute.

Uses a mocked provider (injected via conftest) so no real OpenAI calls are made.
"""

import pytest
from fastapi.testclient import TestClient

from app.domain.exceptions import (
    AIAuthError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)
from app.domain.models import AIResponse
from app.domain.tasks import AITask

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_returns_200(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": "Tell me about Vienna."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "information_preparation"
    assert "mocked:" in body["content"]
    assert body["provider"] == "mock"
    assert body["model"] == "mock-model"
    assert body["latency_ms"] >= 0
    assert body["usage"]["total_tokens"] == 15


def test_execute_accepts_metadata(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={
            "task": "information_structuring",
            "input_text": "structure this",
            "metadata": {"language": "de"},
        },
    )
    assert resp.status_code == 200


def test_execute_accepts_all_task_types(client: TestClient) -> None:
    for task in AITask:
        resp = client.post(
            "/internal/ai/execute",
            json={"task": task.value, "input_text": "test"},
        )
        assert resp.status_code == 200, f"Failed for task={task}"


# ---------------------------------------------------------------------------
# Request ID header
# ---------------------------------------------------------------------------


def test_response_includes_x_request_id(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": "hello"},
    )
    assert "X-Request-ID" in resp.headers


def test_caller_request_id_is_echoed(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": "hello"},
        headers={"X-Request-ID": "trace-abc-123"},
    )
    assert resp.headers["X-Request-ID"] == "trace-abc-123"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_input_text_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert body["request_id"] is not None


def test_empty_input_text_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": ""},
    )
    assert resp.status_code == 422


def test_invalid_task_value_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "nonexistent_task", "input_text": "hello"},
    )
    assert resp.status_code == 422


def test_missing_task_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/internal/ai/execute",
        json={"input_text": "hello"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error handler mapping (monkeypatched execute_task)
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_client(monkeypatch, mock_provider):
    """Client where execute_task can be replaced per test."""
    import app.api.routes as routes_mod
    import app.interfaces.internal_routes as route_mod

    monkeypatch.setattr(route_mod, "get_active_provider", lambda cfg: mock_provider)
    monkeypatch.setattr(routes_mod, "get_active_provider", lambda cfg: mock_provider)

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    return app, monkeypatch


def test_timeout_returns_504(patched_client, mock_provider) -> None:
    app, mp = patched_client
    import app.interfaces.internal_routes as route_mod

    async def _raise(*a, **kw):
        raise AITimeoutError("timed out")

    mp.setattr(route_mod, "execute_task", _raise)
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": "hi"},
    )
    assert resp.status_code == 504
    assert resp.json()["error"] == "ai_timeout"


def test_rate_limit_returns_429(patched_client, mock_provider) -> None:
    app, mp = patched_client
    import app.interfaces.internal_routes as route_mod

    async def _raise(*a, **kw):
        raise AIRateLimitError("quota")

    mp.setattr(route_mod, "execute_task", _raise)
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "voice_assistant", "input_text": "hi"},
    )
    assert resp.status_code == 429
    assert resp.json()["error"] == "ai_rate_limit"


def test_auth_error_returns_503(patched_client, mock_provider) -> None:
    app, mp = patched_client
    import app.interfaces.internal_routes as route_mod

    async def _raise(*a, **kw):
        raise AIAuthError("bad key")

    mp.setattr(route_mod, "execute_task", _raise)
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_structuring", "input_text": "hi"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"] == "ai_auth_error"


def test_provider_error_returns_502(patched_client, mock_provider) -> None:
    app, mp = patched_client
    import app.interfaces.internal_routes as route_mod

    async def _raise(*a, **kw):
        raise AIProviderError("upstream")

    mp.setattr(route_mod, "execute_task", _raise)
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/internal/ai/execute",
        json={"task": "information_preparation", "input_text": "hi"},
    )
    assert resp.status_code == 502
    assert resp.json()["error"] == "ai_provider_error"


# ---------------------------------------------------------------------------
# Health + readiness endpoints (sanity check in integration context)
# ---------------------------------------------------------------------------


def test_health_still_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_returns_200(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["provider"] == "openai"
