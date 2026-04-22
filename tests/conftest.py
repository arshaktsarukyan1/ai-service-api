"""Shared fixtures for the entire test suite."""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure required env vars are present before any app module is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")
os.environ.setdefault("ARANGO_PASSWORD", "test-password")

from app.domain.models import AIRequest, AIResponse, AIUsage  # noqa: E402


class MockProvider:
    """A no-op provider that satisfies the AIProvider protocol."""

    name = "mock"

    async def execute(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            task=request.task,
            content=f"mocked:{request.input_text}",
            provider="mock",
            model="mock-model",
            usage=AIUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            latency_ms=10,
        )

    async def ping(self) -> None:
        return


@pytest.fixture(scope="session")
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def client(monkeypatch, mock_provider: MockProvider) -> TestClient:
    """TestClient with the active provider replaced by MockProvider."""
    import app.interfaces.internal_routes as route_mod

    monkeypatch.setattr(route_mod, "get_active_provider", lambda cfg: mock_provider)

    # Also patch the readiness endpoint's provider lookup.
    import app.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "get_active_provider", lambda cfg: mock_provider)

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)
