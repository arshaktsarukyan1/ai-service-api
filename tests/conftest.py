import os

import pytest
from fastapi.testclient import TestClient

# env vars must be set before any app module is imported
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")
os.environ.setdefault("ARANGO_PASSWORD", "test-password")

from app.domain.models import AIRequest, AIResponse, AIUsage  # noqa: E402
from app.domain.tasks import AITask  # noqa: E402

_FAQ_JSON = (
    '{"faqs": [{"question": "What is the project about?", '
    '"answer": "See the description field in the data."}]}'
)


class MockProvider:
    name = "mock"

    async def execute(self, request: AIRequest) -> AIResponse:
        if request.task is AITask.faq_generation:
            content = _FAQ_JSON
        else:
            content = f"mocked:{request.input_text}"
        return AIResponse(
            task=request.task,
            content=content,
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

    import app.api.routes as routes_mod
    import app.interfaces.faq_routes as faq_routes

    monkeypatch.setattr(routes_mod, "get_active_provider", lambda cfg: mock_provider)
    monkeypatch.setattr(faq_routes, "get_active_provider", lambda cfg: mock_provider)

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)
