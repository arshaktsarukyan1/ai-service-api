import pytest
from fastapi.testclient import TestClient


def test_faq_list_locations(client: TestClient) -> None:
    resp = client.get("/faq/locations")
    assert resp.status_code == 200
    body = resp.json()
    assert "locations" in body
    ids = {x["id"] for x in body["locations"]}
    assert "de-berlin-hbf-upgrade" in ids


def test_faq_get_known_location(client: TestClient) -> None:
    resp = client.get("/faq/de-berlin-hbf-upgrade")
    assert resp.status_code == 200
    out = resp.json()
    assert out["location"]["id"] == "de-berlin-hbf-upgrade"
    assert out["location"]["name"]
    assert len(out["faqs"]) >= 1
    assert out["faqs"][0]["question"]
    assert out["faqs"][0]["answer"]
    assert out["generation"]["model"] == "mock-model"
    assert out["generation"]["provider"] == "mock"
    assert out["generation"]["source"] == "ai_generated"
    assert "X-Request-ID" in resp.headers


def test_faq_unknown_location_returns_404(client: TestClient) -> None:
    resp = client.get("/faq/does-not-exist")
    assert resp.status_code == 404
    err = resp.json()
    assert err["error"] == "location_not_found"
    assert "does-not-exist" in err["detail"]


def test_faq_parse_error_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings
    from app.domain.models import AIRequest, AIResponse, AIUsage
    from app.interfaces import faq_routes
    from app.main import create_app

    class BadFaqProvider:
        name = "bad"

        async def execute(self, request: AIRequest) -> AIResponse:
            return AIResponse(
                task=request.task,
                content="this is not valid faq json",
                provider=BadFaqProvider.name,
                model="m",
                usage=AIUsage(),
                latency_ms=1,
            )

    monkeypatch.setattr(
        faq_routes, "get_active_provider", lambda _cfg: BadFaqProvider()
    )
    get_settings.cache_clear()
    app = create_app()
    c2 = TestClient(app, raise_server_exceptions=False)
    resp = c2.get("/faq/de-berlin-hbf-upgrade")
    assert resp.status_code == 502
    assert resp.json()["error"] == "faq_parse_error"
