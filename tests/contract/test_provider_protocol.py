"""Contract tests: verify that provider implementations satisfy AIProvider protocol.

These tests check structural compliance without making real API calls.
"""

import inspect

from app.domain.models import AIRequest, AIResponse, AIUsage
from app.domain.provider import AIProvider
from app.domain.tasks import AITask


# ---------------------------------------------------------------------------
# MockProvider (from conftest) satisfies the protocol
# ---------------------------------------------------------------------------

def test_mock_provider_satisfies_protocol(mock_provider) -> None:
    assert isinstance(mock_provider, AIProvider)


def test_mock_provider_has_name_attribute(mock_provider) -> None:
    assert isinstance(mock_provider.name, str)
    assert len(mock_provider.name) > 0


def test_mock_provider_execute_is_async(mock_provider) -> None:
    assert inspect.iscoroutinefunction(mock_provider.execute)


async def test_mock_provider_returns_ai_response(mock_provider) -> None:
    request = AIRequest(task=AITask.information_preparation, input_text="test")
    response = await mock_provider.execute(request)
    assert isinstance(response, AIResponse)
    assert response.task == AITask.information_preparation
    assert isinstance(response.content, str)
    assert isinstance(response.provider, str)
    assert isinstance(response.model, str)
    assert response.latency_ms >= 0


# ---------------------------------------------------------------------------
# OpenAIProvider class structure (without instantiation)
# ---------------------------------------------------------------------------

def test_openai_provider_class_has_name_attribute() -> None:
    from app.infrastructure.openai_provider import OpenAIProvider

    assert hasattr(OpenAIProvider, "name")
    assert isinstance(OpenAIProvider.name, str)
    assert OpenAIProvider.name == "openai"


def test_openai_provider_execute_is_async() -> None:
    from app.infrastructure.openai_provider import OpenAIProvider

    assert inspect.iscoroutinefunction(OpenAIProvider.execute)


def test_openai_provider_ping_is_async() -> None:
    from app.infrastructure.openai_provider import OpenAIProvider

    assert inspect.iscoroutinefunction(OpenAIProvider.ping)


def test_openai_provider_raises_auth_error_without_key(monkeypatch) -> None:
    """Confirm AIAuthError is raised when the env var is not set."""
    import os

    from app.domain.exceptions import AIAuthError
    from app.infrastructure.config_schema import ProviderConfig
    from app.infrastructure.openai_provider import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig(api_key_env="OPENAI_API_KEY", default_model="gpt-4o-mini")
    with pytest.raises(AIAuthError):
        OpenAIProvider(config=config)


# ---------------------------------------------------------------------------
# AIResponse shape contract
# ---------------------------------------------------------------------------

def test_ai_response_required_fields() -> None:
    resp = AIResponse(
        task=AITask.voice_assistant,
        content="hello",
        provider="openai",
        model="gpt-4o",
        latency_ms=50,
    )
    assert resp.task == AITask.voice_assistant
    assert resp.usage is None


def test_ai_response_with_usage() -> None:
    usage = AIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp = AIResponse(
        task=AITask.information_structuring,
        content="structured output",
        provider="openai",
        model="gpt-4o-mini",
        usage=usage,
        latency_ms=120,
    )
    assert resp.usage.total_tokens == 15


def test_ai_response_latency_cannot_be_negative() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AIResponse(
            task=AITask.information_preparation,
            content="x",
            provider="openai",
            model="m",
            latency_ms=-1,
        )


import pytest  # noqa: E402 (needed for raises in the last tests)
