"""Unit tests for the AIService orchestrator (execute_task)."""

import pytest

from app.application.ai_service import execute_task
from app.domain.exceptions import AIProviderError, AIUnsupportedTaskError
from app.domain.models import AIRequest, AIResponse, AIUsage
from app.domain.tasks import AITask
from app.infrastructure.config_schema import ProviderConfig, RetryConfig


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        task_models={AITask.voice_assistant: "gpt-4o"},
    )


class _GoodProvider:
    name = "mock"

    async def execute(self, request: AIRequest) -> AIResponse:
        return AIResponse(
            task=request.task,
            content=f"result:{request.input_text}",
            provider="mock",
            model="mock-model",
            usage=AIUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            latency_ms=7,
        )


class _FailingProvider:
    name = "mock"

    async def execute(self, request: AIRequest) -> AIResponse:
        raise AIProviderError("upstream failure")


async def test_execute_task_happy_path(provider_config: ProviderConfig) -> None:
    resp = await execute_task(
        AITask.information_preparation,
        "What is Vienna?",
        provider=_GoodProvider(),
        provider_config=provider_config,
    )
    assert resp.task == AITask.information_preparation
    assert resp.content == "result:What is Vienna?"
    assert resp.provider == "mock"
    assert resp.latency_ms == 7


async def test_execute_task_passes_metadata(provider_config: ProviderConfig) -> None:
    captured: dict = {}

    class _CapturingProvider:
        name = "mock"

        async def execute(self, request: AIRequest) -> AIResponse:
            captured.update(request.metadata)
            return AIResponse(
                task=request.task,
                content="ok",
                provider="mock",
                model="m",
                latency_ms=1,
            )

    await execute_task(
        AITask.information_structuring,
        "structure this",
        provider=_CapturingProvider(),
        provider_config=provider_config,
        metadata={"language": "de", "location_id": "42"},
    )
    assert captured["language"] == "de"
    assert captured["location_id"] == "42"


async def test_execute_task_unsupported_task_raises(
    provider_config: ProviderConfig,
) -> None:
    import app.application.ai_service as svc

    original = svc._SUPPORTED_TASKS
    svc._SUPPORTED_TASKS = frozenset()
    try:
        with pytest.raises(AIUnsupportedTaskError):
            await execute_task(
                AITask.information_preparation,
                "hello",
                provider=_GoodProvider(),
                provider_config=provider_config,
            )
    finally:
        svc._SUPPORTED_TASKS = original


async def test_execute_task_propagates_provider_error(
    provider_config: ProviderConfig,
) -> None:
    with pytest.raises(AIProviderError, match="upstream failure"):
        await execute_task(
            AITask.information_preparation,
            "hello",
            provider=_FailingProvider(),
            provider_config=provider_config,
        )


async def test_execute_task_uses_default_model_when_no_override(
    provider_config: ProviderConfig,
) -> None:
    """information_structuring is not in task_models → falls back to default_model."""
    from app.application.ai_service import _resolve_model

    model = _resolve_model(AITask.information_structuring, provider_config)
    assert model == "gpt-4o-mini"


async def test_execute_task_uses_task_model_override(
    provider_config: ProviderConfig,
) -> None:
    from app.application.ai_service import _resolve_model

    model = _resolve_model(AITask.voice_assistant, provider_config)
    assert model == "gpt-4o"


async def test_execute_task_empty_metadata_default(
    provider_config: ProviderConfig,
) -> None:
    captured: dict = {}

    class _Cap:
        name = "mock"

        async def execute(self, request: AIRequest) -> AIResponse:
            captured["meta"] = request.metadata
            return AIResponse(
                task=request.task, content="x", provider="mock", model="m", latency_ms=1
            )

    await execute_task(
        AITask.information_preparation,
        "hi",
        provider=_Cap(),
        provider_config=provider_config,
    )
    assert captured["meta"] == {}
