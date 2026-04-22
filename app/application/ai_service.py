import logging
from typing import Any

from app.domain.exceptions import AIUnsupportedTaskError
from app.domain.models import AIRequest, AIResponse
from app.domain.provider import AIProvider
from app.domain.tasks import AITask
from app.infrastructure.config_schema import ProviderConfig

logger = logging.getLogger(__name__)

_SUPPORTED_TASKS: frozenset[AITask] = frozenset(AITask)


def _resolve_model(task: AITask, provider_config: ProviderConfig) -> str:
    return provider_config.task_models.get(task, provider_config.default_model)


async def execute_task(
    task: AITask,
    input_text: str,
    *,
    provider: AIProvider,
    provider_config: ProviderConfig,
    metadata: dict[str, Any] | None = None,
) -> AIResponse:
    if task not in _SUPPORTED_TASKS:
        raise AIUnsupportedTaskError(
            f"Task '{task}' is not supported. Supported tasks: {sorted(_SUPPORTED_TASKS)}"
        )

    model = _resolve_model(task, provider_config)
    logger.info(
        "Executing task=%s provider=%s model=%s",
        task,
        provider.name,
        model,
    )

    request = AIRequest(task=task, input_text=input_text, metadata=metadata or {})
    response = await provider.execute(request)

    logger.info(
        "Task=%s completed provider=%s model=%s latency_ms=%d tokens=%s",
        task,
        response.provider,
        response.model,
        response.latency_ms,
        response.usage.total_tokens if response.usage else "n/a",
    )
    return response
