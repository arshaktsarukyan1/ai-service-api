import logging

from fastapi import APIRouter

from app.application.ai_service import execute_task
from app.infrastructure.openai_provider import get_active_provider
from app.infrastructure.yaml_config import AiConfigDep
from app.interfaces.schemas import (
    ActiveProviderResponse,
    ExecuteRequest,
    ExecuteResponse,
    UsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])


@router.get(
    "/ai/provider",
    response_model=ActiveProviderResponse,
    summary="Get active AI provider",
)
async def get_active_ai_provider(config: AiConfigDep) -> ActiveProviderResponse:
    active_provider = config.active_provider
    provider_config = config.providers[active_provider]
    return ActiveProviderResponse(
        active_provider=active_provider,
        default_model=provider_config.default_model,
    )


@router.post(
    "/ai/execute",
    response_model=ExecuteResponse,
    summary="Execute an AI task",
    description=(
        "Submit a task and input text to the active AI provider. "
        "Returns normalized content, provider/model details, usage, and latency."
    ),
)
async def execute_ai_task(body: ExecuteRequest, config: AiConfigDep) -> ExecuteResponse:
    provider = get_active_provider(config)
    provider_config = config.providers[config.active_provider]

    result = await execute_task(
        task=body.task,
        input_text=body.input_text,
        provider=provider,
        provider_config=provider_config,
        metadata=body.metadata,
    )

    usage = (
        UsageResponse(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        )
        if result.usage
        else None
    )

    return ExecuteResponse(
        task=result.task,
        content=result.content,
        provider=result.provider,
        model=result.model,
        usage=usage,
        latency_ms=result.latency_ms,
    )
