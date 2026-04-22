"""Internal AI execution API.

Intended for server-to-server use only — not exposed to end users directly.
All domain exceptions bubble up to the global handlers registered in main.py.
"""

import logging

from fastapi import APIRouter

from app.application.ai_service import execute_task
from app.infrastructure.openai_provider import get_active_provider
from app.infrastructure.yaml_config import AiConfigDep
from app.interfaces.schemas import ExecuteRequest, ExecuteResponse, UsageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post(
    "/ai/execute",
    response_model=ExecuteResponse,
    summary="Execute an AI task",
    description=(
        "Submit a task and input text to the active AI provider. "
        "Returns a normalized response including content, model, provider, usage, and latency."
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
