import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.domain.exceptions import AIAuthError, AIProviderError, AITimeoutError
from app.infrastructure.openai_provider import get_active_provider
from app.infrastructure.yaml_config import AiConfigDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["health"])
def get_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    tags=["health"],
    summary="Readiness check",
    description=(
        "Verifies that the AI config is loaded and the active provider is wired. "
        "Pass `?ping=true` to additionally verify provider connectivity "
        "(makes a lightweight API call — use sparingly)."
    ),
)
async def get_readiness(
    config: AiConfigDep,
    ping: bool = Query(False, description="Trigger a live provider connectivity check."),
) -> JSONResponse:
    active = config.active_provider
    provider_config = config.providers.get(active)
    if provider_config is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": f"Active provider '{active}' not found in config.",
            },
        )

    payload: dict = {"status": "ready", "provider": active, "model": provider_config.default_model}

    if not ping:
        return JSONResponse(content=payload)

    provider = get_active_provider(config)
    try:
        await provider.ping()
        payload["ping"] = "ok"
    except AIAuthError as exc:
        logger.error("Readiness ping auth failure: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Provider authentication failed."},
        )
    except (AITimeoutError, AIProviderError) as exc:
        logger.warning("Readiness ping failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Provider not reachable."},
        )

    return JSONResponse(content=payload)
