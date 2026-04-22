"""Global FastAPI exception handlers.

Each handler:
1. Extracts request_id from request.state (set by RequestIDMiddleware).
2. Logs at the appropriate level with structured context.
3. Returns a consistent ErrorResponse JSON body with the correct HTTP status.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    AIAuthError,
    AIProviderError,
    AIRateLimitError,
    AIServiceError,
    AITimeoutError,
    AIUnsupportedTaskError,
)
from app.interfaces.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _json_error(
    status_code: int,
    error: str,
    detail: str,
    request_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error,
            detail=detail,
            request_id=request_id,
        ).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application."""

    @app.exception_handler(AIUnsupportedTaskError)
    async def handle_unsupported_task(
        request: Request, exc: AIUnsupportedTaskError
    ) -> JSONResponse:
        rid = _request_id(request)
        logger.warning("Unsupported task request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "unsupported_task",
            str(exc),
            rid,
        )

    @app.exception_handler(AIAuthError)
    async def handle_auth_error(request: Request, exc: AIAuthError) -> JSONResponse:
        rid = _request_id(request)
        logger.error("AI auth error request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ai_auth_error",
            "AI service is not properly configured. Contact the administrator.",
            rid,
        )

    @app.exception_handler(AIRateLimitError)
    async def handle_rate_limit(
        request: Request, exc: AIRateLimitError
    ) -> JSONResponse:
        rid = _request_id(request)
        logger.warning("AI rate limit request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "ai_rate_limit",
            "AI provider rate limit exceeded. Please retry after a moment.",
            rid,
        )

    @app.exception_handler(AITimeoutError)
    async def handle_timeout(request: Request, exc: AITimeoutError) -> JSONResponse:
        rid = _request_id(request)
        logger.warning("AI timeout request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "ai_timeout",
            "AI provider did not respond in time. Please retry.",
            rid,
        )

    @app.exception_handler(AIProviderError)
    async def handle_provider_error(
        request: Request, exc: AIProviderError
    ) -> JSONResponse:
        rid = _request_id(request)
        logger.error("AI provider error request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_502_BAD_GATEWAY,
            "ai_provider_error",
            "AI provider returned an unexpected error. Please retry.",
            rid,
        )

    @app.exception_handler(AIServiceError)
    async def handle_service_error(
        request: Request, exc: AIServiceError
    ) -> JSONResponse:
        rid = _request_id(request)
        logger.error("AI service error request_id=%s detail=%s", rid, exc)
        return _json_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "ai_service_error",
            "An internal AI service error occurred.",
            rid,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        rid = _request_id(request)
        errors = exc.errors()
        logger.warning(
            "Request validation failed request_id=%s errors=%s", rid, errors
        )
        first = errors[0] if errors else {}
        location = " → ".join(str(loc) for loc in first.get("loc", []))
        detail = f"{first.get('msg', 'Validation error')} (field: {location})"
        return _json_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            detail,
            rid,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        logger.exception(
            "Unhandled exception request_id=%s type=%s", rid, type(exc).__name__
        )
        return _json_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected error occurred.",
            rid,
        )
