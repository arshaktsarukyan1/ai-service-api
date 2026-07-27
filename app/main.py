import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from app.api.routes import router as health_router
from app.core.config import get_settings
from app.core.error_handlers import register_error_handlers
from app.core.logging_config import configure_logging
from app.core.middleware import RequestIDMiddleware
from app.infrastructure.yaml_config import load_ai_config
from app.interfaces.faq_routes import router as faq_router
from app.interfaces.internal_routes import router as internal_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        cfg = load_ai_config(settings.ai_config_path)
        logger.info(
            "AI config loaded: active_provider=%s path=%s",
            cfg.active_provider,
            settings.ai_config_path,
        )
    except FileNotFoundError as exc:
        logger.critical("AI config file not found: %s — cannot start.", exc)
        sys.exit(1)
    except ValidationError as exc:
        logger.critical("AI config validation failed — cannot start.\n%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected error loading AI config: %s — cannot start.", exc)
        sys.exit(1)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    app.include_router(health_router)
    app.include_router(faq_router)
    app.include_router(internal_router)

    return app


app = create_app()
