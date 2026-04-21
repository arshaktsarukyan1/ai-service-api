import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import ValidationError

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.infrastructure.yaml_config import load_ai_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        cfg = load_ai_config(settings.ai_config_path)
        logger.info(
            "AI config loaded: active_provider=%s, path=%s",
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
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
