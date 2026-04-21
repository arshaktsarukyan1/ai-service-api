from functools import lru_cache
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import Depends

from app.core.config import get_settings
from app.infrastructure.config_schema import AIProvidersConfig

DEFAULT_CONFIG_PATH = Path("config/ai_providers.yaml")


@lru_cache(maxsize=1)
def load_ai_config(path: Path = DEFAULT_CONFIG_PATH) -> AIProvidersConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AIProvidersConfig.model_validate(raw)


def get_ai_config() -> AIProvidersConfig:
    return load_ai_config(get_settings().ai_config_path)


AiConfigDep = Annotated[AIProvidersConfig, Depends(get_ai_config)]
