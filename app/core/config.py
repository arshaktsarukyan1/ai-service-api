import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    app_name: str = "AI Service Backend"
    app_version: str = "0.1.0"
    ai_config_path: Path = Path(
        os.environ.get("AI_CONFIG_PATH", "config/ai_providers.yaml")
    )
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
