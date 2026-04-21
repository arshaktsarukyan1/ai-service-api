from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "AI Service Backend"
    app_version: str = "0.1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
