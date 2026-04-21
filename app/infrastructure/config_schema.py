from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.tasks import AITask


class RetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int = Field(ge=1, le=10, default=3)
    delay_seconds: float = Field(ge=0.1, le=30.0, default=1.0)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key_env: str
    default_model: str
    task_models: dict[AITask, str] = Field(default_factory=dict)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: int = Field(ge=1, le=128000, default=2048)
    timeout_seconds: int = Field(ge=1, le=300, default=30)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class ArangoDBConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "localhost"
    port: int = Field(ge=1, le=65535, default=8529)
    username: str = "root"
    password_env: str = "ARANGO_PASSWORD"
    database: str = "ai_service"


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    arangodb: ArangoDBConfig | None = None


class AIProvidersConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_provider: str
    providers: dict[str, ProviderConfig]
    database: DatabaseConfig | None = None

    @model_validator(mode="after")
    def active_provider_must_exist(self) -> "AIProvidersConfig":
        if self.active_provider not in self.providers:
            raise ValueError(
                f"active_provider '{self.active_provider}' not found in providers"
            )
        return self
