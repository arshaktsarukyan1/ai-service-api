from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.tasks import AITask


class RetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int = Field(ge=1, le=10, default=3)


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


class VoiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stt_model: str = Field(default="gpt-4o-mini-transcribe", min_length=1)
    tts_model: str = Field(default="gpt-4o-mini-tts", min_length=1)
    tts_voice: str = Field(default="alloy", min_length=1)
    language: str = Field(
        default="de",
        min_length=2,
        max_length=12,
        pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]+)*$",
    )
    input_format: Literal["webm", "wav", "mp3", "m4a", "ogg"] = "webm"
    output_format: Literal["mp3", "wav", "opus", "aac", "flac"] = "mp3"
    max_audio_bytes: int = Field(default=5_000_000, ge=1, le=25_000_000)
    session_timeout_seconds: int = Field(default=60, ge=5, le=600)


class GeoFencingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_radius_meters: int = Field(default=100, ge=1, le=100_000)
    min_radius_meters: int = Field(default=10, ge=1, le=100_000)
    max_radius_meters: int = Field(default=5000, ge=1, le=100_000)
    exit_hysteresis_meters: int = Field(default=25, ge=0, le=100_000)
    trigger_cooldown_seconds: int = Field(default=60, ge=0, le=86_400)
    max_acceptable_accuracy_meters: int = Field(default=100, ge=1, le=100_000)

    @model_validator(mode="after")
    def radius_values_must_be_consistent(self) -> GeoFencingConfig:
        if self.min_radius_meters > self.default_radius_meters:
            raise ValueError(
                "min_radius_meters must be less than or equal to "
                "default_radius_meters"
            )
        if self.default_radius_meters > self.max_radius_meters:
            raise ValueError(
                "default_radius_meters must be less than or equal to "
                "max_radius_meters"
            )
        if self.exit_hysteresis_meters > self.max_radius_meters:
            raise ValueError(
                "exit_hysteresis_meters must be less than or equal to "
                "max_radius_meters"
            )
        return self


class AIProvidersConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_provider: str
    providers: dict[str, ProviderConfig]
    database: DatabaseConfig | None = None
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    geofencing: GeoFencingConfig = Field(default_factory=GeoFencingConfig)

    @model_validator(mode="after")
    def active_provider_must_exist(self) -> AIProvidersConfig:
        if self.active_provider not in self.providers:
            raise ValueError(
                f"active_provider '{self.active_provider}' not found in providers"
            )
        return self
