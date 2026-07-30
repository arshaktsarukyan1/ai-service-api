from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import AIUsage


class VoiceTriggerSource(StrEnum):
    explicit_user_request = "explicit_user_request"
    proximity_alert = "proximity_alert"
    app_event = "app_event"
    system_event = "system_event"


class AudioInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: bytes = Field(min_length=1)
    format: str = Field(default="webm", min_length=1)
    mime_type: str = Field(default="audio/webm", min_length=1)
    sample_rate_hz: int | None = Field(default=None, ge=1)


class AudioOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: bytes = Field(min_length=1)
    format: str = Field(default="mp3", min_length=1)
    mime_type: str = Field(default="audio/mpeg", min_length=1)


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    language: str = Field(default="de", min_length=2)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Intent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(default="voice_assistant", min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class VoiceTrigger(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: VoiceTriggerSource = VoiceTriggerSource.explicit_user_request
    location_id: str | None = Field(default=None, min_length=1)
    coordinates: dict[str, float] | None = None
    event_type: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoiceSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    language: str = Field(default="de", min_length=2)
    trigger: VoiceTrigger = Field(default_factory=VoiceTrigger)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SpeechSynthesisOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    voice: str = Field(min_length=1)
    language: str = Field(default="de", min_length=2)
    output_format: str = Field(default="mp3", min_length=1)
    instructions: str | None = Field(default=None, min_length=1)


class VoiceTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: VoiceSession
    transcript: Transcript
    intent: Intent
    response_text: str = Field(min_length=1)
    audio: AudioOutput
    provider: str
    model: str
    usage: AIUsage | None = None
    latency_ms: int = Field(ge=0)
