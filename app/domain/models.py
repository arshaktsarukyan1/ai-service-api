from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.tasks import AITask


class AIUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class AIRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: AITask
    input_text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AIResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: AITask
    content: str
    provider: str
    model: str
    usage: AIUsage | None = None
    latency_ms: int = Field(ge=0)
