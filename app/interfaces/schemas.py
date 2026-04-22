"""Transport-layer schemas for the internal AI service API.

These are intentionally separate from domain models so that HTTP field names,
documentation, and serialization choices can evolve independently of the
domain contracts.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.domain.tasks import AITask


class ExecuteRequest(BaseModel):
    task: AITask = Field(description="The AI task type to perform.")
    input_text: str = Field(min_length=1, description="Text input for the AI model.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form metadata (language, location ID, etc.).",
    )


class UsageResponse(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ExecuteResponse(BaseModel):
    task: AITask
    content: str = Field(description="The AI-generated text output.")
    provider: str = Field(
        description="Name of the AI provider that handled the request."
    )
    model: str = Field(description="Exact model identifier returned by the provider.")
    usage: UsageResponse | None = Field(
        default=None, description="Token usage statistics."
    )
    latency_ms: int = Field(
        ge=0, description="End-to-end provider call latency in milliseconds."
    )


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error type (snake_case).")
    detail: str = Field(description="Human-readable error description.")
    request_id: str | None = Field(default=None, description="Request trace ID.")
