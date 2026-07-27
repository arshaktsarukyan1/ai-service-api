from typing import Any

from pydantic import BaseModel, Field

from app.domain.faq import FaqItem
from app.domain.location import ConstructionSiteLocation
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


class ActiveProviderResponse(BaseModel):
    active_provider: str = Field(description="Configured active AI provider name.")
    default_model: str = Field(description="Default model for the active provider.")


class ErrorResponse(BaseModel):
    error: str = Field(description="Machine-readable error type (snake_case).")
    detail: str = Field(description="Human-readable error description.")
    request_id: str | None = Field(default=None, description="Request trace ID.")


class FaqLocationSummary(BaseModel):
    id: str
    name: str
    start_date: str
    expected_end_date: str
    description: str
    costs: str
    initiator: str
    address: str
    area: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @classmethod
    def from_domain(cls, location: ConstructionSiteLocation) -> FaqLocationSummary:
        return cls(
            id=location.id,
            name=location.name,
            start_date=location.start_date.isoformat(),
            expected_end_date=location.expected_end_date.isoformat(),
            description=location.description,
            costs=location.costs,
            initiator=location.initiator,
            address=location.address,
            area=location.area,
            latitude=location.latitude,
            longitude=location.longitude,
        )


class FaqMetadata(BaseModel):
    model: str
    provider: str
    latency_ms: int = Field(ge=0)
    source: str = Field(default="ai_generated")
    total_tokens: int | None = None


class FaqByLocationResponse(BaseModel):
    location: FaqLocationSummary
    faqs: list[FaqItem] = Field(min_length=1)
    generation: FaqMetadata


class DevLocationListItem(BaseModel):
    id: str
    name: str


class DevLocationListResponse(BaseModel):
    locations: list[DevLocationListItem]
