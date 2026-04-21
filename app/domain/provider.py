from typing import Protocol, runtime_checkable

from app.domain.models import AIRequest, AIResponse


@runtime_checkable
class AIProvider(Protocol):
    name: str

    async def execute(self, request: AIRequest) -> AIResponse: ...
