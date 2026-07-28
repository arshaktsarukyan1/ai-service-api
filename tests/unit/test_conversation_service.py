from app.application.conversation_service import (
    build_voice_assistant_prompt,
    generate_voice_response,
)
from app.domain.location import ConstructionSiteLocation
from app.domain.models import AIRequest, AIResponse
from app.domain.tasks import AITask
from app.domain.voice import Transcript, VoiceSession, VoiceTrigger, VoiceTriggerSource
from app.infrastructure.config_schema import ProviderConfig


def _location() -> ConstructionSiteLocation:
    return ConstructionSiteLocation(
        id="de-berlin-hbf-upgrade",
        name="Berlin Central Station Upgrade",
        start_date="2024-09-01",
        expected_end_date="2027-12-15",
        description="Platform and accessibility upgrades.",
        costs="EUR 38.5 million",
        initiator="Deutsche Bahn",
        address="Invalidenstraße 1, Berlin",
        area="Mitte",
        latitude=52.525,
        longitude=13.3692,
    )


class _CapturingProvider:
    name = "mock"

    def __init__(self) -> None:
        self.request: AIRequest | None = None

    async def execute(self, request: AIRequest) -> AIResponse:
        self.request = request
        return AIResponse(
            task=request.task,
            content="Die Bauarbeiten betreffen die Bahnsteige.",
            provider="mock",
            model="mock-model",
            latency_ms=5,
        )


def test_build_voice_prompt_is_german_first_and_includes_trigger_context() -> None:
    session = VoiceSession(
        id="session-1",
        trigger=VoiceTrigger(
            source=VoiceTriggerSource.proximity_alert,
            location_id="de-berlin-hbf-upgrade",
            event_type="nearby_construction",
        ),
    )
    prompt = build_voice_assistant_prompt(
        Transcript(text="Was passiert hier?", language="de"),
        session,
        location=_location(),
    )
    assert "German-first" in prompt
    assert "proximity_alert" in prompt
    assert "de-berlin-hbf-upgrade" in prompt
    assert "Berlin Central Station Upgrade" in prompt


async def test_generate_voice_response_uses_voice_assistant_task_and_metadata() -> None:
    provider = _CapturingProvider()
    session = VoiceSession(
        id="session-1",
        trigger=VoiceTrigger(
            source=VoiceTriggerSource.app_event,
            location_id="site-1",
            event_type="location_entered",
        ),
    )
    response = await generate_voice_response(
        Transcript(text="Sag mir was dazu.", language="de"),
        session,
        provider=provider,
        provider_config=ProviderConfig(
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4o-mini",
        ),
    )
    assert response.content == "Die Bauarbeiten betreffen die Bahnsteige."
    assert provider.request is not None
    assert provider.request.task is AITask.voice_assistant
    assert provider.request.metadata["session_id"] == "session-1"
    assert provider.request.metadata["trigger_source"] == "app_event"
    assert provider.request.metadata["event_type"] == "location_entered"
