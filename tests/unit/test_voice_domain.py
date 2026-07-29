import pytest
from pydantic import ValidationError

from app.domain.voice import (
    AudioInput,
    AudioOutput,
    Intent,
    SpeechSynthesisOptions,
    Transcript,
    VoiceSession,
    VoiceTrigger,
    VoiceTriggerSource,
    VoiceTurn,
)
from app.domain.voice_provider import SpeechToTextService, TextToSpeechService


class _MockSpeechToTextService:
    async def transcribe(self, audio: AudioInput) -> Transcript:
        return Transcript(text="Hallo", language="de")


class _MockTextToSpeechService:
    async def synthesize(
        self,
        text: str,
        options: SpeechSynthesisOptions,
    ) -> AudioOutput:
        return AudioOutput(content=text.encode(), format=options.output_format)


def test_audio_input_requires_content() -> None:
    with pytest.raises(ValidationError):
        AudioInput(content=b"")


def test_audio_input_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValidationError):
        AudioInput(content=b"audio", sample_rate_hz=0)


def test_audio_output_requires_content() -> None:
    with pytest.raises(ValidationError):
        AudioOutput(content=b"")


def test_transcript_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Transcript(text="")


def test_transcript_rejects_confidence_outside_range() -> None:
    with pytest.raises(ValidationError):
        Transcript(text="Hallo", confidence=1.1)


def test_intent_defaults_and_parameters_are_isolated() -> None:
    first = Intent()
    second = Intent()
    first.parameters["location_id"] = "site-1"
    assert first.name == "voice_assistant"
    assert second.parameters == {}


def test_voice_trigger_accepts_structured_location_metadata() -> None:
    trigger = VoiceTrigger(
        source=VoiceTriggerSource.system_event,
        location_id="site-1",
        coordinates={"latitude": 52.5, "longitude": 13.4},
        event_type="nearby_construction",
        metadata={"severity": "info"},
    )
    assert trigger.coordinates == {"latitude": 52.5, "longitude": 13.4}
    assert trigger.metadata["severity"] == "info"


def test_voice_session_defaults_to_german_user_request() -> None:
    session = VoiceSession()
    assert session.language == "de"
    assert session.trigger.source is VoiceTriggerSource.explicit_user_request


def test_voice_sessions_get_distinct_default_ids() -> None:
    assert VoiceSession().id != VoiceSession().id


def test_voice_session_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        VoiceSession(id="")


def test_speech_synthesis_options_require_voice() -> None:
    with pytest.raises(ValidationError):
        SpeechSynthesisOptions(voice="")


def test_voice_turn_holds_normalized_voice_result() -> None:
    session = VoiceSession(
        trigger=VoiceTrigger(
            source=VoiceTriggerSource.proximity_alert,
            location_id="site-1",
        )
    )
    turn = VoiceTurn(
        session=session,
        transcript=Transcript(text="Was ist hier los?", language="de"),
        intent={"name": "construction_update"},
        response_text="Hier gibt es Bauarbeiten.",
        audio=AudioOutput(content=b"audio"),
        provider="mock",
        model="mock-model",
        latency_ms=42,
    )
    assert turn.session.trigger.location_id == "site-1"
    assert turn.intent.name == "construction_update"
    assert turn.audio.mime_type == "audio/mpeg"


def test_voice_turn_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        VoiceTurn(
            session=VoiceSession(),
            transcript=Transcript(text="Hallo"),
            intent=Intent(),
            response_text="Antwort",
            audio=AudioOutput(content=b"audio"),
            provider="mock",
            model="mock-model",
            latency_ms=-1,
        )


def test_voice_provider_protocols_are_runtime_checkable() -> None:
    assert isinstance(_MockSpeechToTextService(), SpeechToTextService)
    assert isinstance(_MockTextToSpeechService(), TextToSpeechService)
