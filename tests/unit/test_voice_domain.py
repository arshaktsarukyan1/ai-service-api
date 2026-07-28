import pytest
from pydantic import ValidationError

from app.domain.voice import (
    AudioInput,
    AudioOutput,
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


def test_voice_session_defaults_to_german_user_request() -> None:
    session = VoiceSession()
    assert session.language == "de"
    assert session.trigger.source is VoiceTriggerSource.explicit_user_request


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


def test_voice_provider_protocols_are_runtime_checkable() -> None:
    assert isinstance(_MockSpeechToTextService(), SpeechToTextService)
    assert isinstance(_MockTextToSpeechService(), TextToSpeechService)
