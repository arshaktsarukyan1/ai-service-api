import pytest

from app.application.voice_service import (
    VoiceService,
    build_trigger_transcript,
    resolve_voice_intent,
    validate_audio_input,
)
from app.domain.exceptions import VoiceAudioValidationError, VoiceSynthesisError
from app.domain.models import AIRequest, AIResponse, AIUsage
from app.domain.voice import (
    AudioInput,
    AudioOutput,
    SpeechSynthesisOptions,
    Transcript,
    VoiceSession,
    VoiceTrigger,
    VoiceTriggerSource,
)
from app.infrastructure.config_schema import ProviderConfig, VoiceConfig


class _SpeechToText:
    async def transcribe(self, audio: AudioInput) -> Transcript:
        return Transcript(text="Was ist hier?", language="de")


class _TextToSpeech:
    def __init__(self) -> None:
        self.text = ""
        self.options: SpeechSynthesisOptions | None = None

    async def synthesize(
        self,
        text: str,
        options: SpeechSynthesisOptions,
    ) -> AudioOutput:
        self.text = text
        self.options = options
        return AudioOutput(content=b"audio", format=options.output_format)


class _AIProvider:
    name = "mock"

    def __init__(self, content: str = "Hier gibt es Bauarbeiten.") -> None:
        self.content = content
        self.request: AIRequest | None = None

    async def execute(self, request: AIRequest) -> AIResponse:
        self.request = request
        return AIResponse(
            task=request.task,
            content=self.content,
            provider="mock",
            model="mock-model",
            usage=AIUsage(total_tokens=12),
            latency_ms=8,
        )


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )


def test_validate_audio_input_accepts_configured_audio() -> None:
    validate_audio_input(
        AudioInput(content=b"abc", format="webm", mime_type="audio/webm"),
        VoiceConfig(input_format="webm", max_audio_bytes=3),
    )


def test_validate_audio_input_rejects_oversized_audio() -> None:
    with pytest.raises(VoiceAudioValidationError, match="exceeds max size"):
        validate_audio_input(
            AudioInput(content=b"abcd", format="webm", mime_type="audio/webm"),
            VoiceConfig(input_format="webm", max_audio_bytes=3),
        )


def test_validate_audio_input_rejects_unconfigured_format() -> None:
    with pytest.raises(VoiceAudioValidationError, match="Unsupported audio format"):
        validate_audio_input(
            AudioInput(content=b"abc", format="wav", mime_type="audio/wav"),
            VoiceConfig(input_format="webm"),
        )


def test_resolve_voice_intent_for_app_trigger() -> None:
    intent = resolve_voice_intent(
        VoiceSession(
            trigger=VoiceTrigger(
                source=VoiceTriggerSource.app_event,
                event_type="location_entered",
                metadata={"location_id": "site-1"},
            )
        )
    )
    assert intent.name == "location_entered"
    assert intent.confidence == 1.0
    assert intent.parameters["location_id"] == "site-1"


def test_build_trigger_transcript_includes_event_and_location_id() -> None:
    transcript = build_trigger_transcript(
        VoiceSession(
            trigger=VoiceTrigger(
                source=VoiceTriggerSource.proximity_alert,
                location_id="site-1",
                event_type="nearby_construction",
            )
        )
    )
    assert transcript.text == (
        "App-triggered voice event 'nearby_construction' for location site-1."
    )
    assert transcript.confidence == 1.0


async def test_voice_service_process_turn_runs_full_pipeline() -> None:
    tts = _TextToSpeech()
    ai_provider = _AIProvider()
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=tts,
        ai_provider=ai_provider,
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    session = VoiceSession(id="session-1")

    turn = await service.process_turn(
        AudioInput(content=b"abc", format="webm", mime_type="audio/webm"),
        session=session,
    )

    assert turn.session.id == "session-1"
    assert turn.transcript.text == "Was ist hier?"
    assert turn.response_text == "Hier gibt es Bauarbeiten."
    assert turn.audio.content == b"audio"
    assert turn.provider == "mock"
    assert turn.usage.total_tokens == 12
    assert tts.text == "Hier gibt es Bauarbeiten."
    assert tts.options is not None
    assert tts.options.language == "de"
    assert tts.options.output_format == "mp3"
    assert ai_provider.request is not None
    assert ai_provider.request.metadata["session_id"] == "session-1"


async def test_voice_service_process_trigger_runs_without_stt() -> None:
    tts = _TextToSpeech()
    ai_provider = _AIProvider()
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=tts,
        ai_provider=ai_provider,
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    session = VoiceSession(
        id="session-1",
        trigger=VoiceTrigger(
            source=VoiceTriggerSource.app_event,
            location_id="site-1",
            event_type="location_entered",
        ),
    )

    turn = await service.process_trigger(session=session)

    assert turn.session.id == "session-1"
    assert turn.intent.name == "location_entered"
    assert "location_entered" in turn.transcript.text
    assert turn.audio.content == b"audio"
    assert ai_provider.request is not None
    assert "App-triggered voice event" in ai_provider.request.input_text


async def test_voice_service_rejects_empty_ai_response_before_tts() -> None:
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=_TextToSpeech(),
        ai_provider=_AIProvider(content=" "),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    with pytest.raises(VoiceSynthesisError, match="AI response was empty"):
        await service.process_turn(
            AudioInput(content=b"abc", format="webm", mime_type="audio/webm"),
        )
