import pytest

from app.application.voice_service import (
    VoiceService,
    build_trigger_transcript,
    resolve_voice_intent,
    validate_audio_input,
)
from app.domain.exceptions import (
    AIProviderError,
    VoiceAudioValidationError,
    VoiceSynthesisError,
    VoiceTranscriptionError,
)
from app.domain.location import ConstructionSiteLocation
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


class _FailingSpeechToText:
    async def transcribe(self, audio: AudioInput) -> Transcript:
        raise VoiceTranscriptionError("stt failed")


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


class _FailingTextToSpeech:
    async def synthesize(
        self,
        text: str,
        options: SpeechSynthesisOptions,
    ) -> AudioOutput:
        raise VoiceSynthesisError("tts failed")


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


class _FailingAIProvider:
    name = "mock"

    async def execute(self, request: AIRequest) -> AIResponse:
        raise AIProviderError("ai failed")


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )


def _location() -> ConstructionSiteLocation:
    return ConstructionSiteLocation(
        id="site-1",
        name="Berlin Station Upgrade",
        start_date="2024-09-01",
        expected_end_date="2027-12-15",
        description="Accessibility upgrades.",
        costs="EUR 38.5 million",
        initiator="Deutsche Bahn",
        address="Invalidenstrasse 1, Berlin",
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


def test_validate_audio_input_rejects_non_audio_mime_type() -> None:
    with pytest.raises(VoiceAudioValidationError, match="Unsupported audio MIME type"):
        validate_audio_input(
            AudioInput(content=b"abc", format="webm", mime_type="application/json"),
            VoiceConfig(input_format="webm"),
        )


def test_resolve_voice_intent_defaults_to_voice_assistant() -> None:
    intent = resolve_voice_intent(VoiceSession())
    assert intent.name == "voice_assistant"
    assert intent.confidence is None
    assert intent.parameters == {}


def test_resolve_voice_intent_for_proximity_alert() -> None:
    intent = resolve_voice_intent(
        VoiceSession(
            trigger=VoiceTrigger(
                source=VoiceTriggerSource.proximity_alert,
                metadata={"distance_meters": 120},
            )
        )
    )
    assert intent.name == "proximity_alert"
    assert intent.confidence == 1.0
    assert intent.parameters["distance_meters"] == 120


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


def test_resolve_voice_intent_for_system_event_without_event_type() -> None:
    intent = resolve_voice_intent(
        VoiceSession(trigger=VoiceTrigger(source=VoiceTriggerSource.system_event))
    )
    assert intent.name == "system_event"
    assert intent.confidence == 1.0


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


def test_build_trigger_transcript_prefers_location_name_when_available() -> None:
    transcript = build_trigger_transcript(
        VoiceSession(
            trigger=VoiceTrigger(
                source=VoiceTriggerSource.app_event,
                location_id="site-1",
                event_type="location_entered",
            )
        ),
        _location(),
    )
    assert "Berlin Station Upgrade (site-1)" in transcript.text


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


async def test_voice_service_process_turn_uses_default_session() -> None:
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=_TextToSpeech(),
        ai_provider=_AIProvider(),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(language="de"),
    )

    turn = await service.process_turn(
        AudioInput(content=b"abc", format="webm", mime_type="audio/webm")
    )

    assert turn.session.language == "de"
    assert turn.session.trigger.source is VoiceTriggerSource.explicit_user_request


async def test_voice_service_process_turn_uses_custom_tts_config() -> None:
    tts = _TextToSpeech()
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=tts,
        ai_provider=_AIProvider(),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(tts_voice="verse", output_format="wav"),
    )

    turn = await service.process_turn(
        AudioInput(content=b"abc", format="webm", mime_type="audio/webm")
    )

    assert turn.audio.format == "wav"
    assert tts.options is not None
    assert tts.options.voice == "verse"


async def test_voice_service_process_turn_propagates_stt_failure() -> None:
    service = VoiceService(
        speech_to_text=_FailingSpeechToText(),
        text_to_speech=_TextToSpeech(),
        ai_provider=_AIProvider(),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    with pytest.raises(VoiceTranscriptionError, match="stt failed"):
        await service.process_turn(
            AudioInput(content=b"abc", format="webm", mime_type="audio/webm")
        )


async def test_voice_service_process_turn_propagates_ai_failure() -> None:
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=_TextToSpeech(),
        ai_provider=_FailingAIProvider(),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    with pytest.raises(AIProviderError, match="ai failed"):
        await service.process_turn(
            AudioInput(content=b"abc", format="webm", mime_type="audio/webm")
        )


async def test_voice_service_process_turn_propagates_tts_failure() -> None:
    service = VoiceService(
        speech_to_text=_SpeechToText(),
        text_to_speech=_FailingTextToSpeech(),
        ai_provider=_AIProvider(),
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    with pytest.raises(VoiceSynthesisError, match="tts failed"):
        await service.process_turn(
            AudioInput(content=b"abc", format="webm", mime_type="audio/webm")
        )


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


async def test_voice_service_process_trigger_uses_location_context() -> None:
    ai_provider = _AIProvider()
    service = VoiceService(
        speech_to_text=_FailingSpeechToText(),
        text_to_speech=_TextToSpeech(),
        ai_provider=ai_provider,
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
    )
    session = VoiceSession(
        id="session-1",
        trigger=VoiceTrigger(
            source=VoiceTriggerSource.proximity_alert,
            location_id="site-1",
            event_type="nearby_construction",
        ),
    )

    turn = await service.process_trigger(session=session, location=_location())

    assert "Berlin Station Upgrade" in turn.transcript.text
    assert ai_provider.request is not None
    assert "Berlin Station Upgrade" in ai_provider.request.input_text


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
