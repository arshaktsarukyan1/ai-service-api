import time

from app.application.conversation_service import generate_voice_response
from app.domain.exceptions import VoiceAudioValidationError, VoiceSynthesisError
from app.domain.location import ConstructionSiteLocation
from app.domain.models import AIResponse
from app.domain.provider import AIProvider
from app.domain.voice import (
    AudioInput,
    Intent,
    SpeechSynthesisOptions,
    Transcript,
    VoiceSession,
    VoiceTriggerSource,
    VoiceTurn,
)
from app.domain.voice_provider import SpeechToTextService, TextToSpeechService
from app.infrastructure.config_schema import ProviderConfig, VoiceConfig


def validate_audio_input(audio: AudioInput, voice_config: VoiceConfig) -> None:
    if len(audio.content) > voice_config.max_audio_bytes:
        raise VoiceAudioValidationError(
            f"Audio input exceeds max size of {voice_config.max_audio_bytes} bytes."
        )
    if audio.format != voice_config.input_format:
        raise VoiceAudioValidationError(
            f"Unsupported audio format '{audio.format}'. "
            f"Expected '{voice_config.input_format}'."
        )
    if not audio.mime_type.startswith("audio/"):
        raise VoiceAudioValidationError(
            f"Unsupported audio MIME type '{audio.mime_type}'."
        )


def resolve_voice_intent(session: VoiceSession) -> Intent:
    trigger = session.trigger
    if trigger.source is VoiceTriggerSource.proximity_alert:
        return Intent(
            name="proximity_alert",
            confidence=1.0,
            parameters=trigger.metadata,
        )
    event_sources = {VoiceTriggerSource.app_event, VoiceTriggerSource.system_event}
    if trigger.source in event_sources:
        return Intent(
            name=trigger.event_type or trigger.source.value,
            confidence=1.0,
            parameters=trigger.metadata,
        )
    return Intent(name="voice_assistant", parameters=trigger.metadata)


def build_trigger_transcript(
    session: VoiceSession,
    location: ConstructionSiteLocation | None = None,
) -> Transcript:
    trigger = session.trigger
    location_text = (
        f" for location {location.name} ({location.id})"
        if location
        else f" for location {trigger.location_id}"
        if trigger.location_id
        else ""
    )
    event_text = trigger.event_type or trigger.source.value
    return Transcript(
        text=f"App-triggered voice event '{event_text}'{location_text}.",
        language=session.language,
        confidence=1.0,
    )


class VoiceService:
    def __init__(
        self,
        *,
        speech_to_text: SpeechToTextService,
        text_to_speech: TextToSpeechService,
        ai_provider: AIProvider,
        provider_config: ProviderConfig,
        voice_config: VoiceConfig,
    ) -> None:
        self._speech_to_text = speech_to_text
        self._text_to_speech = text_to_speech
        self._ai_provider = ai_provider
        self._provider_config = provider_config
        self._voice_config = voice_config

    async def _synthesize_turn(
        self,
        *,
        session: VoiceSession,
        transcript: Transcript,
        intent: Intent,
        ai_response: AIResponse,
        start: float,
    ) -> VoiceTurn:
        if not ai_response.content.strip():
            raise VoiceSynthesisError("AI response was empty; cannot synthesize audio.")

        audio_output = await self._text_to_speech.synthesize(
            ai_response.content,
            SpeechSynthesisOptions(
                voice=self._voice_config.tts_voice,
                language=session.language,
                output_format=self._voice_config.output_format,
                instructions="Sprich natürlich, klar und freundlich auf Deutsch.",
            ),
        )

        return VoiceTurn(
            session=session,
            transcript=transcript,
            intent=intent,
            response_text=ai_response.content,
            audio=audio_output,
            provider=ai_response.provider,
            model=ai_response.model,
            usage=ai_response.usage,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    async def process_turn(
        self,
        audio: AudioInput,
        *,
        session: VoiceSession | None = None,
        location: ConstructionSiteLocation | None = None,
    ) -> VoiceTurn:
        active_session = session or VoiceSession(language=self._voice_config.language)
        start = time.monotonic()

        validate_audio_input(audio, self._voice_config)
        transcript = await self._speech_to_text.transcribe(audio)
        intent = resolve_voice_intent(active_session)
        ai_response = await generate_voice_response(
            transcript,
            active_session,
            provider=self._ai_provider,
            provider_config=self._provider_config,
            location=location,
        )
        return await self._synthesize_turn(
            session=active_session,
            transcript=transcript,
            intent=intent,
            ai_response=ai_response,
            start=start,
        )

    async def process_trigger(
        self,
        *,
        session: VoiceSession,
        location: ConstructionSiteLocation | None = None,
    ) -> VoiceTurn:
        start = time.monotonic()
        transcript = build_trigger_transcript(session, location)
        intent = resolve_voice_intent(session)
        ai_response = await generate_voice_response(
            transcript,
            session,
            provider=self._ai_provider,
            provider_config=self._provider_config,
            location=location,
        )
        return await self._synthesize_turn(
            session=session,
            transcript=transcript,
            intent=intent,
            ai_response=ai_response,
            start=start,
        )
