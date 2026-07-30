import os
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.domain.exceptions import (
    AIAuthError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    VoiceSynthesisError,
    VoiceTranscriptionError,
)
from app.domain.voice import AudioInput, AudioOutput, SpeechSynthesisOptions, Transcript
from app.infrastructure.config_schema import ProviderConfig, VoiceConfig

_AUDIO_MIME_TYPES = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "wav": "audio/wav",
}


def _create_client(provider_config: ProviderConfig) -> AsyncOpenAI:
    api_key = os.environ.get(provider_config.api_key_env)
    if not api_key:
        raise AIAuthError(
            f"Environment variable '{provider_config.api_key_env}' is not set "
            "or empty. "
            "Set it before starting the service."
        )
    return AsyncOpenAI(
        api_key=api_key,
        timeout=float(provider_config.timeout_seconds),
        max_retries=max(0, provider_config.retry.attempts - 1),
    )


def _raise_provider_error(
    exc: Exception,
    *,
    operation: str,
    timeout_seconds: int,
) -> None:
    if isinstance(exc, AuthenticationError):
        raise AIAuthError(f"OpenAI {operation} authentication failed: {exc}") from exc
    if isinstance(exc, RateLimitError):
        raise AIRateLimitError(
            f"OpenAI {operation} rate limit exceeded: {exc}"
        ) from exc
    if isinstance(exc, APITimeoutError):
        raise AITimeoutError(
            f"OpenAI {operation} timed out after {timeout_seconds}s: {exc}"
        ) from exc
    if isinstance(exc, APIConnectionError):
        raise AIProviderError(f"OpenAI {operation} connection error: {exc}") from exc
    if isinstance(exc, BadRequestError):
        raise AIProviderError(f"OpenAI {operation} bad request: {exc}") from exc
    if isinstance(exc, APIError):
        raise AIProviderError(f"OpenAI {operation} API error: {exc}") from exc
    raise exc


class OpenAISpeechToTextService:
    def __init__(
        self,
        *,
        provider_config: ProviderConfig,
        voice_config: VoiceConfig,
        client: Any | None = None,
    ) -> None:
        self._provider_config = provider_config
        self._voice_config = voice_config
        self._client = client or _create_client(provider_config)

    async def transcribe(self, audio: AudioInput) -> Transcript:
        file_name = f"voice-input.{audio.format}"
        try:
            result = await self._client.audio.transcriptions.create(
                file=(file_name, audio.content, audio.mime_type),
                model=self._voice_config.stt_model,
                language=self._voice_config.language,
                response_format="json",
            )
        except Exception as exc:
            _raise_provider_error(
                exc,
                operation="speech-to-text",
                timeout_seconds=self._provider_config.timeout_seconds,
            )
            raise

        text = result if isinstance(result, str) else getattr(result, "text", "")
        if not isinstance(text, str) or not text.strip():
            raise VoiceTranscriptionError(
                "Speech-to-text returned an empty transcript."
            )

        return Transcript(text=text.strip(), language=self._voice_config.language)


class OpenAITextToSpeechService:
    def __init__(
        self,
        *,
        provider_config: ProviderConfig,
        voice_config: VoiceConfig,
        client: Any | None = None,
    ) -> None:
        self._provider_config = provider_config
        self._voice_config = voice_config
        self._client = client or _create_client(provider_config)

    async def synthesize(
        self,
        text: str,
        options: SpeechSynthesisOptions,
    ) -> AudioOutput:
        try:
            result = await self._client.audio.speech.create(
                input=text,
                model=self._voice_config.tts_model,
                voice=options.voice,
                instructions=options.instructions,
                response_format=options.output_format,
            )
        except Exception as exc:
            _raise_provider_error(
                exc,
                operation="text-to-speech",
                timeout_seconds=self._provider_config.timeout_seconds,
            )
            raise

        content = result.read() if callable(getattr(result, "read", None)) else None
        if not content and isinstance(getattr(result, "content", None), bytes):
            content = result.content
        if not content:
            raise VoiceSynthesisError("Text-to-speech returned empty audio content.")

        return AudioOutput(
            content=content,
            format=options.output_format,
            mime_type=_AUDIO_MIME_TYPES.get(
                options.output_format,
                "application/octet-stream",
            ),
        )
