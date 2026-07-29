from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
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
from app.domain.voice import AudioInput, SpeechSynthesisOptions
from app.infrastructure.config_schema import ProviderConfig, VoiceConfig
from app.infrastructure.openai_audio_provider import (
    OpenAISpeechToTextService,
    OpenAITextToSpeechService,
    _create_client,
)


class _Transcriptions:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or SimpleNamespace(text=" Hallo Welt ")
        self.error = error
        self.kwargs = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


class _Speech:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or _BinaryResult(b"audio")
        self.error = error
        self.kwargs = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


class _Audio:
    def __init__(self, transcriptions=None, speech=None) -> None:
        self.transcriptions = transcriptions or _Transcriptions()
        self.speech = speech or _Speech()


class _Client:
    def __init__(self, audio: _Audio) -> None:
        self.audio = audio


class _BinaryResult:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self) -> bytes:
        return self._content


def _provider_config() -> ProviderConfig:
    return ProviderConfig(api_key_env="OPENAI_API_KEY", default_model="gpt-4o-mini")


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio")
    return httpx.Response(status_code, request=request)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/audio")


def test_create_client_requires_configured_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_OPENAI_KEY", raising=False)
    with pytest.raises(AIAuthError, match="MISSING_OPENAI_KEY"):
        _create_client(
            ProviderConfig(
                api_key_env="MISSING_OPENAI_KEY",
                default_model="gpt-4o-mini",
            )
        )


async def test_openai_stt_transcribes_audio_and_passes_config() -> None:
    transcriptions = _Transcriptions()
    service = OpenAISpeechToTextService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(language="de", stt_model="gpt-4o-transcribe"),
        client=_Client(_Audio(transcriptions=transcriptions)),
    )

    transcript = await service.transcribe(
        AudioInput(content=b"audio", format="webm", mime_type="audio/webm")
    )

    assert transcript.text == "Hallo Welt"
    assert transcript.language == "de"
    assert transcriptions.kwargs["model"] == "gpt-4o-transcribe"
    assert transcriptions.kwargs["language"] == "de"
    assert transcriptions.kwargs["file"][0] == "voice-input.webm"


async def test_openai_stt_accepts_string_response() -> None:
    service = OpenAISpeechToTextService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(language="de"),
        client=_Client(_Audio(transcriptions=_Transcriptions(result=" Guten Tag "))),
    )

    transcript = await service.transcribe(
        AudioInput(content=b"audio", format="webm", mime_type="audio/webm")
    )

    assert transcript.text == "Guten Tag"


async def test_openai_stt_rejects_empty_transcript() -> None:
    service = OpenAISpeechToTextService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(transcriptions=_Transcriptions(result="   "))),
    )

    with pytest.raises(VoiceTranscriptionError, match="empty transcript"):
        await service.transcribe(
            AudioInput(content=b"audio", format="webm", mime_type="audio/webm")
        )


async def test_openai_stt_maps_authentication_error() -> None:
    service = OpenAISpeechToTextService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(
            _Audio(
                transcriptions=_Transcriptions(
                    error=AuthenticationError(
                        "bad key",
                        response=_response(401),
                        body=None,
                    )
                )
            )
        ),
    )

    with pytest.raises(AIAuthError, match="speech-to-text authentication failed"):
        await service.transcribe(
            AudioInput(content=b"audio", format="webm", mime_type="audio/webm")
        )


@pytest.mark.parametrize(
    ("error", "expected_exception", "match"),
    [
        (
            RateLimitError("rate limit", response=_response(429), body=None),
            AIRateLimitError,
            "rate limit exceeded",
        ),
        (
            APITimeoutError(_request()),
            AITimeoutError,
            "timed out",
        ),
        (
            APIConnectionError(request=_request()),
            AIProviderError,
            "connection error",
        ),
        (
            APIError("api failed", request=_request(), body=None),
            AIProviderError,
            "API error",
        ),
    ],
)
async def test_openai_stt_maps_provider_errors(
    error: Exception,
    expected_exception: type[Exception],
    match: str,
) -> None:
    service = OpenAISpeechToTextService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(transcriptions=_Transcriptions(error=error))),
    )

    with pytest.raises(expected_exception, match=match):
        await service.transcribe(
            AudioInput(content=b"audio", format="webm", mime_type="audio/webm")
        )


async def test_openai_tts_synthesizes_audio_and_passes_config() -> None:
    speech = _Speech()
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(tts_model="gpt-4o-mini-tts"),
        client=_Client(_Audio(speech=speech)),
    )

    audio = await service.synthesize(
        "Hallo",
        SpeechSynthesisOptions(
            voice="alloy",
            language="de",
            output_format="mp3",
            instructions="Sprich klar.",
        ),
    )

    assert audio.content == b"audio"
    assert audio.format == "mp3"
    assert audio.mime_type == "audio/mpeg"
    assert speech.kwargs["input"] == "Hallo"
    assert speech.kwargs["model"] == "gpt-4o-mini-tts"
    assert speech.kwargs["voice"] == "alloy"
    assert speech.kwargs["response_format"] == "mp3"


class _ContentOnlyResult:
    def __init__(self, content: bytes) -> None:
        self.content = content


async def test_openai_tts_accepts_content_attribute_response() -> None:
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(speech=_Speech(result=_ContentOnlyResult(b"audio")))),
    )

    audio = await service.synthesize("Hallo", SpeechSynthesisOptions(voice="alloy"))

    assert audio.content == b"audio"


async def test_openai_tts_uses_octet_stream_for_unknown_format() -> None:
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(speech=_Speech())),
    )

    audio = await service.synthesize(
        "Hallo",
        SpeechSynthesisOptions(voice="alloy", output_format="pcm"),
    )

    assert audio.mime_type == "application/octet-stream"


async def test_openai_tts_maps_bad_request_error() -> None:
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(
            _Audio(
                speech=_Speech(
                    error=BadRequestError(
                        "bad audio request",
                        response=_response(400),
                        body=None,
                    )
                )
            )
        ),
    )

    with pytest.raises(AIProviderError, match="text-to-speech bad request"):
        await service.synthesize(
            "Hallo",
            SpeechSynthesisOptions(voice="alloy"),
        )


@pytest.mark.parametrize(
    ("error", "expected_exception", "match"),
    [
        (
            AuthenticationError("bad key", response=_response(401), body=None),
            AIAuthError,
            "text-to-speech authentication failed",
        ),
        (
            RateLimitError("rate limit", response=_response(429), body=None),
            AIRateLimitError,
            "rate limit exceeded",
        ),
        (
            APITimeoutError(_request()),
            AITimeoutError,
            "timed out",
        ),
        (
            APIConnectionError(request=_request()),
            AIProviderError,
            "connection error",
        ),
    ],
)
async def test_openai_tts_maps_provider_errors(
    error: Exception,
    expected_exception: type[Exception],
    match: str,
) -> None:
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(speech=_Speech(error=error))),
    )

    with pytest.raises(expected_exception, match=match):
        await service.synthesize("Hallo", SpeechSynthesisOptions(voice="alloy"))


async def test_openai_tts_rejects_empty_audio_response() -> None:
    service = OpenAITextToSpeechService(
        provider_config=_provider_config(),
        voice_config=VoiceConfig(),
        client=_Client(_Audio(speech=_Speech(result=_BinaryResult(b"")))),
    )

    with pytest.raises(VoiceSynthesisError, match="empty audio content"):
        await service.synthesize(
            "Hallo",
            SpeechSynthesisOptions(voice="alloy"),
        )
