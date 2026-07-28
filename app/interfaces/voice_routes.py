import base64
import binascii
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.application.voice_service import VoiceService
from app.domain.exceptions import (
    AIAuthError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    LocationNotFoundError,
    VoiceAudioValidationError,
    VoiceServiceError,
    VoiceSessionError,
    VoiceSynthesisError,
    VoiceTranscriptionError,
)
from app.domain.voice import AudioInput, VoiceSession, VoiceTrigger
from app.infrastructure.config_schema import AIProvidersConfig
from app.infrastructure.dev_location_repository import get_construction_site
from app.infrastructure.openai_audio_provider import (
    OpenAISpeechToTextService,
    OpenAITextToSpeechService,
)
from app.infrastructure.openai_provider import get_active_provider
from app.infrastructure.yaml_config import AiConfigDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

_INPUT_MIME_TYPES = {
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}


def create_voice_service(config: AIProvidersConfig) -> VoiceService:
    provider_config = config.providers[config.active_provider]
    provider = get_active_provider(config)
    return VoiceService(
        speech_to_text=OpenAISpeechToTextService(
            provider_config=provider_config,
            voice_config=config.voice,
        ),
        text_to_speech=OpenAITextToSpeechService(
            provider_config=provider_config,
            voice_config=config.voice,
        ),
        ai_provider=provider,
        provider_config=provider_config,
        voice_config=config.voice,
    )


def _mime_type_for_audio_format(audio_format: str) -> str:
    return _INPUT_MIME_TYPES.get(audio_format, f"audio/{audio_format}")


async def _send_event(
    websocket: WebSocket,
    event_type: str,
    *,
    session_id: str,
    request_id: str,
    **payload: Any,
) -> None:
    await websocket.send_json(
        {
            "type": event_type,
            "session_id": session_id,
            "request_id": request_id,
            **payload,
        }
    )


async def _send_error(
    websocket: WebSocket,
    *,
    session_id: str,
    request_id: str,
    error: str,
    detail: str,
) -> None:
    await _send_event(
        websocket,
        "error",
        session_id=session_id,
        request_id=request_id,
        error=error,
        detail=detail,
    )


def _error_from_exception(exc: Exception) -> tuple[str, str, int]:
    if isinstance(exc, LocationNotFoundError):
        return "location_not_found", str(exc), status.WS_1008_POLICY_VIOLATION
    if isinstance(exc, VoiceAudioValidationError):
        return "voice_audio_validation_error", str(exc), status.WS_1008_POLICY_VIOLATION
    if isinstance(exc, VoiceSessionError):
        return "voice_session_error", str(exc), status.WS_1008_POLICY_VIOLATION
    if isinstance(exc, VoiceTranscriptionError):
        return (
            "voice_transcription_error",
            "Speech-to-text processing failed. Please retry.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, VoiceSynthesisError):
        return (
            "voice_synthesis_error",
            "Text-to-speech processing failed. Please retry.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, AIAuthError):
        return (
            "ai_auth_error",
            "AI service is not properly configured. Contact the administrator.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, AIRateLimitError):
        return (
            "ai_rate_limit",
            "AI provider rate limit exceeded. Please retry after a moment.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, AITimeoutError):
        return (
            "ai_timeout",
            "AI provider did not respond in time. Please retry.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, AIProviderError):
        return (
            "ai_provider_error",
            "AI provider returned an unexpected error. Please retry.",
            status.WS_1011_INTERNAL_ERROR,
        )
    if isinstance(exc, VoiceServiceError):
        return (
            "voice_service_error",
            "An internal voice service error occurred.",
            status.WS_1011_INTERNAL_ERROR,
        )
    return (
        "internal_error",
        "An unexpected error occurred.",
        status.WS_1011_INTERNAL_ERROR,
    )


def _require_session(
    session: VoiceSession | None,
    payload: dict[str, Any],
) -> VoiceSession:
    if session is None:
        raise VoiceSessionError("Send 'session.start' before audio events.")
    payload_session_id = payload.get("session_id")
    if payload_session_id and payload_session_id != session.id:
        raise VoiceSessionError(
            f"Session id mismatch. Active session is '{session.id}'."
        )
    return session


def _parse_json_message(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VoiceSessionError("WebSocket message must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise VoiceSessionError("WebSocket message must be a JSON object.")
    return payload


def _decode_audio_chunk(payload: dict[str, Any]) -> bytes:
    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        raise VoiceAudioValidationError("'audio_base64' must be a non-empty string.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VoiceAudioValidationError("Audio chunk is not valid base64.") from exc


def _start_session(
    payload: dict[str, Any],
    config: AIProvidersConfig,
) -> tuple[VoiceSession, str, str]:
    trigger_raw = payload.get("trigger") or {}
    if not isinstance(trigger_raw, dict):
        raise VoiceSessionError("'trigger' must be a JSON object when provided.")
    session = VoiceSession(
        id=str(payload.get("session_id") or uuid4().hex),
        language=str(payload.get("language") or config.voice.language),
        trigger=VoiceTrigger.model_validate(trigger_raw),
    )
    audio_format = str(payload.get("audio_format") or config.voice.input_format)
    mime_type = str(
        payload.get("mime_type") or _mime_type_for_audio_format(audio_format)
    )
    return session, audio_format, mime_type


async def _handle_commit(
    websocket: WebSocket,
    *,
    config: AIProvidersConfig,
    session: VoiceSession,
    request_id: str,
    audio_chunks: list[bytes],
    audio_format: str,
    mime_type: str,
) -> None:
    if not audio_chunks:
        raise VoiceAudioValidationError("No audio chunks were received.")

    location = None
    if session.trigger.location_id:
        location = get_construction_site(session.trigger.location_id)
        if location is None:
            raise LocationNotFoundError(
                f"Unknown location_id={session.trigger.location_id!r}."
            )

    service = create_voice_service(config)
    turn = await service.process_turn(
        AudioInput(
            content=b"".join(audio_chunks),
            format=audio_format,
            mime_type=mime_type,
        ),
        session=session,
        location=location,
    )

    await _send_turn_events(websocket, request_id=request_id, turn=turn)


async def _send_turn_events(
    websocket: WebSocket,
    *,
    request_id: str,
    turn,
) -> None:
    await _send_event(
        websocket,
        "transcript.final",
        session_id=turn.session.id,
        request_id=request_id,
        text=turn.transcript.text,
        language=turn.transcript.language,
        confidence=turn.transcript.confidence,
    )
    await _send_event(
        websocket,
        "assistant.text",
        session_id=turn.session.id,
        request_id=request_id,
        text=turn.response_text,
    )
    await _send_event(
        websocket,
        "audio.output",
        session_id=turn.session.id,
        request_id=request_id,
        audio_base64=base64.b64encode(turn.audio.content).decode("ascii"),
        format=turn.audio.format,
        mime_type=turn.audio.mime_type,
    )
    await _send_event(
        websocket,
        "turn.complete",
        session_id=turn.session.id,
        request_id=request_id,
        latency_ms=turn.latency_ms,
        provider=turn.provider,
        model=turn.model,
        total_tokens=turn.usage.total_tokens if turn.usage else None,
    )


async def _handle_trigger_commit(
    websocket: WebSocket,
    *,
    config: AIProvidersConfig,
    session: VoiceSession,
    request_id: str,
) -> None:
    location = None
    if session.trigger.location_id:
        location = get_construction_site(session.trigger.location_id)
        if location is None:
            raise LocationNotFoundError(
                f"Unknown location_id={session.trigger.location_id!r}."
            )

    service = create_voice_service(config)
    turn = await service.process_trigger(session=session, location=location)
    await _send_turn_events(websocket, request_id=request_id, turn=turn)


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket, config: AiConfigDep) -> None:
    await websocket.accept()

    request_id = uuid4().hex
    session: VoiceSession | None = None
    session_id = "unknown"
    audio_format = config.voice.input_format
    mime_type = _mime_type_for_audio_format(audio_format)
    audio_chunks: list[bytes] = []
    audio_size = 0

    try:
        while True:
            payload = _parse_json_message(await websocket.receive_text())
            event_type = payload.get("type")

            if event_type == "session.start":
                if session is not None:
                    raise VoiceSessionError("Voice session has already started.")
                session, audio_format, mime_type = _start_session(payload, config)
                session_id = session.id
                await _send_event(
                    websocket,
                    "session.ready",
                    session_id=session.id,
                    request_id=request_id,
                    language=session.language,
                    audio_format=audio_format,
                    max_audio_bytes=config.voice.max_audio_bytes,
                )
                continue

            if event_type == "audio.chunk":
                active_session = _require_session(session, payload)
                chunk = _decode_audio_chunk(payload)
                audio_size += len(chunk)
                if audio_size > config.voice.max_audio_bytes:
                    raise VoiceAudioValidationError(
                        "Audio input exceeds max size of "
                        f"{config.voice.max_audio_bytes} bytes."
                    )
                audio_chunks.append(chunk)
                session_id = active_session.id
                continue

            if event_type == "audio.commit":
                active_session = _require_session(session, payload)
                await _handle_commit(
                    websocket,
                    config=config,
                    session=active_session,
                    request_id=request_id,
                    audio_chunks=audio_chunks,
                    audio_format=audio_format,
                    mime_type=mime_type,
                )
                audio_chunks.clear()
                audio_size = 0
                session_id = active_session.id
                continue

            if event_type == "trigger.commit":
                active_session = _require_session(session, payload)
                await _handle_trigger_commit(
                    websocket,
                    config=config,
                    session=active_session,
                    request_id=request_id,
                )
                audio_chunks.clear()
                audio_size = 0
                session_id = active_session.id
                continue

            raise VoiceSessionError(f"Unsupported voice event type: {event_type!r}.")

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected session_id=%s", session_id)
    except (ValidationError, VoiceSessionError, VoiceAudioValidationError) as exc:
        error, detail, close_code = _error_from_exception(
            exc if isinstance(exc, Exception) else VoiceSessionError(str(exc))
        )
        await _send_error(
            websocket,
            session_id=session_id,
            request_id=request_id,
            error=error,
            detail=detail,
        )
        await websocket.close(code=close_code)
    except Exception as exc:
        logger.exception(
            "Voice WebSocket failed session_id=%s type=%s",
            session_id,
            type(exc).__name__,
        )
        error, detail, close_code = _error_from_exception(exc)
        await _send_error(
            websocket,
            session_id=session_id,
            request_id=request_id,
            error=error,
            detail=detail,
        )
        await websocket.close(code=close_code)
