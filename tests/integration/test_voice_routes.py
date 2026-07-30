import base64
from typing import Any

from fastapi.testclient import TestClient

from app.domain.exceptions import AIProviderError
from app.domain.models import AIUsage
from app.domain.voice import (
    AudioInput,
    AudioOutput,
    Intent,
    Transcript,
    VoiceSession,
    VoiceTurn,
)
from app.infrastructure.config_schema import (
    AIProvidersConfig,
    ProviderConfig,
    VoiceConfig,
)
from app.infrastructure.yaml_config import get_ai_config


class _MockVoiceService:
    def __init__(self) -> None:
        self.audio: AudioInput | None = None
        self.session: VoiceSession | None = None

    async def process_turn(
        self,
        audio: AudioInput,
        *,
        session: VoiceSession | None = None,
        location=None,
    ) -> VoiceTurn:
        active_session = session or VoiceSession()
        self.audio = audio
        self.session = active_session
        return VoiceTurn(
            session=active_session,
            transcript=Transcript(text="Was passiert hier?", language="de"),
            intent=Intent(name="voice_assistant"),
            response_text="Hier gibt es Bauarbeiten.",
            audio=AudioOutput(content=b"audio-bytes", format="mp3"),
            provider="mock",
            model="mock-model",
            usage=AIUsage(total_tokens=12),
            latency_ms=25,
        )

    async def process_trigger(
        self,
        *,
        session: VoiceSession,
        location=None,
    ) -> VoiceTurn:
        self.session = session
        return VoiceTurn(
            session=session,
            transcript=Transcript(
                text="App-triggered voice event 'location_entered'.",
                language="de",
                confidence=1.0,
            ),
            intent=Intent(name=session.trigger.event_type or "app_event"),
            response_text="Achtung, Baustelle in der Nähe.",
            audio=AudioOutput(content=b"trigger-audio", format="mp3"),
            provider="mock",
            model="mock-model",
            usage=AIUsage(total_tokens=9),
            latency_ms=18,
        )


class _FailingVoiceService:
    async def process_turn(
        self,
        audio: AudioInput,
        *,
        session: VoiceSession | None = None,
        location=None,
    ) -> VoiceTurn:
        raise AIProviderError("upstream voice failure")

    async def process_trigger(
        self,
        *,
        session: VoiceSession,
        location=None,
    ) -> VoiceTurn:
        raise AIProviderError("upstream voice failure")


def _config(*, max_audio_bytes: int = 5_000_000) -> AIProvidersConfig:
    return AIProvidersConfig(
        active_provider="openai",
        providers={
            "openai": ProviderConfig(
                api_key_env="OPENAI_API_KEY",
                default_model="gpt-4o-mini",
            )
        },
        voice=VoiceConfig(max_audio_bytes=max_audio_bytes),
    )


def _client(
    monkeypatch,
    config: AIProvidersConfig | None = None,
    service=None,
) -> TestClient:
    import app.interfaces.voice_routes as voice_routes

    voice_service = service or _MockVoiceService()
    monkeypatch.setattr(
        voice_routes,
        "create_voice_service",
        lambda cfg: voice_service,
    )

    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_ai_config] = lambda: config or _config()
    return TestClient(app, raise_server_exceptions=False)


def _audio_chunk(content: bytes = b"abc") -> str:
    return base64.b64encode(content).decode("ascii")


def _start_session(ws, **extra: Any) -> dict[str, Any]:
    ws.send_json(
        {
            "type": "session.start",
            "session_id": "session-1",
            "language": "de",
            **extra,
        }
    )
    return ws.receive_json()


def test_voice_websocket_successful_turn(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ready = _start_session(ws)
        assert ready["type"] == "session.ready"
        assert ready["session_id"] == "session-1"
        assert ready["language"] == "de"
        assert ready["audio_format"] == "webm"

        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(),
            }
        )
        ws.send_json({"type": "audio.commit", "session_id": "session-1"})

        transcript = ws.receive_json()
        assistant = ws.receive_json()
        audio = ws.receive_json()
        complete = ws.receive_json()

        assert transcript["type"] == "transcript.final"
        assert transcript["text"] == "Was passiert hier?"
        assert assistant["type"] == "assistant.text"
        assert assistant["text"] == "Hier gibt es Bauarbeiten."
        assert audio["type"] == "audio.output"
        assert base64.b64decode(audio["audio_base64"]) == b"audio-bytes"
        assert complete["type"] == "turn.complete"
        assert complete["provider"] == "mock"
        assert complete["model"] == "mock-model"
        assert complete["total_tokens"] == 12


def test_voice_websocket_generates_default_session_id(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "session.start", "language": "de"})
        ready = ws.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["session_id"]
        assert ready["max_audio_bytes"] == 5_000_000


def test_voice_websocket_successful_trigger_commit(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(
            ws,
            trigger={
                "source": "app_event",
                "location_id": "de-berlin-hbf-upgrade",
                "event_type": "location_entered",
            },
        )
        ws.send_json({"type": "trigger.commit", "session_id": "session-1"})

        transcript = ws.receive_json()
        assistant = ws.receive_json()
        audio = ws.receive_json()
        complete = ws.receive_json()

        assert transcript["type"] == "transcript.final"
        assert "location_entered" in transcript["text"]
        assert assistant["text"] == "Achtung, Baustelle in der Nähe."
        assert base64.b64decode(audio["audio_base64"]) == b"trigger-audio"
        assert complete["total_tokens"] == 9


def test_voice_websocket_rejects_trigger_before_session(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "trigger.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "session.start" in error["detail"]


def test_voice_websocket_rejects_audio_before_session(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(),
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "session.start" in error["detail"]


def test_voice_websocket_rejects_non_object_json(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text("[]")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "JSON object" in error["detail"]


def test_voice_websocket_rejects_invalid_json(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text("not-json")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "valid JSON" in error["detail"]


def test_voice_websocket_rejects_duplicate_session_start(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json({"type": "session.start", "session_id": "session-2"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "already started" in error["detail"]


def test_voice_websocket_rejects_session_id_mismatch(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "other-session",
                "audio_base64": _audio_chunk(),
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "Session id mismatch" in error["detail"]


def test_voice_websocket_rejects_missing_audio_base64(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json({"type": "audio.chunk", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_audio_validation_error"
        assert "audio_base64" in error["detail"]


def test_voice_websocket_rejects_invalid_base64_audio(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": "not base64",
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_audio_validation_error"
        assert "valid base64" in error["detail"]


def test_voice_websocket_rejects_oversized_audio(monkeypatch) -> None:
    client = _client(monkeypatch, _config(max_audio_bytes=2))

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(b"abc"),
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_audio_validation_error"
        assert "exceeds max size" in error["detail"]


def test_voice_websocket_rejects_invalid_trigger_payload(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "session_id": "session-1",
                "trigger": "not-object",
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "trigger" in error["detail"]


def test_voice_websocket_rejects_invalid_trigger_source(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "session_id": "session-1",
                "trigger": {"source": "invalid"},
            }
        )
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"


def test_voice_websocket_rejects_unknown_location(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(
            ws,
            trigger={
                "source": "app_event",
                "location_id": "missing-location",
                "event_type": "location_entered",
            },
        )
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(),
            }
        )
        ws.send_json({"type": "audio.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "location_not_found"
        assert "missing-location" in error["detail"]


def test_voice_websocket_rejects_unknown_location_for_trigger_commit(
    monkeypatch,
) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(
            ws,
            trigger={
                "source": "app_event",
                "location_id": "missing-location",
                "event_type": "location_entered",
            },
        )
        ws.send_json({"type": "trigger.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "location_not_found"


def test_voice_websocket_rejects_commit_without_audio(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json({"type": "audio.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_audio_validation_error"
        assert "No audio chunks" in error["detail"]


def test_voice_websocket_rejects_unsupported_event_type(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json({"type": "unknown.event", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_session_error"
        assert "Unsupported voice event type" in error["detail"]


def test_voice_websocket_maps_provider_failure(monkeypatch) -> None:
    client = _client(monkeypatch, service=_FailingVoiceService())

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(ws)
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(),
            }
        )
        ws.send_json({"type": "audio.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "ai_provider_error"
        assert "unexpected error" in error["detail"]


def test_voice_websocket_maps_trigger_provider_failure(monkeypatch) -> None:
    client = _client(monkeypatch, service=_FailingVoiceService())

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(
            ws,
            trigger={
                "source": "app_event",
                "location_id": "de-berlin-hbf-upgrade",
                "event_type": "location_entered",
            },
        )
        ws.send_json({"type": "trigger.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "ai_provider_error"


def test_voice_websocket_clears_audio_buffer_after_trigger_commit(monkeypatch) -> None:
    client = _client(monkeypatch)

    with client.websocket_connect("/ws/voice") as ws:
        _start_session(
            ws,
            trigger={
                "source": "app_event",
                "location_id": "de-berlin-hbf-upgrade",
                "event_type": "location_entered",
            },
        )
        ws.send_json(
            {
                "type": "audio.chunk",
                "session_id": "session-1",
                "audio_base64": _audio_chunk(),
            }
        )
        ws.send_json({"type": "trigger.commit", "session_id": "session-1"})
        for _ in range(4):
            ws.receive_json()

        ws.send_json({"type": "audio.commit", "session_id": "session-1"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == "voice_audio_validation_error"
        assert "No audio chunks" in error["detail"]
