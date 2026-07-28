# AI Service API

FastAPI backend for AI task execution, FAQ generation, and voice interaction.

## Requirements

- Python 3.14
- `uv`
- OpenAI API key

## Setup

```bash
uv sync
cp .env.example .env
```

Set at least:

```env
OPENAI_API_KEY=sk-...
```

Optional config:

- `AI_CONFIG_PATH=config/ai_providers.yaml`
- `LOG_LEVEL=INFO`
- `ARANGO_PASSWORD=...`

## Run

```bash
uv run uvicorn app.main:app --reload
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`

## Main Endpoints

- `GET /health`
- `GET /ready`
- `GET /faq/locations`
- `GET /faq/{location_id}`
- `GET /internal/ai/provider`
- `POST /internal/ai/execute`
- `WS /ws/voice`

Example:

```bash
curl -X POST http://127.0.0.1:8000/internal/ai/execute \
  -H "Content-Type: application/json" \
  -d '{"task":"information_preparation","input_text":"Summarize ArangoDB."}'
```

## Configuration

AI provider settings live in `config/ai_providers.yaml`.

The YAML file stores environment variable names such as `OPENAI_API_KEY` and
`ARANGO_PASSWORD`, not secret values.

Voice settings are in the `voice` block:

- `stt_model`: speech-to-text model, default `gpt-4o-mini-transcribe`
- `tts_model`: text-to-speech model, default `gpt-4o-mini-tts`
- `tts_voice`: TTS voice, default `alloy`
- `language`: default voice language, default `de`
- `input_format`: browser audio format, default `webm`
- `output_format`: generated audio format, default `mp3`
- `max_audio_bytes`: maximum audio per turn

## Voice WebSocket

Connect to:

```text
ws://127.0.0.1:8000/ws/voice
```

Client events:

```json
{"type":"session.start","session_id":"demo","language":"de","audio_format":"webm"}
{"type":"audio.chunk","session_id":"demo","audio_base64":"..."}
{"type":"audio.commit","session_id":"demo"}
{"type":"trigger.commit","session_id":"demo"}
```

Server events:

- `session.ready`
- `transcript.final`
- `assistant.text`
- `audio.output`
- `turn.complete`
- `error`

The Nuxt console uses this endpoint in the Voice Tester section.
Use `trigger.commit` for app/proximity events that should produce speech without
microphone audio.

## Database and Migrations

ArangoDB settings are present in `config/ai_providers.yaml`, but this repo does
not currently include a migration framework or migration directory.

There are no migrations to run yet.

## Checks

```bash
uv run pytest
uv run ruff check .
```
