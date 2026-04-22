# AI Service — Task 1

Autonomous AI service backend built with FastAPI and Python 3.14.  
Provides a configurable, provider-agnostic AI execution layer consumed by the
FAQ, Voice, and Geo-Fencing task packages.

---

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- ArangoDB (optional — required for Tasks 2 and 3)

---

## Quick start

```bash
# 1. Create and sync the virtual environment
uv venv --python 3.14
uv sync

# 2. Copy the example env file and fill in your credentials
cp .env.example .env
# Edit .env: set OPENAI_API_KEY at minimum

# 3. Start the server
uv run uvicorn app.main:app --reload
```

Open:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Readiness: <http://127.0.0.1:8000/ready>

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key. Never commit this value. |
| `ARANGO_PASSWORD` | No | — | ArangoDB root password (used in Tasks 2–3). |
| `AI_CONFIG_PATH` | No | `config/ai_providers.yaml` | Path to the YAML provider config file. |
| `LOG_LEVEL` | No | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

All variables can be placed in a `.env` file at the project root.  
The `.env` file must not be committed to version control.

---

## YAML configuration format

The provider config lives in `config/ai_providers.yaml`.  
A different path can be set via `AI_CONFIG_PATH`.

```yaml
# Which provider is active. Must match a key in `providers`.
active_provider: openai

providers:
  openai:
    # Name of the environment variable holding the API key (value is never stored here).
    api_key_env: OPENAI_API_KEY

    # Default model used when a task has no specific override.
    default_model: gpt-4o-mini

    # Optional per-task model overrides.
    task_models:
      information_preparation: gpt-4o-mini
      information_structuring: gpt-4o-mini
      voice_assistant: gpt-4o   # higher-capability model for voice

    temperature: 0.7        # 0.0–2.0
    max_tokens: 2048        # 1–128000
    timeout_seconds: 30     # 1–300
    retry:
      attempts: 3           # 1–10 (first attempt + N-1 retries)

# Optional database block — required for Tasks 2 and 3.
database:
  arangodb:
    host: localhost
    port: 8529
    username: root
    password_env: ARANGO_PASSWORD   # env var name, not the value
    database: ai_service
```

**Security note**: `api_key_env` and `password_env` store the *name* of an
environment variable, not the secret itself. Secrets are read at runtime from
the process environment.

---

## Architecture

```
app/
  domain/          Pure contracts: AITask, AIRequest, AIResponse, AIProvider protocol
  application/     Provider-agnostic orchestration: execute_task()
  infrastructure/  OpenAI adapter, YAML config loader, config schema
  interfaces/      HTTP transport: routes, request/response schemas
  core/            Settings, logging, middleware, error handlers
  api/             Health and readiness endpoints
```

Functional flow:

```
POST /internal/ai/execute
  → interfaces/internal_routes.py
  → application/ai_service.execute_task()
  → infrastructure/openai_provider.OpenAIProvider.execute()
  → OpenAI API
  → AIResponse (normalized)
```

---

## How to add a new AI provider

1. **Create an adapter** in `app/infrastructure/`:

```python
# app/infrastructure/anthropic_provider.py
import os
from app.domain.exceptions import AIAuthError, AIProviderError
from app.domain.models import AIRequest, AIResponse
from app.domain.tasks import AITask
from app.infrastructure.config_schema import ProviderConfig

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise AIAuthError(f"'{config.api_key_env}' env var is not set.")
        # initialise your SDK client here

    async def ping(self) -> None:
        # lightweight connectivity check
        ...

    async def execute(self, request: AIRequest) -> AIResponse:
        # call the provider, map to AIResponse
        ...
```

2. **Register it** in `app/infrastructure/openai_provider.py` inside
   `get_active_provider()`:

```python
if name == "anthropic":
    from app.infrastructure.anthropic_provider import AnthropicProvider
    _provider_cache[name] = AnthropicProvider(config=provider_config)
```

3. **Add a config block** in `config/ai_providers.yaml`:

```yaml
active_provider: anthropic
providers:
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-3-5-sonnet-20241022
```

No changes to domain contracts or the application service are needed.

---

## API reference

### `GET /health`

Returns `200 OK` when the process is alive.

```json
{"status": "ok"}
```

### `GET /ready`

Returns `200 OK` when config is loaded and the active provider is wired.  
Returns `503` if provider configuration is missing.

```bash
curl http://localhost:8000/ready
# {"status": "ready", "provider": "openai", "model": "gpt-4o-mini"}
```

Add `?ping=true` to make a lightweight live connectivity check (calls
`models.list()`). Use sparingly — this makes a real API call.

```bash
curl "http://localhost:8000/ready?ping=true"
# {"status": "ready", "provider": "openai", "model": "gpt-4o-mini", "ping": "ok"}
```

### `POST /internal/ai/execute`

Execute an AI task. Intended for server-to-server use.

**Request body**

```json
{
  "task": "information_preparation",
  "input_text": "Summarize the key attractions in Vienna.",
  "metadata": {"language": "en", "location_id": "42"}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `task` | string | Yes | One of: `information_preparation`, `information_structuring`, `voice_assistant` |
| `input_text` | string | Yes | Non-empty prompt text |
| `metadata` | object | No | Free-form key-value pairs forwarded to the provider |

**Response body (200)**

```json
{
  "task": "information_preparation",
  "content": "Vienna is known for its imperial palaces...",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 142,
    "total_tokens": 160
  },
  "latency_ms": 1243
}
```

**Error body (all non-2xx responses)**

```json
{
  "error": "ai_timeout",
  "detail": "AI provider did not respond in time. Please retry.",
  "request_id": "f3b2a6b1-..."
}
```

**curl example**

```bash
curl -X POST http://localhost:8000/internal/ai/execute \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id" \
  -d '{"task": "information_preparation", "input_text": "What is ArangoDB?"}'
```

---

## Error codes

| HTTP status | `error` field | Cause |
|---|---|---|
| 422 | `validation_error` | Missing or invalid request field |
| 422 | `unsupported_task` | Task value not in the supported set |
| 429 | `ai_rate_limit` | OpenAI rate limit or quota exceeded |
| 502 | `ai_provider_error` | Unexpected upstream API error |
| 503 | `ai_auth_error` | API key missing, invalid, or revoked |
| 504 | `ai_timeout` | Provider call exceeded `timeout_seconds` |
| 500 | `internal_error` | Unexpected server-side error |

All error responses include a `request_id` field for log correlation.  
Pass `X-Request-ID` in your request to carry your own trace ID through.

---

## Running tests

```bash
uv run pytest
```

Test layout:

```
tests/
  unit/         YAML loader, AIService routing, config schema validation
  integration/  Full HTTP round-trips with mocked provider
  contract/     AIProvider protocol compliance, AIResponse shape
```

---

## Known limits and failure modes

| Limit | Value | Notes |
|---|---|---|
| `max_tokens` | 2048 (default) | Configurable up to 128 000 in YAML |
| `timeout_seconds` | 30 (default) | Configurable up to 300 in YAML |
| Retry attempts | 3 (default) | SDK handles exponential backoff |
| Concurrent requests | Unlimited | AsyncOpenAI uses an httpx connection pool |

**Known failure modes**

- **Missing `OPENAI_API_KEY`**: service raises `AIAuthError` on first request and
  returns `503`. Set the env var and restart.
- **Invalid YAML config**: process exits at startup with a `CRITICAL` log
  describing the validation error.
- **Config file not found**: process exits at startup with a `CRITICAL` log
  showing the expected path. Set `AI_CONFIG_PATH` to the correct location.
- **Rate limit (429 from OpenAI)**: SDK retries automatically up to
  `retry.attempts - 1` times. If retries are exhausted, the
  service returns `429` to the caller.
- **Timeouts**: if OpenAI does not respond within `timeout_seconds`, the SDK
  raises a timeout which becomes a `504` response. Tune `timeout_seconds` for
  large `max_tokens` values.
- **Token overruns**: if `input_text` is very large and exceeds the model's
  context window minus `max_tokens`, OpenAI returns a `BadRequestError` which
  maps to `502`. Reduce input size or increase the model's context limit.
