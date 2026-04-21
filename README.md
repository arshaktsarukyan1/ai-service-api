# AI Service Backend

FastAPI project prepared for Python 3.14.

## Requirements

- Python 3.14
- uv package manager

## Quick start

1. Install Python 3.14 locally.
2. Create and sync the environment:

```bash
uv venv --python 3.14
uv sync
```

3. Run the API:

```bash
uv run uvicorn app.main:app --reload
```

4. Open:

- API docs: <http://127.0.0.1:8000/docs>
- Health endpoint: <http://127.0.0.1:8000/health>

## Architecture

Task 1 uses a layered structure to keep provider integrations replaceable and
future task packages (FAQ, Voice, Geo-Fencing) decoupled from SDK details.

- `app/domain/`: stable contracts and provider protocol (`AITask`, `AIRequest`,
  `AIResponse`, `AIProvider`)
- `app/application/`: use-case orchestration (`AIService`)
- `app/infrastructure/`: provider adapters and configuration loaders (OpenAI,
  YAML)
- `app/interfaces/`: internal transport layer (API routes and schemas)

Functional flow:

`Controller -> AIService -> AIProvider (OpenAI) -> AIResponse`

Extensibility seams:

- Add new providers by implementing `AIProvider`
- Add new AI capabilities by extending `AITask`
