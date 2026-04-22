import os
import time

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
)
from app.domain.models import AIRequest, AIResponse, AIUsage
from app.domain.tasks import AITask
from app.infrastructure.config_schema import AIProvidersConfig, ProviderConfig


class OpenAIProvider:
    name = "openai"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise AIAuthError(
                f"Environment variable '{config.api_key_env}' is not set or empty. "
                "Set it before starting the service."
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=float(config.timeout_seconds),
            max_retries=max(0, config.retry.attempts - 1),
        )

    def _resolve_model(self, task: AITask) -> str:
        return self._config.task_models.get(task, self._config.default_model)

    async def ping(self) -> None:
        try:
            await self._client.models.list()
        except AuthenticationError as exc:
            raise AIAuthError(f"OpenAI authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise AITimeoutError(
                f"OpenAI call timed out after {self._config.timeout_seconds}s: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise AIProviderError(f"OpenAI connection error: {exc}") from exc
        except APIError as exc:
            raise AIProviderError(
                f"OpenAI API error ({exc.status_code}): {exc}"
            ) from exc

    async def execute(self, request: AIRequest) -> AIResponse:
        model = self._resolve_model(request.task)
        start = time.monotonic()
        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.input_text}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except AuthenticationError as exc:
            raise AIAuthError(f"OpenAI authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise AIRateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise AITimeoutError(
                f"OpenAI call timed out after {self._config.timeout_seconds}s: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise AIProviderError(f"OpenAI connection error: {exc}") from exc
        except BadRequestError as exc:
            raise AIProviderError(f"OpenAI bad request: {exc}") from exc
        except APIError as exc:
            raise AIProviderError(
                f"OpenAI API error ({exc.status_code}): {exc}"
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        choice = completion.choices[0]
        usage = (
            AIUsage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            if completion.usage
            else None
        )

        return AIResponse(
            task=request.task,
            content=choice.message.content or "",
            provider=self.name,
            model=completion.model,
            usage=usage,
            latency_ms=latency_ms,
        )


_provider_cache: dict[str, OpenAIProvider] = {}


def get_active_provider(config: AIProvidersConfig) -> OpenAIProvider:
    name = config.active_provider
    if name not in _provider_cache:
        provider_config = config.providers[name]
        if name == "openai":
            _provider_cache[name] = OpenAIProvider(config=provider_config)
        else:
            raise AIProviderError(f"Unknown provider: '{name}'")
    return _provider_cache[name]
