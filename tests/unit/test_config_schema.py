"""Unit tests for Pydantic config schema validation."""

import pytest
from pydantic import ValidationError

from app.infrastructure.config_schema import (
    AIProvidersConfig,
    ArangoDBConfig,
    DatabaseConfig,
    ProviderConfig,
    RetryConfig,
)


def _valid_provider() -> dict:
    return {"api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"}


# --- RetryConfig ---


def test_retry_config_defaults() -> None:
    cfg = RetryConfig()
    assert cfg.attempts == 3
    assert cfg.delay_seconds == 1.0


def test_retry_config_attempts_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(attempts=0)


def test_retry_config_attempts_above_maximum_raises() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(attempts=11)


# --- ProviderConfig ---


def test_provider_config_valid() -> None:
    cfg = ProviderConfig(**_valid_provider())
    assert cfg.api_key_env == "OPENAI_API_KEY"
    assert cfg.default_model == "gpt-4o-mini"


def test_provider_config_temperature_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(**_valid_provider(), temperature=3.0)


def test_provider_config_max_tokens_zero_raises() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(**_valid_provider(), max_tokens=0)


def test_provider_config_defaults() -> None:
    cfg = ProviderConfig(**_valid_provider())
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 2048
    assert cfg.timeout_seconds == 30


# --- ArangoDBConfig ---


def test_arango_config_defaults() -> None:
    cfg = ArangoDBConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 8529
    assert cfg.username == "root"


def test_arango_config_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        ArangoDBConfig(port=0)


# --- AIProvidersConfig ---


def test_providers_config_valid() -> None:
    cfg = AIProvidersConfig(
        active_provider="openai",
        providers={"openai": ProviderConfig(**_valid_provider())},
    )
    assert cfg.active_provider == "openai"


def test_active_provider_not_in_providers_raises() -> None:
    with pytest.raises(ValidationError, match="not found in providers"):
        AIProvidersConfig(
            active_provider="missing",
            providers={"openai": ProviderConfig(**_valid_provider())},
        )


def test_providers_config_database_optional() -> None:
    cfg = AIProvidersConfig(
        active_provider="openai",
        providers={"openai": ProviderConfig(**_valid_provider())},
    )
    assert cfg.database is None


def test_providers_config_with_database() -> None:
    cfg = AIProvidersConfig(
        active_provider="openai",
        providers={"openai": ProviderConfig(**_valid_provider())},
        database=DatabaseConfig(arangodb=ArangoDBConfig()),
    )
    assert cfg.database.arangodb.port == 8529
