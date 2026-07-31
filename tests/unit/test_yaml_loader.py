"""Unit tests for the YAML configuration loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.infrastructure.yaml_config import load_ai_config

MINIMAL_VALID_YAML = """\
active_provider: openai
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o-mini
"""

FULL_YAML = """\
active_provider: openai
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o-mini
    task_models:
      information_preparation: gpt-4o-mini
      voice_assistant: gpt-4o
    temperature: 0.5
    max_tokens: 1024
    timeout_seconds: 20
    retry:
      attempts: 2
database:
  arangodb:
    host: localhost
    port: 8529
    username: root
    password_env: ARANGO_PASSWORD
    database: ai_service
voice:
  stt_model: gpt-4o-mini-transcribe
  tts_model: gpt-4o-mini-tts
  tts_voice: alloy
  language: de
  input_format: webm
  output_format: mp3
  max_audio_bytes: 5000000
  session_timeout_seconds: 60
geofencing:
  default_radius_meters: 250
  min_radius_meters: 25
  max_radius_meters: 1000
  exit_hysteresis_meters: 50
  trigger_cooldown_seconds: 120
  max_acceptable_accuracy_meters: 75
"""


@pytest.fixture(autouse=True)
def clear_cache():
    load_ai_config.cache_clear()
    yield
    load_ai_config.cache_clear()


def test_load_minimal_valid_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(MINIMAL_VALID_YAML)
    cfg = load_ai_config(cfg_file)
    assert cfg.active_provider == "openai"
    assert "openai" in cfg.providers
    assert cfg.providers["openai"].default_model == "gpt-4o-mini"


def test_load_full_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(FULL_YAML)
    cfg = load_ai_config(cfg_file)
    assert cfg.providers["openai"].temperature == 0.5
    assert cfg.providers["openai"].max_tokens == 1024
    assert cfg.providers["openai"].retry.attempts == 2
    assert cfg.database is not None
    assert cfg.database.arangodb is not None
    assert cfg.database.arangodb.port == 8529
    assert cfg.voice.language == "de"
    assert cfg.voice.stt_model == "gpt-4o-mini-transcribe"
    assert cfg.geofencing.default_radius_meters == 250
    assert cfg.geofencing.max_acceptable_accuracy_meters == 75


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ai_config(tmp_path / "nonexistent.yaml")


def test_unknown_active_provider_raises_validation_error(tmp_path: Path) -> None:
    bad_yaml = """\
active_provider: unknown
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o-mini
"""
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(bad_yaml)
    with pytest.raises(ValidationError, match="not found in providers"):
        load_ai_config(cfg_file)


def test_missing_required_provider_field_raises_validation_error(
    tmp_path: Path,
) -> None:
    # api_key_env and default_model are required
    bad_yaml = """\
active_provider: openai
providers:
  openai:
    api_key_env: OPENAI_API_KEY
"""
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_ai_config(cfg_file)


def test_retry_config_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(MINIMAL_VALID_YAML)
    cfg = load_ai_config(cfg_file)
    retry = cfg.providers["openai"].retry
    assert retry.attempts == 3


def test_database_section_is_optional(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(MINIMAL_VALID_YAML)
    cfg = load_ai_config(cfg_file)
    assert cfg.database is None


def test_voice_section_defaults_when_omitted(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(MINIMAL_VALID_YAML)
    cfg = load_ai_config(cfg_file)
    assert cfg.voice.tts_model == "gpt-4o-mini-tts"


def test_geofencing_section_defaults_when_omitted(tmp_path: Path) -> None:
    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(MINIMAL_VALID_YAML)
    cfg = load_ai_config(cfg_file)
    assert cfg.geofencing.default_radius_meters == 100


def test_invalid_voice_section_raises_validation_error(tmp_path: Path) -> None:
    bad_yaml = """\
active_provider: openai
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o-mini
voice:
  input_format: raw
"""
    cfg_file = tmp_path / "bad_voice.yaml"
    cfg_file.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_ai_config(cfg_file)


def test_task_models_override(tmp_path: Path) -> None:
    from app.domain.tasks import AITask

    cfg_file = tmp_path / "ai.yaml"
    cfg_file.write_text(FULL_YAML)
    cfg = load_ai_config(cfg_file)
    task_models = cfg.providers["openai"].task_models
    assert task_models[AITask.voice_assistant] == "gpt-4o"
    assert task_models[AITask.information_preparation] == "gpt-4o-mini"
