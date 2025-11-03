"""Tests for config module."""

from pathlib import Path

import pytest

from src.config import Config


def test_config_from_yaml_default() -> None:
    """Test loading default configuration."""
    config = Config.from_yaml()

    assert config.llm.provider == "lm_studio"
    assert config.llm.base_url == "http://127.0.0.1:1234/v1"
    assert config.retry.max_attempts == 5


def test_llm_config_lm_studio() -> None:
    """Test LM Studio configuration."""
    config = Config.from_yaml()

    assert config.llm.provider == "lm_studio"
    assert config.llm.models == ["openai/gpt-oss-20b"]
    assert config.llm.api_key == "lm-studio"
    assert config.llm.temperature == 0.3


def test_retry_config() -> None:
    """Test retry configuration."""
    config = Config.from_yaml()

    assert config.retry.max_attempts == 5
    assert config.retry.max_wait_time == 300
    assert config.retry.exponential_base == 2


def test_analysis_config() -> None:
    """Test analysis configuration."""
    config = Config.from_yaml()

    assert config.analysis_max_chunk_tokens == 18000
    assert config.analysis_batch_size == 10
    assert config.analysis_parallel_requests == 3


def test_output_config() -> None:
    """Test output configuration."""
    config = Config.from_yaml()

    assert config.output_default_dir == "output"
    assert "markdown" in config.output_formats
    assert "csv" in config.output_formats


def test_logging_config() -> None:
    """Test logging configuration."""
    config = Config.from_yaml()

    assert config.logging_level == "INFO"
    assert config.logging_format == "json"


def test_config_file_not_found() -> None:
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(Path("/nonexistent/config.yaml"))
