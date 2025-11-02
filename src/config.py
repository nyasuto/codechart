"""Configuration management for CodeChart."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str  # "lm_studio" or "openai"
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


@dataclass
class RetryConfig:
    """Retry strategy configuration."""

    max_attempts: int
    max_wait_time: int
    exponential_base: int


@dataclass
class Config:
    """Main configuration class."""

    llm: LLMConfig
    retry: RetryConfig
    analysis_max_chunk_tokens: int
    analysis_batch_size: int
    analysis_parallel_requests: int
    output_default_dir: str
    output_formats: list[str]
    logging_level: str
    logging_format: str
    logging_file: str

    @classmethod
    def from_yaml(cls, config_path: Path | str | None = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses default config.

        Returns:
            Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Config":
        """Parse configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config object
        """
        # Determine which provider to use
        provider = data["api"]["provider"]

        if provider == "lm_studio":
            lm_config = data["api"]["lm_studio"]
            llm = LLMConfig(
                provider="lm_studio",
                base_url=lm_config["base_url"],
                model=lm_config["model"],
                api_key=lm_config["api_key"],
                temperature=lm_config["temperature"],
                max_tokens=lm_config["max_tokens"],
                timeout=lm_config["timeout"],
                max_retries=lm_config["max_retries"],
            )
        elif provider == "openai":
            openai_config = data["api"]["openai"]
            api_key_env = openai_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)

            if not api_key:
                raise ValueError(f"Environment variable {api_key_env} not set for OpenAI API")

            llm = LLMConfig(
                provider="openai",
                base_url="https://api.openai.com/v1",
                model=openai_config["model"],
                api_key=api_key,
                temperature=openai_config["temperature"],
                max_tokens=openai_config["max_tokens"],
                timeout=openai_config["timeout"],
                max_retries=openai_config["max_retries"],
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Retry configuration
        retry_data = data.get("retry", {})
        retry = RetryConfig(
            max_attempts=retry_data.get("max_attempts", 5),
            max_wait_time=retry_data.get("max_wait_time", 300),
            exponential_base=retry_data.get("exponential_base", 2),
        )

        # Analysis settings
        analysis = data.get("analysis", {})

        # Output settings
        output = data.get("output", {})

        # Logging settings
        logging = data.get("logging", {})

        return cls(
            llm=llm,
            retry=retry,
            analysis_max_chunk_tokens=analysis.get("max_chunk_tokens", 18000),
            analysis_batch_size=analysis.get("batch_size", 10),
            analysis_parallel_requests=analysis.get("parallel_requests", 3),
            output_default_dir=output.get("default_dir", "output"),
            output_formats=output.get("formats", ["markdown", "csv"]),
            logging_level=logging.get("level", "INFO"),
            logging_format=logging.get("format", "json"),
            logging_file=logging.get("file", "codechart.log"),
        )


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        Config object
    """
    return Config.from_yaml(config_path)
