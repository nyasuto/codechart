"""Retry strategy for LLM API calls."""

import time
from collections.abc import Callable
from typing import Any, TypeVar

import backoff
from openai import APIConnectionError, APITimeoutError, RateLimitError

from src.config import RetryConfig

T = TypeVar("T")


class RetryStrategy:
    """Retry strategy with exponential backoff."""

    def __init__(self, config: RetryConfig):
        """Initialize retry strategy.

        Args:
            config: Retry configuration
        """
        self.config = config
        self._retry_count = 0

    def _on_backoff(self, details: dict[str, Any]) -> None:
        """Callback for backoff events.

        Args:
            details: Backoff details
        """
        self._retry_count += 1
        wait_time = details.get("wait", 0)
        tries = details.get("tries", 0)
        print(
            f"Retry {tries}/{self.config.max_attempts} after {wait_time:.2f}s "
            f"(attempt {self._retry_count})"
        )

    def _on_giveup(self, details: dict[str, Any]) -> None:
        """Callback for giving up.

        Args:
            details: Backoff details
        """
        tries = details.get("tries", 0)
        print(f"Giving up after {tries} attempts")

    def get_decorator(self) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Get retry decorator for API calls.

        Returns:
            Decorator function
        """
        return backoff.on_exception(
            backoff.expo,
            (RateLimitError, APITimeoutError, APIConnectionError),
            max_tries=self.config.max_attempts,
            max_time=self.config.max_wait_time,
            base=self.config.exponential_base,
            on_backoff=self._on_backoff,
            on_giveup=self._on_giveup,
        )

    def reset_count(self) -> None:
        """Reset retry counter."""
        self._retry_count = 0

    @property
    def retry_count(self) -> int:
        """Get current retry count.

        Returns:
            Number of retries performed
        """
        return self._retry_count


def create_retry_decorator(config: RetryConfig) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Create a retry decorator with the given configuration.

    Args:
        config: Retry configuration

    Returns:
        Decorator function
    """
    strategy = RetryStrategy(config)
    return strategy.get_decorator()


def simple_retry(
    func: Callable[..., T], max_attempts: int = 3, delay: float = 1.0
) -> Callable[..., T]:
    """Simple retry wrapper without backoff library.

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between retries in seconds

    Returns:
        Wrapped function
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    wait_time = delay * (2**attempt)
                    print(f"Retry {attempt + 1}/{max_attempts} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"Giving up after {max_attempts} attempts")

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry logic")

    return wrapper
