"""Token counting utilities using tiktoken."""

from functools import lru_cache
from typing import Protocol

import tiktoken


class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        ...


class TiktokenCounter:
    """Token counter using OpenAI's tiktoken library."""

    def __init__(self, model: str = "gpt-4", cache_size: int = 2048) -> None:
        """Initialize token counter.

        Args:
            model: Model name for tiktoken encoding
            cache_size: Size of LRU cache for token counts
        """
        self.model = model
        self.cache_size = cache_size
        self.encoding = tiktoken.encoding_for_model(model)

        # Create cached counting method
        self._count_cached = lru_cache(maxsize=cache_size)(self._count_impl)

    def count(self, text: str) -> int:
        """Count tokens in text with caching.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return self._count_cached(text)

    def _count_impl(self, text: str) -> int:
        """Implementation of token counting.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts.

        Args:
            texts: List of texts to count tokens for

        Returns:
            List of token counts
        """
        return [self.count(text) for text in texts]

    def get_cache_info(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, size, and max size
        """
        info = self._count_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "max_size": info.maxsize if info.maxsize is not None else 0,
        }

    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._count_cached.cache_clear()


class EstimatedTokenCounter:
    """Fast estimated token counter for C/C++ code.

    Uses character-based estimation:
    - Average C/C++ code: ~0.4 tokens per character
    - Accuracy: ~95%
    """

    TOKENS_PER_CHAR = 0.4

    def count(self, text: str) -> int:
        """Estimate token count.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        return int(len(text) * self.TOKENS_PER_CHAR)

    def count_batch(self, texts: list[str]) -> list[int]:
        """Estimate tokens for multiple texts.

        Args:
            texts: List of texts to estimate tokens for

        Returns:
            List of estimated token counts
        """
        return [self.count(text) for text in texts]


def create_token_counter(
    mode: str = "accurate", model: str = "gpt-4", cache_size: int = 2048
) -> TokenCounter:
    """Factory function to create a token counter.

    Args:
        mode: 'accurate' for tiktoken, 'fast' for estimation
        model: Model name for tiktoken encoding
        cache_size: Size of LRU cache (for accurate mode)

    Returns:
        TokenCounter instance

    Raises:
        ValueError: If mode is not 'accurate' or 'fast'
    """
    if mode == "accurate":
        return TiktokenCounter(model=model, cache_size=cache_size)
    elif mode == "fast":
        return EstimatedTokenCounter()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'accurate' or 'fast'")
