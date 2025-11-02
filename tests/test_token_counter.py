"""Tests for token_counter module."""

import pytest

from src.token_counter import (
    EstimatedTokenCounter,
    TiktokenCounter,
    create_token_counter,
)


def test_tiktoken_counter_basic() -> None:
    """Test basic token counting with tiktoken."""
    counter = TiktokenCounter()

    # Simple text
    count = counter.count("Hello, world!")
    assert count > 0
    assert isinstance(count, int)


def test_tiktoken_counter_caching() -> None:
    """Test that caching works correctly."""
    counter = TiktokenCounter(cache_size=10)

    text = "int add(int a, int b) { return a + b; }"

    # First call - cache miss
    count1 = counter.count(text)
    info1 = counter.get_cache_info()

    # Second call - cache hit
    count2 = counter.count(text)
    info2 = counter.get_cache_info()

    assert count1 == count2
    assert info2["hits"] > info1["hits"]


def test_tiktoken_counter_batch() -> None:
    """Test batch token counting."""
    counter = TiktokenCounter()

    texts = [
        "int main() { return 0; }",
        "void foo() {}",
        "int add(int a, int b) { return a + b; }",
    ]

    counts = counter.count_batch(texts)

    assert len(counts) == len(texts)
    assert all(count > 0 for count in counts)


def test_tiktoken_counter_cache_clear() -> None:
    """Test cache clearing."""
    counter = TiktokenCounter(cache_size=10)

    counter.count("test")
    info_before = counter.get_cache_info()
    assert info_before["size"] > 0

    counter.clear_cache()
    info_after = counter.get_cache_info()
    assert info_after["size"] == 0


def test_estimated_token_counter() -> None:
    """Test estimated token counter."""
    counter = EstimatedTokenCounter()

    text = "int main() { return 0; }"
    count = counter.count(text)

    assert count > 0
    assert isinstance(count, int)
    # Estimate should be roughly 0.4 tokens per character
    assert count == int(len(text) * 0.4)


def test_estimated_counter_batch() -> None:
    """Test batch estimation."""
    counter = EstimatedTokenCounter()

    texts = ["test1", "test2", "test3"]
    counts = counter.count_batch(texts)

    assert len(counts) == len(texts)
    assert all(count > 0 for count in counts)


def test_create_token_counter_accurate() -> None:
    """Test creating accurate token counter."""
    counter = create_token_counter(mode="accurate")
    assert isinstance(counter, TiktokenCounter)


def test_create_token_counter_fast() -> None:
    """Test creating fast token counter."""
    counter = create_token_counter(mode="fast")
    assert isinstance(counter, EstimatedTokenCounter)


def test_create_token_counter_invalid_mode() -> None:
    """Test invalid mode raises error."""
    with pytest.raises(ValueError):
        create_token_counter(mode="invalid")
