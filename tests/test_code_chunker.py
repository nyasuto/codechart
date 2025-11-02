"""Tests for code_chunker module."""

from pathlib import Path

import pytest

from src.ast_parser import FunctionNode, ParsedCode
from src.code_chunker import CodeChunk, CodeChunker
from src.token_counter import EstimatedTokenCounter


def test_code_chunk_from_function() -> None:
    """Test creating CodeChunk from FunctionNode."""
    func = FunctionNode(
        name="test_func",
        signature="int test_func(int a)",
        body="int test_func(int a) { return a; }",
        start_line=1,
        end_line=3,
        params=["a"],
        return_type="int",
    )

    chunk = CodeChunk.from_function(func, tokens=10, file_path=Path("test.c"))

    assert chunk.type == "function"
    assert chunk.name == "test_func"
    assert chunk.tokens == 10
    assert chunk.metadata["function_name"] == "test_func"
    assert chunk.metadata["file_name"] == "test.c"


def test_code_chunker_initialization() -> None:
    """Test CodeChunker initialization."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter, max_tokens=1000)

    assert chunker.max_tokens == 1000
    assert chunker.token_counter is counter


def test_code_chunker_simple_functions() -> None:
    """Test chunking simple functions."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter, max_tokens=1000)

    # Create parsed code with two simple functions
    func1 = FunctionNode(
        name="func1",
        signature="int func1()",
        body="int func1() { return 1; }",
        start_line=1,
        end_line=1,
    )

    func2 = FunctionNode(
        name="func2",
        signature="int func2()",
        body="int func2() { return 2; }",
        start_line=3,
        end_line=3,
    )

    parsed = ParsedCode(functions=[func1, func2])

    chunks = chunker.chunk_functions(parsed)

    assert len(chunks) == 2
    assert chunks[0].name == "func1"
    assert chunks[1].name == "func2"


def test_code_chunker_large_function_skipped(capsys: pytest.CaptureFixture) -> None:
    """Test that large functions are skipped with warning."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter, max_tokens=10)  # Very small limit

    # Create a function with many characters
    large_body = "int large_func() {" + " " * 100 + "return 0; }"
    large_func = FunctionNode(
        name="large_func",
        signature="int large_func()",
        body=large_body,
        start_line=1,
        end_line=1,
    )

    parsed = ParsedCode(functions=[large_func])

    chunks = chunker.chunk_functions(parsed)

    # Should be skipped
    assert len(chunks) == 0

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "large_func" in captured.out


def test_code_chunker_get_stats_empty() -> None:
    """Test getting stats for empty chunk list."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter)

    stats = chunker.get_chunk_stats([])

    assert stats["total_chunks"] == 0
    assert stats["total_tokens"] == 0


def test_code_chunker_get_stats() -> None:
    """Test getting chunk statistics."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter)

    # Create mock chunks
    chunks = [
        CodeChunk(id="1", type="function", name="f1", code="code1", tokens=100),
        CodeChunk(id="2", type="function", name="f2", code="code2", tokens=200),
        CodeChunk(id="3", type="function", name="f3", code="code3", tokens=150),
    ]

    stats = chunker.get_chunk_stats(chunks)

    assert stats["total_chunks"] == 3
    assert stats["total_tokens"] == 450
    assert stats["min_tokens"] == 100
    assert stats["max_tokens"] == 200


def test_code_chunker_chunk_file() -> None:
    """Test chunking from file path."""
    counter = EstimatedTokenCounter()
    chunker = CodeChunker(counter)

    func = FunctionNode(
        name="test",
        signature="int test()",
        body="int test() { return 0; }",
        start_line=1,
        end_line=1,
    )

    parsed = ParsedCode(functions=[func])

    chunks = chunker.chunk_file(parsed, Path("test.c"))

    assert len(chunks) == 1
    assert chunks[0].metadata["file_name"] == "test.c"
