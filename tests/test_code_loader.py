"""Tests for code_loader module."""

from pathlib import Path

import pytest

from src.code_loader import CodeFile, CodeLoader


def test_code_file_properties() -> None:
    """Test CodeFile dataclass properties."""
    code_file = CodeFile(
        path=Path("test.c"),
        content="int main() {\n    return 0;\n}\n",
        language="c",
        hash="abc123",
    )

    assert code_file.name == "test.c"
    assert code_file.line_count == 3


def test_code_loader_initialization() -> None:
    """Test CodeLoader initialization."""
    loader = CodeLoader(include_headers=True)
    assert loader.include_headers is True

    loader = CodeLoader(include_headers=False)
    assert loader.include_headers is False


def test_detect_language() -> None:
    """Test language detection from file extension."""
    loader = CodeLoader()

    assert loader._detect_language(Path("test.c")) == "c"
    assert loader._detect_language(Path("test.cpp")) == "cpp"
    assert loader._detect_language(Path("test.cc")) == "cpp"
    assert loader._detect_language(Path("test.h")) == "c"
    assert loader._detect_language(Path("test.hpp")) == "cpp"


def test_calculate_hash() -> None:
    """Test SHA-256 hash calculation."""
    loader = CodeLoader()

    hash1 = loader._calculate_hash("test content")
    hash2 = loader._calculate_hash("test content")
    hash3 = loader._calculate_hash("different content")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA-256 hex digest length


def test_is_valid_file() -> None:
    """Test file extension validation."""
    loader = CodeLoader(include_headers=True)

    assert loader._is_valid_file(Path("test.c")) is True
    assert loader._is_valid_file(Path("test.cpp")) is True
    assert loader._is_valid_file(Path("test.h")) is True
    assert loader._is_valid_file(Path("test.txt")) is False
    assert loader._is_valid_file(Path("test.py")) is False


def test_is_valid_file_without_headers() -> None:
    """Test file extension validation without headers."""
    loader = CodeLoader(include_headers=False)

    assert loader._is_valid_file(Path("test.c")) is True
    assert loader._is_valid_file(Path("test.cpp")) is True
    assert loader._is_valid_file(Path("test.h")) is False
