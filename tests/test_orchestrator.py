"""Tests for orchestrator module."""

from pathlib import Path

import pytest

from src.config import Config
from src.orchestrator import AnalysisStats, Orchestrator


def test_analysis_stats_creation() -> None:
    """Test AnalysisStats creation."""
    stats = AnalysisStats(
        total_files=10,
        successful_files=9,
        failed_files=1,
        total_chunks=50,
        successful_chunks=48,
        failed_chunks=2,
    )

    assert stats.total_files == 10
    assert stats.successful_files == 9
    assert stats.failed_chunks == 2


def test_analysis_stats_error_rate() -> None:
    """Test error rate calculation."""
    stats = AnalysisStats(
        total_chunks=100,
        successful_chunks=95,
        failed_chunks=5,
    )

    assert stats.error_rate == 0.05
    assert stats.success_rate == 0.95


def test_analysis_stats_zero_chunks() -> None:
    """Test error rate with zero chunks."""
    stats = AnalysisStats()

    assert stats.error_rate == 0.0
    assert stats.success_rate == 1.0


def test_orchestrator_initialization() -> None:
    """Test Orchestrator initialization."""
    config = Config.from_yaml()
    orchestrator = Orchestrator(config=config)

    assert orchestrator.config == config
    assert orchestrator.loader is not None
    assert orchestrator.chunker is not None
    assert orchestrator.analyzer is not None


def test_orchestrator_initialization_default_config() -> None:
    """Test Orchestrator initialization with default config."""
    orchestrator = Orchestrator()

    assert orchestrator.config is not None
    assert orchestrator.loader is not None


def test_analyze_directory_nonexistent(tmp_path: Path) -> None:
    """Test analyzing non-existent directory."""
    orchestrator = Orchestrator()
    nonexistent = tmp_path / "nonexistent"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        orchestrator.analyze_directory(
            source_dir=nonexistent,
            output_dir=tmp_path / "output",
        )


def test_analyze_directory_empty(tmp_path: Path) -> None:
    """Test analyzing empty directory."""
    orchestrator = Orchestrator()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    stats = orchestrator.analyze_directory(
        source_dir=empty_dir,
        output_dir=tmp_path / "output",
    )

    assert stats.total_files == 0
    assert stats.total_chunks == 0


def test_analyze_directory_with_simple_c_file(tmp_path: Path) -> None:
    """Test analyzing directory with a simple C file."""
    # Create a simple C file
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    c_file = source_dir / "test.c"
    c_file.write_text(
        """
int add(int a, int b) {
    return a + b;
}
""",
        encoding="utf-8",
    )

    orchestrator = Orchestrator()

    # Note: This test requires LLM to be available
    # For unit testing, we might want to mock the LLM analyzer
    # For now, this test will be skipped if LLM is not available
    pytest.skip("Requires LLM API access")

    stats = orchestrator.analyze_directory(
        source_dir=source_dir,
        output_dir=tmp_path / "output",
        project_name="TestProject",
    )

    assert stats.total_files == 1
    assert stats.successful_files == 1
    assert stats.total_chunks > 0
