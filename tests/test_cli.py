"""Tests for CLI module."""

from click.testing import CliRunner

from src.cli import analyze, info, main


def test_main_help() -> None:
    """Test main command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "CodeChart" in result.output
    assert "analyze" in result.output


def test_analyze_help() -> None:
    """Test analyze command help."""
    runner = CliRunner()
    result = runner.invoke(analyze, ["--help"])

    assert result.exit_code == 0
    assert "SOURCE_DIR" in result.output
    assert "--output" in result.output
    assert "--config" in result.output


def test_analyze_nonexistent_directory() -> None:
    """Test analyze with non-existent directory."""
    runner = CliRunner()
    result = runner.invoke(analyze, ["/nonexistent/path"])

    assert result.exit_code == 2  # Click error code for invalid path
    assert "does not exist" in result.output.lower() or "error" in result.output.lower()


def test_analyze_dry_run_not_implemented() -> None:
    """Test dry-run mode (not yet implemented)."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create a dummy directory
        import os

        os.mkdir("test_src")

        result = runner.invoke(analyze, ["test_src", "--dry-run"])

        assert result.exit_code == 1
        assert "not yet implemented" in result.output.lower()


def test_info_command() -> None:
    """Test info command."""
    runner = CliRunner()
    result = runner.invoke(info)

    assert result.exit_code == 0
    assert "Configuration" in result.output
    assert "LLM Provider" in result.output
    assert "Model" in result.output


def test_main_version() -> None:
    """Test version option."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
