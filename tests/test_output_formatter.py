"""Tests for output_formatter module."""

import csv
from pathlib import Path

from src.llm_analyzer import AnalysisResult
from src.output_formatter import FileInfo, OutputFormatter, ProjectStats


def create_test_result(
    chunk_id: str = "test.c:add",
    chunk_name: str = "add",
    summary: str = "Test function",
) -> AnalysisResult:
    """Create a test analysis result."""
    return AnalysisResult(
        chunk_id=chunk_id,
        chunk_name=chunk_name,
        summary=summary,
        purpose="Testing purpose",
        algorithm="Simple algorithm",
        complexity="O(1)",
        dependencies=["helper_func"],
        potential_issues=["Potential overflow"],
        improvements=["Add validation"],
        raw_response='{"summary": "test"}',
        tokens_used=100,
    )


def test_file_info_creation() -> None:
    """Test FileInfo dataclass creation."""
    file_info = FileInfo(name="test.c", path="src/test.c", function_count=5, line_count=100)

    assert file_info.name == "test.c"
    assert file_info.function_count == 5
    assert file_info.line_count == 100


def test_project_stats_creation() -> None:
    """Test ProjectStats dataclass creation."""
    stats = ProjectStats(
        total_files=10,
        total_functions=50,
        total_lines=1000,
        avg_complexity="O(n)",
        total_tokens=5000,
    )

    assert stats.total_files == 10
    assert stats.total_functions == 50
    assert stats.avg_complexity == "O(n)"


def test_output_formatter_initialization(tmp_path: Path) -> None:
    """Test OutputFormatter initialization."""
    formatter = OutputFormatter(tmp_path)

    assert formatter.output_dir == tmp_path
    assert (tmp_path / "files").exists()
    assert (tmp_path / "metrics").exists()


def test_generate_project_summary(tmp_path: Path) -> None:
    """Test project summary generation."""
    formatter = OutputFormatter(tmp_path)

    results = [
        create_test_result("file1.c:func1", "func1"),
        create_test_result("file1.c:func2", "func2"),
        create_test_result("file2.c:func3", "func3"),
    ]

    file_results = {
        "file1.c": [results[0], results[1]],
        "file2.c": [results[2]],
    }

    output_path = formatter.generate_project_summary("TestProject", results, file_results)

    assert output_path.exists()
    assert output_path.name == "README.md"

    content = output_path.read_text()
    assert "TestProject" in content
    assert "総ファイル数" in content
    assert "総関数数" in content
    assert "file1.c" in content


def test_generate_file_doc(tmp_path: Path) -> None:
    """Test file documentation generation."""
    formatter = OutputFormatter(tmp_path)

    results = [
        create_test_result("test.c:add", "add", "Adds two numbers"),
        create_test_result("test.c:subtract", "subtract", "Subtracts numbers"),
    ]

    output_path = formatter.generate_file_doc("src/test.c", results)

    assert output_path.exists()
    assert output_path.name == "test.c.md"

    content = output_path.read_text()
    assert "test.c" in content
    assert "add" in content
    assert "subtract" in content
    assert "目的" in content
    assert "アルゴリズム" in content


def test_generate_function_csv(tmp_path: Path) -> None:
    """Test function CSV generation."""
    formatter = OutputFormatter(tmp_path)

    results = [
        create_test_result("file1.c:func1", "func1"),
        create_test_result("file1.c:func2", "func2"),
    ]

    output_path = formatter.generate_function_csv(results)

    assert output_path.exists()
    assert output_path.name == "functions.csv"

    # Read and validate CSV
    with output_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 3  # Header + 2 data rows
    assert "ファイル" in rows[0]
    assert "関数名" in rows[0]
    assert "func1" in rows[1]
    assert "func2" in rows[2]


def test_generate_metrics_csv(tmp_path: Path) -> None:
    """Test metrics CSV generation."""
    formatter = OutputFormatter(tmp_path)

    results = [
        create_test_result("file1.c:func1", "func1"),
        create_test_result("file2.c:func2", "func2"),
    ]

    file_results = {"file1.c": [results[0]], "file2.c": [results[1]]}

    output_path = formatter.generate_metrics_csv(results, file_results)

    assert output_path.exists()
    assert output_path.name == "metrics.csv"

    # Read and validate CSV
    with output_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) > 1  # Header + data rows
    assert "カテゴリ" in rows[0]
    assert "メトリクス" in rows[0]

    # Check for key metrics
    categories = [row[0] for row in rows[1:]]
    assert "プロジェクト" in categories
    assert "品質" in categories


def test_empty_results(tmp_path: Path) -> None:
    """Test handling of empty results."""
    formatter = OutputFormatter(tmp_path)

    results: list[AnalysisResult] = []
    file_results: dict[str, list[AnalysisResult]] = {}

    # Should not raise errors
    summary_path = formatter.generate_project_summary("Empty", results, file_results)
    assert summary_path.exists()

    csv_path = formatter.generate_function_csv(results)
    assert csv_path.exists()

    # CSV should have header only
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # Header only
