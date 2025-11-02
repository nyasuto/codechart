"""Output formatter for generating technical documentation."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.llm_analyzer import AnalysisResult


@dataclass
class FileInfo:
    """Information about a file."""

    name: str
    path: str
    function_count: int
    line_count: int


@dataclass
class ProjectStats:
    """Project statistics."""

    total_files: int
    total_functions: int
    total_lines: int
    avg_complexity: str
    total_tokens: int


class OutputFormatter:
    """Formatter for generating Markdown and CSV outputs."""

    def __init__(self, output_dir: Path, template_dir: Path | None = None):
        """Initialize output formatter.

        Args:
            output_dir: Output directory for generated files
            template_dir: Template directory (defaults to ./templates)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "files").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)

        # Setup Jinja2 environment
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "templates"

        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True
        )

    def generate_project_summary(
        self,
        project_name: str,
        results: Sequence[AnalysisResult],
        file_results: dict[str, list[AnalysisResult]],
    ) -> Path:
        """Generate project summary markdown.

        Args:
            project_name: Name of the project
            results: All analysis results
            file_results: Results grouped by file

        Returns:
            Path to generated README.md
        """
        # Calculate statistics
        stats = self._calculate_stats(results, file_results)

        # Create file info list
        files = []
        for file_path, file_results_list in file_results.items():
            file_name = Path(file_path).name
            files.append(
                FileInfo(
                    name=file_name,
                    path=file_path,
                    function_count=len(file_results_list),
                    line_count=sum(r.tokens_used for r in file_results_list),  # Approximation
                )
            )

        # Extract critical issues
        critical_issues = []
        for result in results:
            if result.potential_issues:
                for issue in result.potential_issues:
                    critical_issues.append(
                        {
                            "function_name": result.chunk_name,
                            "file_name": result.chunk_id.split(":")[0]
                            if ":" in result.chunk_id
                            else "unknown",
                            "description": issue,
                        }
                    )

        # Render template
        template = self.template_env.get_template("project_summary.md.j2")
        content = template.render(
            project_name=project_name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stats=stats,
            files=files,
            critical_issues=critical_issues[:10],  # Top 10 issues
        )

        # Write to file
        output_path = self.output_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")

        return output_path

    def generate_file_doc(
        self, file_path: str, results: Sequence[AnalysisResult], language: str = "c"
    ) -> Path:
        """Generate file detail markdown.

        Args:
            file_path: Path to the source file
            results: Analysis results for this file
            language: Programming language

        Returns:
            Path to generated markdown file
        """
        file_name = Path(file_path).name

        # Create file summary
        if results:
            file_summary = f"このファイルには{len(results)}個の関数が含まれています。"
        else:
            file_summary = "このファイルには関数が含まれていません。"

        # Prepare function data
        functions = []
        for result in results:
            functions.append(
                {
                    "signature": result.chunk_name,
                    "start_line": 1,  # Will be populated from metadata if available
                    "end_line": 100,  # Will be populated from metadata if available
                    "purpose": result.purpose,
                    "algorithm": result.algorithm,
                    "complexity": result.complexity,
                    "dependencies": result.dependencies,
                    "potential_issues": result.potential_issues,
                    "improvements": result.improvements,
                }
            )

        # Render template
        template = self.template_env.get_template("file_detail.md.j2")
        content = template.render(
            file_name=file_name,
            file_path=file_path,
            language=language,
            line_count=sum(r.tokens_used for r in results),  # Approximation
            function_count=len(results),
            file_summary=file_summary,
            functions=functions,
        )

        # Write to file
        output_path = self.output_dir / "files" / f"{file_name}.md"
        output_path.write_text(content, encoding="utf-8")

        return output_path

    def generate_function_csv(self, results: Sequence[AnalysisResult]) -> Path:
        """Generate function list CSV.

        Args:
            results: Analysis results

        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / "metrics" / "functions.csv"

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "ファイル",
                    "関数名",
                    "時間複雑度",
                    "依存関数数",
                    "潜在的問題数",
                    "改善提案数",
                    "トークン数",
                ]
            )

            # Data rows
            for result in results:
                file_name = result.chunk_id.split(":")[0] if ":" in result.chunk_id else "unknown"
                writer.writerow(
                    [
                        file_name,
                        result.chunk_name,
                        result.complexity,
                        len(result.dependencies),
                        len(result.potential_issues),
                        len(result.improvements),
                        result.tokens_used,
                    ]
                )

        return output_path

    def generate_metrics_csv(
        self, results: Sequence[AnalysisResult], file_results: dict[str, list[AnalysisResult]]
    ) -> Path:
        """Generate metrics summary CSV.

        Args:
            results: All analysis results
            file_results: Results grouped by file

        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / "metrics" / "metrics.csv"

        stats = self._calculate_stats(results, file_results)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["カテゴリ", "メトリクス", "値"])

            # Project metrics
            writer.writerow(["プロジェクト", "総ファイル数", stats.total_files])
            writer.writerow(["プロジェクト", "総関数数", stats.total_functions])
            writer.writerow(["プロジェクト", "総行数", stats.total_lines])

            # Quality metrics
            writer.writerow(["品質", "平均複雑度", stats.avg_complexity])
            writer.writerow(
                [
                    "品質",
                    "潜在的問題数",
                    sum(len(r.potential_issues) for r in results),
                ]
            )

            # Token usage
            writer.writerow(["トークン", "総使用トークン数", stats.total_tokens])

        return output_path

    def append_function_to_csv(self, result: AnalysisResult) -> Path:
        """Append a single function result to CSV (incremental mode).

        Args:
            result: Single analysis result

        Returns:
            Path to CSV file
        """
        output_path = self.output_dir / "metrics" / "functions.csv"

        # Check if file exists, if not create with header
        file_exists = output_path.exists()

        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(
                    [
                        "ファイル",
                        "関数名",
                        "時間複雑度",
                        "依存関数数",
                        "潜在的問題数",
                        "改善提案数",
                        "トークン数",
                    ]
                )

            # Write data row
            file_name = result.chunk_id.split(":")[0] if ":" in result.chunk_id else "unknown"
            writer.writerow(
                [
                    file_name,
                    result.chunk_name,
                    result.complexity,
                    len(result.dependencies),
                    len(result.potential_issues),
                    len(result.improvements),
                    result.tokens_used,
                ]
            )

        return output_path

    def append_function_to_file_doc(
        self, file_path: str, result: AnalysisResult, language: str = "c"
    ) -> Path:
        """Append a single function to file documentation (incremental mode).

        Args:
            file_path: Path to the source file
            result: Single analysis result
            language: Programming language

        Returns:
            Path to generated markdown file
        """
        file_name = Path(file_path).name
        output_path = self.output_dir / "files" / f"{file_name}.md"

        # Check if file exists
        file_exists = output_path.exists()

        if not file_exists:
            # Create file with header
            header = f"""# {file_name}

**パス**: `{file_path}`
**言語**: {language}

---

## 関数一覧

"""
            output_path.write_text(header, encoding="utf-8")

        # Append function documentation
        function_doc = f"""### `{result.chunk_name}`

#### 目的
{result.purpose}

#### アルゴリズム
{result.algorithm}

#### 時間計算量
{result.complexity}

"""

        if result.dependencies:
            function_doc += "#### 依存関数\n"
            for dep in result.dependencies:
                function_doc += f"- `{dep}`\n"
            function_doc += "\n"

        if result.potential_issues:
            function_doc += "#### ⚠️ 潜在的問題\n"
            for issue in result.potential_issues:
                function_doc += f"- {issue}\n"
            function_doc += "\n"

        if result.improvements:
            function_doc += "#### 💡 改善提案\n"
            for improvement in result.improvements:
                function_doc += f"- {improvement}\n"
            function_doc += "\n"

        function_doc += "---\n\n"

        # Append to file
        with output_path.open("a", encoding="utf-8") as f:
            f.write(function_doc)

        return output_path

    def finalize_file_doc(self, file_path: str) -> Path:
        """Finalize file documentation by adding footer.

        Args:
            file_path: Path to the source file

        Returns:
            Path to markdown file
        """
        file_name = Path(file_path).name
        output_path = self.output_dir / "files" / f"{file_name}.md"

        if output_path.exists():
            footer = "\n---\n\n*この文書は CodeChart により自動生成されました。*\n"
            with output_path.open("a", encoding="utf-8") as f:
                f.write(footer)

        return output_path

    def _calculate_stats(
        self, results: Sequence[AnalysisResult], file_results: dict[str, list[AnalysisResult]]
    ) -> ProjectStats:
        """Calculate project statistics.

        Args:
            results: All analysis results
            file_results: Results grouped by file

        Returns:
            Project statistics
        """
        total_files = len(file_results)
        total_functions = len(results)
        total_lines = sum(r.tokens_used for r in results)  # Approximation
        total_tokens = sum(r.tokens_used for r in results)

        # Calculate average complexity (simple approximation)
        complexities = [r.complexity for r in results if r.complexity]
        if complexities:
            # Extract O(n) patterns and approximate
            avg_complexity = "O(n)"  # Simplified for now
        else:
            avg_complexity = "N/A"

        return ProjectStats(
            total_files=total_files,
            total_functions=total_functions,
            total_lines=total_lines,
            avg_complexity=avg_complexity,
            total_tokens=total_tokens,
        )
