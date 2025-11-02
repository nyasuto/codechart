"""Orchestrator for the entire analysis pipeline."""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.ast_parser import ParseError, create_parser
from src.code_chunker import CodeChunker
from src.code_loader import CodeLoader
from src.config import Config
from src.llm_analyzer import AnalysisResult, LLMAnalyzer
from src.output_formatter import OutputFormatter
from src.token_counter import create_token_counter


@dataclass
class AnalysisStats:
    """Statistics for analysis run."""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_chunks == 0:
            return 0.0
        return self.failed_chunks / self.total_chunks

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - self.error_rate


class Orchestrator:
    """Orchestrates the entire code analysis pipeline."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize orchestrator.

        Args:
            config: Configuration object (loads default if None)
        """
        self.config = config or Config.from_yaml()

        # Initialize components
        self.loader = CodeLoader(include_headers=True)
        self.token_counter = create_token_counter(mode="fast")
        self.chunker = CodeChunker(
            token_counter=self.token_counter,
            max_tokens=self.config.analysis_max_chunk_tokens,
        )
        self.analyzer = LLMAnalyzer(config=self.config)

        # Statistics
        self.stats = AnalysisStats()

    def analyze_directory(
        self,
        source_dir: Path,
        output_dir: Path,
        project_name: str | None = None,
    ) -> AnalysisStats:
        """Analyze all C/C++ files in a directory.

        Args:
            source_dir: Directory containing source files
            output_dir: Output directory for results
            project_name: Project name (uses directory name if None)

        Returns:
            Analysis statistics
        """
        start_time = time.time()

        # Use directory name as project name if not provided
        if project_name is None:
            project_name = source_dir.name

        print(f"Starting analysis of '{project_name}'...")

        # Discover files
        print(f"Discovering files in {source_dir}...")
        file_paths = self.loader.discover_files(source_dir)
        self.stats.total_files = len(file_paths)
        print(f"Found {self.stats.total_files} files")

        if self.stats.total_files == 0:
            print("No C/C++ files found!")
            return self.stats

        # Process all files
        all_results: list[AnalysisResult] = []
        file_results: dict[str, list[AnalysisResult]] = defaultdict(list)

        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{self.stats.total_files}] Processing {file_path.name}...")
            results = self._process_file(file_path)

            if results:
                all_results.extend(results)
                file_results[str(file_path.relative_to(source_dir))] = results
                self.stats.successful_files += 1
            else:
                self.stats.failed_files += 1

        # Generate output
        if all_results:
            print("\nGenerating documentation...")
            formatter = OutputFormatter(output_dir)

            # Generate project summary
            summary_path = formatter.generate_project_summary(
                project_name=project_name,
                results=all_results,
                file_results=dict(file_results),
            )
            print(f"Generated: {summary_path}")

            # Generate file documentation
            for file_path, results in file_results.items():
                doc_path = formatter.generate_file_doc(
                    file_path=file_path,
                    results=results,
                    language="c",  # TODO: Detect language properly
                )
                print(f"Generated: {doc_path}")

            # Generate CSV files
            functions_csv = formatter.generate_function_csv(all_results)
            print(f"Generated: {functions_csv}")

            metrics_csv = formatter.generate_metrics_csv(all_results, dict(file_results))
            print(f"Generated: {metrics_csv}")

        # Calculate final stats
        self.stats.total_time = time.time() - start_time

        # Print summary
        self._print_summary()

        return self.stats

    def _process_file(self, file_path: Path) -> list[AnalysisResult]:
        """Process a single source file.

        Args:
            file_path: Path to source file

        Returns:
            List of analysis results
        """
        results: list[AnalysisResult] = []

        try:
            # Load file
            code_file = self.loader.load_file(file_path)
            print(f"  Loaded: {code_file.line_count} lines")

            # Parse code
            parser = create_parser(code_file.language)
            parsed = parser.parse(code_file.content)
            print(f"  Parsed: {len(parsed.functions)} functions")

            if len(parsed.functions) == 0:
                print("  No functions found, skipping...")
                return results

            # Chunk code
            chunks = self.chunker.chunk_file(parsed, file_path)
            print(f"  Chunked: {len(chunks)} chunks")
            self.stats.total_chunks += len(chunks)

            if len(chunks) == 0:
                print("  No chunks generated, skipping...")
                return results

            # Analyze each chunk
            for j, chunk in enumerate(chunks, 1):
                try:
                    print(f"    Analyzing chunk {j}/{len(chunks)}: {chunk.name}...", end=" ")
                    result = self.analyzer.analyze_chunk(chunk)
                    results.append(result)
                    self.stats.successful_chunks += 1
                    self.stats.total_tokens += result.tokens_used
                    print("✓")
                except Exception as e:
                    print(f"✗ ({e})")
                    self.stats.failed_chunks += 1
                    self.stats.errors.append(f"{file_path.name}:{chunk.name} - {e}")

        except ParseError as e:
            error_msg = f"{file_path.name}: Parse error - {e}"
            print(f"  Error: {error_msg}")
            self.stats.errors.append(error_msg)
        except Exception as e:
            error_msg = f"{file_path.name}: Unexpected error - {e}"
            print(f"  Error: {error_msg}")
            self.stats.errors.append(error_msg)

        return results

    def _print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)

        print("\nFiles:")
        print(f"  Total:      {self.stats.total_files}")
        print(f"  Successful: {self.stats.successful_files}")
        print(f"  Failed:     {self.stats.failed_files}")

        print("\nChunks:")
        print(f"  Total:      {self.stats.total_chunks}")
        print(f"  Successful: {self.stats.successful_chunks}")
        print(f"  Failed:     {self.stats.failed_chunks}")

        if self.stats.total_chunks > 0:
            print(f"  Success Rate: {self.stats.success_rate:.1%}")
            print(f"  Error Rate:   {self.stats.error_rate:.1%}")

        print("\nTokens:")
        print(f"  Total: {self.stats.total_tokens:,}")

        print("\nTime:")
        print(f"  Total: {self.stats.total_time:.2f}s")

        if self.stats.errors:
            print(f"\nErrors ({len(self.stats.errors)}):")
            for error in self.stats.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.stats.errors) > 10:
                print(f"  ... and {len(self.stats.errors) - 10} more")

        print("=" * 60)
