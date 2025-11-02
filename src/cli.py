"""Command-line interface for CodeChart."""

import sys
from pathlib import Path

import click

from src.config import Config
from src.orchestrator import Orchestrator


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """CodeChart - C/C++ Legacy Code Analysis Pipeline.

    Analyze C/C++ code and generate technical documentation using LLM.
    """
    pass


@main.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default="output",
    help="Output directory for generated documentation (default: output)",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration file (default: config/default.yaml)",
)
@click.option(
    "-n",
    "--name",
    "project_name",
    type=str,
    default=None,
    help="Project name (default: source directory name)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without calling LLM API (token counting only)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output",
)
def analyze(
    source_dir: Path,
    output_dir: Path,
    config_path: Path | None,
    project_name: str | None,
    dry_run: bool,
    debug: bool,
) -> None:
    """Analyze C/C++ source code and generate technical documentation.

    SOURCE_DIR: Directory containing C/C++ source files to analyze
    """
    try:
        # Load configuration
        if config_path:
            config = Config.from_yaml(config_path)
        else:
            config = Config.from_yaml()

        # TODO: Implement dry-run mode
        if dry_run:
            click.echo("Dry-run mode is not yet implemented", err=True)
            sys.exit(1)

        # TODO: Implement debug mode
        if debug:
            click.echo("Debug mode enabled")

        # Create orchestrator
        orchestrator = Orchestrator(config=config)

        # Run analysis
        click.echo("\nCodeChart v0.1.0")
        click.echo(f"Source: {source_dir}")
        click.echo(f"Output: {output_dir}")
        click.echo()

        stats = orchestrator.analyze_directory(
            source_dir=source_dir,
            output_dir=output_dir,
            project_name=project_name,
        )

        # Check success criteria
        if stats.error_rate > 0.05:  # Error rate > 5%
            click.echo(
                f"\n⚠️  Warning: Error rate ({stats.error_rate:.1%}) exceeds 5% threshold",
                err=True,
            )
            sys.exit(1)

        if stats.total_chunks == 0:
            click.echo("\n⚠️  Warning: No code chunks were analyzed", err=True)
            sys.exit(1)

        click.echo("\n✓ Analysis completed successfully!")
        click.echo(f"  Documentation generated in: {output_dir}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n\nAnalysis interrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if debug:
            raise
        sys.exit(1)


@main.command()
def info() -> None:
    """Display configuration and environment information."""
    try:
        config = Config.from_yaml()

        click.echo("CodeChart Configuration:")
        click.echo(f"  LLM Provider: {config.llm.provider}")
        click.echo(f"  Model: {config.llm.model}")
        click.echo(f"  Base URL: {config.llm.base_url}")
        click.echo(f"  Max Chunk Tokens: {config.analysis_max_chunk_tokens:,}")
        click.echo(f"  Batch Size: {config.analysis_batch_size}")
        click.echo(f"  Parallel Requests: {config.analysis_parallel_requests}")
        click.echo(f"  Output Formats: {', '.join(config.output_formats)}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
