"""Code chunking functionality for breaking code into analyzable pieces."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from src.ast_parser import FunctionNode, ParsedCode
from src.token_counter import TokenCounter


@dataclass
class CodeChunk:
    """Represents a chunk of code for analysis."""

    id: str  # Unique identifier (SHA-256)
    type: str  # 'function', 'struct', 'header', etc.
    name: str  # Function name, struct name, etc.
    code: str  # Code content
    tokens: int  # Token count
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_function(
        cls,
        func: FunctionNode,
        tokens: int,
        file_path: Path | None = None,
    ) -> "CodeChunk":
        """Create a chunk from a function node.

        Args:
            func: Function node
            tokens: Token count for the function
            file_path: Optional file path

        Returns:
            CodeChunk instance
        """
        chunk_id = hashlib.sha256(func.body.encode("utf-8")).hexdigest()

        metadata = {
            "function_name": func.name,
            "start_line": func.start_line,
            "end_line": func.end_line,
            "line_count": func.line_count,
            "params": func.params,
            "return_type": func.return_type,
        }

        if file_path:
            metadata["file_path"] = str(file_path)
            metadata["file_name"] = file_path.name

        return cls(
            id=chunk_id,
            type="function",
            name=func.name,
            code=func.body,
            tokens=tokens,
            metadata=metadata,
        )


class CodeChunker:
    """Chunks code into pieces suitable for LLM analysis."""

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 18000,
    ) -> None:
        """Initialize code chunker.

        Args:
            token_counter: Token counter instance
            max_tokens: Maximum tokens per chunk
        """
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def chunk_functions(
        self,
        parsed: ParsedCode,
        file_path: Path | None = None,
    ) -> list[CodeChunk]:
        """Chunk functions from parsed code.

        Args:
            parsed: Parsed code object
            file_path: Optional file path for metadata

        Returns:
            List of code chunks
        """
        chunks: list[CodeChunk] = []

        for func in parsed.functions:
            # Count tokens for the function
            tokens = self.token_counter.count(func.body)

            if tokens <= self.max_tokens:
                # Function fits in one chunk
                chunk = CodeChunk.from_function(func, tokens, file_path)
                chunks.append(chunk)
            else:
                # Function is too large - skip for now (Phase 1)
                # Will be handled in Phase 2 with block splitting
                print(
                    f"Warning: Function '{func.name}' exceeds token limit "
                    f"({tokens} > {self.max_tokens}). Skipping for now."
                )
                continue

        return chunks

    def chunk_file(
        self,
        parsed: ParsedCode,
        file_path: Path,
    ) -> list[CodeChunk]:
        """Chunk all code from a file.

        Args:
            parsed: Parsed code object
            file_path: File path

        Returns:
            List of code chunks
        """
        return self.chunk_functions(parsed, file_path)

    def get_chunk_stats(self, chunks: list[CodeChunk]) -> dict:
        """Get statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }

        token_counts = [chunk.tokens for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) // len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }


class ChunkingError(Exception):
    """Exception raised when chunking fails."""

    pass
