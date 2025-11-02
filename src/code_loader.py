"""Code file loader for C/C++ source files."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeFile:
    """Represents a loaded source code file."""

    path: Path
    content: str
    language: str  # 'c' or 'cpp'
    hash: str  # SHA-256 hash of content
    encoding: str = "utf-8"

    @property
    def name(self) -> str:
        """Get file name."""
        return self.path.name

    @property
    def line_count(self) -> int:
        """Get number of lines in the file."""
        return len(self.content.splitlines())


class CodeLoader:
    """Loads C/C++ source files from the filesystem."""

    # File extensions for C/C++ files
    C_EXTENSIONS = {".c"}
    CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c++"}
    HEADER_EXTENSIONS = {".h", ".hpp", ".hxx", ".h++"}

    def __init__(self, include_headers: bool = True) -> None:
        """Initialize code loader.

        Args:
            include_headers: Whether to include header files
        """
        self.include_headers = include_headers
        self._valid_extensions = (
            self.C_EXTENSIONS | self.CPP_EXTENSIONS | self.HEADER_EXTENSIONS
            if include_headers
            else self.C_EXTENSIONS | self.CPP_EXTENSIONS
        )

    def discover_files(self, root_path: Path) -> list[Path]:
        """Discover all C/C++ files in a directory recursively.

        Args:
            root_path: Root directory to search

        Returns:
            List of file paths found
        """
        if not root_path.exists():
            raise FileNotFoundError(f"Path does not exist: {root_path}")

        if root_path.is_file():
            return [root_path] if self._is_valid_file(root_path) else []

        # Recursively find all valid files
        return sorted(
            [
                f
                for f in root_path.rglob("*")
                if f.is_file() and self._is_valid_file(f)
            ]
        )

    def load_file(self, file_path: Path) -> CodeFile:
        """Load a single source file.

        Args:
            file_path: Path to the file to load

        Returns:
            CodeFile object

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try UTF-8 first, then fall back to latin-1
        encoding = "utf-8"
        try:
            content = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"
            content = file_path.read_text(encoding=encoding)

        # Determine language
        language = self._detect_language(file_path)

        # Calculate hash
        file_hash = self._calculate_hash(content)

        return CodeFile(
            path=file_path,
            content=content,
            language=language,
            hash=file_hash,
            encoding=encoding,
        )

    def load_batch(self, file_paths: list[Path]) -> Iterator[CodeFile]:
        """Load multiple files in batch.

        Args:
            file_paths: List of file paths to load

        Yields:
            CodeFile objects
        """
        for file_path in file_paths:
            try:
                yield self.load_file(file_path)
            except (FileNotFoundError, UnicodeDecodeError) as e:
                # Log error but continue with other files
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file has a valid C/C++ extension.

        Args:
            file_path: Path to check

        Returns:
            True if valid, False otherwise
        """
        return file_path.suffix.lower() in self._valid_extensions

    def _detect_language(self, file_path: Path) -> str:
        """Detect whether file is C or C++.

        Args:
            file_path: Path to the file

        Returns:
            'c' or 'cpp'
        """
        suffix = file_path.suffix.lower()
        if suffix in self.C_EXTENSIONS:
            return "c"
        elif suffix in self.CPP_EXTENSIONS:
            return "cpp"
        elif suffix in self.HEADER_EXTENSIONS:
            # For headers, assume C++ if .hpp/.hxx, otherwise C
            return "cpp" if suffix in {".hpp", ".hxx", ".h++"} else "c"
        return "c"  # Default to C

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content.

        Args:
            content: File content

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
