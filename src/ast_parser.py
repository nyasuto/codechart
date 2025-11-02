"""AST parsing for C/C++ code using pycparser."""

from dataclasses import dataclass, field
from typing import Protocol

from pycparser import c_ast, c_generator, c_parser


@dataclass
class FunctionNode:
    """Represents a function extracted from source code."""

    name: str
    signature: str
    body: str
    start_line: int
    end_line: int
    params: list[str] = field(default_factory=list)
    return_type: str = ""

    @property
    def line_count(self) -> int:
        """Get number of lines in the function."""
        return self.end_line - self.start_line + 1


@dataclass
class ParsedCode:
    """Container for parsed code elements."""

    functions: list[FunctionNode] = field(default_factory=list)
    structs: list[dict] = field(default_factory=list)
    globals: list[dict] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)


class ASTParser(Protocol):
    """Protocol for AST parser implementations."""

    def parse(self, code: str) -> ParsedCode:
        """Parse code and extract AST information.

        Args:
            code: Source code to parse

        Returns:
            ParsedCode object
        """
        ...


class PycparserAdapter:
    """AST parser using pycparser for C code."""

    def __init__(self) -> None:
        """Initialize pycparser adapter."""
        self.parser = c_parser.CParser()
        self.generator = c_generator.CGenerator()

    def parse(self, code: str) -> ParsedCode:
        """Parse C code using pycparser.

        Args:
            code: C source code

        Returns:
            ParsedCode object

        Raises:
            ParseError: If code cannot be parsed
        """
        try:
            ast = self.parser.parse(code, filename="<string>")
        except Exception as e:
            raise ParseError(f"Failed to parse code: {e}") from e

        result = ParsedCode()

        # Extract functions
        for ext in ast.ext:
            if isinstance(ext, c_ast.FuncDef):
                func_node = self._extract_function(ext)
                if func_node:
                    result.functions.append(func_node)

        return result

    def _extract_function(self, func_def: c_ast.FuncDef) -> FunctionNode | None:
        """Extract function information from AST node.

        Args:
            func_def: Function definition AST node

        Returns:
            FunctionNode object or None if extraction fails
        """
        try:
            # Get function name
            name = func_def.decl.name

            # Generate signature
            signature = self.generator.visit(func_def.decl)

            # Generate body
            body = self.generator.visit(func_def)

            # Get line numbers (if available)
            start_line = getattr(func_def.coord, "line", 0) if func_def.coord else 0
            # Estimate end line based on body
            end_line = start_line + body.count("\n")

            # Extract parameters
            params = self._extract_params(func_def.decl)

            # Extract return type
            return_type = self._extract_return_type(func_def.decl)

            return FunctionNode(
                name=name,
                signature=signature,
                body=body,
                start_line=start_line,
                end_line=end_line,
                params=params,
                return_type=return_type,
            )
        except Exception:
            return None

    def _extract_params(self, decl: c_ast.Decl) -> list[str]:
        """Extract parameter names from function declaration.

        Args:
            decl: Function declaration node

        Returns:
            List of parameter names
        """
        params = []
        if hasattr(decl.type, "args") and decl.type.args:
            for param in decl.type.args.params:
                if hasattr(param, "name") and param.name:
                    params.append(param.name)
        return params

    def _extract_return_type(self, decl: c_ast.Decl) -> str:
        """Extract return type from function declaration.

        Args:
            decl: Function declaration node

        Returns:
            Return type as string
        """
        try:
            if hasattr(decl.type, "type"):
                return self.generator.visit(decl.type.type)
        except Exception:
            pass
        return ""


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass


def create_parser(language: str = "c") -> ASTParser:
    """Factory function to create an AST parser.

    Args:
        language: Programming language ('c' or 'cpp')

    Returns:
        ASTParser instance

    Raises:
        ValueError: If language is not supported
    """
    if language == "c":
        return PycparserAdapter()
    elif language == "cpp":
        # C++ support will be added in Phase 2 using libclang
        raise NotImplementedError("C++ parsing will be supported in Phase 2")
    else:
        raise ValueError(f"Unsupported language: {language}")
