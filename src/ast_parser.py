"""AST parsing for C/C++ code using tree-sitter and pycparser."""

import re
from dataclasses import dataclass, field
from typing import Protocol

import tree_sitter_c as tsc
from pycparser import c_ast, c_generator, c_parser
from tree_sitter import Language, Parser, Query, QueryCursor


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

    def _preprocess(self, code: str) -> str:
        """Simple preprocessing to remove comments and directives.

        Args:
            code: Raw C code

        Returns:
            Preprocessed code
        """
        # Remove single-line comments
        code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # Remove preprocessor directives (keep empty lines for line numbers)
        code = re.sub(r"^\s*#.*?$", "", code, flags=re.MULTILINE)

        return code

    def parse(self, code: str) -> ParsedCode:
        """Parse C code using pycparser.

        Args:
            code: C source code

        Returns:
            ParsedCode object

        Raises:
            ParseError: If code cannot be parsed
        """
        # Preprocess code to remove comments and directives
        code = self._preprocess(code)

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
                return str(self.generator.visit(decl.type.type))
        except Exception:
            pass
        return ""


class ParseError(Exception):
    """Exception raised when parsing fails."""

    pass


class TreeSitterAdapter:
    """AST parser using tree-sitter for C code."""

    def __init__(self) -> None:
        """Initialize tree-sitter adapter."""
        self.language = Language(tsc.language())
        self.parser = Parser(self.language)

        # Query for function definitions
        self.function_query = Query(
            self.language,
            """
            (function_definition
                declarator: (function_declarator
                              declarator: (identifier) @name)
                body: (compound_statement) @body) @function
            """,
        )

    def parse(self, code: str) -> ParsedCode:
        """Parse C code using tree-sitter.

        Args:
            code: C source code

        Returns:
            ParsedCode object

        Raises:
            ParseError: If code cannot be parsed
        """
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
        except Exception as e:
            raise ParseError(f"Failed to parse code: {e}") from e

        result = ParsedCode()
        code_bytes = bytes(code, "utf8")

        # Extract functions using query with cursor
        cursor = QueryCursor(self.function_query)
        matches = cursor.matches(tree.root_node)

        # Process each match
        for _pattern_index, captures_dict in matches:
            current_func = {}

            # Convert captures dict to our format
            if "function" in captures_dict:
                current_func["function"] = captures_dict["function"][0]
            if "name" in captures_dict:
                current_func["name"] = captures_dict["name"][0]
            if "body" in captures_dict:
                current_func["body"] = captures_dict["body"][0]

            if current_func:
                func_node = self._create_function_node(current_func, code_bytes)
                if func_node:
                    result.functions.append(func_node)

        return result

    def _create_function_node(self, captures: dict, code_bytes: bytes) -> FunctionNode | None:
        """Create FunctionNode from captures.

        Args:
            captures: Dictionary with function, name, and body nodes
            code_bytes: Source code as bytes

        Returns:
            FunctionNode or None if extraction fails
        """
        try:
            func_node = captures["function"]
            name_node = captures["name"]

            # Extract name
            name = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf8")

            # Extract full function body
            body = code_bytes[func_node.start_byte : func_node.end_byte].decode("utf8")

            # Get line numbers
            start_line = func_node.start_point[0] + 1
            end_line = func_node.end_point[0] + 1

            # Extract signature (declarator part)
            # Find the declarator node
            declarator_node = func_node.child_by_field_name("declarator")
            if declarator_node:
                signature = code_bytes[
                    declarator_node.start_byte : declarator_node.end_byte
                ].decode("utf8")
            else:
                signature = name

            # Extract parameters
            params = self._extract_params(declarator_node, code_bytes)

            # Extract return type (everything before the declarator)
            return_type = self._extract_return_type(func_node, declarator_node, code_bytes)

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

    def _extract_params(self, declarator_node, code_bytes: bytes) -> list[str]:
        """Extract parameter names from function declarator.

        Args:
            declarator_node: Function declarator node
            code_bytes: Source code as bytes

        Returns:
            List of parameter names
        """
        params = []
        if not declarator_node:
            return params

        # Find parameter list
        param_list = declarator_node.child_by_field_name("parameters")
        if param_list:
            for child in param_list.children:
                if child.type == "parameter_declaration":
                    # Find identifier in parameter
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            param_name = code_bytes[subchild.start_byte : subchild.end_byte].decode(
                                "utf8"
                            )
                            params.append(param_name)
                            break

        return params

    def _extract_return_type(self, func_node, declarator_node, code_bytes: bytes) -> str:
        """Extract return type from function definition.

        Args:
            func_node: Function definition node
            declarator_node: Function declarator node
            code_bytes: Source code as bytes

        Returns:
            Return type as string
        """
        try:
            # Return type is everything before the declarator
            if declarator_node:
                return_type_end = declarator_node.start_byte
                # Find the first child (usually the type)
                type_node = func_node.children[0]
                if type_node and type_node.end_byte <= return_type_end:
                    return_type = code_bytes[type_node.start_byte : type_node.end_byte].decode(
                        "utf8"
                    )
                    return return_type.strip()
        except Exception:
            pass
        return ""


def create_parser(language: str = "c", use_tree_sitter: bool = True) -> ASTParser:
    """Factory function to create an AST parser.

    Args:
        language: Programming language ('c' or 'cpp')
        use_tree_sitter: Use tree-sitter (default) or pycparser

    Returns:
        ASTParser instance

    Raises:
        ValueError: If language is not supported
    """
    if language == "c":
        if use_tree_sitter:
            return TreeSitterAdapter()
        else:
            return PycparserAdapter()
    elif language == "cpp":
        # C++ support will be added in Phase 2 using libclang
        raise NotImplementedError("C++ parsing will be supported in Phase 2")
    else:
        raise ValueError(f"Unsupported language: {language}")
