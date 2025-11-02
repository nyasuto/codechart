"""Tests for ast_parser module."""

import pytest

from src.ast_parser import ParseError, PycparserAdapter, create_parser


def test_pycparser_simple_function() -> None:
    """Test parsing a simple function."""
    parser = PycparserAdapter()

    code = """
    int add(int a, int b) {
        return a + b;
    }
    """

    parsed = parser.parse(code)

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert func.name == "add"
    assert "int" in func.signature
    assert "add" in func.signature


def test_pycparser_multiple_functions() -> None:
    """Test parsing multiple functions."""
    parser = PycparserAdapter()

    code = """
    int add(int a, int b) {
        return a + b;
    }

    int subtract(int a, int b) {
        return a - b;
    }
    """

    parsed = parser.parse(code)

    assert len(parsed.functions) == 2
    assert parsed.functions[0].name == "add"
    assert parsed.functions[1].name == "subtract"


def test_pycparser_function_params() -> None:
    """Test extraction of function parameters."""
    parser = PycparserAdapter()

    code = """
    int multiply(int x, int y) {
        return x * y;
    }
    """

    parsed = parser.parse(code)

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert "x" in func.params
    assert "y" in func.params


def test_pycparser_void_function() -> None:
    """Test parsing void function."""
    parser = PycparserAdapter()

    code = """
    void hello() {
        /* Do nothing */
    }
    """

    parsed = parser.parse(code)

    assert len(parsed.functions) == 1
    func = parsed.functions[0]
    assert func.name == "hello"


def test_pycparser_invalid_code() -> None:
    """Test parsing invalid code raises error."""
    parser = PycparserAdapter()

    invalid_code = "this is not valid C code { } [ ]"

    with pytest.raises(ParseError):
        parser.parse(invalid_code)


def test_create_parser_c() -> None:
    """Test creating parser for C."""
    parser = create_parser("c")
    assert isinstance(parser, PycparserAdapter)


def test_create_parser_cpp_not_implemented() -> None:
    """Test C++ parser not implemented yet."""
    with pytest.raises(NotImplementedError):
        create_parser("cpp")


def test_create_parser_invalid_language() -> None:
    """Test invalid language raises error."""
    with pytest.raises(ValueError):
        create_parser("python")
