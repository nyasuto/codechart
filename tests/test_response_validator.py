"""Tests for response_validator module."""

import json

import pytest
from pydantic import ValidationError

from src.response_validator import AnalysisResponse, ResponseValidator


def test_analysis_response_creation() -> None:
    """Test AnalysisResponse model creation."""
    response = AnalysisResponse(
        summary="Test summary",
        purpose="Test purpose",
        algorithm="Test algorithm",
        complexity="O(1)",
        dependencies=["func1", "func2"],
        potential_issues=["issue1"],
        improvements=["improvement1"],
    )

    assert response.summary == "Test summary"
    assert len(response.dependencies) == 2
    assert response.complexity == "O(1)"


def test_validate_valid_json() -> None:
    """Test validation of valid JSON response."""
    response_text = """
    {
        "summary": "Test function",
        "purpose": "Testing",
        "algorithm": "Simple loop",
        "complexity": "O(n)",
        "dependencies": ["helper"],
        "potential_issues": [],
        "improvements": ["Add docs"]
    }
    """

    result = ResponseValidator.validate(response_text)

    assert result.summary == "Test function"
    assert result.complexity == "O(n)"
    assert len(result.dependencies) == 1


def test_validate_json_with_markdown_fence() -> None:
    """Test validation of JSON within markdown code fence."""
    response_text = """
    Here is the analysis:
    ```json
    {
        "summary": "Test function",
        "purpose": "Testing",
        "algorithm": "Simple",
        "complexity": "O(1)",
        "dependencies": [],
        "potential_issues": [],
        "improvements": []
    }
    ```
    """

    result = ResponseValidator.validate(response_text)

    assert result.summary == "Test function"
    assert result.complexity == "O(1)"


def test_validate_json_embedded_in_text() -> None:
    """Test validation of JSON embedded in text."""
    response_text = """
    Some text before
    {
        "summary": "Test",
        "purpose": "Testing",
        "algorithm": "Simple",
        "complexity": "O(1)",
        "dependencies": [],
        "potential_issues": [],
        "improvements": []
    }
    Some text after
    """

    result = ResponseValidator.validate(response_text)

    assert result.summary == "Test"


def test_validate_invalid_json() -> None:
    """Test validation failure with invalid JSON."""
    response_text = "This is not JSON at all"

    with pytest.raises(json.JSONDecodeError):
        ResponseValidator.validate(response_text)


def test_validate_missing_required_fields() -> None:
    """Test validation failure with missing required fields."""
    response_text = """
    {
        "summary": "Test"
    }
    """

    with pytest.raises(ValidationError):
        ResponseValidator.validate(response_text)


def test_validate_safe_success() -> None:
    """Test safe validation with valid response."""
    response_text = """
    {
        "summary": "Test",
        "purpose": "Testing",
        "algorithm": "Simple",
        "complexity": "O(1)",
        "dependencies": [],
        "potential_issues": [],
        "improvements": []
    }
    """

    result = ResponseValidator.validate_safe(response_text)

    assert result is not None
    assert result.summary == "Test"


def test_validate_safe_failure() -> None:
    """Test safe validation with invalid response."""
    response_text = "Not valid JSON"

    result = ResponseValidator.validate_safe(response_text)

    assert result is None
