"""Tests for response_validator module."""

import json

import pytest
from pydantic import ValidationError

from src.response_validator import (
    AnalysisResponse,
    BehaviorSchema,
    CallGraphSchema,
    DataFlowSchema,
    ResponseValidator,
)


def test_analysis_response_creation() -> None:
    """Test AnalysisResponse model creation."""
    response = AnalysisResponse(
        function_role="Test function role",
        behavior=BehaviorSchema(
            normal_case="Normal behavior",
            special_cases=[],
            error_cases=[],
        ),
        data_flow=DataFlowSchema(
            inputs="Test inputs",
            outputs="Test outputs",
            side_effects="None",
        ),
        call_graph=CallGraphSchema(
            calls=["func1", "func2"],
            called_by=[],
        ),
        state_management="No state",
        assumptions="None",
        notes="Test notes",
    )

    assert response.function_role == "Test function role"
    assert len(response.call_graph.calls) == 2


def test_validate_valid_json() -> None:
    """Test validation of valid JSON response."""
    response_text = """
    {
        "function_role": "Test function",
        "behavior": {
            "normal_case": "Normal operation",
            "special_cases": [],
            "error_cases": []
        },
        "data_flow": {
            "inputs": "Test input",
            "outputs": "Test output",
            "side_effects": "None"
        },
        "call_graph": {
            "calls": ["helper"],
            "called_by": []
        },
        "state_management": "No state",
        "assumptions": "None",
        "notes": ""
    }
    """

    result = ResponseValidator.validate(response_text)

    assert result.function_role == "Test function"
    assert result.behavior.normal_case == "Normal operation"
    assert len(result.call_graph.calls) == 1


def test_validate_json_with_markdown_fence() -> None:
    """Test validation of JSON within markdown code fence."""
    response_text = """
    Some text before
    ```json
    {
        "function_role": "Fenced function",
        "behavior": {"normal_case": "Normal", "special_cases": [], "error_cases": []},
        "data_flow": {"inputs": "In", "outputs": "Out", "side_effects": "None"},
        "call_graph": {"calls": [], "called_by": []},
        "state_management": "None",
        "assumptions": "None",
        "notes": ""
    }
    ```
    Some text after
    """

    result = ResponseValidator.validate(response_text)
    assert result.function_role == "Fenced function"


def test_validate_json_embedded_in_text() -> None:
    """Test validation of JSON embedded in text."""
    response_text = """
    Here is the analysis:
    {
        "function_role": "Embedded function",
        "behavior": {"normal_case": "Normal", "special_cases": [], "error_cases": []},
        "data_flow": {"inputs": "In", "outputs": "Out", "side_effects": "None"},
        "call_graph": {"calls": [], "called_by": []},
        "state_management": "None",
        "assumptions": "None",
        "notes": ""
    }
    That's it!
    """

    result = ResponseValidator.validate(response_text)
    assert result.function_role == "Embedded function"


def test_validate_invalid_json() -> None:
    """Test validation with invalid JSON."""
    response_text = "This is not JSON at all"

    with pytest.raises(json.JSONDecodeError):
        ResponseValidator.validate(response_text)


def test_validate_missing_required_fields() -> None:
    """Test validation with missing required fields."""
    response_text = """
    {
        "function_role": "Incomplete"
    }
    """

    with pytest.raises(ValidationError):
        ResponseValidator.validate(response_text)


def test_validate_safe_success() -> None:
    """Test safe validation with valid response."""
    response_text = """
    {
        "function_role": "Safe function",
        "behavior": {"normal_case": "Normal", "special_cases": [], "error_cases": []},
        "data_flow": {"inputs": "In", "outputs": "Out", "side_effects": "None"},
        "call_graph": {"calls": [], "called_by": []},
        "state_management": "None",
        "assumptions": "None",
        "notes": ""
    }
    """

    result = ResponseValidator.validate_safe(response_text)

    assert result is not None
    assert result.function_role == "Safe function"


def test_validate_safe_failure() -> None:
    """Test safe validation with invalid response."""
    response_text = "Not valid JSON"

    result = ResponseValidator.validate_safe(response_text)

    assert result is None
