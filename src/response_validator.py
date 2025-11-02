"""Response validation using Pydantic."""

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class AnalysisResponse(BaseModel):
    """Schema for LLM analysis response."""

    summary: str = Field(..., min_length=1, description="Brief function description")
    purpose: str = Field(..., min_length=1, description="Purpose of the function")
    algorithm: str = Field(..., description="Algorithm used")
    complexity: str = Field(..., description="Time complexity in O-notation")
    dependencies: list[str] = Field(default_factory=list, description="List of called functions")
    potential_issues: list[str] = Field(
        default_factory=list, description="List of potential issues"
    )
    improvements: list[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )


class ResponseValidator:
    """Validator for LLM responses."""

    @staticmethod
    def validate(response_text: str) -> AnalysisResponse:
        """Validate and parse LLM response.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Validated AnalysisResponse object

        Raises:
            ValidationError: If response doesn't match schema
            json.JSONDecodeError: If response is not valid JSON
        """
        # Try to extract JSON from response
        json_data = ResponseValidator._extract_json(response_text)

        # Validate against schema
        return AnalysisResponse(**json_data)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from response text.

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                return json.loads(json_str)

        # Try to find JSON block without fence
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    @staticmethod
    def validate_safe(response_text: str) -> AnalysisResponse | None:
        """Safely validate response, returning None on error.

        Args:
            response_text: Raw response text from LLM

        Returns:
            AnalysisResponse object or None if validation fails
        """
        try:
            return ResponseValidator.validate(response_text)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Warning: Response validation failed: {e}")
            return None
