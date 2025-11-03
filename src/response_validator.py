"""Response validation using Pydantic."""

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class BehaviorSchema(BaseModel):
    """Schema for function behavior."""

    normal_case: str = Field(..., min_length=1, description="Normal case behavior")
    special_cases: list[str] = Field(default_factory=list, description="Special cases")
    error_cases: list[str] = Field(default_factory=list, description="Error cases")


class DataFlowSchema(BaseModel):
    """Schema for data flow."""

    inputs: str = Field(..., description="Input parameters and their meaning")
    outputs: str = Field(..., description="Return values and output parameters")
    side_effects: str = Field(..., description="Side effects")


class CallGraphSchema(BaseModel):
    """Schema for call graph."""

    calls: list[str] = Field(default_factory=list, description="Functions called by this function")
    called_by: list[str] = Field(
        default_factory=list, description="Expected callers of this function"
    )


class AnalysisResponse(BaseModel):
    """Schema for LLM analysis response."""

    function_role: str = Field(..., min_length=1, description="Role of this function in the system")
    behavior: BehaviorSchema = Field(..., description="Function behavior")
    data_flow: DataFlowSchema = Field(..., description="Data flow information")
    call_graph: CallGraphSchema = Field(..., description="Call graph information")
    state_management: str = Field(..., description="State management approach")
    assumptions: str = Field(..., description="Implicit assumptions")
    notes: str = Field(default="", description="Additional important information")


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
        # Try to parse directly (fast path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract first complete JSON object using raw_decode
        # This handles cases where LLM adds extra text after JSON
        decoder = json.JSONDecoder()
        text = text.strip()

        # Find first {
        start_idx = text.find("{")
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found in response", text, 0)

        try:
            # raw_decode returns (obj, end_index)
            obj, end_idx = decoder.raw_decode(text, start_idx)
            return obj
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    obj, _ = decoder.raw_decode(json_str)
                    return obj
                except json.JSONDecodeError:
                    pass

        # Last resort: extract from first { to last }
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

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
