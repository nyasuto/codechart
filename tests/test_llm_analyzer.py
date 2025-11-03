"""Tests for llm_analyzer module."""

from unittest.mock import MagicMock, patch

from src.code_chunker import CodeChunk
from src.config import Config
from src.llm_analyzer import AnalysisResult, Behavior, CallGraph, DataFlow, LLMAnalyzer


def test_analysis_result_creation() -> None:
    """Test AnalysisResult dataclass creation."""
    result = AnalysisResult(
        chunk_id="test123",
        chunk_name="test_func",
        function_role="Test function for demonstration",
        behavior=Behavior(
            normal_case="Returns sum of two integers",
            special_cases=[],
            error_cases=[],
        ),
        data_flow=DataFlow(
            inputs="Two integer parameters a and b",
            outputs="Integer sum",
            side_effects="None",
        ),
        call_graph=CallGraph(
            calls=["helper_func"],
            called_by=[],
        ),
        state_management="No state",
        assumptions="Input integers are valid",
        notes="Simple addition function",
        raw_response='{"function_role": "test"}',
        tokens_used=100,
    )

    assert result.chunk_id == "test123"
    assert result.function_role == "Test function for demonstration"
    assert len(result.call_graph.calls) == 1


def test_llm_analyzer_initialization() -> None:
    """Test LLMAnalyzer initialization."""
    config = Config.from_yaml()
    analyzer = LLMAnalyzer(config)

    assert analyzer.config == config
    assert analyzer.total_tokens_used == 0


@patch("src.llm_analyzer.OpenAI")
def test_analyze_chunk_success(mock_openai_class: MagicMock) -> None:
    """Test successful chunk analysis with mocked OpenAI."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "function_role": "Addition function for testing",
        "behavior": {
            "normal_case": "Adds two numbers",
            "special_cases": [],
            "error_cases": []
        },
        "data_flow": {
            "inputs": "Two integers a and b",
            "outputs": "Integer result",
            "side_effects": "None"
        },
        "call_graph": {
            "calls": [],
            "called_by": ["main"]
        },
        "state_management": "No state",
        "assumptions": "Valid integer inputs",
        "notes": "Simple function"
    }
    """
    mock_response.usage.total_tokens = 150

    mock_client.chat.completions.create.return_value = mock_response

    # Create analyzer and test
    config = Config.from_yaml()
    analyzer = LLMAnalyzer(config)

    chunk = CodeChunk(
        id="test_chunk",
        type="function",
        name="test_func",
        code="int add(int a, int b) { return a + b; }",
        tokens=20,
    )

    result = analyzer.analyze_chunk(chunk)

    assert result.chunk_id == "test_chunk"
    assert result.function_role == "Addition function for testing"
    assert result.data_flow.inputs == "Two integers a and b"
    assert result.tokens_used == 150
    assert analyzer.get_total_tokens() == 150


@patch("src.llm_analyzer.OpenAI")
def test_analyze_chunk_validation_failure(mock_openai_class: MagicMock) -> None:
    """Test chunk analysis with invalid JSON response."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is not valid JSON"
    mock_response.usage.total_tokens = 50

    mock_client.chat.completions.create.return_value = mock_response

    # Create analyzer and test
    config = Config.from_yaml()
    analyzer = LLMAnalyzer(config)

    chunk = CodeChunk(
        id="test_chunk",
        type="function",
        name="test_func",
        code="int add(int a, int b) { return a + b; }",
        tokens=20,
    )

    result = analyzer.analyze_chunk(chunk)

    # Should return partial result with error message
    assert "【解析失敗】" in result.function_role
    assert result.data_flow.inputs == "【解析失敗】"
    assert "【解析失敗】" in result.assumptions


def test_token_counting() -> None:
    """Test token counting functionality."""
    config = Config.from_yaml()
    analyzer = LLMAnalyzer(config)

    assert analyzer.get_total_tokens() == 0

    # Manually set tokens (in real test would be from API call)
    analyzer.total_tokens_used = 100
    assert analyzer.get_total_tokens() == 100

    analyzer.reset_token_count()
    assert analyzer.get_total_tokens() == 0
