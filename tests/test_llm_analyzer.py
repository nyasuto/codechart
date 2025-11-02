"""Tests for llm_analyzer module."""

from unittest.mock import MagicMock, patch

from src.code_chunker import CodeChunk
from src.config import Config
from src.llm_analyzer import AnalysisResult, LLMAnalyzer


def test_analysis_result_creation() -> None:
    """Test AnalysisResult dataclass creation."""
    result = AnalysisResult(
        chunk_id="test123",
        chunk_name="test_func",
        summary="Test function",
        purpose="Testing",
        algorithm="Simple loop",
        complexity="O(n)",
        dependencies=["helper_func"],
        potential_issues=["None"],
        improvements=["Add error handling"],
        raw_response='{"summary": "test"}',
        tokens_used=100,
    )

    assert result.chunk_id == "test123"
    assert result.complexity == "O(n)"
    assert len(result.dependencies) == 1


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
        "summary": "This is a test function",
        "purpose": "To demonstrate testing",
        "algorithm": "Simple addition",
        "complexity": "O(1)",
        "dependencies": [],
        "potential_issues": [],
        "improvements": ["Add input validation"]
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
    assert result.summary == "This is a test function"
    assert result.complexity == "O(1)"
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
    assert "Analysis failed" in result.summary
    assert result.complexity == "Unknown"
    assert "Analysis validation failed" in result.potential_issues


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
