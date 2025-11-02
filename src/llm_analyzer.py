"""LLM-based code analyzer."""

from dataclasses import dataclass

from openai import OpenAI

from src.code_chunker import CodeChunk
from src.config import Config
from src.prompt_templates import PromptTemplates
from src.response_validator import ResponseValidator
from src.retry_strategy import RetryStrategy


@dataclass
class AnalysisResult:
    """Result of code analysis."""

    chunk_id: str
    chunk_name: str
    summary: str
    purpose: str
    algorithm: str
    complexity: str
    dependencies: list[str]
    potential_issues: list[str]
    improvements: list[str]
    raw_response: str
    tokens_used: int


class LLMAnalyzer:
    """Analyzer using LLM for code analysis."""

    def __init__(self, config: Config):
        """Initialize LLM analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
        self.retry_strategy = RetryStrategy(config.retry)
        self.validator = ResponseValidator()
        self.total_tokens_used = 0

    def analyze_chunk(self, chunk: CodeChunk, context: str = "") -> AnalysisResult:
        """Analyze a code chunk.

        Args:
            chunk: Code chunk to analyze
            context: Additional context

        Returns:
            Analysis result

        Raises:
            Exception: If analysis fails after retries
        """
        # Format prompt
        user_prompt = PromptTemplates.format_function_analysis(chunk.code, context)

        # Call LLM with retry
        retry_decorator = self.retry_strategy.get_decorator()
        call_llm = retry_decorator(self._call_llm)

        response_text, tokens_used = call_llm(user_prompt)

        # Validate response
        try:
            validated = self.validator.validate(response_text)
        except Exception as e:
            print(f"Warning: Validation failed for chunk {chunk.id}: {e}")
            # Return partial result
            return AnalysisResult(
                chunk_id=chunk.id,
                chunk_name=chunk.name,
                summary="Analysis failed: " + str(e),
                purpose="Unknown",
                algorithm="Unknown",
                complexity="Unknown",
                dependencies=[],
                potential_issues=["Analysis validation failed"],
                improvements=[],
                raw_response=response_text,
                tokens_used=tokens_used,
            )

        # Track tokens
        self.total_tokens_used += tokens_used

        return AnalysisResult(
            chunk_id=chunk.id,
            chunk_name=chunk.name,
            summary=validated.summary,
            purpose=validated.purpose,
            algorithm=validated.algorithm,
            complexity=validated.complexity,
            dependencies=validated.dependencies,
            potential_issues=validated.potential_issues,
            improvements=validated.improvements,
            raw_response=response_text,
            tokens_used=tokens_used,
        )

    def _call_llm(self, user_prompt: str) -> tuple[str, int]:
        """Call LLM API.

        Args:
            user_prompt: User prompt

        Returns:
            Tuple of (response text, tokens used)

        Raises:
            OpenAI API exceptions on error
        """
        response = self.client.chat.completions.create(
            model=self.config.llm.model,
            messages=[
                {"role": "system", "content": PromptTemplates.get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=self.config.llm.timeout,
        )

        response_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        return response_text, tokens_used

    def get_total_tokens(self) -> int:
        """Get total tokens used.

        Returns:
            Total tokens used
        """
        return self.total_tokens_used

    def reset_token_count(self) -> None:
        """Reset token counter."""
        self.total_tokens_used = 0
