"""LLM-based code analyzer."""

import threading
from dataclasses import dataclass

from openai import OpenAI

from src.code_chunker import CodeChunk
from src.config import Config
from src.prompt_templates import PromptTemplates
from src.response_validator import ResponseValidator
from src.retry_strategy import RetryStrategy


@dataclass
class Behavior:
    """Function behavior description."""

    normal_case: str
    special_cases: list[str]
    error_cases: list[str]


@dataclass
class DataFlow:
    """Data flow information."""

    inputs: str
    outputs: str
    side_effects: str


@dataclass
class CallGraph:
    """Call graph information."""

    calls: list[str]
    called_by: list[str]


@dataclass
class AnalysisResult:
    """Result of code analysis."""

    chunk_id: str
    chunk_name: str
    function_role: str
    behavior: Behavior
    data_flow: DataFlow
    call_graph: CallGraph
    state_management: str
    assumptions: str
    notes: str
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

        # 複数モデルのラウンドロビン用カウンター
        self.models = config.llm.models
        self.model_index = 0
        self.model_lock = threading.Lock()  # スレッドセーフなモデル選択用

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
                function_role="Analysis failed: " + str(e),
                behavior=Behavior(
                    normal_case="Unknown",
                    special_cases=[],
                    error_cases=[],
                ),
                data_flow=DataFlow(
                    inputs="Unknown",
                    outputs="Unknown",
                    side_effects="Unknown",
                ),
                call_graph=CallGraph(
                    calls=[],
                    called_by=[],
                ),
                state_management="Unknown",
                assumptions="Analysis validation failed",
                notes="",
                raw_response=response_text,
                tokens_used=tokens_used,
            )

        # Track tokens
        self.total_tokens_used += tokens_used

        return AnalysisResult(
            chunk_id=chunk.id,
            chunk_name=chunk.name,
            function_role=validated.function_role,
            behavior=Behavior(
                normal_case=validated.behavior.normal_case,
                special_cases=validated.behavior.special_cases,
                error_cases=validated.behavior.error_cases,
            ),
            data_flow=DataFlow(
                inputs=validated.data_flow.inputs,
                outputs=validated.data_flow.outputs,
                side_effects=validated.data_flow.side_effects,
            ),
            call_graph=CallGraph(
                calls=validated.call_graph.calls,
                called_by=validated.call_graph.called_by,
            ),
            state_management=validated.state_management,
            assumptions=validated.assumptions,
            notes=validated.notes,
            raw_response=response_text,
            tokens_used=tokens_used,
        )

    def _get_next_model(self) -> str:
        """Get next model in round-robin fashion (thread-safe).

        Returns:
            Model name for the next request
        """
        with self.model_lock:
            model = self.models[self.model_index % len(self.models)]
            self.model_index += 1
            print(f"[LLM] Using model instance: {model} (request #{self.model_index})")
        return model

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
            model=self._get_next_model(),
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
