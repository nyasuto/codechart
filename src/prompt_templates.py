"""Prompt templates for LLM analysis."""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for code analysis."""

    SYSTEM_PROMPT = """あなたはC/C++コード解析の専門家です。
提供されたコードを詳細に分析し、以下の観点から評価してください：
1. 機能と目的
2. アルゴリズムの複雑度
3. 潜在的なバグやメモリリーク
4. 改善提案

必ずJSON形式で出力してください。"""

    FUNCTION_ANALYSIS = """以下のC/C++関数を解析してください。

## コード
```c
{code}
```

## コンテキスト
{context}

## 出力形式（JSON）
必ず以下のJSON形式で回答してください。他のテキストは含めないでください。

{{
  "summary": "関数の簡潔な説明（1-2文）",
  "purpose": "この関数の目的",
  "algorithm": "使用されているアルゴリズム",
  "complexity": "時間計算量（O記法）",
  "dependencies": ["呼び出している関数のリスト"],
  "potential_issues": ["潜在的な問題点のリスト"],
  "improvements": ["改善提案のリスト"]
}}"""

    @staticmethod
    def format_function_analysis(code: str, context: str = "") -> str:
        """Format function analysis prompt.

        Args:
            code: Source code to analyze
            context: Additional context (type definitions, macros, etc.)

        Returns:
            Formatted prompt
        """
        if not context:
            context = "コンテキストなし"

        return PromptTemplates.FUNCTION_ANALYSIS.format(code=code, context=context)

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt.

        Returns:
            System prompt string
        """
        return PromptTemplates.SYSTEM_PROMPT
