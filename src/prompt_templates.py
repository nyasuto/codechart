"""Prompt templates for LLM analysis."""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for code analysis."""

    SYSTEM_PROMPT = """あなたはレガシーC/C++コードの解析専門家です。
提供されたコードを詳細に分析し、コードの動作と役割を理解するための情報を抽出してください。

重要: 改善提案や複雑度の評価は不要です。あくまでコードの理解に必要な情報のみを提供してください。

必ずJSON形式で出力してください。"""

    FUNCTION_ANALYSIS = """以下のC/C++関数の動作を詳細に解析してください。

## コード
```c
{code}
```

## コンテキスト
{context}

## 出力形式（JSON）
以下のJSON形式で、この関数の動作と役割を説明してください。
他のテキストは含めず、JSONのみを出力してください。

{{
  "function_role": "システム全体でのこの関数の役割",
  "behavior": {{
    "normal_case": "正常系の動作",
    "special_cases": ["特殊ケースとその処理（配列形式）"],
    "error_cases": ["エラーケースとその処理（配列形式）"]
  }},
  "data_flow": {{
    "inputs": "入力パラメータとその意味",
    "outputs": "戻り値・出力パラメータ",
    "side_effects": "副作用（状態変更、I/O等）"
  }},
  "call_graph": {{
    "calls": ["この関数が呼び出す関数のリスト"],
    "called_by": ["想定される呼び出し元（推測可能な場合）"]
  }},
  "state_management": "状態管理の方法（グローバル変数、静的変数等）",
  "assumptions": "暗黙の前提条件・仮定",
  "notes": "理解に重要なその他の情報"
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
