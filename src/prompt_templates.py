"""Prompt templates for LLM analysis."""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for code analysis."""

    SYSTEM_PROMPT = """あなたはレガシーC/C++コードの解析専門家です。
提供されたコードを詳細に分析し、コードの動作と役割を理解するための情報を抽出してください。

重要: 改善提案や複雑度の評価は不要です。あくまでコードの理解に必要な情報のみを提供してください。

出力形式の厳格なルール:
- 必ず1つのJSON objectのみを出力してください
- JSON以外のテキスト（説明、コメント、マークダウン等）は一切含めないでください
- コードブロック（```json など）も使用しないでください
- 回答の最初の文字は必ず "{" で、最後の文字は必ず "}" です
- 完成前に、出力が有効なJSONであることを自己検証してください"""

    FUNCTION_ANALYSIS = """以下のC/C++関数の動作を詳細に解析してください。

## コード
```c
{code}
```

## コンテキスト
{context}

## 出力形式（JSON）
以下のJSON形式で、この関数の動作と役割を説明してください。
**重要**: 他のテキストは含めず、JSONのみを出力してください。

### 型の注意事項
- `function_role`, `state_management`, `assumptions`, `notes` → **文字列型**
- `normal_case`, `inputs`, `outputs`, `side_effects` → **文字列型**
- `special_cases`, `error_cases`, `calls`, `called_by` → **文字列の配列**（各要素が文字列）

### JSON構造
{{
  "function_role": "システム全体でのこの関数の役割",
  "behavior": {{
    "normal_case": "正常系の動作",
    "special_cases": ["特殊ケース1の説明", "特殊ケース2の説明"],
    "error_cases": ["エラーケース1の説明", "エラーケース2の説明"]
  }},
  "data_flow": {{
    "inputs": "入力パラメータとその意味（文字列で記述）",
    "outputs": "戻り値・出力パラメータ（文字列で記述）",
    "side_effects": "副作用の説明、または「なし」"
  }},
  "call_graph": {{
    "calls": ["呼び出す関数1", "呼び出す関数2"],
    "called_by": ["想定される呼び出し元1", "想定される呼び出し元2"]
  }},
  "state_management": "状態管理の方法（グローバル変数、静的変数等）、または「なし」",
  "assumptions": "暗黙の前提条件・仮定",
  "notes": "理解に重要なその他の情報"
}}

### 良い例
{{
  "function_role": "2つの整数を加算して結果を返す",
  "behavior": {{
    "normal_case": "引数aとbを加算し、その結果を返す",
    "special_cases": [
      "a または b が INT_MAX に近い場合、オーバーフローの可能性がある",
      "負の数の加算も正しく処理される"
    ],
    "error_cases": []
  }},
  "data_flow": {{
    "inputs": "a, b: 加算する2つの整数（int型）",
    "outputs": "戻り値: a + b の計算結果（int型）",
    "side_effects": "なし"
  }},
  "call_graph": {{
    "calls": [],
    "called_by": []
  }},
  "state_management": "なし",
  "assumptions": "int型のオーバーフローチェックは呼び出し側で行う",
  "notes": "単純な算術演算関数"
}}

### 避けるべき例（悪い例）
{{
  "data_flow": {{
    "inputs": [{{"name": "a", "type": "int"}}],  // ❌ 辞書の配列ではなく文字列で
    "outputs": {{"name": "result"}},             // ❌ 辞書ではなく文字列で
    "side_effects": []                           // ❌ 空配列ではなく「なし」という文字列
  }},
  "behavior": {{
    "special_cases": [
      ["a = 0", "結果は b"]  // ❌ 配列の中に配列を入れない。1つの文字列にまとめる
    ]
  }}
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
