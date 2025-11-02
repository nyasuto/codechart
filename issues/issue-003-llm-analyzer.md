# Issue #3: GPT-4解析モジュール（llm_analyzer.py）の基本実装

## 概要
GPT-4 APIを使用してコードチャンクを解析し、技術文書の基礎となる情報を抽出する。

## 詳細説明
OpenAI APIクライアントを実装し、以下を実現：
- プロンプトテンプレート管理
- レート制限対応（指数バックオフリトライ）
- JSON形式での構造化レスポンス
- エラーハンドリング

## 受け入れ条件（Acceptance Criteria）
- [ ] `src/llm_analyzer.py`: LLM解析ロジック
- [ ] `src/prompt_templates.py`: プロンプトテンプレート
- [ ] `src/retry_strategy.py`: リトライ機構（o3提案）
- [ ] APIキーの環境変数管理
- [ ] レスポンスのJSON Schemaバリデーション
- [ ] 単体テスト（モックAPI使用、カバレッジ80%以上）

## タスク

### 1. Prompt Templates実装
```python
# src/prompt_templates.py
class PromptTemplate:
    ANALYZE_FUNCTION = """
以下のC/C++関数を解析してください。

## コード
```c
{code}
```

## コンテキスト
{context}

## 出力形式（JSON）
{{
  "summary": "関数の簡潔な説明（1-2文）",
  "purpose": "この関数の目的",
  "algorithm": "使用されているアルゴリズム",
  "complexity": "時間計算量（O記法）",
  "dependencies": ["呼び出している関数のリスト"],
  "potential_issues": ["潜在的な問題点のリスト"],
  "improvements": ["改善提案のリスト"]
}}
"""
```

### 2. LLM Analyzer実装
```python
# src/llm_analyzer.py
@dataclass
class AnalysisResult:
    chunk_id: str
    summary: str
    purpose: str
    algorithm: str
    complexity: str
    dependencies: list[str]
    potential_issues: list[str]
    improvements: list[str]
    raw_response: dict

class LLMAnalyzer:
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.api_key)
        self.model = config.model
        self.temperature = config.temperature

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, Timeout),
        max_tries=5
    )
    def analyze_chunk(self, chunk: CodeChunk) -> AnalysisResult:
        """チャンクを解析"""
```

### 3. Retry Strategy実装
```python
# src/retry_strategy.py
import backoff
from openai import RateLimitError, Timeout

def get_retry_decorator():
    """o3提案: 指数バックオフリトライ"""
    return backoff.on_exception(
        backoff.expo,
        (RateLimitError, Timeout),
        max_tries=5,
        max_time=300,
        on_backoff=log_retry_attempt
    )
```

### 4. レスポンスバリデーション
```python
# src/response_validator.py
from pydantic import BaseModel

class AnalysisSchema(BaseModel):
    summary: str
    purpose: str
    algorithm: str
    complexity: str
    dependencies: list[str]
    potential_issues: list[str]
    improvements: list[str]

def validate_response(response: dict) -> AnalysisSchema:
    """Pydanticでバリデーション"""
    return AnalysisSchema(**response)
```

## 推定工数
3日（24時間）

## 依存関係
- #1: プロジェクト構造とCI/CD環境構築
- #2: コード分割モジュール

## 優先度
P0 (Critical)

## ラベル
- type: feature
- priority: critical
- phase: 1
- epic: 基本パイプライン構築

## 技術的課題

### 1. トークン制限
- システムプロンプト: 約200トークン
- プロンプトテンプレート: 約300トークン
- コード: 最大17,500トークン
- **合計上限: 18,000トークン（余裕を持たせる）**

### 2. レート制限
- 10,000 TPM (tokens per minute)
- 実装: トークンバケットアルゴリズム
- リトライ: 指数バックオフ（1s, 2s, 4s, 8s, 16s）

### 3. コスト管理
```python
class CostTracker:
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """GPT-4 Turboの料金計算"""
        input_cost = input_tokens / 1000 * 0.01
        output_cost = output_tokens / 1000 * 0.03
        return input_cost + output_cost
```

## テストケース
```python
def test_analyze_simple_function(mock_openai):
    """モックAPIでの基本テスト"""
    chunk = create_test_chunk("int add(int a, int b) { return a + b; }")
    result = analyzer.analyze_chunk(chunk)

    assert result.summary != ""
    assert result.complexity != ""

def test_retry_on_rate_limit(mock_openai):
    """レート制限時のリトライ確認"""
    mock_openai.side_effect = [
        RateLimitError("Rate limit"),
        RateLimitError("Rate limit"),
        valid_response
    ]
    result = analyzer.analyze_chunk(chunk)
    assert result is not None
```

## セキュリティ考慮事項
- APIキーは環境変数で管理（`.env`ファイル使用）
- コード内にハードコードしない
- `.gitignore`に`.env`追加

## 参考資料
- OpenAI API Reference: https://platform.openai.com/docs/api-reference
- backoff library: https://github.com/litl/backoff
- Pydantic: https://docs.pydantic.dev/
