# システムアーキテクチャ設計書

## バージョン: 1.0
## 更新日: 2025-11-02

---

## 1. システム概要

### 1.1 アーキテクチャ原則

**設計原則:**
1. **モジュール性**: 各コンポーネントは独立して開発・テスト可能
2. **拡張性**: ベクトルDB・パーサーの交換が容易
3. **信頼性**: エラーハンドリングとリトライ機構の徹底
4. **可観測性**: ログ・メトリクスによるボトルネック特定
5. **コスト効率**: トークン使用量の最小化とキャッシュ活用

**o3フィードバック反映項目:**
- VectorStore抽象化インターフェース（Phase 1から）
- リトライ・エラーハンドリングの明示的実装
- 設定管理の外部化（YAML/TOML）
- プロンプト評価フレームワーク
- 増分キャッシュ機構

---

## 2. システム全体像

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                          │
│                     (src/cli.py, src/api.py)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Orchestrator                                │
│                 (src/orchestrator.py)                           │
│  • パイプライン制御  • 並列処理  • 進捗管理                      │
└─┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────┐
│Code │ │AST  │ │LLM  │ │RAG  │ │ Output   │
│Load │ │Parse│ │     │ │     │ │ Format   │
└─────┘ └─────┘ └─────┘ └─────┘ └──────────┘
   │        │        │        │         │
   ▼        ▼        ▼        ▼         ▼
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌─────────┐
│File│ │AST │ │GPT-│ │Vec- │ │JSON/MD  │
│Sys │ │Tree│ │ 4  │ │torDB│ │ Report  │
└────┘ └────┘ └────┘ └────┘ └─────────┘
```

---

## 3. コンポーネント詳細設計

### 3.1 Code Loader（コードローダー）

**責務:**
- ソースファイルの検出と読み込み
- ファイル種別判定（C/C++/ヘッダ）
- エンコーディング処理

**インターフェース:**
```python
# src/code_loader.py
from pathlib import Path
from typing import Iterator

class CodeFile:
    path: Path
    content: str
    language: str  # 'c' or 'cpp'
    encoding: str
    hash: str      # SHA-256 for caching

class CodeLoader:
    def __init__(self, config: Config):
        self.config = config

    def discover_files(self, root_path: Path) -> list[Path]:
        """指定ディレクトリ配下のC/C++ファイルを検出"""
        pass

    def load_file(self, file_path: Path) -> CodeFile:
        """ファイルを読み込みCodeFileオブジェクトを返す"""
        pass

    def load_batch(self, file_paths: list[Path]) -> Iterator[CodeFile]:
        """複数ファイルをバッチで読み込み（並列対応）"""
        pass
```

**実装詳細:**
- `pathlib`でクロスプラットフォーム対応
- `chardet`でエンコーディング自動検出
- `.gitignore`形式の除外ルール対応
- ファイルハッシュでキャッシュ管理

---

### 3.2 AST Parser（構文解析）

**責務:**
- C/C++コードのAST生成
- 関数・構造体・マクロの抽出
- 依存関係グラフ構築

**インターフェース:**
```python
# src/ast_parser.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FunctionNode:
    name: str
    signature: str
    body: str
    start_line: int
    end_line: int
    dependencies: list[str]  # 呼び出している関数
    complexity: int          # Cyclomatic Complexity

@dataclass
class ParsedCode:
    functions: list[FunctionNode]
    structs: list[dict]
    macros: list[dict]
    includes: list[str]
    global_vars: list[dict]

class ASTParser(ABC):
    @abstractmethod
    def parse(self, code: str) -> ParsedCode:
        """コードを解析してAST情報を返す"""
        pass

class PycparserAdapter(ASTParser):
    """Phase 1: pycparser実装"""
    def parse(self, code: str) -> ParsedCode:
        # pycparserロジック
        pass

class LibclangAdapter(ASTParser):
    """Phase 2: libclang実装"""
    def parse(self, code: str) -> ParsedCode:
        # libclangロジック
        pass

class ParserFactory:
    @staticmethod
    def create(language: str, phase: int) -> ASTParser:
        """言語とPhaseに応じてパーサーを選択"""
        if phase == 1 or language == 'c':
            return PycparserAdapter()
        else:
            return LibclangAdapter()
```

---

### 3.3 Code Chunker（コード分割）

**責務:**
- 関数単位でのチャンク生成
- トークン数計算と制限管理
- コンテキスト情報の付与

**インターフェース:**
```python
# src/code_chunker.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class CodeChunk:
    id: str               # 一意ID（SHA-256）
    type: str             # 'function', 'struct', 'header'
    name: str             # 関数名など
    code: str             # コード本体
    context: str          # 追加コンテキスト（ヘッダ定義等）
    tokens: int           # トークン数
    metadata: dict        # ファイルパス、行番号等

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class CodeChunker:
    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 18000  # バッファ考慮
    ):
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def chunk_functions(
        self,
        parsed: ParsedCode,
        strategy: str = 'hybrid'  # 'function', 'block', 'hybrid'
    ) -> list[CodeChunk]:
        """関数リストをチャンクに分割"""
        pass

    def add_context(
        self,
        chunk: CodeChunk,
        parsed: ParsedCode
    ) -> CodeChunk:
        """チャンクに関連する型定義・マクロを追加"""
        # o3提案: ヘッダコンテキスト追加
        pass

    def create_overlap_window(
        self,
        chunks: list[CodeChunk],
        overlap_ratio: float = 0.25
    ) -> list[CodeChunk]:
        """隣接チャンクとのオーバーラップ生成"""
        pass
```

**トークンカウンター実装:**
```python
# src/token_counter.py
import tiktoken
from functools import lru_cache

class TiktokenCounter:
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)

    @lru_cache(maxsize=2048)
    def count(self, text: str) -> int:
        """キャッシュ付きトークン計算"""
        return len(self.encoding.encode(text))

class EstimatedCounter:
    """高速概算（C/C++コードは平均0.4トークン/文字）"""
    def count(self, text: str) -> int:
        return int(len(text) * 0.4)
```

---

### 3.4 LLM Analyzer（LLM解析）

**責務:**
- GPT-4 APIへのリクエスト送信
- プロンプトテンプレート管理
- レート制限・リトライ処理
- レスポンス検証

**インターフェース:**
```python
# src/llm_analyzer.py
from typing import Optional
import backoff
from openai import OpenAI, RateLimitError

@dataclass
class AnalysisResult:
    chunk_id: str
    summary: str          # 関数の要約
    purpose: str          # 目的
    complexity: str       # 複雑度評価
    dependencies: list[str]
    issues: list[str]     # 潜在的問題
    suggestions: list[str]
    raw_response: dict

class PromptTemplate:
    ANALYZE_FUNCTION = """
以下のC/C++関数を解析してください。

## コード
```c
{code}
```

## コンテキスト
{context}

## 解析項目
1. この関数の目的と役割
2. アルゴリズムの複雑度
3. 潜在的なバグやメモリリーク
4. 改善提案

JSON形式で回答してください。
"""

class LLMAnalyzer:
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.api_key)
        self.model = config.model
        self.temperature = config.temperature

    @backoff.on_exception(
        backoff.expo,
        RateLimitError,
        max_tries=5,
        max_time=300
    )
    def analyze_chunk(
        self,
        chunk: CodeChunk,
        prompt_template: str = PromptTemplate.ANALYZE_FUNCTION
    ) -> AnalysisResult:
        """チャンクを解析"""
        prompt = prompt_template.format(
            code=chunk.code,
            context=chunk.context
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "C/C++コード解析の専門家"},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        return self._parse_response(response, chunk.id)

    def batch_analyze(
        self,
        chunks: list[CodeChunk],
        max_workers: int = 5
    ) -> list[AnalysisResult]:
        """複数チャンクを並列解析（asyncio使用）"""
        pass
```

---

### 3.5 RAG Manager（検索拡張生成）

**責務:**
- ベクトルDBへのチャンク登録
- 類似コード検索
- コンテキスト拡張

**インターフェース（抽象化）:**
```python
# src/vector_store.py
from abc import ABC, abstractmethod
from typing import Protocol

class VectorStore(Protocol):
    """ベクトルDB抽象化インターフェース（o3提案）"""

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """チャンクとベクトルを登録"""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> list[tuple[CodeChunk, float]]:
        """類似検索（チャンクとスコアを返す）"""
        ...

    def delete_by_file(self, file_path: str) -> None:
        """ファイル単位で削除（増分更新用）"""
        ...

# FAISS実装（Phase 1から使用）
class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int = 1536):
        import faiss
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_chunk: dict[int, CodeChunk] = {}

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        import numpy as np
        vectors = np.array(embeddings).astype('float32')
        start_id = len(self.id_to_chunk)
        self.index.add(vectors)
        for i, chunk in enumerate(chunks):
            self.id_to_chunk[start_id + i] = chunk

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> list[tuple[CodeChunk, float]]:
        import numpy as np
        query_vec = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.id_to_chunk:
                results.append((self.id_to_chunk[idx], float(dist)))
        return results

# ChromaDB実装（テスト用のみ）
class ChromaDBVectorStore(VectorStore):
    """単体テスト用の軽量実装"""
    pass

# RAG Manager
class RAGManager:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.vector_store = vector_store
        self.client = OpenAI()
        self.embedding_model = embedding_model

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """チャンクをベクトル化"""
        texts = [f"{c.name}\n{c.code}" for c in chunks]
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [r.embedding for r in response.data]

    def index_code(self, chunks: list[CodeChunk]) -> None:
        """コードをインデックス化"""
        embeddings = self.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)

    def find_similar(
        self,
        query: str,
        top_k: int = 5
    ) -> list[CodeChunk]:
        """類似コード検索"""
        query_embedding = self.embed_chunks([
            CodeChunk(id='query', type='query', name='', code=query,
                     context='', tokens=0, metadata={})
        ])[0]

        results = self.vector_store.search(query_embedding, top_k)
        return [chunk for chunk, _ in results]
```

---

### 3.6 Cache Manager（キャッシュ管理）

**責務（o3提案による追加）:**
- 解析済みチャンクのキャッシュ
- ファイルハッシュベースの増分更新
- コスト削減

**インターフェース:**
```python
# src/cache_manager.py
import hashlib
import json
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_file_hash(self, file_path: Path) -> str:
        """ファイルのSHA-256ハッシュ計算"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def is_cached(self, file_hash: str) -> bool:
        """キャッシュが存在するか確認"""
        cache_file = self.cache_dir / f"{file_hash}.json"
        return cache_file.exists()

    def load_cache(self, file_hash: str) -> Optional[list[AnalysisResult]]:
        """キャッシュから解析結果を読み込み"""
        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def save_cache(self, file_hash: str, results: list[AnalysisResult]) -> None:
        """解析結果をキャッシュに保存"""
        cache_file = self.cache_dir / f"{file_hash}.json"
        with open(cache_file, 'w') as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
```

---

### 3.7 Orchestrator（オーケストレーター）

**責務:**
- パイプライン全体の制御
- 並列処理の管理
- 進捗報告

**インターフェース:**
```python
# src/orchestrator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class PipelineStats:
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    cached_chunks: int = 0
    api_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.loader = CodeLoader(config)
        self.parser = ParserFactory.create(config.language, config.phase)
        self.chunker = CodeChunker(TiktokenCounter(), config.max_tokens)
        self.llm = LLMAnalyzer(config)
        self.rag = RAGManager(FAISSVectorStore())
        self.cache = CacheManager(config.cache_dir)
        self.stats = PipelineStats()

    def run(self, root_path: Path) -> dict:
        """パイプライン実行"""
        # 1. ファイル検出
        files = self.loader.discover_files(root_path)
        self.stats.total_files = len(files)

        all_results = []

        # 2. 並列処理
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_file, f): f
                for f in files
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                results = future.result()
                all_results.extend(results)

        # 3. RAGインデックス構築
        if self.config.enable_rag:
            chunks = self._results_to_chunks(all_results)
            self.rag.index_code(chunks)

        return {
            'results': all_results,
            'stats': self.stats.__dict__
        }

    def _process_file(self, file_path: Path) -> list[AnalysisResult]:
        """単一ファイルの処理"""
        # キャッシュチェック
        file_hash = self.cache.get_file_hash(file_path)
        if cached := self.cache.load_cache(file_hash):
            self.stats.cached_chunks += len(cached)
            return cached

        # 新規処理
        code_file = self.loader.load_file(file_path)
        parsed = self.parser.parse(code_file.content)
        chunks = self.chunker.chunk_functions(parsed)

        results = []
        for chunk in chunks:
            result = self.llm.analyze_chunk(chunk)
            results.append(result)
            self.stats.api_calls += 1
            self.stats.total_tokens += chunk.tokens

        # キャッシュ保存
        self.cache.save_cache(file_hash, results)
        return results
```

---

## 4. データフロー

### 4.1 標準処理フロー

```
[ソースコード] → [Code Loader]
                      ↓
                 [AST Parser]
                      ↓
                 [Code Chunker] → [Token Counter]
                      ↓
                 [Cache Check] → (Hit) → [結果返却]
                      ↓ (Miss)
                 [LLM Analyzer] → [GPT-4 API]
                      ↓
                 [Cache Save]
                      ↓
                 [RAG Manager] → [Vector DB]
                      ↓
                 [Output Formatter] → [JSON/Markdown]
```

### 4.2 増分更新フロー

```
[変更ファイル検出] → [ハッシュ計算]
                         ↓
                    [差分抽出]
                         ↓
                    [旧キャッシュ削除]
                         ↓
                    [再解析]
                         ↓
                    [Vector DB更新]
```

---

## 5. 設定管理

### 5.1 設定ファイル（config/default.yaml）

```yaml
# o3提案: 設定の外部化
api:
  provider: openai
  model: gpt-4-turbo-preview
  api_key_env: OPENAI_API_KEY
  max_tokens: 18000
  temperature: 0.2
  timeout: 60

parser:
  phase: 1  # 1: pycparser, 2+: libclang
  preprocess: true
  include_headers: true

chunking:
  strategy: hybrid  # function, block, hybrid
  overlap_ratio: 0.25
  max_chunk_tokens: 18000

vector_store:
  type: faiss  # faiss, chroma, weaviate
  dimension: 1536
  index_path: data/faiss.index

cache:
  enabled: true
  directory: .cache/analysis
  ttl_days: 30

output:
  formats: [json, markdown]
  directory: output
  include_source: false

performance:
  max_workers: 5
  batch_size: 10
  rate_limit_rpm: 500  # requests per minute
```

---

## 6. エラーハンドリング戦略（o3提案）

### 6.1 API呼び出しエラー

```python
# src/retry_strategy.py
import backoff
from openai import APIError, RateLimitError, Timeout

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, Timeout),
    max_tries=5,
    max_time=300,
    on_backoff=lambda details: logger.warning(
        f"Retry {details['tries']}/{details['max_tries']} "
        f"after {details['wait']:.1f}s"
    )
)
def call_api_with_retry(func, *args, **kwargs):
    """指数バックオフでリトライ"""
    return func(*args, **kwargs)
```

### 6.2 パースエラー

```python
class ParsingError(Exception):
    def __init__(self, file_path: Path, line: int, message: str):
        self.file_path = file_path
        self.line = line
        self.message = message

def parse_with_fallback(code: str) -> ParsedCode:
    """pycparser失敗時にlibclangへフォールバック"""
    try:
        return pycparser_parse(code)
    except ParsingError as e:
        logger.warning(f"pycparser failed, trying libclang: {e}")
        return libclang_parse(code)
```

---

## 7. 監視・ログ

### 7.1 ログ設定

```python
# src/logging_config.py
import logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 使用例
logger.info(
    "chunk_analyzed",
    chunk_id=chunk.id,
    tokens=chunk.tokens,
    cost=estimated_cost,
    duration_ms=duration
)
```

### 7.2 メトリクス収集

```python
# src/metrics.py
from dataclasses import dataclass
from time import time

@dataclass
class Metrics:
    api_latency_ms: list[float]
    token_usage: list[int]
    cache_hit_rate: float
    error_rate: float

    def report(self):
        return {
            'avg_latency_ms': sum(self.api_latency_ms) / len(self.api_latency_ms),
            'total_tokens': sum(self.token_usage),
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate
        }
```

---

## 8. テスト戦略

### 8.1 単体テスト

```python
# tests/test_chunker.py
import pytest

def test_function_chunking():
    code = """
    int add(int a, int b) { return a + b; }
    int sub(int a, int b) { return a - b; }
    """
    parsed = parser.parse(code)
    chunks = chunker.chunk_functions(parsed)

    assert len(chunks) == 2
    assert chunks[0].name == "add"
    assert chunks[1].name == "sub"

def test_token_limit_enforcement():
    # 巨大関数のテスト
    large_code = generate_large_function(30000)  # トークン
    chunks = chunker.chunk_functions(large_code)

    for chunk in chunks:
        assert chunk.tokens <= 18000
```

### 8.2 統合テスト

```python
# tests/test_e2e.py
def test_cjson_analysis():
    """cJSONでのE2Eテスト"""
    orchestrator = Orchestrator(config)
    results = orchestrator.run(Path("data/cjson"))

    assert results['stats']['total_files'] > 0
    assert results['stats']['api_calls'] > 0
    assert results['stats']['error_rate'] < 0.05
```

---

## 9. デプロイメント

### 9.1 ディレクトリ構造

```
codechart/
├── src/
│   ├── __init__.py
│   ├── cli.py              # CLIエントリーポイント
│   ├── orchestrator.py     # パイプライン制御
│   ├── code_loader.py
│   ├── ast_parser.py
│   ├── code_chunker.py
│   ├── token_counter.py
│   ├── llm_analyzer.py
│   ├── vector_store.py
│   ├── rag_manager.py
│   ├── cache_manager.py
│   ├── output_formatter.py
│   └── config.py
├── config/
│   ├── default.yaml
│   └── production.yaml
├── tests/
│   ├── test_*.py
│   └── fixtures/
├── docs/
├── data/                   # 入力ソースコード
├── output/                 # 解析結果
├── .cache/                 # キャッシュ
├── pyproject.toml          # Poetry設定
└── README.md
```

### 9.2 Docker構成

```dockerfile
# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY . .
CMD ["poetry", "run", "python", "-m", "src.cli"]
```

---

## 10. Phase別実装計画

### Phase 1（Week 1-2）
- ✅ Code Loader
- ✅ AST Parser (pycparser)
- ✅ Code Chunker (関数単位)
- ✅ LLM Analyzer (基本実装)
- ✅ Cache Manager
- ⏳ Output Formatter

### Phase 2（Week 3-4）
- ⏳ AST Parser (libclang追加)
- ⏳ Code Chunker (ハイブリッド戦略)
- ⏳ Orchestrator (並列処理)
- ⏳ 静的解析統合

### Phase 3（Week 5-6）
- ⏳ RAG Manager
- ✅ Vector Store (FAISS)
- ⏳ コンテキスト拡張
- ⏳ 検索精度評価

### Phase 4（Week 7-8）
- ⏳ 性能最適化
- ⏳ エラーハンドリング強化
- ⏳ ドキュメント整備
- ⏳ プロダクション準備

---

## 11. 参考資料

- pycparser Documentation: https://github.com/eliben/pycparser
- FAISS Wiki: https://github.com/facebookresearch/faiss/wiki
- OpenAI API Best Practices: https://platform.openai.com/docs/guides/best-practices
