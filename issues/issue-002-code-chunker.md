# Issue #2: コード分割モジュール（code_chunker.py）の実装

## 概要
C/C++コードを関数単位で分割し、GPT-4のトークン制限内に収まるチャンクを生成する。

## 詳細説明
pycparserを使用してC言語のASTを解析し、以下を実現：
- 関数単位での抽出
- トークン数計算（tiktoken）
- 2万トークン以下のチャンク生成
- メタデータ付与（ファイル名、行番号、関数名）

## 受け入れ条件（Acceptance Criteria）
- [ ] `src/code_loader.py`: ファイル読み込み機能
- [ ] `src/ast_parser.py`: pycparserラッパー
- [ ] `src/token_counter.py`: tiktokenラッパー（キャッシュ付き）
- [ ] `src/code_chunker.py`: チャンキングロジック
- [ ] 単体テスト（カバレッジ80%以上）
- [ ] cJSONの任意の関数で動作確認

## タスク

### 1. Code Loader実装
```python
class CodeFile:
    path: Path
    content: str
    language: str  # 'c' or 'cpp'
    hash: str      # SHA-256

class CodeLoader:
    def discover_files(self, root: Path) -> list[Path]:
        """C/C++ファイルを再帰検索"""

    def load_file(self, path: Path) -> CodeFile:
        """ファイル読み込み＋ハッシュ計算"""
```

### 2. AST Parser実装
```python
@dataclass
class FunctionNode:
    name: str
    signature: str
    body: str
    start_line: int
    end_line: int

class PycparserAdapter:
    def parse(self, code: str) -> ParsedCode:
        """pycparserでAST解析"""
```

### 3. Token Counter実装
```python
class TiktokenCounter:
    @lru_cache(maxsize=2048)
    def count(self, text: str) -> int:
        """キャッシュ付きトークン計算"""
```

### 4. Code Chunker実装
```python
class CodeChunker:
    def chunk_functions(
        self,
        parsed: ParsedCode,
        max_tokens: int = 18000
    ) -> list[CodeChunk]:
        """関数単位でチャンク生成"""
```

## 推定工数
3日（24時間）

## 依存関係
- #1: プロジェクト構造とCI/CD環境構築

## 優先度
P0 (Critical)

## ラベル
- type: feature
- priority: critical
- phase: 1
- epic: 基本パイプライン構築

## 技術的課題
1. **pycparserのプリプロセッサ問題**
   - マクロ展開が必要なコードはgcc -E前処理
   - エラー時のフォールバック処理

2. **巨大関数の処理**
   - Phase 1では一旦スキップ（警告のみ）
   - Phase 2でブロック分割対応

3. **文字エンコーディング**
   - chardetで自動検出
   - UTF-8前提だが、Shift_JIS等も考慮

## テストケース
```python
def test_simple_function():
    code = "int add(int a, int b) { return a + b; }"
    chunks = chunker.chunk_functions(parser.parse(code))
    assert len(chunks) == 1
    assert chunks[0].name == "add"

def test_token_limit():
    large_code = generate_large_code(30000)  # トークン
    chunks = chunker.chunk_functions(large_code)
    for chunk in chunks:
        assert chunk.tokens <= 18000
```

## 参考資料
- pycparser examples: https://github.com/eliben/pycparser/tree/master/examples
- tiktoken: https://github.com/openai/tiktoken
