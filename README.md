# CodeChart - Legacy Code Analyzer Pipeline
C/C++レガシーコード解析パイプライン

[![CI](https://github.com/nyasuto/codechart/actions/workflows/ci.yml/badge.svg)](https://github.com/nyasuto/codechart/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概要
大規模C/C++コードベースを段階的に解析し、高品質な技術文書（Markdown/CSV）を自動生成するツール。
GPT-4 APIによる意味解析と、社内RAGシステムへの最適化された出力を実現します。

## 主要機能
- **コード自動分割（チャンキング）**: 関数単位でトークン制限内に分割
- **LLMによる理解・解析**: 関数の役割、動作、データフロー、コールグラフを抽出
- **技術文書生成**: Markdown（詳細説明）+ CSV（メトリクス）
- **並列処理**: 複数LLMインスタンスによる高速解析
- **増分解析**: キャッシュ機構によるコスト削減

## 開発フェーズ
1. **Phase 1** (完了): 1万行規模のOSSで基本機能検証（cJSON）
2. **Phase 2** (現在): 理解特化型への改訂、並列処理、増分解析
3. **Phase 3**: 技術文書生成機能の強化
4. **Phase 4**: 大規模コードベース（3-5万行）での本番運用

## 技術スタック
- **言語**: Python 3.12+
- **パッケージ管理**: uv
- **AST解析**: pycparser → tree-sitter（将来）
- **LLM**: OpenAI-compatible API (GPT-4, CodeLlama, etc.)
- **文書生成**: Markdown、CSV
- **品質管理**: pytest、ruff、GitHub Actions

---

## セットアップ

### 必須要件
- Python 3.12 以上
- OpenAI API キー（GPT-4アクセス権限）
- Git

### 1. リポジトリのクローン
```bash
git clone https://github.com/nyasuto/codechart.git
cd codechart
```

### 2. 開発環境のセットアップ
```bash
# uvと依存関係を自動インストール
make dev-setup
```

このコマンドは以下を実行します：
- uvのインストール（未インストールの場合）
- プロジェクトの依存関係インストール（`uv sync --all-extras`）

#### 手動でuvをインストールする場合
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# インストール後
uv sync --all-extras
```

### 3. 環境変数の設定
```bash
# .envファイルを作成
cp .env.example .env  # サンプルがある場合

# または直接設定
export OPENAI_API_KEY="your-api-key-here"
```

### 4. 設定ファイルの確認
```bash
# デフォルト設定を確認
cat config/default.yaml

# 必要に応じてカスタマイズ（オプション）
cp config/default.yaml config/custom.yaml
```

---

## 使い方

### 基本的な解析実行
```bash
# cJSONプロジェクトを解析
uv run codechart analyze data/cJSON -o output/cjson

# カスタム設定を使用
uv run codechart analyze data/cJSON -c config/custom.yaml

# ドライラン（APIコールなし、トークン計算のみ）
uv run codechart analyze data/cJSON --dry-run
```

### 出力ファイル
```
output/cjson/
├── README.md                    # プロジェクトサマリー
├── files/
│   ├── cJSON.c.md              # ファイル詳細
│   └── ...
└── metrics/
    ├── functions.csv           # 関数一覧
    └── metrics.csv             # メトリクスサマリー
```

---

## 推奨する静的解析ツール

CodeChartは**LLMによる理解・解析**に特化しています。
品質評価やメトリクス計算には、以下の優れた静的解析ツールとの併用を推奨します。

### コード品質・静的解析

| ツール | 用途 | 特徴 |
|--------|------|------|
| [**Cppcheck**](http://cppcheck.sourceforge.net/) | 静的解析 | メモリリーク、未定義動作、バッファオーバーフロー検出 |
| [**Clang-Tidy**](https://clang.llvm.org/extra/clang-tidy/) | Linter | モダンC++への移行支援、コーディング規約チェック |
| [**SonarQube**](https://www.sonarqube.org/) | 総合品質管理 | 技術的負債、セキュリティ脆弱性の可視化 |

### メトリクス計算

| ツール | 用途 | 特徴 |
|--------|------|------|
| [**Lizard**](https://github.com/terryyin/lizard) | 複雑度計算 | Cyclomatic Complexity、関数行数、ネスト深度 |
| [**CCCC**](http://cccc.sourceforge.net/) | OO メトリクス | C++向けのオブジェクト指向メトリクス |

### コールグラフ・依存関係

| ツール | 用途 | 特徴 |
|--------|------|------|
| [**cscope**](http://cscope.sourceforge.net/) | コードナビゲーション | 関数呼び出し、変数参照の検索 |
| [**Doxygen**](https://www.doxygen.nl/) | ドキュメント生成 | コールグラフ、クラス図の自動生成 |
| [**Egypt**](https://www.gson.org/egypt/) | コールグラフ | ctagsベースのコールグラフ可視化 |

### 使用例

```bash
# CodeChartで理解・技術文書生成
uv run codechart analyze src/ -o output/docs

# Lizardで複雑度メトリクス計算
lizard src/ -o output/metrics.html

# Cppcheckで静的解析
cppcheck --enable=all --xml src/ 2> output/cppcheck.xml

# Doxygenでコールグラフ生成
doxygen Doxyfile
```

この組み合わせにより、**理解**（CodeChart）と**品質評価**（静的解析ツール）の両面から、レガシーコードを効果的に分析できます。

---

## 開発

### 利用可能なコマンド
```bash
make help          # コマンド一覧を表示
make install       # 依存関係をインストール
make test          # テストを実行
make lint          # ruff checkを実行
make format        # コードをフォーマット
make format-check  # フォーマットチェック（CI用）
make quality       # すべての品質チェックを実行
make clean         # キャッシュを削除
```

### テストの実行
```bash
# 全テスト実行
make test

# または直接pytest実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=src --cov-report=html

# 特定のテストのみ
uv run pytest tests/test_chunker.py
```

### コード品質チェック
```bash
# リント
make lint

# フォーマット
make format

# フォーマットチェック
make format-check
```

### すべての品質チェックを実行
```bash
# GitHub Actionsと同じチェックをローカルで実行
make quality
```

---

## ディレクトリ構成
```
codechart/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD設定
├── src/                        # ソースコード（Phase 1で実装）
│   ├── __init__.py
│   ├── cli.py                  # CLIエントリーポイント
│   ├── code_loader.py          # ファイル読み込み
│   ├── ast_parser.py           # AST解析
│   ├── code_chunker.py         # チャンキング
│   ├── token_counter.py        # トークン計算
│   ├── llm_analyzer.py         # LLM解析
│   ├── cache_manager.py        # キャッシュ管理
│   └── output_formatter.py     # 文書生成
├── config/
│   └── default.yaml            # デフォルト設定
├── templates/                  # Jinja2テンプレート
├── tests/                      # テストコード
├── docs/                       # ドキュメント
│   ├── architecture.md         # アーキテクチャ設計
│   └── tech_research.md        # 技術調査
├── data/                       # 入力ソースコード
├── output/                     # 解析結果（.gitignore）
├── .cache/                     # キャッシュ（.gitignore）
├── Makefile                    # ビルドコマンド
├── pyproject.toml              # プロジェクト設定
├── uv.lock                     # 依存関係ロックファイル
└── README.md                   # このファイル
```

---

## コントリビューション

### Issue報告
バグ報告や機能要望は [GitHub Issues](https://github.com/nyasuto/codechart/issues) へ。

### 開発フロー
1. Issueを確認・作成
2. 機能ブランチを作成（`feat/issue-X-feature-name`）
3. コード実装とテスト追加
4. 品質チェック実施（`make quality`）
5. Pull Request作成

詳細は [development_plan.md](development_plan.md) を参照。

---

## ライセンス
MIT License

---

## 参考資料
- [開発計画書](development_plan.md)
- [アーキテクチャ設計](docs/architecture.md)
- [技術調査レポート](docs/tech_research.md)
- [GitHub Issues](https://github.com/nyasuto/codechart/issues)
