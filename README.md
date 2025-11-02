# CodeChart - Legacy Code Analyzer Pipeline
C/C++レガシーコード解析パイプライン

[![CI](https://github.com/nyasuto/codechart/actions/workflows/ci.yml/badge.svg)](https://github.com/nyasuto/codechart/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概要
大規模C/C++コードベースを段階的に解析し、高品質な技術文書（Markdown/CSV）を自動生成するツール。
GPT-4 APIによる意味解析と、社内RAGシステムへの最適化された出力を実現します。

## 主要機能
- **コード自動分割（チャンキング）**: 関数単位でトークン制限内に分割
- **GPT-4による意味解析**: 目的、アルゴリズム、潜在的問題を抽出
- **技術文書生成**: Markdown（詳細説明）+ CSV（メトリクス）
- **増分解析**: キャッシュ機構によるコスト削減
- **静的解析統合**: Cppcheck、ctags（Phase 2）

## 開発フェーズ
1. **Phase 1** (現在): 1万行規模のOSSで基本機能検証（cJSON）
2. **Phase 2**: 3万行規模への拡張 + 静的解析統合
3. **Phase 3**: 技術文書生成機能の強化
4. **Phase 4**: 5万行規模（Redis相当）での本番運用

## 技術スタック
- **言語**: Python 3.10+
- **AST解析**: pycparser（Phase 1）→ libclang（Phase 2+）
- **LLM**: GPT-4 Turbo API（18,000トークン/チャンク）
- **文書生成**: Jinja2（Markdown）、CSV
- **品質管理**: pytest、ruff、mypy

---

## セットアップ

### 必須要件
- Python 3.10 以上
- OpenAI API キー（GPT-4アクセス権限）
- Git

### 1. リポジトリのクローン
```bash
git clone https://github.com/nyasuto/codechart.git
cd codechart
```

### 2. 依存関係のインストール

#### オプションA: Poetry（推奨）
```bash
# Poetryのインストール（未インストールの場合）
curl -sSL https://install.python-poetry.org | python3 -

# 依存関係のインストール
poetry install

# 仮想環境の有効化
poetry shell
```

#### オプションB: pip
```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# 開発用依存関係（テスト・リント用）
pip install -r requirements-dev.txt
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
python -m src.cli analyze data/cJSON -o output/cjson

# カスタム設定を使用
python -m src.cli analyze data/cJSON -c config/custom.yaml

# ドライラン（APIコールなし、トークン計算のみ）
python -m src.cli analyze data/cJSON --dry-run
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

## 開発

### テストの実行
```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov=src --cov-report=html

# 特定のテストのみ
pytest tests/test_chunker.py
```

### コード品質チェック
```bash
# リント
ruff check src tests

# フォーマット
ruff format src tests

# 型チェック
mypy src
```

### すべての品質チェックを実行
```bash
# GitHub Actionsと同じチェックをローカルで実行
pytest --cov=src --cov-report=term-missing
ruff check src tests
ruff format --check src tests
mypy src
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
├── pyproject.toml              # プロジェクト設定
├── requirements.txt            # 本番依存関係
├── requirements-dev.txt        # 開発依存関係
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
4. 品質チェック実施（pytest、ruff、mypy）
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