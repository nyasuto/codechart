# Issue #1: プロジェクト構造とCI/CD環境構築

## 概要
プロジェクトの基本構造とCI/CD環境を構築し、開発体制を整える。

## 詳細説明
Python開発のベストプラクティスに従い、以下を構築：
- Poetry/pipenvによる依存管理
- GitHub Actionsでのテスト・リンター自動実行
- 開発環境のDocker化（オプション）
- 設定管理の外部化（YAML）

## 受け入れ条件（Acceptance Criteria）
- [ ] `pyproject.toml`でPython 3.10+の依存関係定義
- [ ] 必須ライブラリのインストール確認
  - pycparser
  - tiktoken
  - openai
  - pyyaml
  - pytest, ruff, mypy
- [ ] `.github/workflows/ci.yml`でテスト・リント自動実行
- [ ] `config/default.yaml`で設定管理
- [ ] README.mdにセットアップ手順記載
- [ ] `.gitignore`で不要ファイル除外（.cache/, output/）

## タスク
1. Poetryプロジェクト初期化
   ```bash
   poetry init
   poetry add pycparser tiktoken openai pyyaml
   poetry add -D pytest pytest-cov ruff mypy
   ```

2. ディレクトリ構造作成
   ```
   codechart/
   ├── src/
   ├── config/
   ├── tests/
   ├── docs/
   ├── data/
   ├── output/
   └── .cache/
   ```

3. GitHub Actions CI設定
   - pytest実行
   - ruffリント
   - mypy型チェック
   - カバレッジレポート

4. config/default.yamlサンプル作成

## 推定工数
2日（16時間）

## 依存関係
なし（最初のタスク）

## 優先度
P0 (Critical)

## ラベル
- type: ci/cd
- priority: critical
- phase: 1
- epic: 基本パイプライン構築

## 担当者
（未定）

## 技術メモ
o3提案を反映：
- 設定の外部化（YAML）
- CI/CDの早期構築
- 開発環境の標準化

## 参考資料
- Poetry: https://python-poetry.org/
- GitHub Actions: https://docs.github.com/en/actions
