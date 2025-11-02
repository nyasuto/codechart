.PHONY: help install dev-setup test lint format format-check quality clean

# デフォルトターゲット
.DEFAULT_GOAL := help

help: ## ヘルプを表示
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 依存関係をインストール
	uv sync

dev-setup: ## 開発環境をセットアップ（初回）
	@echo "Installing uv..."
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Installing dependencies..."
	uv sync --all-extras
	@echo "Setup complete! Run 'make test' to verify."

test: ## テストを実行
	uv run pytest

lint: ## コードをlint
	uv run ruff check src tests

format: ## コードをフォーマット
	uv run ruff format src tests

format-check: ## フォーマットをチェック（CIで使用）
	uv run ruff format --check src tests

quality: lint format-check test ## 全品質チェックを実行

clean: ## キャッシュファイルを削除
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
