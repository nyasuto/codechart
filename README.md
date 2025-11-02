# Legacy Code Analyzer Pipeline
C/C++レガシーコード解析パイプライン

## 概要
大規模C/C++コードベースを段階的に解析するためのツール。
GPT-4 APIとRAGを組み合わせた解析システムの構築。

## 目標
1. **Phase 1**: 1万行規模のOSSで基本機能検証
2. **Phase 2**: 3万行規模への拡張
3. **Phase 3**: 5万行規模（Redis相当）での本格運用

## 主要機能
- ソースコード自動分割（チャンキング）
- 静的解析ツールとの連携
- GPT-4 APIによる意味解析
- RAGによる類似コード検索
- 解析結果の構造化出力

## 技術スタック
- 言語: Python 3.10+
- 静的解析: Cppcheck, ctags
- LLM: GPT-4 API (1-2万トークン制限)
- ベクトルDB: ChromaDB or FAISS
- AST解析: pycparser, clang

## 制約事項
- 最大入力: 2万トークン/リクエスト
- 自社API環境での動作
- エージェント機能は使用不可

## PoC対象候補（1万行規模）
- cJSON (C, 約8,000行) - JSONパーサー
- miniz (C, 約10,000行) - 圧縮ライブラリ
- mongoose (C, 約15,000行の一部) - Webサーバー

## ディレクトリ構成
```
project/
├── src/           # パイプラインコード
├── config/        # 設定ファイル
├── data/          # 入力ソースコード
├── output/        # 解析結果
├── tests/         # テストコード
└── docs/          # ドキュメント
```