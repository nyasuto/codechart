# Issue #5: cJSONでの初期動作検証

## 概要
実際のOSS（cJSON、約8,000行）で基本パイプラインの動作を検証し、Phase 1の完了を確認する。

## 詳細説明
cJSONリポジトリをクローンし、以下を実施：
1. 全関数の解析成功を確認
2. 技術文書（Markdown/CSV）の品質確認
3. 性能測定（処理時間、トークン消費、コスト）
4. 問題点の洗い出しと改善

## 受け入れ条件（Acceptance Criteria）
- [ ] cJSONリポジトリのクローンと前処理
- [ ] 全C言語ファイルの解析成功（エラー率5%以下）
- [ ] Markdown/CSV出力の生成確認
- [ ] 性能ベンチマークレポート作成
- [ ] 課題リストと改善案の文書化
- [ ] デモ実施（社内RAGチームへの共有）

## タスク

### 1. cJSON準備
```bash
cd data
git clone https://github.com/DaveGamble/cJSON.git
cd cJSON

# ファイル数確認
find . -name "*.c" -o -name "*.h" | wc -l

# 行数確認
find . -name "*.c" -o -name "*.h" | xargs wc -l
```

### 2. 解析実行
```bash
# 基本実行
poetry run python -m src.cli analyze data/cJSON -o output/cjson

# デバッグモード
poetry run python -m src.cli analyze data/cJSON -o output/cjson --debug

# ドライラン（APIコールなし、トークン計算のみ）
poetry run python -m src.cli analyze data/cJSON --dry-run
```

### 3. 結果検証

#### 成功基準チェックリスト
- [ ] 全ファイルが解析された
- [ ] エラー率が5%以下
- [ ] `output/cjson/README.md`が生成された
- [ ] `output/cjson/files/`に各ファイルのMDが生成
- [ ] `output/cjson/metrics/functions.csv`が生成
- [ ] CSVに全関数が含まれている

#### 品質確認
```python
# 検証スクリプト
def verify_output():
    # Markdown検証
    readme = Path("output/cjson/README.md")
    assert readme.exists()
    content = readme.read_text()
    assert "cJSON" in content
    assert "## 概要" in content

    # CSV検証
    functions_csv = Path("output/cjson/metrics/functions.csv")
    df = pd.read_csv(functions_csv)
    assert len(df) > 0
    assert '関数名' in df.columns
    assert '複雑度' in df.columns

    # 手動レビュー
    sample_func = df.sample(5)
    for _, row in sample_func.iterrows():
        file_path = Path(f"output/cjson/files/{row['ファイル']}.md")
        assert file_path.exists()
```

### 4. 性能測定

#### ベンチマークスクリプト
```python
# benchmarks/benchmark_cjson.py
import time
from src.orchestrator import Orchestrator

def benchmark():
    start_time = time.time()

    orchestrator = Orchestrator(config)
    results = orchestrator.run(Path("data/cJSON"))

    elapsed = time.time() - start_time

    report = {
        '処理時間': f"{elapsed:.2f}秒",
        '総ファイル数': results['stats']['total_files'],
        '総関数数': results['stats']['total_chunks'],
        'APIコール数': results['stats']['api_calls'],
        '総トークン数': results['stats']['total_tokens'],
        '推定コスト': f"${results['stats']['total_cost']:.2f}",
        'キャッシュヒット率': f"{results['stats']['cache_hit_rate']:.1%}",
        'エラー率': f"{results['stats']['error_rate']:.1%}"
    }

    return report
```

#### 期待値
| 項目 | 目標 | 備考 |
|------|------|------|
| 処理時間 | 30分以内 | 8,000行規模 |
| APIコール数 | 50-100回 | 関数数に依存 |
| 総トークン数 | 100K-200K | 入力のみ |
| 推定コスト | $5以下 | 初回実行 |
| キャッシュヒット率 | 0% → 100% | 2回目実行 |
| エラー率 | 5%以下 | パース失敗など |

### 5. 問題点の洗い出し

#### チェック項目
- [ ] パース失敗したファイル・関数の特定
- [ ] トークン超過したチャンクの確認
- [ ] GPT-4のレスポンス品質評価
- [ ] 出力文書の可読性確認
- [ ] 社内RAGチームからのフィードバック

#### 問題例と対策
| 問題 | 対策 | Issue |
|------|------|-------|
| マクロ展開失敗 | gcc -E前処理 | #2の改善 |
| トークン超過 | チャンク分割改善 | #2の改善 |
| レスポンス遅延 | 並列処理 | Phase 2 |
| 文書品質低い | プロンプト改善 | #3の改善 |

## 推定工数
2日（16時間）

## 依存関係
- #1: プロジェクト構造とCI/CD環境構築
- #2: コード分割モジュール
- #3: GPT-4解析モジュール
- #4: 技術文書出力機能

## 優先度
P0 (Critical)

## ラベル
- type: test
- priority: critical
- phase: 1
- epic: 基本パイプライン構築

## 成果物
1. **性能レポート** (`docs/benchmark_cjson.md`)
   - 処理時間、トークン消費、コスト
   - ボトルネック分析

2. **品質レポート** (`docs/quality_cjson.md`)
   - 解析成功率
   - 文書品質評価
   - 社内RAGチームからのフィードバック

3. **課題リスト** (`docs/issues_phase1.md`)
   - 発見された問題
   - Phase 2での改善案

4. **デモ資料** (`docs/demo_phase1.pdf`)
   - 社内共有用スライド
   - 生成された技術文書のサンプル

## デモシナリオ
1. cJSONプロジェクト概要説明
2. 実行コマンドのデモ
3. 生成されたMarkdown文書の閲覧
4. CSVメトリクスの確認
5. 社内RAGシステムへの取り込みデモ（RAGチーム協力）
6. Q&Aセッション

## 参考資料
- cJSON Repository: https://github.com/DaveGamble/cJSON
