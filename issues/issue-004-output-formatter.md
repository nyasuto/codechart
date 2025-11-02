# Issue #4: 技術文書出力機能（Markdown/CSV）の実装

## 概要
GPT-4の解析結果を社内RAGシステムに最適化されたMarkdown/CSV形式で出力する。

## 詳細説明
解析結果を以下の形式で出力：
1. **Markdown文書**: プロジェクト概要、ファイル詳細、関数説明
2. **CSV**: 関数一覧、メトリクス、依存関係
3. **階層構造**: プロジェクト → ファイル → 関数

## 受け入れ条件（Acceptance Criteria）
- [ ] `src/output_formatter.py`: 出力フォーマッター
- [ ] Markdown形式でプロジェクトサマリー生成
- [ ] ファイル単位の詳細技術文書（Markdown）
- [ ] 関数一覧CSV（メトリクス付き）
- [ ] 依存関係CSV
- [ ] 設定ファイルで出力形式を選択可能
- [ ] テンプレートエンジン（Jinja2）の活用

## タスク

### 1. Output Formatter実装
```python
# src/output_formatter.py
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class OutputFormatter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.template_env = Environment(
            loader=FileSystemLoader('templates')
        )

    def generate_project_summary(
        self,
        results: list[AnalysisResult],
        stats: dict
    ) -> Path:
        """プロジェクトサマリーMarkdown生成"""

    def generate_file_doc(
        self,
        file_path: Path,
        results: list[AnalysisResult]
    ) -> Path:
        """ファイル単位の技術文書生成"""

    def generate_function_csv(
        self,
        results: list[AnalysisResult]
    ) -> Path:
        """関数一覧CSV生成"""

    def generate_metrics_csv(
        self,
        results: list[AnalysisResult]
    ) -> Path:
        """メトリクスCSV生成"""
```

### 2. Markdownテンプレート作成

#### `templates/project_summary.md.j2`
```markdown
# {{ project_name }} 技術文書

## プロジェクト概要
- **総ファイル数**: {{ stats.total_files }}
- **総関数数**: {{ stats.total_functions }}
- **総行数**: {{ stats.total_lines }}
- **平均複雑度**: {{ stats.avg_complexity }}

## ファイル一覧
{% for file in files %}
- [{{ file.name }}]({{ file.path }})
  - 関数数: {{ file.function_count }}
  - 行数: {{ file.line_count }}
{% endfor %}

## 潜在的問題
{% for issue in critical_issues %}
- **{{ issue.severity }}**: {{ issue.description }} ({{ issue.location }})
{% endfor %}
```

#### `templates/file_detail.md.j2`
```markdown
# {{ file_name }}

## 概要
{{ file_summary }}

## 関数一覧

{% for func in functions %}
### `{{ func.signature }}`

**目的:** {{ func.purpose }}

**アルゴリズム:** {{ func.algorithm }}

**複雑度:** {{ func.complexity }}

**依存関数:**
{% for dep in func.dependencies %}
- `{{ dep }}`
{% endfor %}

{% if func.potential_issues %}
**潜在的問題:**
{% for issue in func.potential_issues %}
- {{ issue }}
{% endfor %}
{% endif %}

{% if func.improvements %}
**改善提案:**
{% for improvement in func.improvements %}
- {{ improvement }}
{% endfor %}
{% endif %}

---
{% endfor %}
```

### 3. CSV出力実装

#### 関数一覧CSV
```python
def generate_function_csv(results: list[AnalysisResult]) -> Path:
    """
    カラム:
    - ファイル名
    - 関数名
    - 行数
    - 複雑度（O記法）
    - 循環的複雑度
    - 引数数
    - 依存関数数
    - 潜在的問題数
    """
    import csv
    output_path = self.output_dir / "functions.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ファイル', '関数名', '開始行', '終了行', '行数',
            '時間複雑度', '循環的複雑度', '引数数',
            '依存関数数', '潜在的問題数'
        ])

        for result in results:
            writer.writerow([
                result.metadata['file_path'],
                result.metadata['function_name'],
                result.metadata['start_line'],
                result.metadata['end_line'],
                result.metadata['line_count'],
                result.complexity,
                result.metadata.get('cyclomatic_complexity', 'N/A'),
                result.metadata.get('param_count', 0),
                len(result.dependencies),
                len(result.potential_issues)
            ])

    return output_path
```

#### メトリクスサマリーCSV
```csv
カテゴリ,メトリクス,値
プロジェクト,総ファイル数,15
プロジェクト,総関数数,120
プロジェクト,総行数,8543
品質,平均複雑度,3.2
品質,高複雑度関数数,8
品質,潜在的問題数,23
```

## 推定工数
2日（16時間）

## 依存関係
- #2: コード分割モジュール
- #3: GPT-4解析モジュール

## 優先度
P0 (Critical)

## ラベル
- type: feature
- priority: critical
- phase: 1
- epic: 基本パイプライン構築

## 技術的考慮事項

### 1. RAGシステム最適化
社内RAGチームと連携し、以下を確認：
- チャンクサイズの推奨値
- メタデータの必須項目
- ファイル分割粒度（1ファイル vs 関数ごと）

### 2. Markdown構造
```
output/
├── README.md                    # プロジェクトサマリー
├── files/
│   ├── math_utils.c.md         # ファイル詳細
│   ├── string_utils.c.md
│   └── ...
├── metrics/
│   ├── functions.csv           # 関数一覧
│   ├── metrics.csv             # メトリクスサマリー
│   └── dependencies.csv        # 依存関係
└── graphs/
    └── dependency_graph.mmd    # Mermaid図（Phase 3）
```

### 3. 文字エンコーディング
- すべてUTF-8で出力
- BOM無し（Linuxツールとの互換性）

## テストケース
```python
def test_markdown_generation():
    results = [create_test_result()]
    formatter = OutputFormatter(Path('output'))

    md_path = formatter.generate_file_doc('test.c', results)

    assert md_path.exists()
    content = md_path.read_text()
    assert '## 概要' in content
    assert '### `add`' in content

def test_csv_generation():
    results = [create_test_result()]
    csv_path = formatter.generate_function_csv(results)

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == len(results)
    assert 'ファイル' in df.columns
```

## 参考資料
- Jinja2: https://jinja.palletsprojects.com/
- Python csv module: https://docs.python.org/3/library/csv.html
