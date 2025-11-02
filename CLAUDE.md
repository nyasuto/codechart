# Development Policy

## Project Overview

TBD

## Issue 実装フロー

Issue を実装する前に、以下を評価して判断すること：

### 複雑度チェック（各項目 Yes=1 点）

- [ ] 新しい技術/ライブラリが必要
- [ ] 3 つ以上の機能に影響する
- [ ] DB 設計の変更を含む
- [ ] 認証/支払い/個人情報を扱う
- [ ] パフォーマンスがクリティカル

### 自動判定

- **3 点以上**: 設計書を先に提示 → 承認待ち
- **1-2 点**: 実装方針（3 行）を示して即実装
- **0 点**: 何も言わずに即実装

## 🚨 CRITICAL: PR Merge Policy

**ABSOLUTE RULE: Claude MUST NEVER merge PRs automatically**

- ✅ **ALLOWED**: Create PRs using `gh pr create`
- ✅ **ALLOWED**: Watch PR checks with `gh pr checks --watch`
- ❌ **FORBIDDEN**: Use `gh pr merge` or any merge commands
- ❌ **FORBIDDEN**: Automatic merging regardless of PR size or type
- ❌ **FORBIDDEN**: Merging even if all CI checks pass
- ✅ **REQUIRED**: Human must review and merge ALL PRs manually

**WHY**: Human review is essential for quality control, architectural decisions, and understanding changes.

## Git Workflow

### main ブランチへの直接コミット禁止

- ❌ main ブランチへの直接コミットは禁止
- ✅ 全ての変更は feature ブランチから開始
- ✅ Pull Request を経由してマージ

### ブランチ命名規則

- `feat/X-description` - 新機能
- `fix/X-description` - バグ修正
- `docs/X-description` - ドキュメント
- `ci/X-description` - CI/CD 設定
- `refactor/X-description` - リファクタリング

### 開発フロー

```bash
# 1. featureブランチを作成
git checkout -b feat/issue-X-feature-name

# 2. 変更を実装

# 3. ローカルでテスト
make quality

# 4. コミット
git add .
git commit -m "feat: 機能の説明"

# 5. プッシュ
git push origin feat/issue-X-feature-name

# 6. Pull Request作成
gh pr create --title "feat: 機能の説明" --body "詳細..."

# 7. CI/CD通過を確認
gh pr checks --watch

# 8. ⚠️ 人間によるレビューとマージを待つ
```


## Code Quality Standards

- All tests must pass
- Conventional commit messages
- Minimum 70% test coverage for new features
