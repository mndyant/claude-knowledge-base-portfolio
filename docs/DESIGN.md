# 設計書

**プロジェクト名**: claude-knowledge-base  
**バージョン**: 1.2.0  
**作成日**: 2026-05-26  
**更新日**: 2026-07-18  
**ステータス**: 稼働中（本番運用・GitHub Actionsで毎日自動更新）

> 本書は初期設計の記録として残しているため、パラメータ等の数値は当初案の
> ままの箇所がある。実装後に確定した値（チャンクサイズtarget=400/max=480/
> overlap=50など）は `CLAUDE.md` および `docs/ENGINEERING_NOTES.md` を正とする。

---

## 開発環境

### 構成概要

| 項目 | 内容 |
|------|------|
| ホストOS | Windows（WSL2・Docker不使用） |
| 実行環境 | Windows ネイティブ Python + venv |
| シェル | PowerShell |
| エディタ | VSCode（Windows）|
| 実装担当 | Claude Code |
| 設計・レビュー | Claude（claude.ai） |

### 役割分担

| フェーズ | ツール | 用途 |
|---------|--------|------|
| 計画・設計 | Claude | アーキテクチャ検討・設計書作成・レビュー |
| 実装 | Claude Code | スクリプト・テストコード・設定ファイルの生成 |
| 遠隔実装 | Claude Code + Remote Control | iPhone経由でローカルセッションを操作 |
| CI/CD | GitHub Actions | 自動取得・ベクトル化・テスト実行 |

---

### セットアップ手順（PowerShell）

```powershell
# リポジトリに移動
cd C:\projects\claude-knowledge-base

# venv作成・有効化
python -m venv venv
.\venv\Scripts\Activate.ps1

# 依存パッケージインストール
pip install -r requirements.txt

# 初回データ取得・ベクトル化
python scripts/fetch.py
python scripts/embed.py

# 検索API起動
uvicorn scripts.search_api:app --reload --port 8001
# → http://localhost:8001/docs でSwagger UI確認
```

> **Note**: PowerShellの実行ポリシーエラーが出た場合：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### venv運用ルール

```powershell
# 作業開始時（毎回）
.\venv\Scripts\Activate.ps1

# パッケージ追加時
pip install {package}
pip freeze > requirements.txt  # 必ず更新

# 作業終了時
deactivate
```

---

### APIキー管理

**Windowsユーザー環境変数で管理する（コードに直接書かない）。**

#### 設定方法

```powershell
# PowerShellから設定（永続化）
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-xxxx", "User")

# 設定確認
[System.Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
```

またはGUI：
```
Windowsの設定 → システム → バージョン情報
→ システムの詳細設定 → 環境変数
→ ユーザー環境変数 → 新規
  変数名: ANTHROPIC_API_KEY
  変数値: sk-ant-xxxx
```

> **Note**: 設定後はPowerShell・VSCodeを再起動しないと反映されない。

#### .envファイルの扱い

ローカル開発の補助として`.env`を使う場合：

```bash
# .env（リポジトリルート）
ANTHROPIC_API_KEY=sk-ant-xxxx
```

`.env`は必ず`.gitignore`に含める（コミット禁止）。

#### GitHub Actionsでのキー管理

```
GitHubリポジトリ → Settings → Secrets and variables → Actions
→ New repository secret
  Name: ANTHROPIC_API_KEY
  Value: sk-ant-xxxx
```

update.ymlで参照：
```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

### VSCode設定

#### 推奨拡張機能（.vscode/extensions.json）

```json
{
  "recommendations": [
    "ms-python.python",                  // Python基本
    "ms-python.vscode-pylance",          // 型チェック・補完
    "ms-python.black-formatter",         // コードフォーマット
    "charliermarsh.ruff",                // Linter
    "ms-python.debugpy",                 // デバッガ
    "rangav.vscode-thunder-client",      // APIテスト（Postman代替）
    "fastapi.fastapi-vscode",            // FastAPIルート一覧
    "github.vscode-github-actions"       // GitHub Actions構文チェック
  ]
}
```

#### VSCode設定（.vscode/settings.json）

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  }
}
```

---

### 推奨ライブラリ一覧（requirements.txt）

```
# 取得・整形
requests==2.32.*
beautifulsoup4==4.12.*
feedparser==6.0.*
tenacity==8.*              # リトライ処理

# ベクトル化
sentence-transformers==3.*
torch==2.*                 # CPU版で十分

# ベクトルDB（ローカルファイルDB・サーバー不要）
chromadb==1.5.*

# API
fastapi==0.115.*
uvicorn[standard]==0.30.*
pydantic==2.*

# 環境変数
python-dotenv==1.*         # .envファイル読み込み

# テスト
pytest==8.*
pytest-cov
requests-mock
httpx

# 開発ツール
black
ruff
```

---

### .gitignore 設定

```
# Python
__pycache__/
*.pyc
venv/

# 環境変数（コミット禁止）
.env

# モデルキャッシュ（容量大・再ダウンロード可）
.cache/

# ChromaDBデータ
# chroma-data/  ← Git管理する場合はコメントアウト
```

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [要件定義](#2-要件定義)
3. [システム構成](#3-システム構成)
4. [コンポーネント設計](#4-コンポーネント設計)
5. [データ設計](#5-データ設計)
6. [API設計](#6-api設計)
7. [CI/CDパイプライン設計](#7-cicdパイプライン設計)
8. [エラーハンドリング設計](#8-エラーハンドリング設計)
9. [技術選定根拠](#9-技術選定根拠)

---

## 1. プロジェクト概要

### 1.1 背景・目的

Claude.aiは学習データのカットオフ以降の最新情報を持たない。  
AnthropicはAPIドキュメント・ブログ・リリースノートを継続的に更新しているが、  
これらを手動で追跡するのはコストが高い。

本システムは以下を自動化する：

- Anthropic公式情報の定期取得・整形
- ベクトルDB（ChromaDB）への登録
- FastAPIによる意味検索エンドポイントの提供

### 1.2 スコープ

**対象（In Scope）**

- Anthropic公式ドキュメント・ブログ・リリースノートの取得
- テキストのチャンク分割・ベクトル化・ChromaDB登録
- 意味検索APIの提供
- GitHub Actionsによる自動更新

**対象外（Out of Scope）**

- Anthropic Academy動画コンテンツの自動取得
- 多言語翻訳の自動化
- クラウドデプロイ（ローカル運用のみ）
- 認証・認可機能

---

## 2. 要件定義

### 2.1 機能要件

| ID | 要件 | 優先度 |
|----|------|--------|
| FR-001 | llms-full.txtを取得しMarkdownに整形できること | Must |
| FR-002 | 公式ブログ記事を取得・保存できること | Must |
| FR-003 | GitHubリリースノートを取得・保存できること | Must |
| FR-004 | テキストをチャンク分割しベクトル化できること | Must |
| FR-005 | ChromaDBにベクトルを永続化できること | Must |
| FR-006 | 自然言語クエリで類似チャンクを検索できること | Must |
| FR-007 | 検索結果にソースURLとスコアを含めること | Must |
| FR-008 | 前回取得との差分のみ更新できること | Should |
| FR-009 | GitHub Actionsで毎日自動実行されること | Must |
| FR-010 | FastAPI経由でHTTPリクエストで検索できること | Must |

### 2.2 非機能要件

| ID | 要件 | 基準値 |
|----|------|--------|
| NFR-001 | 検索レスポンスタイム | 2秒以内（top_k=5） |
| NFR-002 | 取得スクリプトの実行時間 | 10分以内 |
| NFR-003 | 運用コスト | 月額0円 |
| NFR-004 | Python依存パッケージ | requirements.txtで管理 |
| NFR-005 | エラー発生時の通知 | GitHub Actionsのログに記録 |
| NFR-006 | APIキー | Windowsユーザー環境変数で管理・コードに直接記述しない |

---

## 3. システム構成

### 3.1 全体アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                  GitHub Actions                  │
│  ┌────────────┐  ┌────────────┐                  │
│  │ fetch.py   │→ │ embed.py   │                  │
│  │ (取得・整形) │  │ (ベクトル化) │                  │
│  └────────────┘  └─────┬──────┘                  │
│                        │ commit & push            │
└────────────────────────┼────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   GitHubリポジトリ   │
              │  ┌───────────────┐  │
              │  │ knowledge/    │  │
              │  │ chroma-data/  │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │ git pull（手動 or 自動）
                         ▼
              ┌──────────────────────┐
              │  Windows + venv      │
              │  ┌────────────────┐  │
              │  │ search_api.py  │  │
              │  │ (FastAPI)      │  │
              │  └───────┬────────┘  │
              └──────────┼───────────┘
                    ┌────┴────┐
                    ▼         ▼
               Claude Code  ブラウザ
               （質問応答）  (Swagger UI)
```

### 3.2 データフロー

```
[取得フェーズ]
外部ソース → fetch.py → knowledge/{カテゴリ}/*.md

[ベクトル化フェーズ]
knowledge/*.md → チャンク分割（512トークン / overlap 50）
              → multilingual-e5-base でエンベディング生成
              → ChromaDB（chroma-data/）に保存

[検索フェーズ]
クエリ文字列 → エンベディング生成
            → ChromaDB でコサイン類似度検索
            → 上位k件を返却（content / source / score）
```

---

## 4. コンポーネント設計

### 4.1 fetch.py

**責務**: 外部ソースからの情報取得・Markdown整形・ファイル保存

```python
fetch_llms_full_txt() -> None
fetch_blog_posts() -> None
fetch_release_notes() -> None
fetch_cookbooks() -> None
is_updated(filepath: str, content: str) -> bool
```

**エラー処理**

- HTTPエラー（4xx/5xx）: ログ出力後スキップ（他ソースの処理は継続）
- タイムアウト: 30秒で打ち切り・tenacityでリトライ2回
- パースエラー: ログ出力後スキップ

---

### 4.2 embed.py

**責務**: Markdownファイルのチャンク分割・ベクトル化・ChromaDB登録

```python
load_documents(knowledge_dir: str) -> list[Document]
split_chunks(documents: list[Document]) -> list[Chunk]
embed_and_store(chunks: list[Chunk]) -> None
get_collection_stats() -> dict
```

**チャンク分割仕様**

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| chunk_size | 512トークン | multilingual-e5-baseの最大入力512に合わせる |
| overlap | 50トークン | 文脈の連続性を保持 |
| 分割基準 | 改行・見出しを優先 | 意味的まとまりを維持 |

---

### 4.3 search_api.py

**責務**: FastAPIによるベクトル検索エンドポイントの提供

```
POST /search    クエリに対して類似チャンクを返す
GET  /health    ヘルスチェック・コレクション統計
GET  /sources   登録済みソース一覧
```

---

## 5. データ設計

### 5.1 knowledgeディレクトリ構造

```
knowledge/
├── docs/
│   ├── claude-code/
│   ├── api/
│   └── build-with-claude/
├── blog/
├── release-notes/
└── academy/（手動追加）
```

### 5.2 Markdownファイルのメタデータ形式

```yaml
---
source_url: https://docs.anthropic.com/en/docs/claude-code/overview
fetched_at: 2026-05-26T00:00:00+09:00
category: docs
section: claude-code
---
```

### 5.3 ChromaDBコレクション設計

| 項目 | 値 |
|------|-----|
| コレクション名 | anthropic_docs |
| エンベディングモデル | multilingual-e5-base |
| 次元数 | 768 |
| 距離関数 | cosine |
| 永続化パス | ./chroma-data/ |

---

## 6. API設計

### 6.1 POST /search

**リクエスト**
```json
{
  "query": "Claude Codeでサブエージェントを使う方法",
  "top_k": 5,
  "category": "docs"
}
```

**レスポンス**
```json
{
  "results": [
    {
      "content": "サブエージェントはTask toolを使って...",
      "source": "knowledge/docs/claude-code/subagents.md",
      "score": 0.923,
      "section": "サブエージェントの基本"
    }
  ],
  "total": 5,
  "elapsed_ms": 312
}
```

---

## 7. CI/CDパイプライン設計

### 7.1 update.yml

```yaml
トリガー:
  - schedule: cron '0 15 * * *'  # 毎日00:00 JST
  - workflow_dispatch

ジョブ: update
  実行環境: ubuntu-latest
  タイムアウト: 30分

ステップ:
  1. actions/checkout@v4
  2. actions/setup-python@v5 (python-version: '3.11')
  3. pip install -r requirements.txt
  4. python scripts/fetch.py
  5. python scripts/embed.py
  6. git diff --quiet || git commit
  7. git push
```

### 7.2 シークレット管理

| シークレット名 | 用途 | 設定箇所 |
|--------------|------|---------|
| GITHUB_TOKEN | コミット・プッシュ | 自動付与 |
| ANTHROPIC_API_KEY | 将来的なAPI利用 | GitHub Secrets |

---

## 8. エラーハンドリング設計

| エラー種別 | 発生箇所 | 対応 |
|-----------|---------|------|
| HTTPタイムアウト | fetch.py | tenacityでリトライ2回後スキップ |
| HTTPステータスエラー | fetch.py | スキップ・ログ出力（他ソース継続） |
| パースエラー | fetch.py | スキップ・ログ出力 |
| ChromaDB接続エラー | embed.py / search_api.py | 例外送出・プロセス終了 |
| モデルロードエラー | embed.py | 例外送出・プロセス終了 |
| クエリバリデーションエラー | search_api.py | 422返却 |

---

## 9. 技術選定根拠

### 9.1 実行環境: Windowsネイティブ venv

| 比較項目 | Windowsネイティブ venv | WSL2 | Docker |
|---------|----------------------|------|--------|
| PCへの負荷 | 軽い | 中 | 重い |
| セットアップ | 簡単 | 中程度 | 複雑 |
| Claude Code対応 | ✅ | ✅ | △ |
| PowerShell統合 | ◎ | △ | △ |

**選定理由**: 最も軽量・シンプル。ChromaDBはWindowsネイティブで動作するためコンテナ不要。

### 9.2 ベクトルDB: ChromaDB

完全無料・ローカルファイルDB・pip一発・Gitバージョン管理可能。

### 9.3 エンベディング: multilingual-e5-base

日本語対応・完全無料・ローカル動作の3条件を満たす。

### 9.4 検索API: FastAPI

Swagger UI自動生成・Pydantic型バリデーション・将来的なClaude Codeツール連携を考慮。
