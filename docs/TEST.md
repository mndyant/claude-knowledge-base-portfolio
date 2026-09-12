# テスト仕様書

**プロジェクト名**: claude-knowledge-base  
**バージョン**: 1.0.0  
**作成日**: 2026-05-26  
**更新日**: 2026-07-18

> 本書が詳細にケース設計しているのは初期実装（fetch / embed / search_api /
> 結合 / E2E）の22件。その後、検索品質評価（`eval_retrieval`）・回答生成
> （`rag_answer`）・hallucination評価（`eval_answers`）・チャンク方式比較
> （`eval_chunking`）・reranker（`rerank`）・日本語ダイジェスト（`digest`）
> のテストが追加され、2026-07-18時点の総数は **141件**
> （`pytest tests/ --collect-only -q` で確認可能）。新規追加分は個別の
> TC表を持たず、各テストファイルのdocstring・アサーションが仕様を兼ねる。

---

## 目次

1. [テスト方針](#1-テスト方針)
2. [テスト環境](#2-テスト環境)
3. [単体テスト仕様](#3-単体テスト仕様)
4. [結合テスト仕様](#4-結合テスト仕様)
5. [E2Eテスト仕様](#5-e2eテスト仕様)
6. [テストケース一覧](#6-テストケース一覧)
7. [実行方法](#7-実行方法)

---

## 1. テスト方針

### 1.1 テストの目的

| 目的 | 内容 |
|------|------|
| 正常系の保証 | 各コンポーネントが仕様通りに動作することを確認する |
| 異常系の堅牢性 | ネットワークエラー・不正入力に対して適切に処理することを確認する |
| 回帰防止 | 変更による既存機能の破壊を検出する |

### 1.2 テストレベルと責務

```
E2Eテスト       GitHub Actions実行 → ChromaDB登録 → API検索 の一連を検証
    │
結合テスト       fetch.py → embed.py / embed.py → search_api.py の連携を検証
    │
単体テスト       各関数の入出力・エラーハンドリングを検証
```

### 1.3 カバレッジ目標

| テストレベル | 対象 | 目標カバレッジ |
|-------------|------|-------------|
| 単体テスト | scripts/*.py | 80%以上 |
| 結合テスト | スクリプト間連携 | 主要フロー100% |
| E2Eテスト | パイプライン全体 | 正常系100% |

### 1.4 モック方針

- 外部HTTP通信は**すべてモック**（requests-mock使用）
- ChromaDBは**テスト用一時コレクション**を使用（本番コレクションを汚染しない）
- sentence-transformersモデルは**軽量モデルに差し替え**（CI時間短縮）

---

## 2. テスト環境

### 2.1 必要パッケージ

```
pytest==8.x
pytest-cov
requests-mock
httpx              # FastAPIテストクライアント用
```

### 2.2 環境変数

| 変数名 | テスト時の値 | 説明 |
|--------|------------|------|
| CHROMA_TEST_DIR | ./chroma_test/ | テスト用ChromaDBパス |
| EMBED_MODEL | paraphrase-MiniLM-L3-v2 | CI用軽量モデル |

### 2.3 フィクスチャ

```python
# tests/conftest.py

@pytest.fixture
def mock_llms_txt():
    """llms-full.txtのモックレスポンスを返す"""

@pytest.fixture
def sample_chunks():
    """テスト用チャンクリストを返す"""

@pytest.fixture
def test_chroma_client():
    """テスト用ChromaDBクライアントを生成し、テスト後に削除する"""

@pytest.fixture
def api_client():
    """FastAPIテストクライアントを返す"""
```

---

## 3. 単体テスト仕様

### 3.1 test_fetch.py

#### TC-F-001: llms-full.txt 正常取得

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch_llms_full_txt()` |
| 前提条件 | HTTPレスポンス200・有効なMarkdown本文 |
| 入力 | モックレスポンス（Markdownテキスト） |
| 期待結果 | `knowledge/docs/` 配下にファイルが生成される |
| 確認方法 | ファイル存在確認・内容の非空確認 |

#### TC-F-002: HTTPタイムアウト時のリトライ

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch_llms_full_txt()` |
| 前提条件 | 1回目・2回目タイムアウト / 3回目成功 |
| 入力 | モック: Timeout → Timeout → 200 |
| 期待結果 | リトライ2回後に成功・ファイル生成 |
| 確認方法 | ファイル生成確認・requests呼び出し回数の確認 |

#### TC-F-003: HTTP 500エラー時のスキップ

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch_llms_full_txt()` |
| 前提条件 | HTTPレスポンス500 |
| 入力 | モックレスポンス 500 |
| 期待結果 | 例外を送出せず処理継続・ファイル非生成 |
| 確認方法 | 例外なし確認・ファイル非存在確認 |

#### TC-F-004: 差分なし時のスキップ

| 項目 | 内容 |
|------|------|
| テスト対象 | `is_updated()` |
| 前提条件 | 既存ファイルと取得内容が同一 |
| 入力 | 同一ハッシュのコンテンツ |
| 期待結果 | `False` を返す |
| 確認方法 | 戻り値の確認 |

#### TC-F-005: 差分あり時の更新検出

| 項目 | 内容 |
|------|------|
| テスト対象 | `is_updated()` |
| 前提条件 | 既存ファイルと取得内容が異なる |
| 入力 | 異なるハッシュのコンテンツ |
| 期待結果 | `True` を返す |
| 確認方法 | 戻り値の確認 |

#### TC-F-006: ブログ記事の正常取得

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch_blog_posts()` |
| 前提条件 | RSSフィード正常レスポンス（3件） |
| 入力 | モックRSSフィード（記事3件） |
| 期待結果 | `knowledge/blog/` 配下に3ファイル生成 |
| 確認方法 | ファイル数確認・Frontmatter存在確認 |

---

### 3.2 test_embed.py

#### TC-E-001: Markdownファイルの正常読み込み

| 項目 | 内容 |
|------|------|
| テスト対象 | `load_documents()` |
| 前提条件 | knowledge/配下に3ファイル存在 |
| 入力 | テスト用Markdownファイル3件 |
| 期待結果 | Documentオブジェクト3件返却 |
| 確認方法 | 返却リストの長さ・content非空確認 |

#### TC-E-002: チャンク分割の境界値

| 項目 | 内容 |
|------|------|
| テスト対象 | `split_chunks()` |
| 前提条件 | 1,024トークンのテキスト1件 |
| 入力 | 1,024トークンのMarkdown |
| 期待結果 | チャンク数が2以上・各チャンク512トークン以内 |
| 確認方法 | チャンク数・各チャンクのトークン数確認 |

#### TC-E-003: チャンクのオーバーラップ確認

| 項目 | 内容 |
|------|------|
| テスト対象 | `split_chunks()` |
| 前提条件 | 連続するテキスト |
| 入力 | 1,024トークンのMarkdown |
| 期待結果 | 隣接チャンク間に50トークンの重複が存在する |
| 確認方法 | 隣接チャンクの末尾・先頭テキストの重複確認 |

#### TC-E-004: ChromaDBへの正常登録

| 項目 | 内容 |
|------|------|
| テスト対象 | `embed_and_store()` |
| 前提条件 | テスト用ChromaDBコレクション・チャンク3件 |
| 入力 | サンプルチャンク3件 |
| 期待結果 | ChromaDBに3件登録・IDが正しく付与される |
| 確認方法 | `collection.count()` で件数確認 |

#### TC-E-005: 既存ドキュメントのupsert

| 項目 | 内容 |
|------|------|
| テスト対象 | `embed_and_store()` |
| 前提条件 | 同一IDのチャンクが既存 |
| 入力 | 同一IDで内容が異なるチャンク |
| 期待結果 | 件数変化なし・内容が更新される |
| 確認方法 | 件数確認・内容の変更確認 |

---

### 3.3 test_search_api.py

#### TC-S-001: 正常な検索リクエスト

| 項目 | 内容 |
|------|------|
| テスト対象 | `POST /search` |
| 前提条件 | ChromaDBに10件登録済み |
| 入力 | `{"query": "Claude Code", "top_k": 3}` |
| 期待結果 | 200 / results配列3件 / 各要素にcontent・source・score |
| 確認方法 | レスポンスステータス・JSONスキーマ確認 |

#### TC-S-002: top_kの上限バリデーション

| 項目 | 内容 |
|------|------|
| テスト対象 | `POST /search` |
| 前提条件 | なし |
| 入力 | `{"query": "test", "top_k": 100}` |
| 期待結果 | 422 Unprocessable Entity |
| 確認方法 | レスポンスステータス確認 |

#### TC-S-003: 空クエリのバリデーション

| 項目 | 内容 |
|------|------|
| テスト対象 | `POST /search` |
| 前提条件 | なし |
| 入力 | `{"query": "", "top_k": 5}` |
| 期待結果 | 422 Unprocessable Entity |
| 確認方法 | レスポンスステータス確認 |

#### TC-S-004: カテゴリフィルタ

| 項目 | 内容 |
|------|------|
| テスト対象 | `POST /search` |
| 前提条件 | docs・blog混在で10件登録済み |
| 入力 | `{"query": "test", "top_k": 5, "category": "docs"}` |
| 期待結果 | 全resultsのsourceが `knowledge/docs/` 配下 |
| 確認方法 | 各resultのsourceパス確認 |

#### TC-S-005: ヘルスチェック

| 項目 | 内容 |
|------|------|
| テスト対象 | `GET /health` |
| 前提条件 | ChromaDB正常稼働 |
| 入力 | なし |
| 期待結果 | 200 / `{"status": "ok", "total_documents": N}` |
| 確認方法 | ステータス・JSONキー存在確認 |

#### TC-S-006: ChromaDB未初期化時のエラー

| 項目 | 内容 |
|------|------|
| テスト対象 | `POST /search` |
| 前提条件 | ChromaDBコレクション未作成 |
| 入力 | `{"query": "test", "top_k": 5}` |
| 期待結果 | 503 / errorメッセージに案内文言含む |
| 確認方法 | ステータス・エラーメッセージ確認 |

---

## 4. 結合テスト仕様

### 4.1 fetch → embed 連携テスト

#### TC-I-001: 取得からベクトル登録までの一連フロー

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch.py` → `embed.py` |
| 前提条件 | モックHTTPサーバー起動・テスト用ChromaDB |
| 手順 | 1. `fetch_llms_full_txt()` 実行<br>2. 生成されたファイルを `embed_and_store()` に渡す |
| 期待結果 | ChromaDBにチャンクが1件以上登録される |
| 確認方法 | `get_collection_stats()` で件数確認 |

#### TC-I-002: 差分更新の正確性

| 項目 | 内容 |
|------|------|
| テスト対象 | `fetch.py` → `embed.py` |
| 前提条件 | 初回取得・登録済み（5チャンク） |
| 手順 | 1. 1ファイルのみ内容変更<br>2. `fetch.py` 再実行<br>3. `embed.py` 再実行 |
| 期待結果 | 変更ファイル分のみupsert・総件数変化なし |
| 確認方法 | ChromaDB件数・変更済みチャンクの内容確認 |

---

### 4.2 embed → search_api 連携テスト

#### TC-I-003: 登録データの検索可能性

| 項目 | 内容 |
|------|------|
| テスト対象 | `embed.py` → `search_api.py` |
| 前提条件 | 既知テキストをChromaDBに登録済み |
| 手順 | 1. 既知テキストのキーワードでPOST /search |
| 期待結果 | 既知テキストが検索結果1位に含まれる |
| 確認方法 | results[0].scoreが0.8以上 |

---

## 5. E2Eテスト仕様

### 5.1 パイプライン全体テスト

#### TC-E2E-001: GitHub Actions相当の一連実行

| 項目 | 内容 |
|------|------|
| テスト対象 | fetch.py → embed.py → search_api（起動確認） |
| 前提条件 | モックHTTPサーバー・テスト用ChromaDB |
| 手順 | 1. `python scripts/fetch.py` 実行<br>2. `python scripts/embed.py` 実行<br>3. `uvicorn scripts.search_api:app` 起動<br>4. `GET /health` でステータス確認<br>5. `POST /search` で検索実行 |
| 期待結果 | 全ステップが正常終了・検索結果が1件以上返却 |
| 確認方法 | 各ステップの終了コード・APIレスポンス確認 |

#### TC-E2E-002: 差分更新の継続実行

| 項目 | 内容 |
|------|------|
| テスト対象 | パイプライン2回実行 |
| 手順 | 1. 1回目: fetch → embed 実行・件数N記録<br>2. ソース内容を一部変更<br>3. 2回目: fetch → embed 実行 |
| 期待結果 | 2回目実行後の件数がNと同じ（重複登録なし） |
| 確認方法 | `get_collection_stats()` の件数比較 |

---

## 6. テストケース一覧

| ID | 対象 | 種別 | 内容 | 優先度 |
|----|------|------|------|--------|
| TC-F-001 | fetch.py | 単体 | llms-full.txt 正常取得 | High |
| TC-F-002 | fetch.py | 単体 | タイムアウト時リトライ | High |
| TC-F-003 | fetch.py | 単体 | HTTP500エラー時スキップ | High |
| TC-F-004 | fetch.py | 単体 | 差分なし時スキップ | Medium |
| TC-F-005 | fetch.py | 単体 | 差分あり時更新検出 | Medium |
| TC-F-006 | fetch.py | 単体 | ブログ記事正常取得 | Medium |
| TC-E-001 | embed.py | 単体 | Markdownファイル読み込み | High |
| TC-E-002 | embed.py | 単体 | チャンク分割境界値 | High |
| TC-E-003 | embed.py | 単体 | オーバーラップ確認 | Medium |
| TC-E-004 | embed.py | 単体 | ChromaDB正常登録 | High |
| TC-E-005 | embed.py | 単体 | 既存ドキュメントupsert | High |
| TC-S-001 | search_api.py | 単体 | 正常検索リクエスト | High |
| TC-S-002 | search_api.py | 単体 | top_k上限バリデーション | Medium |
| TC-S-003 | search_api.py | 単体 | 空クエリバリデーション | Medium |
| TC-S-004 | search_api.py | 単体 | カテゴリフィルタ | Medium |
| TC-S-005 | search_api.py | 単体 | ヘルスチェック | High |
| TC-S-006 | search_api.py | 単体 | DB未初期化エラー | High |
| TC-I-001 | fetch→embed | 結合 | 取得〜登録フロー | High |
| TC-I-002 | fetch→embed | 結合 | 差分更新の正確性 | High |
| TC-I-003 | embed→API | 結合 | 登録データの検索可能性 | High |
| TC-E2E-001 | パイプライン全体 | E2E | 一連実行 | High |
| TC-E2E-002 | パイプライン全体 | E2E | 差分更新継続実行 | Medium |

**合計: 22件**（High: 14件 / Medium: 8件、初期実装分のみ。全体141件については冒頭の注記を参照）

---

## 7. 実行方法

### 7.1 全テスト実行

```bash
pytest tests/ -v
```

### 7.2 カバレッジ計測付き実行

```bash
pytest tests/ --cov=scripts --cov-report=term-missing
```

### 7.3 テストレベル別実行

```bash
# 単体テストのみ
pytest tests/test_fetch.py tests/test_embed.py tests/test_search_api.py -v

# 結合テストのみ
pytest tests/test_integration.py -v

# E2Eテストのみ
pytest tests/test_e2e.py -v
```

### 7.4 GitHub Actionsでのテスト実行

```yaml
# .github/workflows/test.yml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov requests-mock httpx
      - run: pytest tests/ --cov=scripts --cov-report=xml
```
