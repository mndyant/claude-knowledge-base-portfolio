# Claude Knowledge Base

公式文書を収集・構造を保って分割し、ローカルのベクトルDBから検索するRAGアプリです。日本語の質問から英語の文書を探し、必要に応じて出典付きの回答を生成します。

**見どころ：正常に動く処理と、検索品質が保たれた処理を分けて検証したこと。**
チャンクがモデルの入力上限を超えていた問題を実トークナイザーで特定し、構造保持分割、検索評価、再開可能な差分更新へつなげました。

- [設計と改善の要点](docs/ENGINEERING_NOTES.md)
- [検索・回答品質の評価記録](docs/EVAL.md)
- [Windows検証の範囲](docs/WINDOWS_VALIDATION.md)

![日本語の質問から英語の根拠文書を検索したデモ画面](docs/images/sample-search.png)

自作サンプル4文書での画面です。回答生成を使わず、原文を展開して根拠を確認できます。一般公開の常設デモはなく、以下の手順でローカル起動します。

## まず動かす：小さなサンプル検索

自作サンプル4件を、実際の `multilingual-e5-base` とChromaDBで検索します。個人用DB・取得文書・APIキーは不要です。サンプルの順位は動作確認用で、検索品質ベンチマークとは別です。

Windows PowerShellでリポジトリのルートから実行します。既存モデルのキャッシュがあればネットワークを使いません。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

[検索画面](http://127.0.0.1:8001/) ／ [Swagger UI](http://127.0.0.1:8001/docs)

「なぜチャンクのトークン数を測る必要がある？」を検索し、先頭の文書の原文を開いて根拠を確認できます。サンプルDBは `.portfolio-demo/` に作られ、再実行しても重複登録しません。8001番が使用中なら `--port 18001` を追加してください。

モデルが未取得の場合だけ、先に以下でダウンロードします。モデル取得には通信・ディスク容量・メモリが必要です。API利用料は発生しません。

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
```

モデルを用意したら起動します。キャッシュが見つからない場合は、上のモデル取得を同じ仮想環境・ユーザーで実行してください。

```powershell
.\.venv\Scripts\python.exe -m scripts.portfolio_demo --serve
```

デモは回答生成用の環境変数を使わず、外部LLM・通知サービスを呼びません。終了は `Ctrl+C` です。

## 構成

```mermaid
flowchart LR
    A[公式文書の取得] --> B[構造を保った分割]
    B --> C[e5によるベクトル化]
    C --> D[ChromaDB]
    Q[日本語の質問] --> E[FastAPI検索]
    D --> E
    E --> F[検索結果と根拠文書]
    E --> G[任意の回答生成]
```

| 要素 | 技術・役割 |
|---|---|
| 取得 | requests / Beautiful Soup / tenacity |
| 分割 | Markdown構造保持、e5の実トークナイザーで上限確認 |
| 検索 | sentence-transformers / ChromaDB / FastAPI |
| 画面 | HTML / CSS / JavaScript。検索のみが初期設定 |
| テスト | pytest、HTTPモック、独立したテストDB |
| 更新 | GitHub Actionsの手動実行。テストCIはpush・PRで実行 |

## 検索品質の記録

以下は2026年7月に測定した既存の評価記録です。今回のサンプル4件による動作確認結果とは区別しています。母集団の異なる行同士を直接比較しないでください。

| 評価条件 | Recall@5 | MRR@10 |
|---|---:|---:|
| ベクトル検索・全体評価 | 0.875 | 0.7383 |
| SQLite FTS5・全体評価 | 0.375 | 0.2938 |
| 構造保持200/240・サブセット評価 | 0.925 | 0.8750 |

サブセット内の比較では、400→200トークンへの小チャンク化でMRR@10が0.8078→0.8750（差0.0672）。チャンク数と処理負荷が増えるため、全体への適用は見送っています。詳細・限界・評価条件は [EVAL.md](docs/EVAL.md) を参照してください。

## 公式文書を使う場合

```powershell
.\.venv\Scripts\python.exe scripts/fetch.py
.\.venv\Scripts\python.exe scripts/split_llms_full.py --input knowledge/docs/llms-full.md --output data/processed/chunks.jsonl --stats data/processed/stats.json
.\.venv\Scripts\python.exe scripts/reembed.py --input data/processed/chunks.jsonl
.\.venv\Scripts\python.exe -m uvicorn scripts.search_api:app --host 127.0.0.1 --port 8001
```

全量の取得・分割・埋め込みはサンプルより時間がかかります。収集文書・DB・配信状態は配布物に含みません。分割統計の `oversize_chunk_count == 0` と実トークナイザーの使用を確認してから登録します。

検索APIの例：

```json
{"query":"画像をAPIに渡す方法", "top_k":5, "generate_answer":false}
```

`generate_answer: false` はクエリ展開・回答生成を含め外部LLMを呼びません。通常起動で回答生成を選ぶ場合は `GEMINI_API_KEY` が必要です。既定値はAPI・画面とも無効（検索のみ）で、`generate_answer: true` を明示した場合だけ回答を生成します。利用APIの条件・モデルの提供状況は実接続前に確認してください。

設定は起動プロセスの環境変数から読みます。`.env.example` は変数名の参考で、`.env` の自動読込はしません。APIキーはOSや実行環境の秘密設定で渡し、コマンド例・Git履歴には記載しないでください。検索クエリの上限は4000文字です。

## テスト

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

単体・結合・E2EではHTTPをモックし、モデルを決定的なテストダブルへ置き換えています。実モデルでの検索はサンプルデモで別に確認します。検証環境と未確認範囲は [WINDOWS_VALIDATION.md](docs/WINDOWS_VALIDATION.md) に記載します。

## 公開範囲

更新・配信ワークフローは手動実行です。資格情報・既存DB・取得コンテンツ・個人用の運用資料は同梱しません。コードの公開ライセンスは著作権者表記とともに確定する予定です。公式文書そのものの権利とコードのライセンスは別です。

公開構成では配信状態をGitに保存しないため、手動digest workflowは毎回初回の記録のみとなり、継続配信には使えません。実運用には別途、状態の永続化と通知先の設定が必要です。更新workflowは文書・DBをダウンロード可能なartifactとして公開しません。
