# Claude Knowledge Base：開発ガイド

作業ルールは [AGENTS.md](AGENTS.md) を参照。

- Python / FastAPI / ChromaDB。Windows PowerShellを基本環境とする。
- `python -m scripts.portfolio_demo --serve`：独立サンプルDBと検索画面。
- `python -m pytest tests/ -q`：HTTP・モデルを置き換えた単体／結合テスト。
- 構造保持分割はtarget=400 / max=480 / overlap=50。モデル入力上限512にはprefix等を含む。
- 変更時は [ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md) と [EVAL.md](docs/EVAL.md) を参照する。
- 配信・取得ワークフローは手動実行のみ。実通知をテストに使わない。
- キーの確認は設定の有無だけを表示する。個人用の環境変数・DB・通知先を移さない。
- 未同梱の学習ページは404、検索画面がない場合はSwagger UIへ誘導する。

セットアップと利用者向けの手順は [README.md](README.md) に記載。
