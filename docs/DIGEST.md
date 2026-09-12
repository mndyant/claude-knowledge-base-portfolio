# 日本語ダイジェスト配信（digest.py）設計メモ

**作成日**: 2026-07-10
**関連**: `scripts/digest.py` / `.github/workflows/digest.yml` / `scripts/fetch.py`

RAG（`chroma-data` + `search_api.py`）は疑問が湧いたときに調べる**プル型**。
digest.pyは「何が変わったか気づく」ための**プッシュ型**で、両者は補完関係にある。

## なぜこの設計か

### 実行タイミング: PT業務時間に合わせた1日2回

Anthropicの更新は本社（サンフランシスコ＝太平洋時間）の業務時間に集中する。
`digest.yml`は19:30 UTC（PT昼）と01:30 UTC（PT夕方＝日本の朝10:30）に実行する。
2回目が日本の朝に届くよう設計している。PTの夏冬時間差（±1h）とActionsの
cron遅延は日次ダイジェスト用途では許容範囲。

### ページ検出はURLマーカー基準（見出しの位置に依存しない）

llms-full.txtは2026年7月にフォーマットが変わり（71MB→93MB）、通常docsは
「見出し→URL」、APIリファレンス系（1,583ページ）は「URL→見出し」という
2つの構造が混在するようになった。`split_llms_full.py`のページ境界検出
（`---`の直後がH1、という6月版の仮定）は新形式だと472/1,720ページしか
検出できない。digest.pyは各ページに必ず1つある`**URL:**`マーカー行を
基準にすることで、見出しの前後どちらにあっても両形式に対応している
（`collect_docs_pages`参照）。RAG側のチャンキングは未追従（後述）。

### Geminiは固定モデル名でなく`-latest`エイリアス

2026-07-10、運用中に`gemini-2.5-flash-lite`が404で廃止されているのを発見した。
無人運用のボットで固定モデル名を使うと、モデル廃止のタイミングで突然壊れる。
`gemini-flash-lite-latest`のような`-latest`エイリアスを使い、それでも404の
場合に備えて候補リスト（`MODEL_CANDIDATES`）へのフォールバックも入れている。

### 新ソース追加時は過去分を静かにベースライン化する

既存運用に新しいソース（例: status）を追加すると、そのソースの全件が
「前回状態に存在しない」ため一斉に「新着」として通知されてしまう。
これは実際に発生した問題で、status.claude.com追加時に過去のインシデント
21件が一気に通知される想定外の挙動になっていた。

対策として`diff_items`は、**そのitemのsourceが前回状態に一件も存在しない
場合は通知対象から除外**する（`known_sources`チェック）。除外された項目も
`save_state`では保存されるため、次回以降は正しく差分検知される。これは
「状態ファイル自体が存在しない＝真の初回実行」の初期化ロジックと同じ考え方を
ソース単位に拡張したもの。

新ソースを追加したら、ログの
`新規ソースを検知（status）: 通知せずベースライン化します`
が出ることを確認する（`python scripts/digest.py --dry-run`で確認可能）。

### 情報源の使い分け

| 知りたいこと | ソース | 理由 |
|-------------|--------|------|
| APIドキュメントの変更 | docs（llms-full.txt） | 一次情報そのもの |
| 新機能・方針発表 | blog | 公式RSS（404廃止済み→ミラー使用） |
| Claude CodeのCLI変更 | release-notes | GitHub Releases |
| **モデル廃止・移行スケジュール** | docs内の`model-deprecations`ページ | 既にRAGコーパスに含まれる。新規ソース不要 |
| **インシデント・障害・アクセス一時停止** | **status**（status.claude.com） | docs/blogに載らない運用上のお知らせ。RSS: `https://status.claude.com/history.rss` |

「Fable 5の利用制限延長」のような運用上のお知らせは、実際に
`status.claude.com`のインシデント（例:
"We've suspended access to Claude Mythos 5 and Claude Fable 5"）として
配信されることを2026-07-10に確認済み。

Xポストは最速だが有料APIが必要なため採用しない。この設計（1日2回の
バッチ）では、docs/blog/status自体がXの投稿元（一次情報）であり、
Xはそこへのリンクを告知するだけなので、速度優位はバッチ間隔で吸収される。

## 既知の未対応事項

- **split_llms_full.pyのページ境界検出が新フォーマットに未追従**
  （上記「ページ検出」参照）。次回チャンキング時にURL_MARKER_RE方式へ
  寄せること。詳細は`docs/ENGINEERING_NOTES.md`のの設計メモにも記載
- knowledge/academyディレクトリ（手動追加のAcademyコース）はdigest.pyの
  監視対象に含めていない（手動追加物のため差分通知の対象外という判断）
