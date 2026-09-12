# 検索品質評価サマリー

vector検索（ChromaDB + multilingual-e5-base）、FTS5全文検索（SQLite bm25）、vector_rerank（vector top-30をCrossEncoderでrerankしてtop-10を評価）の3方式の比較。レイテンシはクエリ1件あたりの平均合計時間（vector検索+rerankの合計。vector_rerankのみ計測、単位ms）。

| backend | scope | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | avg latency(ms) |
|---|---|---|---|---|---|---|---|---|
| vector | overall | 40 | 0.625 | 0.825 | 0.85 | 0.85 | 0.7258 | - |
| vector | ja | 25 | 0.56 | 0.8 | 0.84 | 0.84 | 0.6813 | - |
| vector | en | 15 | 0.7333 | 0.8667 | 0.8667 | 0.8667 | 0.8 | - |
| fts | overall | 40 | 0.225 | 0.35 | 0.375 | 0.4 | 0.2938 | - |
| fts | ja | 25 | 0.08 | 0.12 | 0.12 | 0.12 | 0.0933 | - |
| fts | en | 15 | 0.4667 | 0.7333 | 0.8 | 0.8667 | 0.6278 | - |
| vector_rerank | overall | 40 | 0.575 | 0.775 | 0.8 | 0.85 | 0.6869 | 11012 |
| vector_rerank | ja | 25 | 0.52 | 0.72 | 0.76 | 0.8 | 0.6347 | 11012 |
| vector_rerank | en | 15 | 0.6667 | 0.8667 | 0.8667 | 0.9333 | 0.7741 | 11012 |

## チャンク方式比較（ページ単位・サンプル評価）

goldクエリの正解ページを全て含む150ページのサンプル（全1736ページ中、シード20260716）でのページ単位評価。チャンクをページURLへ写像し、重複ページを除いた順位でRecall@k / MRR@10 を算出。詳細は chunking_comparison.json / docs/EVAL.md。

| method | chunks | avg tokens | >512 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| A_structured_400 | 4288 | 351.9 | 0 | 40 | 0.725 | 0.875 | 0.9 | 0.925 | 0.8078 |
| B_structured_200 | 8395 | 181.9 | 0 | 40 | 0.825 | 0.925 | 0.925 | 0.925 | 0.875 |
| C_naive_400 | 4025 | 392.5 | 0 | 40 | 0.625 | 0.9 | 0.95 | 0.95 | 0.7529 |

評価対象外のgold問: 0件
