"""RAG検索品質の評価ハーネス（vector / FTS5 / vector_rerank の比較）。

``eval/dataset.jsonl`` の正解付きクエリ集合に対し、ChromaDBのベクトル検索
（``scripts/search_api.py`` と同じembeddingモデル・prefix・コレクション）、
SQLite FTS5（標準ライブラリのみ・bm25ランキング）、およびvector top-30を
CrossEncoderでrerankする2段構成（``scripts/rerank.py``）の3方式で
top-N検索を行い、Recall@1/3/5/10・MRR@10を算出する。vector_rerankでは
クエリ単位のレイテンシ（vector単体 / rerank追加分）も記録する。

ChromaDB自体は外部の``--db-path``で指定する永続化ディレクトリを直接参照する
（リポジトリ内へはコピーしない）。長時間プロセスが強制終了される環境を想定し、
クエリ単位の中間結果を ``eval/cache/raw_{backend}.jsonl`` へ逐次保存し、
再実行時は処理済みクエリをスキップする。

使い方:
    python scripts/eval_retrieval.py --db-path <chroma-dataディレクトリ>
    python scripts/eval_retrieval.py --db-path <...> --backend vector --top-n 10
    python scripts/eval_retrieval.py --db-path <...> --backend vector_rerank
    python scripts/eval_retrieval.py --db-path <...> --rebuild-index

vector_rerankはrerankerモデルの初回ダウンロードにネットワークが必要
（embed.pyが既定でHFをオフライン化するため ``EMBED_LOCAL_FILES_ONLY=0``
を指定して実行する）。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import rerank as rerank_module  # noqa: E402
from scripts.embed import (
    embed_texts,
    get_chroma_client,
    get_collection_name,
)  # noqa: E402

DEFAULT_DATASET = Path("eval/dataset.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
CACHE_DIR = Path("eval/cache")
FTS_DB_PATH = CACHE_DIR / "fts.sqlite"
PAGE_URL_CACHE = CACHE_DIR / "page_urls.json"

RECALL_LEVELS = (1, 3, 5, 10)
MRR_CUTOFF = 10

# vector_rerankバックエンド: vector検索でこの件数まで粗く候補を絞り込んでから
# CrossEncoderでrerankする（top-30 -> top-10）。
RERANK_CANDIDATE_N = 30

# llms-full.txt由来のチャンクは、ページ内最初のチャンク本文に
# "URL: https://..." という行を含む（split_llms_full.py参照）。
# metadataの"source"は常に"llms-full.txt"で実URLを持たないため、
# ここから実ページURLを復元してurl_contains判定に使う。
URL_RE = re.compile(r"URL:\s*(https?://\S+)")


@dataclass
class GoldItem:
    """評価データセットの1問。"""

    id: str
    query: str
    lang: str
    category: str
    relevant: list[dict[str, str]]
    notes: str = ""


@dataclass
class RetrievedHit:
    """検索結果1件（gold照合前の生データ）。"""

    rank: int
    chunk_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # rerank用の本文（vector_rerankバックエンドのみ使用。他は空文字のまま）。
    document: str = ""


def load_dataset(path: Path) -> list[GoldItem]:
    """評価データセット（JSONL）を読み込む。"""
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            items.append(
                GoldItem(
                    id=record["id"],
                    query=record["query"],
                    lang=record["lang"],
                    category=record.get("category", ""),
                    relevant=record["relevant"],
                    notes=record.get("notes", ""),
                )
            )
    return items


# --------------------------------------------------------------------------
# gold照合ロジック
# --------------------------------------------------------------------------


def build_page_url_index(collection: Any) -> dict[str, str]:
    """page_index -> 実ページURL のマップを作る（初回のみ全件スキャンしキャッシュ）。"""
    if PAGE_URL_CACHE.exists():
        return json.loads(PAGE_URL_CACHE.read_text(encoding="utf-8"))

    data = collection.get(include=["metadatas", "documents"])
    urls: dict[str, str] = {}
    for metadata, document in zip(data["metadatas"], data["documents"], strict=False):
        metadata = metadata or {}
        page_index = str(metadata.get("page_index", ""))
        if not page_index or page_index in urls:
            continue
        match = URL_RE.search(document or "")
        if match:
            urls[page_index] = match.group(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_URL_CACHE.write_text(json.dumps(urls, ensure_ascii=False), encoding="utf-8")
    return urls


def matches_criterion(
    criterion: dict[str, str], metadata: dict[str, Any], resolved_url: str
) -> bool:
    """1つのgold criterionがチャンクのmetadataにマッチするか判定する（大小文字無視）。"""
    value = str(criterion.get("value", "")).lower()
    ctype = criterion.get("type")
    if ctype == "heading_contains":
        return value in str(metadata.get("heading_path", "")).lower()
    if ctype == "url_contains":
        haystack = f"{metadata.get('source', '')} {resolved_url or ''}".lower()
        return value in haystack
    return False


def is_relevant(
    item: GoldItem, metadata: dict[str, Any], page_urls: dict[str, str]
) -> bool:
    """チャンクがgold itemのいずれかのrelevant criterionにマッチするか判定する。"""
    resolved_url = page_urls.get(str(metadata.get("page_index", "")), "")
    return any(matches_criterion(c, metadata, resolved_url) for c in item.relevant)


# --------------------------------------------------------------------------
# vector backend
# --------------------------------------------------------------------------


def vector_search(query: str, collection: Any, top_n: int) -> list[RetrievedHit]:
    """search_api.pyと同じembedding手順（query: prefix・同一コレクション）で検索する。

    ``documents`` はChromaDBのデフォルトincludeに含まれるため常に取得するが、
    vectorバックエンド自体はこれを使わない（vector_rerankがrerank対象の
    本文として利用する）。
    """
    query_embedding = embed_texts([query], is_query=True)
    response = collection.query(query_embeddings=query_embedding, n_results=top_n)

    ids = (response.get("ids") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]

    hits: list[RetrievedHit] = []
    for rank, (chunk_id, metadata, distance, document) in enumerate(
        zip(ids, metadatas, distances, documents, strict=False), start=1
    ):
        hits.append(
            RetrievedHit(
                rank=rank,
                chunk_id=chunk_id,
                score=round(1.0 - float(distance), 4),
                metadata=metadata or {},
                document=document or "",
            )
        )
    return hits


# --------------------------------------------------------------------------
# vector_rerank backend（vector top-30 -> CrossEncoderでrerank -> top-10）
# --------------------------------------------------------------------------


def vector_rerank_search(
    query: str, collection: Any, top_n: int, candidate_n: int = RERANK_CANDIDATE_N
) -> tuple[list[RetrievedHit], float, float]:
    """vector検索でcandidate_n件に絞り込み、CrossEncoderでrerankしてtop_n件を返す。

    戻り値は ``(hits, vector_latency_seconds, rerank_latency_seconds)``。
    レイテンシはvector単体との比較のためにクエリ単位で計測する。
    """
    vector_started = time.time()
    candidates = vector_search(query, collection, candidate_n)
    vector_latency = time.time() - vector_started

    rerank_started = time.time()
    reranked = rerank_module.rerank(
        query,
        [
            {
                "chunk_id": hit.chunk_id,
                "text": hit.document,
                "metadata": hit.metadata,
            }
            for hit in candidates
        ],
        top_k=top_n,
    )
    rerank_latency = time.time() - rerank_started

    hits = [
        RetrievedHit(
            rank=rank,
            chunk_id=item["chunk_id"],
            score=round(float(item["rerank_score"]), 4),
            metadata=item.get("metadata") or {},
        )
        for rank, item in enumerate(reranked, start=1)
    ]
    return hits, vector_latency, rerank_latency


# --------------------------------------------------------------------------
# FTS5 backend
# --------------------------------------------------------------------------


def build_fts_index(collection: Any, rebuild: bool) -> sqlite3.Connection:
    """ChromaDBの全チャンクをSQLite FTS5（unicode61・bm25）へダンプする。

    ``eval/cache/fts.sqlite`` に構築済みならそのまま再利用する
    （``rebuild=True`` の場合のみ作り直す）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild and FTS_DB_PATH.exists():
        FTS_DB_PATH.unlink()

    is_new = not FTS_DB_PATH.exists()
    conn = sqlite3.connect(FTS_DB_PATH)

    if is_new:
        conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "chunk_id UNINDEXED, "
            "content, "
            "heading_path UNINDEXED, "
            "source UNINDEXED, "
            "page_index UNINDEXED, "
            "category UNINDEXED, "
            "tokenize='unicode61'"
            ")"
        )
        data = collection.get(include=["metadatas", "documents"])
        rows = []
        for chunk_id, metadata, document in zip(
            data["ids"], data["metadatas"], data["documents"], strict=False
        ):
            metadata = metadata or {}
            rows.append(
                (
                    chunk_id,
                    document or "",
                    str(metadata.get("heading_path", "")),
                    str(metadata.get("source", "")),
                    str(metadata.get("page_index", "")),
                    str(metadata.get("category", "")),
                )
            )
        conn.executemany(
            "INSERT INTO chunks "
            "(chunk_id, content, heading_path, source, page_index, category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    return conn


def _fts_match_query(query: str) -> str:
    """自然文クエリをFTS5 MATCH構文として安全なOR結合トークン列へ変換する。"""
    tokens = re.findall(r"\w+", query.lower())
    return " OR ".join(tokens)


def fts_search(conn: sqlite3.Connection, query: str, top_n: int) -> list[RetrievedHit]:
    """SQLite FTS5でbm25ランキングのtop-N検索を行う。"""
    match_query = _fts_match_query(query)
    if not match_query:
        return []

    cursor = conn.execute(
        "SELECT chunk_id, heading_path, source, page_index, category, "
        "bm25(chunks) AS raw_score "
        "FROM chunks WHERE chunks MATCH ? ORDER BY raw_score LIMIT ?",
        (match_query, top_n),
    )
    hits: list[RetrievedHit] = []
    for rank, row in enumerate(cursor.fetchall(), start=1):
        chunk_id, heading_path, source, page_index, category, raw_score = row
        hits.append(
            RetrievedHit(
                rank=rank,
                chunk_id=chunk_id,
                # bm25()は小さいほど良いスコアなので、vector側と揃えて符号反転する
                # （大きいほど良い）。
                score=round(-float(raw_score), 4),
                metadata={
                    "heading_path": heading_path,
                    "source": source,
                    "page_index": page_index,
                    "category": category,
                },
            )
        )
    return hits


# --------------------------------------------------------------------------
# 中間結果キャッシュ（クエリ単位・再開可能）
# --------------------------------------------------------------------------


def _raw_cache_path(backend: str) -> Path:
    return CACHE_DIR / f"raw_{backend}.jsonl"


def load_raw_cache(backend: str) -> dict[str, dict[str, Any]]:
    """クエリ単位の中間結果キャッシュを読み込む（存在しなければ空辞書）。"""
    path = _raw_cache_path(backend)
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["id"]] = record
    return records


def append_raw_cache(backend: str, record: dict[str, Any]) -> None:
    """1クエリ分の中間結果を追記する（プロセスが途中で落ちても再実行で再利用できる）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_cache_path(backend)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_query_record(
    item: GoldItem,
    hits: list[RetrievedHit],
    page_urls: dict[str, str],
    latency_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """1クエリ分の検索結果をgold照合し、保存用レコードへ変換する。

    ``latency_seconds`` はvector_rerankバックエンド用の計測値
    （例: ``{"vector_ms": ..., "rerank_ms": ..., "total_ms": ...}``）。
    他バックエンドでは省略する。
    """
    first_hit_rank: int | None = None
    hit_records = []
    for hit in hits:
        relevant = is_relevant(item, hit.metadata, page_urls)
        if relevant and first_hit_rank is None:
            first_hit_rank = hit.rank
        hit_records.append(
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "heading_path": str(hit.metadata.get("heading_path", "")),
                "source": str(hit.metadata.get("source", "")),
                "is_relevant": relevant,
            }
        )
    record: dict[str, Any] = {
        "id": item.id,
        "query": item.query,
        "lang": item.lang,
        "category": item.category,
        "notes": item.notes,
        "first_hit_rank": first_hit_rank,
        "hits": hit_records,
    }
    if latency_seconds:
        record.update(latency_seconds)
    return record


def run_backend(
    backend: str,
    dataset: list[GoldItem],
    top_n: int,
    db_path: str,
    rebuild_index: bool,
) -> dict[str, Any]:
    """1つのbackendで全クエリを評価し、集計結果を返す。"""
    client = get_chroma_client(path=db_path)
    collection = client.get_collection(get_collection_name())
    page_urls = build_page_url_index(collection)

    conn: sqlite3.Connection | None = None
    if backend == "fts":
        conn = build_fts_index(collection, rebuild=rebuild_index)

    cached = load_raw_cache(backend)
    todo = [item for item in dataset if item.id not in cached]
    print(f"[{backend}] {len(cached)} cached, {len(todo)} to run", flush=True)

    started = time.time()
    for index, item in enumerate(todo, start=1):
        latency_seconds: dict[str, float] | None = None
        if backend == "vector":
            hits = vector_search(item.query, collection, top_n)
        elif backend == "vector_rerank":
            hits, vector_latency, rerank_latency = vector_rerank_search(
                item.query, collection, top_n
            )
            latency_seconds = {
                "vector_latency_ms": round(vector_latency * 1000, 2),
                "rerank_latency_ms": round(rerank_latency * 1000, 2),
                "total_latency_ms": round((vector_latency + rerank_latency) * 1000, 2),
            }
        else:
            assert conn is not None
            hits = fts_search(conn, item.query, top_n)

        record = build_query_record(item, hits, page_urls, latency_seconds)
        append_raw_cache(backend, record)
        cached[item.id] = record
        print(
            f"[{backend}] {index}/{len(todo)} {item.id} "
            f"first_hit_rank={record['first_hit_rank']} "
            f"({time.time() - started:.1f}s elapsed)",
            flush=True,
        )

    ordered = [cached[item.id] for item in dataset]
    return aggregate(ordered, top_n)


# --------------------------------------------------------------------------
# 指標集計
# --------------------------------------------------------------------------


def _subset_metrics(records: list[dict[str, Any]], top_n: int) -> dict[str, Any] | None:
    """Recall@k・MRR@10を算出する。"""
    n = len(records)
    if n == 0:
        return None

    metrics: dict[str, Any] = {"n": n}
    for k in RECALL_LEVELS:
        if k > top_n:
            continue
        hit_count = sum(
            1
            for r in records
            if r["first_hit_rank"] is not None and r["first_hit_rank"] <= k
        )
        metrics[f"recall@{k}"] = round(hit_count / n, 4)

    reciprocal_sum = sum(
        (
            (1.0 / r["first_hit_rank"])
            if r["first_hit_rank"] is not None and r["first_hit_rank"] <= MRR_CUTOFF
            else 0.0
        )
        for r in records
    )
    metrics["mrr@10"] = round(reciprocal_sum / n, 4)
    return metrics


_LATENCY_KEYS = ("vector_latency_ms", "rerank_latency_ms", "total_latency_ms")


def _avg_latency(records: list[dict[str, Any]]) -> dict[str, float] | None:
    """クエリ単位のレイテンシ計測値（vector_rerankバックエンドのみ）の平均を取る。

    レコードにレイテンシキーが無いバックエンド（vector・fts）では None を返す。
    """
    present = [r for r in records if "total_latency_ms" in r]
    if not present:
        return None
    return {
        key: round(sum(r[key] for r in present) / len(present), 2)
        for key in _LATENCY_KEYS
    }


def aggregate(records: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    """全体・言語別（ja/en）の指標を集計する。"""
    overall = _subset_metrics(records, top_n)
    by_lang = {
        lang: _subset_metrics([r for r in records if r["lang"] == lang], top_n)
        for lang in ("ja", "en")
    }
    return {
        "top_n": top_n,
        "overall": overall,
        "by_lang": by_lang,
        "avg_latency_ms": _avg_latency(records),
        "queries": records,
    }


# --------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------


def write_result_json(out_dir: Path, backend: str, result: dict[str, Any]) -> Path:
    """backend別の詳細結果をJSONへ書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_{backend}.json"
    payload = {"backend": backend, **result}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_metrics_row(
    backend: str,
    scope: str,
    metrics: dict[str, Any] | None,
    avg_latency_ms: dict[str, float] | None = None,
) -> str:
    if metrics is None:
        return f"| {backend} | {scope} | - | - | - | - | - | - | - |"
    recall = {k: metrics.get(f"recall@{k}", "-") for k in RECALL_LEVELS}
    latency = f"{avg_latency_ms['total_latency_ms']:.0f}" if avg_latency_ms else "-"
    return (
        f"| {backend} | {scope} | {metrics['n']} | "
        f"{recall[1]} | {recall[3]} | {recall[5]} | {recall[10]} | "
        f"{metrics['mrr@10']} | {latency} |"
    )


def write_summary_md(out_dir: Path) -> Path:
    """eval/results/内の retrieval_*.json を読み込み比較表のMarkdownを書き出す。"""
    lines = [
        "# 検索品質評価サマリー",
        "",
        "vector検索（ChromaDB + multilingual-e5-base）、FTS5全文検索"
        "（SQLite bm25）、vector_rerank（vector top-30をCrossEncoderで"
        "rerankしてtop-10を評価）の3方式の比較。レイテンシはクエリ1件"
        "あたりの平均合計時間（vector検索+rerankの合計。vector_rerank"
        "のみ計測、単位ms）。",
        "",
        "| backend | scope | n | Recall@1 | Recall@3 | Recall@5 | "
        "Recall@10 | MRR@10 | avg latency(ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for backend in ("vector", "fts", "vector_rerank"):
        result_path = out_dir / f"retrieval_{backend}.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        avg_latency_ms = result.get("avg_latency_ms")
        lines.append(
            _format_metrics_row(
                backend, "overall", result.get("overall"), avg_latency_ms
            )
        )
        by_lang = result.get("by_lang", {})
        lines.append(
            _format_metrics_row(backend, "ja", by_lang.get("ja"), avg_latency_ms)
        )
        lines.append(
            _format_metrics_row(backend, "en", by_lang.get("en"), avg_latency_ms)
        )

    lines.append("")
    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="RAG検索品質評価ハーネス（vector検索 vs SQLite FTS5全文検索）"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="ChromaDBの永続化ディレクトリ（外部パス。リポジトリ内へはコピーしないこと）",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--backend",
        choices=["vector", "fts", "vector_rerank", "all"],
        default="all",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="FTSインデックス・page urlキャッシュ・クエリ単位の中間結果を破棄して作り直す",
    )
    return parser.parse_args()


def main() -> None:
    """CLIエントリポイント。"""
    args = parse_args()

    if args.rebuild_index:
        for path in (
            FTS_DB_PATH,
            PAGE_URL_CACHE,
            _raw_cache_path("vector"),
            _raw_cache_path("fts"),
            _raw_cache_path("vector_rerank"),
        ):
            if path.exists():
                path.unlink()
                print(f"removed cache: {path}", flush=True)

    dataset = load_dataset(args.dataset)
    print(f"{len(dataset)} queries loaded from {args.dataset}", flush=True)

    backends = ["vector", "fts"] if args.backend == "all" else [args.backend]
    for backend in backends:
        result = run_backend(
            backend, dataset, args.top_n, args.db_path, args.rebuild_index
        )
        out_path = write_result_json(args.out, backend, result)
        print(f"[{backend}] wrote {out_path}", flush=True)
        print(f"[{backend}] overall: {result['overall']}", flush=True)

    summary_path = write_summary_md(args.out)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
