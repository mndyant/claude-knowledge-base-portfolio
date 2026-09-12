"""チャンク分割3方式の検索品質比較ハーネス（ページ単位評価）。

全1,720ページ・31,525チャンクの全量再embeddingは環境制約（CPU約6.3チャンク/秒・
長時間プロセスが強制終了される）で不可のため、goldクエリ（``eval/dataset.jsonl``）
の正解ページを全て含む約150ページのサンプル（乱数シード固定）で比較する。

比較する3方式:

- **A_structured_400**: 現行・構造保持（``split_llms_full.py`` を再利用、
  target=400 / max=480 / overlap=50）
- **B_structured_200**: 小チャンク・構造保持（同スクリプトの既存パラメータで
  target=200 / max=240 / overlap=30。min_tokensは現行のtarget比を保って90）
- **C_naive_400**: naive固定長（Markdown構造を無視した400トークン窓・
  overlap=50。e5トークナイザーで計測し、メタデータはページURLのみ）

方式間でチャンクメタデータの粒度が異なるため、チャンク単位ではなく
**ページ単位**で評価する: top-N検索結果のチャンクを「そのチャンクが属する
ページのURL」へ写像し、重複ページを除いた順位で Recall@k / MRR@10 を計算する。
gold判定（``url_contains`` / ``heading_contains``）は方式に依存しないよう
ページ単位で事前に確定する（``url_contains`` はページURL、``heading_contains``
はページ内の見出しパス「H1 > H2 > H3」に適用）。

ページ切り出しは ``digest.py`` の ``URL_MARKER_RE`` 方式を使う
（``split_llms_full.py`` の ``_is_page_separator`` は2026年7月版
llms-full.txtの新フォーマット（``**URL:**`` マーカー）で検出率が低い
既知問題があるため。``docs/HANDOFF.md`` 参照）。

長時間プロセスが強制終了される環境を想定し、チャンク生成・embeddingは
中間結果を ``--work-dir`` へ逐次永続化し、再実行で続きから進む。
``--time-budget`` 秒を超えると ``INCOMPLETE`` を表示して正常終了するので、
``ALL_DONE`` が出るまで繰り返し実行する。

使い方:
    python scripts/eval_chunking.py --llms-full <llms-full.txt> --work-dir <dir>
    python scripts/eval_chunking.py --llms-full <...> --work-dir <...> --stage report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.digest import URL_MARKER_RE  # noqa: E402
from scripts.eval_retrieval import GoldItem, load_dataset  # noqa: E402
from scripts.split_llms_full import (  # noqa: E402
    DEFAULT_HF_TOKENIZER,
    TokenCounter,
    build_chunk_drafts,
    parse_markdown_blocks,
)

DEFAULT_DATASET = Path("eval/dataset.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
SAMPLE_PAGES_JSON = DEFAULT_RESULTS_DIR / "chunking_sample_pages.json"
COMPARISON_JSON = DEFAULT_RESULTS_DIR / "chunking_comparison.json"
SUMMARY_MD = DEFAULT_RESULTS_DIR / "summary.md"

DEFAULT_SAMPLE_SIZE = 150
DEFAULT_SEED = 20260716
DEFAULT_TIME_BUDGET_SECONDS = 90.0

RECALL_LEVELS = (1, 3, 5, 10)
MRR_CUTOFF = 10
TOP_PAGES = 10
# ページ単位でtop-10を確保するため、チャンク単位ではこの件数まで取得して
# ページへ写像・重複除去する。
CANDIDATE_CHUNKS = 60

EMBED_BATCH_SIZE = 32
# multilingual-e5-base の入力上限。これを超えるチャンクは黙って切り捨てられる。
E5_MAX_TOKENS = 512

# ページ範囲の推定に使う、URLマーカー行から遡る最大行数（digest.pyの
# ``_find_page_title`` と同じ値。形式A「見出しがURL直前にある」ページの
# タイトル行を含めるため）。
PAGE_START_LOOKBACK = 4

# ページタイトル検出に使う見出し行（digest.pyのHEADING_REと同じH1〜H3）。
PAGE_TITLE_RE = re.compile(r"^(#{1,3})\s+(.+)$")

# 評価母集団から除外する巨大ページの本文文字数しきい値。
# 2026年7月版llms-full.txtにはSDK言語別のAPIリファレンス索引ページ
# （/docs/en/api/go/beta など、1ページ最大3.4MB・チャンク2,000個超）が
# 34ページあり、1ページでサンプル評価のembedding予算を支配してしまう。
# goldクエリの実体ページで最大の prompt-caching（約151k文字）は残るよう、
# 155,000文字を上限とする（除外はsample JSONに件数を記録する）。
MAX_PAGE_CHARS = 155_000

# 比較する3方式。structuredはsplit_llms_full.pyの既存パラメータをそのまま渡す。
METHODS: dict[str, dict[str, Any]] = {
    "A_structured_400": {
        "kind": "structured",
        "min_tokens": 180,
        "target_tokens": 400,
        "max_tokens": 480,
        "overlap_tokens": 50,
    },
    "B_structured_200": {
        "kind": "structured",
        "min_tokens": 90,
        "target_tokens": 200,
        "max_tokens": 240,
        "overlap_tokens": 30,
    },
    "C_naive_400": {
        "kind": "naive",
        "window_tokens": 400,
        "overlap_tokens": 50,
    },
}


@dataclass(frozen=True)
class Page:
    """llms-full.txtから切り出した1ページ。"""

    url: str
    text: str


class Stopwatch:
    """プロセス全体の時間予算を管理する（強制終了される前に自主的に止まる）。"""

    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds
        self.started = time.time()

    def expired(self) -> bool:
        return (time.time() - self.started) >= self.budget_seconds


# --------------------------------------------------------------------------
# ページ切り出し（digest.pyのURLマーカー方式）
# --------------------------------------------------------------------------


def _strip_frontmatter(text: str) -> str:
    """先頭のYAML Frontmatterを除去する（digest.pyと同じ簡易方式）。"""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return text


def _page_start_index(lines: list[str], url_index: int) -> int:
    """ページの開始行を返す。

    digest.pyの ``_find_page_title`` と同様に直前方向へ最大4行さかのぼり
    （空行はスキップ）、最初の非空行が見出しならそこをページ先頭にする
    （形式A: タイトルがURL直前にあるページ）。見出しでなければ前ページの
    本文なので巻き込まず、URLマーカー行自身を先頭にする。
    """
    for index in range(url_index - 1, max(-1, url_index - 1 - PAGE_START_LOOKBACK), -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if PAGE_TITLE_RE.match(stripped):
            return index
        break
    return url_index


def split_pages(text: str) -> list[Page]:
    """URLマーカー行を境界にllms-full.txtをページへ分割する。

    ページ範囲は「URLマーカーの直前のタイトル行（あれば）」から
    「次ページの開始行の手前」まで（digest.pyの ``collect_docs_pages`` の
    URLマーカー方式）。同一URLの2回目以降の出現はスキップする。
    """
    lines = (
        _strip_frontmatter(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    url_indices = [
        index for index, line in enumerate(lines) if URL_MARKER_RE.match(line.strip())
    ]
    start_indices = [_page_start_index(lines, url_index) for url_index in url_indices]

    pages: list[Page] = []
    seen_urls: set[str] = set()
    for position, url_index in enumerate(url_indices):
        match = URL_MARKER_RE.match(lines[url_index].strip())
        assert match is not None
        url = match.group(1)

        start = start_indices[position]
        end = (
            start_indices[position + 1]
            if position + 1 < len(url_indices)
            else len(lines)
        )

        if url in seen_urls:
            continue
        seen_urls.add(url)
        body = "\n".join(lines[start:end]).strip()
        if body:
            pages.append(Page(url=url, text=body))
    return pages


def page_heading_paths(page_text: str) -> list[str]:
    """ページ内に現れる見出しパス（"H1 > H2 > H3" 形式）を列挙する。

    ``split_llms_full.parse_markdown_blocks`` を再利用するため、コードフェンス内の
    ``#`` 行を見出しと誤認しない。gold判定の ``heading_contains`` はこのパス
    文字列への部分一致で行う（チャンク方式に依存しないページ単位の判定）。
    """
    paths: list[str] = []
    seen: set[str] = set()
    for block in parse_markdown_blocks(page_text):
        path = " > ".join(block.heading_path)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


# --------------------------------------------------------------------------
# gold判定（ページ単位）とサンプル選定
# --------------------------------------------------------------------------


def page_matches_item(item: GoldItem, url: str, heading_paths: list[str]) -> bool:
    """goldの relevant criteria（OR条件）がページにマッチするか判定する。"""
    url_lower = url.lower()
    paths_lower = [path.lower() for path in heading_paths]
    for criterion in item.relevant:
        value = str(criterion.get("value", "")).lower()
        ctype = criterion.get("type")
        if ctype == "url_contains" and value in url_lower:
            return True
        if ctype == "heading_contains" and any(value in path for path in paths_lower):
            return True
    return False


def build_gold_page_map(
    dataset: list[GoldItem], pages: list[Page]
) -> dict[str, list[str]]:
    """クエリid -> 正解ページURLリスト を全ページ走査で事前に確定する。"""
    headings_by_url = {page.url: page_heading_paths(page.text) for page in pages}
    gold_map: dict[str, list[str]] = {}
    for item in dataset:
        gold_map[item.id] = [
            page.url
            for page in pages
            if page_matches_item(item, page.url, headings_by_url[page.url])
        ]
    return gold_map


def filter_population(
    pages: list[Page], max_chars: int = MAX_PAGE_CHARS
) -> tuple[list[Page], list[str]]:
    """評価母集団を返す。巨大ページは除外し、除外URL一覧も返す。"""
    population = [page for page in pages if len(page.text) <= max_chars]
    excluded = [page.url for page in pages if len(page.text) > max_chars]
    return population, excluded


def select_sample_urls(
    pages: list[Page],
    gold_map: dict[str, list[str]],
    sample_size: int,
    seed: int,
) -> list[str]:
    """gold正解ページ全部＋ランダムページで合計sample_size件のURLを選ぶ。"""
    gold_urls = {url for urls in gold_map.values() for url in urls}
    all_urls = [page.url for page in pages]
    sampled = [url for url in all_urls if url in gold_urls]

    remaining = [url for url in all_urls if url not in gold_urls]
    fill_count = max(0, min(sample_size - len(sampled), len(remaining)))
    rng = random.Random(seed)
    sampled.extend(sorted(rng.sample(remaining, fill_count)))
    return sampled


# --------------------------------------------------------------------------
# チャンク分割（3方式）
# --------------------------------------------------------------------------


def chunk_page_structured(
    page: Page,
    token_counter: TokenCounter,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """split_llms_full.pyの構造保持分割を1ページに適用する（方式A/B）。"""
    blocks = parse_markdown_blocks(page.text)
    drafts = build_chunk_drafts(
        blocks,
        token_counter=token_counter,
        min_tokens=min_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    return [
        {
            "content": draft.content,
            "metadata": {
                "url": page.url,
                "heading_path": " > ".join(draft.heading_path),
            },
        }
        for draft in drafts
        if draft.content.strip()
    ]


def naive_chunk_page(
    text: str,
    tokenizer: Any,
    window_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Markdown構造を無視した固定長トークン窓で分割する（方式C）。

    ``tokenizer`` は ``encode(text, add_special_tokens=False)`` /
    ``decode(ids, skip_special_tokens=True)`` を持つオブジェクト
    （HF tokenizerまたはテスト用ダブル）。
    """
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    stride = max(window_tokens - overlap_tokens, 1)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    contents: list[str] = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + window_tokens]
        content = tokenizer.decode(window, skip_special_tokens=True).strip()
        if content:
            contents.append(content)
        if start + window_tokens >= len(token_ids):
            break
        start += stride
    return contents


def chunk_page_naive(
    page: Page,
    tokenizer: Any,
    window_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """方式C: naive固定長分割。メタデータはページURLのみ付与する。"""
    return [
        {"content": content, "metadata": {"url": page.url}}
        for content in naive_chunk_page(
            page.text, tokenizer, window_tokens, overlap_tokens
        )
    ]


def chunk_id(method: str, url: str, content: str) -> str:
    """内容ハッシュによる冪等なチャンクID（再実行時のupsert・スキップ用）。"""
    digest = hashlib.sha256(f"{url}\x00{content}".encode()).hexdigest()[:16]
    return f"{method}:{digest}"


# --------------------------------------------------------------------------
# ページ単位の指標
# --------------------------------------------------------------------------


def dedupe_urls(urls: list[str]) -> list[str]:
    """チャンク順位を保ったままページURLの重複を除く。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def first_hit_rank(ranked_urls: list[str], gold_urls: set[str]) -> int | None:
    """ページ順位リスト中で最初に正解ページが現れる順位（1-origin）を返す。"""
    for rank, url in enumerate(ranked_urls, start=1):
        if url in gold_urls:
            return rank
    return None


def compute_metrics(first_ranks: list[int | None]) -> dict[str, Any]:
    """ページ単位の Recall@k / MRR@10 を集計する。"""
    n = len(first_ranks)
    metrics: dict[str, Any] = {"n": n}
    if n == 0:
        return metrics
    for k in RECALL_LEVELS:
        hits = sum(1 for rank in first_ranks if rank is not None and rank <= k)
        metrics[f"recall@{k}"] = round(hits / n, 4)
    reciprocal = sum(
        1.0 / rank for rank in first_ranks if rank is not None and rank <= MRR_CUTOFF
    )
    metrics["mrr@10"] = round(reciprocal / n, 4)
    return metrics


# --------------------------------------------------------------------------
# ステージ1: prepare（ページ分割・サンプル選定・チャンク生成）
# --------------------------------------------------------------------------


def _method_chunks_path(work_dir: Path, method: str) -> Path:
    return work_dir / f"chunks_{method}.jsonl"


def _method_done_path(work_dir: Path, method: str) -> Path:
    return work_dir / f"chunks_{method}.done"


def _method_progress_path(work_dir: Path, method: str) -> Path:
    return work_dir / f"chunks_{method}.progress"


def load_or_create_sample(
    pages: list[Page],
    dataset: list[GoldItem],
    sample_path: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """サンプル定義（正解ページマップ＋サンプルURL一覧）を読み込むか新規作成する。

    一度作成したサンプルはファイルで固定し、後続ステージ・再実行はこれを使う
    （llms-full.txtの取得タイミング差で母集団が変わっても結果が揺れないように）。
    """
    if sample_path.exists():
        return json.loads(sample_path.read_text(encoding="utf-8"))

    population, mega_pages = filter_population(pages)
    gold_map = build_gold_page_map(dataset, population)
    sampled_urls = select_sample_urls(population, gold_map, sample_size, seed)
    excluded = sorted(
        item.id for item in dataset if not (set(gold_map[item.id]) & set(sampled_urls))
    )
    sample = {
        "seed": seed,
        "sample_size": sample_size,
        "total_pages": len(pages),
        "max_page_chars": MAX_PAGE_CHARS,
        "mega_page_excluded_count": len(mega_pages),
        "mega_page_excluded_urls": mega_pages,
        "population_page_count": len(population),
        "sampled_page_count": len(sampled_urls),
        "gold_page_urls": gold_map,
        "sampled_urls": sampled_urls,
        "excluded_query_ids": excluded,
    }
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return sample


def generate_method_chunks(
    method: str,
    sampled_pages: list[Page],
    work_dir: Path,
    token_counter: TokenCounter,
    tokenizer: Any,
    stopwatch: Stopwatch,
) -> bool:
    """1方式分のチャンクをJSONLへ生成する（ページ単位で再開可能）。

    Returns:
        全ページ完了なら ``True``、時間切れで中断したら ``False``。
    """
    if _method_done_path(work_dir, method).exists():
        return True

    spec = METHODS[method]
    chunks_path = _method_chunks_path(work_dir, method)
    progress_path = _method_progress_path(work_dir, method)
    done_pages = int(progress_path.read_text()) if progress_path.exists() else 0
    if done_pages == 0 and chunks_path.exists():
        chunks_path.unlink()

    seen_ids: set[str] = set()
    if chunks_path.exists():
        with chunks_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    seen_ids.add(json.loads(line)["id"])

    with chunks_path.open("a", encoding="utf-8", newline="\n") as file:
        for page_index in range(done_pages, len(sampled_pages)):
            if stopwatch.expired():
                print(
                    f"INCOMPLETE stage=prepare method={method} "
                    f"pages={page_index}/{len(sampled_pages)}",
                    flush=True,
                )
                return False

            page = sampled_pages[page_index]
            if spec["kind"] == "structured":
                records = chunk_page_structured(
                    page,
                    token_counter,
                    min_tokens=spec["min_tokens"],
                    target_tokens=spec["target_tokens"],
                    max_tokens=spec["max_tokens"],
                    overlap_tokens=spec["overlap_tokens"],
                )
            else:
                records = chunk_page_naive(
                    page,
                    tokenizer,
                    window_tokens=spec["window_tokens"],
                    overlap_tokens=spec["overlap_tokens"],
                )

            for record in records:
                record_id = chunk_id(method, page.url, record["content"])
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                record["id"] = record_id
                record["metadata"]["token_count"] = token_counter.count(
                    record["content"]
                )
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            file.flush()
            progress_path.write_text(str(page_index + 1))

    _method_done_path(work_dir, method).touch()
    print(f"[prepare] {method}: {len(seen_ids)} chunks done", flush=True)
    return True


def run_prepare(
    llms_full: Path,
    dataset: list[GoldItem],
    work_dir: Path,
    sample_path: Path,
    sample_size: int,
    seed: int,
    stopwatch: Stopwatch,
) -> bool:
    """ページ分割→サンプル選定→3方式のチャンク生成を行う。"""
    pages = split_pages(llms_full.read_text(encoding="utf-8"))
    print(f"[prepare] {len(pages)} pages parsed from {llms_full}", flush=True)

    sample = load_or_create_sample(pages, dataset, sample_path, sample_size, seed)
    sampled_set = set(sample["sampled_urls"])
    sampled_pages = [page for page in pages if page.url in sampled_set]
    print(
        f"[prepare] sample: {len(sampled_pages)} pages "
        f"(excluded queries: {sample['excluded_query_ids']})",
        flush=True,
    )

    token_counter = TokenCounter(hf_model=DEFAULT_HF_TOKENIZER)
    from scripts.split_llms_full import _load_hf_tokenizer

    tokenizer = _load_hf_tokenizer(DEFAULT_HF_TOKENIZER)
    if tokenizer is None:
        raise SystemExit(f"HF tokenizerをロードできません: {DEFAULT_HF_TOKENIZER}")

    for method in METHODS:
        if not generate_method_chunks(
            method, sampled_pages, work_dir, token_counter, tokenizer, stopwatch
        ):
            return False
    return True


# --------------------------------------------------------------------------
# ステージ2: embed（一時ChromaDBコレクションへupsert・再開可能）
# --------------------------------------------------------------------------


def _collection_name(method: str) -> str:
    return f"chunk_eval_{method.lower()}"


def _load_method_chunks(work_dir: Path, method: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with _method_chunks_path(work_dir, method).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _existing_ids(collection: Any, ids: list[str]) -> set[str]:
    """コレクション内に登録済みのIDを返す（バッチ取得）。"""
    existing: set[str] = set()
    for start in range(0, len(ids), 500):
        response = collection.get(ids=ids[start : start + 500], include=[])
        existing.update(response.get("ids") or [])
    return existing


def run_embed(work_dir: Path, stopwatch: Stopwatch) -> bool:
    """3方式のチャンクを方式別コレクションへembedding・upsertする。

    IDは内容ハッシュで冪等のため、再実行時は登録済みチャンクをスキップして
    続きから進む。時間切れ時は ``False`` を返す（正常終了・要再実行）。
    """
    from scripts.embed import embed_texts, get_chroma_client, get_or_create_collection

    client = get_chroma_client(path=str(work_dir / "chroma"))
    for method in METHODS:
        records = _load_method_chunks(work_dir, method)
        collection = get_or_create_collection(client, name=_collection_name(method))
        existing = _existing_ids(collection, [record["id"] for record in records])
        todo = [record for record in records if record["id"] not in existing]
        print(
            f"[embed] {method}: {len(existing)} done, {len(todo)} to embed",
            flush=True,
        )

        for start in range(0, len(todo), EMBED_BATCH_SIZE):
            if stopwatch.expired():
                print(
                    f"INCOMPLETE stage=embed method={method} "
                    f"remaining={len(todo) - start}",
                    flush=True,
                )
                return False
            batch = todo[start : start + EMBED_BATCH_SIZE]
            embeddings = embed_texts([record["content"] for record in batch])
            collection.upsert(
                ids=[record["id"] for record in batch],
                documents=[record["content"] for record in batch],
                metadatas=[record["metadata"] for record in batch],
                embeddings=embeddings,
            )
    return True


# --------------------------------------------------------------------------
# ステージ3: search（gold 40問をページ単位で評価）
# --------------------------------------------------------------------------


def run_search(
    dataset: list[GoldItem],
    work_dir: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """3方式それぞれで全goldクエリを検索し、ページ単位の指標を集計する。"""
    from scripts.embed import embed_texts, get_chroma_client

    client = get_chroma_client(path=str(work_dir / "chroma"))
    sampled_set = set(sample["sampled_urls"])
    gold_map: dict[str, list[str]] = sample["gold_page_urls"]
    excluded: list[str] = sample["excluded_query_ids"]

    targets = [item for item in dataset if set(gold_map.get(item.id, [])) & sampled_set]
    results: dict[str, Any] = {"excluded_query_ids": excluded, "methods": {}}

    for method in METHODS:
        collection = client.get_collection(_collection_name(method))
        per_query: list[dict[str, Any]] = []
        for item in targets:
            gold_urls = set(gold_map[item.id]) & sampled_set
            embedding = embed_texts([item.query], is_query=True)
            response = collection.query(
                query_embeddings=embedding,
                n_results=min(CANDIDATE_CHUNKS, collection.count()),
                include=["metadatas"],
            )
            metadatas = (response.get("metadatas") or [[]])[0]
            ranked_pages = dedupe_urls(
                [str((metadata or {}).get("url", "")) for metadata in metadatas]
            )[:TOP_PAGES]
            rank = first_hit_rank(ranked_pages, gold_urls)
            per_query.append(
                {
                    "id": item.id,
                    "query": item.query,
                    "lang": item.lang,
                    "first_hit_rank": rank,
                    "top_pages": ranked_pages[:5],
                }
            )
        metrics = compute_metrics([record["first_hit_rank"] for record in per_query])
        results["methods"][method] = {"metrics": metrics, "queries": per_query}
        print(f"[search] {method}: {metrics}", flush=True)
    return results


# --------------------------------------------------------------------------
# ステージ4: report（結果JSON・summary.md追記）
# --------------------------------------------------------------------------


def chunk_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """方式ごとのチャンク数・平均トークン・512超過数を集計する。"""
    token_counts = [int(record["metadata"].get("token_count", 0)) for record in records]
    return {
        "chunk_count": len(records),
        "avg_tokens": (
            round(sum(token_counts) / len(token_counts), 1) if token_counts else 0
        ),
        "max_tokens": max(token_counts) if token_counts else 0,
        "over_512_count": sum(1 for count in token_counts if count > E5_MAX_TOKENS),
    }


def write_comparison_json(
    out_path: Path,
    sample: dict[str, Any],
    stats: dict[str, dict[str, Any]],
    search_results: dict[str, Any],
) -> None:
    """機械可読な比較結果JSONを書き出す。"""
    payload = {
        "note": (
            "goldクエリの正解ページを含むサンプルページ（全量再embeddingは"
            "環境制約で不可）でのページ単位評価。チャンク単位ではない。"
        ),
        "sample": {
            "seed": sample["seed"],
            "total_pages": sample["total_pages"],
            "max_page_chars": sample.get("max_page_chars"),
            "mega_page_excluded_count": sample.get("mega_page_excluded_count"),
            "sampled_page_count": sample["sampled_page_count"],
            "excluded_query_ids": search_results["excluded_query_ids"],
        },
        "methods": {
            method: {
                "params": {
                    key: value
                    for key, value in METHODS[method].items()
                    if key != "kind"
                },
                "kind": METHODS[method]["kind"],
                "chunk_stats": stats[method],
                "metrics": search_results["methods"][method]["metrics"],
                "queries": search_results["methods"][method]["queries"],
            }
            for method in METHODS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


CHUNKING_SECTION_HEADER = "## チャンク方式比較（ページ単位・サンプル評価）"


def append_summary_md(
    summary_path: Path,
    sample: dict[str, Any],
    stats: dict[str, dict[str, Any]],
    search_results: dict[str, Any],
) -> None:
    """summary.mdへ比較表セクションを追記する（再実行時は同セクションを置換）。"""
    lines = [
        CHUNKING_SECTION_HEADER,
        "",
        f"goldクエリの正解ページを全て含む{sample['sampled_page_count']}ページの"
        f"サンプル（全{sample['total_pages']}ページ中、シード{sample['seed']}）での"
        "ページ単位評価。チャンクをページURLへ写像し、重複ページを除いた順位で"
        "Recall@k / MRR@10 を算出。詳細は chunking_comparison.json / docs/EVAL.md。",
        "",
        "| method | chunks | avg tokens | >512 | n | Recall@1 | Recall@3 | "
        "Recall@5 | Recall@10 | MRR@10 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        stat = stats[method]
        metrics = search_results["methods"][method]["metrics"]
        recall = {k: metrics.get(f"recall@{k}", "-") for k in RECALL_LEVELS}
        lines.append(
            f"| {method} | {stat['chunk_count']} | {stat['avg_tokens']} | "
            f"{stat['over_512_count']} | {metrics.get('n', 0)} | "
            f"{recall[1]} | {recall[3]} | {recall[5]} | {recall[10]} | "
            f"{metrics.get('mrr@10', '-')} |"
        )
    excluded = search_results["excluded_query_ids"]
    lines.extend(
        [
            "",
            f"評価対象外のgold問: {len(excluded)}件"
            + (f"（{', '.join(excluded)}）" if excluded else ""),
            "",
        ]
    )

    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    if CHUNKING_SECTION_HEADER in existing:
        pattern = re.escape(CHUNKING_SECTION_HEADER) + r".*?(?=\n## |\Z)"
        existing = re.sub(pattern, "", existing, flags=re.DOTALL).rstrip() + "\n"
    content = existing.rstrip() + "\n\n" + "\n".join(lines)
    summary_path.write_text(content.rstrip() + "\n", encoding="utf-8")


def run_report(
    work_dir: Path,
    sample: dict[str, Any],
    search_results: dict[str, Any],
    comparison_path: Path,
    summary_path: Path,
) -> None:
    """結果JSONとsummary.md追記を書き出す。"""
    stats = {
        method: chunk_stats(_load_method_chunks(work_dir, method)) for method in METHODS
    }
    write_comparison_json(comparison_path, sample, stats, search_results)
    append_summary_md(summary_path, sample, stats, search_results)
    print(f"[report] wrote {comparison_path} and {summary_path}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="チャンク分割3方式（構造保持400/構造保持200/naive400）の比較評価"
    )
    parser.add_argument("--llms-full", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="チャンクJSONL・一時ChromaDBの置き場（リポジトリ外の外部パス）",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-pages-json", type=Path, default=SAMPLE_PAGES_JSON)
    parser.add_argument("--comparison-json", type=Path, default=COMPARISON_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
        help="この秒数を超えたら中断して正常終了する（再実行で続きから進む）",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "embed", "search", "report", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    """CLIエントリポイント。ALL_DONEが出るまで繰り返し実行する。"""
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    stopwatch = Stopwatch(args.time_budget)
    dataset = load_dataset(args.dataset)

    stages = (
        ["prepare", "embed", "search", "report"]
        if args.stage == "all"
        else [args.stage]
    )

    if "prepare" in stages:
        if not run_prepare(
            args.llms_full,
            dataset,
            args.work_dir,
            args.sample_pages_json,
            args.sample_size,
            args.seed,
            stopwatch,
        ):
            return

    if "embed" in stages and not run_embed(args.work_dir, stopwatch):
        return

    if "search" in stages or "report" in stages:
        sample = json.loads(args.sample_pages_json.read_text(encoding="utf-8"))
        search_results = run_search(dataset, args.work_dir, sample)
        if "report" in stages:
            run_report(
                args.work_dir,
                sample,
                search_results,
                args.comparison_json,
                args.summary_md,
            )

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
