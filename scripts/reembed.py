"""chunks.jsonlをChromaDBへ全量embeddingする運用スクリプト（中断・再開可能）。

登録済みIDをスキップするため、何度中断されても再実行すれば続きから進む。
コレクションを作り直す場合だけ ``--reset`` を付ける。

チャンクIDはsource_file + 内容ハッシュで決まる安定IDなので（split_llms_full.py参照）、
内容が変わっていないチャンクは前回と同じIDになりembeddingがスキップされる。
逆に元ドキュメントから消えたチャンクはchunks.jsonlに存在しなくなるため、
起動時にコレクション側の同一source由来の余剰IDを検出しdeleteする（prune）。
これがないと、消えた・古くなった内容がベクトルDBに残り続け検索結果を汚染する。

長時間プロセスが強制終了される環境では ``--max-seconds 200`` などで
自主終了させ、完了するまで繰り返し実行する（スライス実行）。

使い方:
    python scripts/reembed.py                # 続きから再開（stale chunkも自動prune）
    python scripts/reembed.py --reset        # コレクションを削除して最初から
    python scripts/reembed.py --max-seconds 200   # 200秒で自主終了（再実行で続き）
    python scripts/reembed.py --no-prune     # pruneせずembeddingだけ行う
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.embed import (
    embed_and_store,
    get_chroma_client,
    get_collection_name,
    get_or_create_collection,
    load_chunks_jsonl,
)

DEFAULT_INPUT = Path("data/processed/chunks.jsonl")
BATCH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="chunks.jsonlを再開可能にembeddingする"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="既存コレクションを削除して最初からやり直す",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="モデル推論バッチサイズ（CPU既定8）",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="embedding開始からこの秒数を超えたら自主終了する（0=無制限）",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="chunks.jsonlに存在しなくなったチャンクの削除をスキップする",
    )
    return parser.parse_args()


def _prune_stale(collection, chunks, existing_ids: set[str]) -> int:
    """chunks.jsonlに存在しない、同一source由来の余剰チャンクを削除する。

    IDのprefix（``source_file:``）でしか判定しないため、markdown経由の
    従来方式（32桁hexの決め打ちID・":"を含まない）で登録されたチャンクは
    誤って削除しない。
    """
    current_ids = {chunk.id for chunk in chunks}
    sources = {chunk.source for chunk in chunks if chunk.source}
    prefixes = tuple(f"{source}:" for source in sources)
    if not prefixes:
        return 0

    stale = [
        chunk_id
        for chunk_id in existing_ids
        if chunk_id not in current_ids and chunk_id.startswith(prefixes)
    ]
    if stale:
        collection.delete(ids=stale)
    return len(stale)


def main() -> None:
    args = parse_args()
    client = get_chroma_client()
    name = get_collection_name()

    if args.reset:
        try:
            client.delete_collection(name)
            print(f"deleted collection: {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 存在しない場合は続行
            print(f"no collection to delete: {exc}", flush=True)

    collection = get_or_create_collection(client, name)
    chunks = load_chunks_jsonl(args.input)
    print(f"{len(chunks)} chunks loaded from {args.input}", flush=True)

    existing = set(collection.get(include=[])["ids"])
    todo = [chunk for chunk in chunks if chunk.id not in existing]
    print(f"{len(existing)} already stored, {len(todo)} to embed", flush=True)

    if not args.no_prune:
        pruned = _prune_stale(collection, chunks, existing)
        if pruned:
            print(f"pruned {pruned} stale chunks no longer in {args.input}", flush=True)

    started = time.time()
    batch_size = max(1, args.batch_size)
    for start in range(0, len(todo), batch_size):
        embed_and_store(
            todo[start : start + batch_size],
            collection=collection,
            batch_size=batch_size,
            encode_batch_size=args.encode_batch_size,
        )
        done = min(start + batch_size, len(todo))
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0.0
        eta_min = (len(todo) - done) / rate / 60 if rate else 0.0
        print(
            f"progress: {done}/{len(todo)} ({rate:.1f} chunks/s, ETA {eta_min:.0f}min)",
            flush=True,
        )
        if args.max_seconds and elapsed > args.max_seconds and done < len(todo):
            print(
                f"time budget reached, {len(todo) - done} remaining. rerun to resume",
                flush=True,
            )
            return

    print(f"done: {collection.count()} documents in collection", flush=True)


if __name__ == "__main__":
    main()
