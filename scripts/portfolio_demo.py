"""公式資料に基づく独自要約を実モデルで検索する、個人データ・外部LLM不要のデモ。"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from scripts.source_notes import CHECKED_AT, NOTES, source_for


def prepare_demo() -> int:
    """専用DBへ固定サンプルをupsertし、登録件数を返す。"""
    # User-provided production settings are deliberately not used by this demo.
    root = Path(__file__).resolve().parents[1]
    os.environ["CHROMA_DIR"] = str(root / ".portfolio-demo" / "chroma")
    os.environ["CHROMA_COLLECTION"] = "portfolio_source_notes"
    os.environ["EMBED_MODEL"] = "intfloat/multilingual-e5-base"
    os.environ["EMBED_LOCAL_FILES_ONLY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ.pop("GEMINI_API_KEY", None)

    from scripts.embed import (
        Chunk,
        embed_and_store,
        get_model,
        get_or_create_collection,
    )

    tokenizer = get_model("intfloat/multilingual-e5-base").tokenizer
    chunks = []
    for note in NOTES:
        content = note["content"]
        tokens = len(tokenizer.encode("passage: " + content, add_special_tokens=True))
        if tokens > 512:
            raise ValueError("Sample exceeds the embedding token limit")
        source = source_for(note)
        chunks.append(
            Chunk(
                id="portfolio-notes-v1:"
                + hashlib.sha256(content.encode("utf-8")).hexdigest(),
                content=content,
                source=source,
                metadata={
                    "source": source,
                    "category": "docs",
                    "heading_path": note["title"],
                    "source_url": note["url"],
                    "sources_checked_at": CHECKED_AT,
                    "token_estimate": tokens,
                    "summary_short": note["summary"],
                },
            )
        )
    collection = get_or_create_collection()
    embed_and_store(chunks, collection=collection)
    stale = set(collection.get(include=[])["ids"]) - {chunk.id for chunk in chunks}
    if stale:
        collection.delete(ids=sorted(stale))
    return collection.count()


def main() -> None:
    """独立したサンプルDBを作成し、必要に応じてローカルAPIを起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve", action="store_true", help="検索画面・Swagger UIを起動"
    )
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    count = prepare_demo()
    print(
        f"Source-note index ready: {count} documents. Real e5 embeddings; no external LLM.",
        flush=True,
    )
    if args.serve:
        import uvicorn

        uvicorn.run("scripts.search_api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
