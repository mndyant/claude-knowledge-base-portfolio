"""結合テスト（TC-I-001〜003）。"""

from __future__ import annotations

from pathlib import Path

from scripts import embed, fetch
from scripts.embed import (
    embed_and_store,
    get_collection_stats,
    get_or_create_collection,
    load_documents,
    split_chunks,
)


def test_fetch_to_embed_flow(requests_mock, mock_llms_txt, test_chroma_client):
    """TC-I-001: 取得→読込→チャンク→登録でChromaDBに1件以上登録される。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, text=mock_llms_txt)
    collection = get_or_create_collection(client=test_chroma_client)

    fetch.fetch_llms_full_txt()
    documents = load_documents("knowledge")
    chunks = split_chunks(documents)
    embed_and_store(chunks, collection=collection)

    assert get_collection_stats(collection=collection)["total_documents"] >= 1


def test_diff_update_keeps_count(requests_mock, mock_llms_txt, test_chroma_client):
    """TC-I-002: 1ファイル変更→再登録で総件数が変わらない（upsert）。"""
    collection = get_or_create_collection(client=test_chroma_client)

    # 初回登録
    Path("knowledge/docs").mkdir(parents=True, exist_ok=True)
    Path("knowledge/docs/a.md").write_text(
        fetch.add_frontmatter("最初の本文", url="https://example.com", category="docs"),
        encoding="utf-8",
    )
    chunks = split_chunks(load_documents("knowledge"))
    embed_and_store(chunks, collection=collection)
    first_count = collection.count()
    assert first_count >= 1

    # 同一ファイルの内容を変更して再登録（チャンク数は同じ＝同一ID）
    Path("knowledge/docs/a.md").write_text(
        fetch.add_frontmatter(
            "変更後の本文", url="https://example.com", category="docs"
        ),
        encoding="utf-8",
    )
    chunks = split_chunks(load_documents("knowledge"))
    embed_and_store(chunks, collection=collection)

    assert collection.count() == first_count


def test_registered_text_is_searchable(test_chroma_client):
    """TC-I-003: 登録した既知テキストが検索1位・score 0.8以上で返る。"""
    collection = get_or_create_collection(client=test_chroma_client)
    known = "Claude Code のサブエージェント機能について解説します。"
    chunks = split_chunks(
        [
            embed.Document(
                content=known, source="knowledge/docs/known.md", category="docs"
            )
        ]
    )
    embed_and_store(chunks, collection=collection)

    query_vec = embed.embed_texts([known], is_query=True)
    result = collection.query(query_embeddings=query_vec, n_results=1)

    score = 1.0 - float(result["distances"][0][0])
    assert score >= 0.8
