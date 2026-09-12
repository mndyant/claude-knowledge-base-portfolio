"""embed.py の単体テスト（TC-E-001〜005）。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import embed
from scripts.embed import (
    Chunk,
    Document,
    _make_chunk_id,
    get_or_create_collection,
    load_documents,
    split_chunks,
)


def test_embed_texts_uses_cpu_friendly_batch_size(monkeypatch):
    """モデル推論にはDB書き込みバッチと独立した小さいバッチを使う。"""
    calls = {}

    class Result:
        def tolist(self):
            return [[1.0]]

    class RecordingModel:
        def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            calls.update(kwargs)
            return Result()

    monkeypatch.delenv("EMBED_ENCODE_BATCH_SIZE", raising=False)
    monkeypatch.setattr(embed, "get_model", lambda *args, **kwargs: RecordingModel())

    embed.embed_texts(["test"], model_name="test-model")

    assert calls["batch_size"] == embed.DEFAULT_ENCODE_BATCH_SIZE
    assert calls["show_progress_bar"] is False


def _write_doc(rel_path: str, body: str) -> None:
    path = Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = "---\nsource_url: https://example.com\ncategory: docs\n---\n\n"
    path.write_text(frontmatter + body, encoding="utf-8")


def test_load_documents_reads_three_files():
    """TC-E-001: knowledge/配下の3ファイルをDocument3件として読み込む。"""
    _write_doc("knowledge/docs/a.md", "ドキュメントAの本文。")
    _write_doc("knowledge/docs/b.md", "ドキュメントBの本文。")
    _write_doc("knowledge/blog/c.md", "ブログCの本文。")

    documents = load_documents("knowledge")

    assert len(documents) == 3
    assert all(d.content.strip() for d in documents)


def test_split_chunks_boundary():
    """TC-E-002: 1024トークンが2チャンク以上・各チャンク512トークン以内。"""
    text = " ".join(f"tok{i}" for i in range(1024))
    documents = [Document(content=text, source="knowledge/docs/big.md")]

    chunks = split_chunks(documents, chunk_size=512, overlap=50)

    tokenizer = embed.get_model(embed.get_model_name()).tokenizer
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(tokenizer.encode(chunk.content)) <= 512


def test_split_chunks_overlap():
    """TC-E-003: 隣接チャンク間に50トークンの重複が存在する。"""
    text = " ".join(f"tok{i}" for i in range(1024))
    documents = [Document(content=text, source="knowledge/docs/big.md")]

    chunks = split_chunks(documents, chunk_size=512, overlap=50)
    tokenizer = embed.get_model(embed.get_model_name()).tokenizer

    first = tokenizer.encode(chunks[0].content)
    second = tokenizer.encode(chunks[1].content)
    assert first[-50:] == second[:50]


def test_embed_and_store_registers_three(test_chroma_client):
    """TC-E-004: チャンク3件をChromaDBに登録できる。"""
    collection = get_or_create_collection(client=test_chroma_client)
    chunks = [
        Chunk(
            id=_make_chunk_id(f"knowledge/docs/d{i}.md", 0),
            content=f"チャンク{i}の本文",
            source=f"knowledge/docs/d{i}.md",
            category="docs",
            chunk_index=0,
            metadata={"source": f"knowledge/docs/d{i}.md", "category": "docs"},
        )
        for i in range(3)
    ]

    embed.embed_and_store(chunks, collection=collection)

    assert collection.count() == 3


def test_embed_and_store_uses_embed_text_but_stores_full_content(
    test_chroma_client, monkeypatch
):
    """embed_textがある場合、埋め込み計算にはそちらを使い、表示用documentは
    content（列挙定数を含む完全な本文）のまま保存する。"""
    captured_texts: list[str] = []

    def fake_embed_texts(texts, **kwargs):  # type: ignore[no-untyped-def]
        captured_texts.extend(texts)
        return [[0.1] for _ in texts]

    monkeypatch.setattr(embed, "embed_texts", fake_embed_texts)

    collection = get_or_create_collection(client=test_chroma_client)
    source = "llms-full.txt:boilerplate"
    full_content = "# Endpoint\n\n- `unique_param`\n- `CONST_A(\"a\")`\n...(省略)..."
    chunk = Chunk(
        id=_make_chunk_id(source, 0),
        content=full_content,
        source=source,
        metadata={"source": source, "category": "docs"},
        embed_text="# Endpoint\n\n- `unique_param`\n[N件の列挙定数を省略]",
    )

    embed.embed_and_store([chunk], collection=collection)

    assert captured_texts == [chunk.embed_text]
    stored = collection.get(ids=[chunk.id])
    assert stored["documents"][0] == full_content


def test_load_chunks_jsonl_reads_embed_text(tmp_path):
    """JSONLの``embed_text``キーがChunk.embed_textへ引き継がれる。"""
    path = tmp_path / "chunks.jsonl"
    record = {
        "id": "llms-full.txt:abc123",
        "content": "完全な本文（列挙定数含む）",
        "embed_text": "省略済みの本文",
        "metadata": {"source_file": "llms-full.txt", "heading_path": "Endpoint"},
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks = embed.load_chunks_jsonl(path)

    assert len(chunks) == 1
    assert chunks[0].content == "完全な本文（列挙定数含む）"
    assert chunks[0].embed_text == "省略済みの本文"
    assert chunks[0].text_for_embedding() == "省略済みの本文"


def test_chunk_text_for_embedding_falls_back_to_content():
    """embed_textが空なら、text_for_embedding()はcontentを返す。"""
    chunk = Chunk(id="x", content="本文そのまま", source="s")
    assert chunk.text_for_embedding() == "本文そのまま"


def test_embed_and_store_upsert(test_chroma_client):
    """TC-E-005: 同一IDのupsertで件数は変わらず内容が更新される。"""
    collection = get_or_create_collection(client=test_chroma_client)
    source = "knowledge/docs/same.md"
    cid = _make_chunk_id(source, 0)
    meta = {"source": source, "category": "docs"}

    original = Chunk(id=cid, content="最初の本文", source=source, metadata=meta)
    embed.embed_and_store([original], collection=collection)
    assert collection.count() == 1

    updated = Chunk(id=cid, content="更新後の本文", source=source, metadata=meta)
    embed.embed_and_store([updated], collection=collection)

    assert collection.count() == 1
    stored = collection.get(ids=[cid])
    assert stored["documents"][0] == "更新後の本文"
