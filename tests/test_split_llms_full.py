"""split_llms_full.pyのRAG向け分割ルールを検証する。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import split_llms_full as splitter
from scripts.embed import load_chunks_jsonl


def _fallback_counter() -> splitter.TokenCounter:
    counter = splitter.TokenCounter()
    counter._encoding = None
    counter.name = "char_count/4"
    return counter


def test_code_fence_is_not_split_midway():
    """コードブロックは途中で分割しない。"""
    markdown = (
        "# Claude API\n\n"
        + ("overview text " * 80)
        + "\n\n```python\n"
        + "\n".join(f"print({index})" for index in range(80))
        + "\n```\n\n"
        + ("tail text " * 120)
    )

    blocks = splitter.parse_markdown_blocks(markdown)
    drafts = splitter.build_chunk_drafts(
        blocks,
        token_counter=_fallback_counter(),
        min_tokens=40,
        target_tokens=80,
        max_tokens=120,
        overlap_tokens=10,
    )

    assert any(draft.has_code for draft in drafts)
    assert all(draft.content.count("```") in {0, 2} for draft in drafts)


def test_heading_path_uses_h1_to_h3():
    """heading_pathはH1/H2/H3から作る。"""
    markdown = "# Messages\n\n## Create\n\n### Parameters\n\nbody"
    blocks = splitter.parse_markdown_blocks(markdown)

    assert blocks[0].heading_path == ("Messages", "Create", "Parameters")


def test_page_separator_only_increments_before_h1():
    """H2前の---はページ区切りではなく、H1前の---だけページ区切りにする。"""
    markdown = (
        "# Page One\n\nbody\n\n---\n\n## Same Page Section\n\nbody\n\n"
        "---\n\n# Page Two\n\nbody"
    )
    blocks = splitter.parse_markdown_blocks(markdown)

    page_by_heading = {block.heading_path[-1]: block.page_index for block in blocks}
    assert page_by_heading["Same Page Section"] == page_by_heading["Page One"]
    assert page_by_heading["Page Two"] == page_by_heading["Page One"] + 1


def test_extract_page_urls_tracks_page_index():
    markdown = (
        "# Intro\n\n---\n\n# First\n\nURL: https://example.com/first\n\n"
        "---\n\n# Second\n\nURL: https://example.com/second\n"
    )

    assert splitter.extract_page_urls(markdown) == {
        1: "https://example.com/first",
        2: "https://example.com/second",
    }


def test_extract_page_urls_ignores_frontmatter_separator():
    markdown = (
        "---\nsource_url: https://example.com/all\ncategory: docs\n---\n\n"
        "# Intro\n\n---\n\n# First\n\nURL: https://example.com/first\n"
    )

    assert splitter.extract_page_urls(markdown) == {
        1: "https://example.com/first"
    }


def test_extract_page_urls_supports_july_format():
    """7月版の「--- → **URL:** → --- → ## 見出し」構造でもページ検出できる。"""
    markdown = (
        "# Intro\n\n"
        "---\n\n**URL:** https://example.com/api-ref-first\n\n---\n\n"
        "## First Endpoint\n\nbody\n\n"
        "---\n\n**URL:** https://example.com/api-ref-second\n\n---\n\n"
        "## Second Endpoint\n\nbody\n"
    )

    assert splitter.extract_page_urls(markdown) == {
        1: "https://example.com/api-ref-first",
        2: "https://example.com/api-ref-second",
    }


def test_extract_page_urls_first_url_wins():
    """本文中の「URL: ...」行（例示画像など）がページURLを上書きしない。"""
    markdown = (
        "# Intro\n\n---\n\n# Vision\n\nURL: https://example.com/vision\n\n"
        "example image:\n\nURL: https://upload.wikimedia.org/ant.jpg\n"
    )

    assert splitter.extract_page_urls(markdown) == {
        1: "https://example.com/vision",
    }


def test_page_separator_detects_july_url_marker():
    """7月版のページ区切り（---直後が**URL:**）でpage_indexが増える。"""
    markdown = (
        "# Page One\n\nbody\n\n"
        "---\n\n**URL:** https://example.com/two\n\n---\n\n## Page Two\n\nbody"
    )
    blocks = splitter.parse_markdown_blocks(markdown)

    page_by_heading = {
        block.heading_path[-1]: block.page_index
        for block in blocks
        if block.heading_path
    }
    assert page_by_heading["Page Two"] == page_by_heading["Page One"] + 1


def test_require_hf_token_counter_rejects_fallback():
    """HF tokenizerを要求したのに劣化した場合はSystemExitで失敗する。"""
    counter = _fallback_counter()
    assert not counter.name.startswith("hf:")

    with pytest.raises(SystemExit):
        splitter.require_hf_token_counter(counter, "intfloat/multilingual-e5-base")

    # 明示的に空文字（HF無効化）を指定した場合は通す
    splitter.require_hf_token_counter(counter, "")


def test_duplicate_chunks_are_removed():
    """同じ本文のchunkはJSONL化前に重複除外する。"""
    drafts = [
        splitter.ChunkDraft("same content", 0, ("A",), False),
        splitter.ChunkDraft("same   content", 0, ("A",), False),
        splitter.ChunkDraft("different content", 0, ("A",), False),
    ]

    records, duplicate_removed = splitter.materialize_records(
        drafts,
        source_file="llms-full.txt",
        token_counter=_fallback_counter(),
    )

    assert len(records) == 2
    assert duplicate_removed == 1


def test_strip_boilerplate_enum_runs_replaces_long_run():
    """5件以上連続する列挙定数の羅列は埋め込み用テキストから間引く。"""
    content = (
        "# Acknowledge Work (Beta) (Java)\n\n"
        "### Parameters\n\n"
        "- `WorkAckParams params`\n"
        "- `Optional<List<AnthropicBeta>> betas`\n"
        'Optional header to specify the beta version(s) you want to use.\n'
        '- `MESSAGE_BATCHES_2024_09_24("message-batches-2024-09-24")`\n'
        '- `PROMPT_CACHING_2024_07_31("prompt-caching-2024-07-31")`\n'
        '- `COMPUTER_USE_2024_10_22("computer-use-2024-10-22")`\n'
        '- `PDFS_2024_09_25("pdfs-2024-09-25")`\n'
        '- `TOKEN_COUNTING_2024_11_01("token-counting-2024-11-01")`\n'
        '- `FILES_API_2025_04_14("files-api-2025-04-14")`\n'
    )

    stripped = splitter.strip_boilerplate_enum_runs(content)

    assert "MESSAGE_BATCHES_2024_09_24" not in stripped
    assert "[6件の列挙定数を省略]" in stripped
    assert "- `WorkAckParams params`" in stripped
    assert "Optional header to specify the beta version(s) you want to use." in stripped


def test_strip_boilerplate_enum_runs_keeps_short_run():
    """しきい値未満の短い列挙は、実際の回答に使われうるため保持する。"""
    content = (
        "- `group_type`: filter\n"
        '- `MODEL_GROUP("model_group")`\n'
        '- `BATCH("batch")`\n'
    )

    stripped = splitter.strip_boilerplate_enum_runs(content)

    assert stripped == content


def test_materialize_records_adds_embed_text_for_boilerplate_chunk():
    """列挙定数を含むチャンクはembed_textを持ち、contentは変更されない。"""
    enum_lines = "\n".join(
        f'- `CONST_{i}("const-{i}")`' for i in range(6)
    )
    content = f"# Endpoint\n\n### Parameters\n\n- `unique_param`\n{enum_lines}"
    drafts = [splitter.ChunkDraft(content, 0, ("Endpoint",), False)]

    records, _ = splitter.materialize_records(
        drafts,
        source_file="llms-full.txt",
        token_counter=_fallback_counter(),
    )

    assert len(records) == 1
    record = records[0]
    assert record["content"] == content
    assert "embed_text" in record
    assert record["embed_text"] != content
    assert "CONST_0" not in record["embed_text"]


def test_non_code_chunks_stay_under_max_tokens():
    """通常テキストはmax_tokens以下に収める。"""
    markdown = "# Long Page\n\n" + ("long sentence. " * 2000)
    counter = _fallback_counter()
    drafts = splitter.build_chunk_drafts(
        splitter.parse_markdown_blocks(markdown),
        token_counter=counter,
        min_tokens=100,
        target_tokens=140,
        max_tokens=160,
        overlap_tokens=20,
    )

    assert drafts
    assert all(
        counter.count(draft.content) <= 160 for draft in drafts if not draft.has_code
    )


def test_load_chunks_jsonl_adds_search_metadata(tmp_path):
    """chunks.jsonlからChromaDB投入用Chunkを復元し、検索API用metadataを補う。"""
    path = tmp_path / "chunks.jsonl"
    record = {
        "id": "chunk-1",
        "content": "Claude API content",
        "metadata": {
            "source_file": "llms-full.txt",
            "page_index": 2,
            "heading_path": "Messages > Create",
            "chunk_index": 7,
            "char_count": 18,
            "token_estimate": 5,
            "has_code": False,
            "summary_short": "Messages > Create: Claude API content",
        },
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks = load_chunks_jsonl(path)

    assert len(chunks) == 1
    assert chunks[0].id == "chunk-1"
    assert chunks[0].source == "llms-full.txt"
    assert chunks[0].metadata["source"] == "llms-full.txt"
    assert chunks[0].metadata["section"] == "Messages > Create"
    assert chunks[0].metadata["category"] == "docs"


def test_write_stats_uses_filenames_for_paths_outside_repository(tmp_path):
    """リポジトリ外の出力先はstatsにファイル名だけを記録する。"""
    stats_path = tmp_path / "stats.json"
    args = SimpleNamespace(
        output=tmp_path / "chunks.jsonl",
        min_tokens=180,
        target_tokens=400,
        max_tokens=480,
        overlap_tokens=50,
    )

    stats = splitter.write_stats(
        records=[],
        stats_path=stats_path,
        source_file="llms-full.txt",
        input_char_count=0,
        block_count=0,
        page_count=0,
        duplicate_removed=0,
        token_counter=_fallback_counter(),
        args=args,
    )

    assert stats["output_file"] == "chunks.jsonl"
    assert stats["stats_file"] == "stats.json"


def test_reuse_existing_ids_keeps_unchanged_chunks(tmp_path):
    output = tmp_path / "chunks.jsonl"
    output.write_text(
        json.dumps({"id": "old-stable-id", "content": "same content"}) + "\n",
        encoding="utf-8",
    )
    records = [
        {"id": "new-id-1", "content": "same  content", "metadata": {}},
        {"id": "new-id-2", "content": "changed content", "metadata": {}},
    ]

    reused = splitter.reuse_existing_ids(records, output)

    assert reused == 1
    assert records[0]["id"] == "old-stable-id"
    assert records[1]["id"] == "new-id-2"
