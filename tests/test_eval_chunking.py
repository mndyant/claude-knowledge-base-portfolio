"""eval_chunking.py（チャンク方式比較ハーネス）の単体テスト。

実モデル・ネットワーク・ChromaDBに依存せず、合成データで
URLマーカーページ分割・naive固定長分割・ページ単位指標を検証する
（conftest.pyのFakeTokenizerを利用）。
"""

from __future__ import annotations

from conftest import FakeTokenizer

from scripts.eval_chunking import (
    METHODS,
    Page,
    build_gold_page_map,
    chunk_id,
    chunk_page_naive,
    chunk_page_structured,
    chunk_stats,
    compute_metrics,
    dedupe_urls,
    filter_population,
    first_hit_rank,
    naive_chunk_page,
    page_heading_paths,
    page_matches_item,
    select_sample_urls,
    split_pages,
)
from scripts.eval_retrieval import GoldItem
from scripts.split_llms_full import TokenCounter

SYNTHETIC_LLMS_FULL = """# Header preamble

Some intro text without URL marker.

# Prompt caching

**URL:** https://platform.claude.com/docs/en/prompt-caching

Prompt caching lets you reuse system prompts.

## 1-hour cache duration

Set ttl to extend the cache lifetime.

---

**URL:** https://platform.claude.com/docs/en/api/messages

---

## Messages API

Create a message with the API.

```python
# not a heading inside code fence
client.messages.create()
```

# Vision

**URL:** https://platform.claude.com/docs/en/vision

Vision page body.

## Limitations

Claude may miss small text in images.
"""


def _gold_item(item_id: str, relevant: list[dict[str, str]]) -> GoldItem:
    return GoldItem(id=item_id, query="q", lang="ja", category="c", relevant=relevant)


# --------------------------------------------------------------------------
# URLマーカーページ分割
# --------------------------------------------------------------------------


class TestSplitPages:
    def test_splits_by_url_marker(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        urls = [page.url for page in pages]
        assert urls == [
            "https://platform.claude.com/docs/en/prompt-caching",
            "https://platform.claude.com/docs/en/api/messages",
            "https://platform.claude.com/docs/en/vision",
        ]

    def test_page_body_includes_title_and_sections(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        caching = pages[0]
        assert "# Prompt caching" in caching.text
        assert "1-hour cache duration" in caching.text
        # 次ページ（Messages API）の本文は含まない
        assert "Messages API" not in caching.text

    def test_format_b_page_with_heading_after_url(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        messages = pages[1]
        assert "## Messages API" in messages.text
        assert "Create a message" in messages.text

    def test_duplicate_url_is_skipped(self) -> None:
        text = (
            "# Page A\n\n**URL:** https://example.com/a\n\nbody a\n\n"
            "# Page A again\n\n**URL:** https://example.com/a\n\nbody dup\n"
        )
        pages = split_pages(text)
        assert len(pages) == 1
        assert pages[0].url == "https://example.com/a"

    def test_no_marker_returns_empty(self) -> None:
        assert split_pages("# Just a heading\n\nbody\n") == []


# --------------------------------------------------------------------------
# ページ内見出しパスとgold判定
# --------------------------------------------------------------------------


class TestGoldPageMatching:
    def test_heading_paths_are_hierarchical(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        paths = page_heading_paths(pages[2].text)
        assert "Vision" in paths
        assert "Vision > Limitations" in paths

    def test_code_fence_heading_is_ignored(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        paths = page_heading_paths(pages[1].text)
        assert not any("not a heading" in path for path in paths)

    def test_url_contains_match(self) -> None:
        item = _gold_item("q1", [{"type": "url_contains", "value": "prompt-caching"}])
        assert page_matches_item(
            item, "https://platform.claude.com/docs/en/prompt-caching", []
        )
        assert not page_matches_item(
            item, "https://platform.claude.com/docs/en/vision", []
        )

    def test_heading_contains_match_is_case_insensitive(self) -> None:
        item = _gold_item(
            "q2", [{"type": "heading_contains", "value": "vision > limitations"}]
        )
        assert page_matches_item(item, "https://x", ["Vision > Limitations"])
        assert not page_matches_item(item, "https://x", ["Vision"])

    def test_build_gold_page_map(self) -> None:
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        dataset = [
            _gold_item("q1", [{"type": "url_contains", "value": "prompt-caching"}]),
            _gold_item(
                "q2", [{"type": "heading_contains", "value": "Vision > Limitations"}]
            ),
            _gold_item("q3", [{"type": "url_contains", "value": "no-such-page"}]),
        ]
        gold_map = build_gold_page_map(dataset, pages)
        assert gold_map["q1"] == ["https://platform.claude.com/docs/en/prompt-caching"]
        assert gold_map["q2"] == ["https://platform.claude.com/docs/en/vision"]
        assert gold_map["q3"] == []


# --------------------------------------------------------------------------
# サンプル選定
# --------------------------------------------------------------------------


class TestSelectSample:
    def _pages(self, count: int) -> list[Page]:
        return [
            Page(url=f"https://example.com/p{i}", text=f"body {i}")
            for i in range(count)
        ]

    def test_gold_pages_always_included(self) -> None:
        pages = self._pages(20)
        gold_map = {"q1": [pages[3].url], "q2": [pages[15].url]}
        sampled = select_sample_urls(pages, gold_map, sample_size=5, seed=1)
        assert pages[3].url in sampled
        assert pages[15].url in sampled
        assert len(sampled) == 5

    def test_seed_makes_selection_deterministic(self) -> None:
        pages = self._pages(50)
        gold_map = {"q1": [pages[0].url]}
        first = select_sample_urls(pages, gold_map, sample_size=10, seed=42)
        second = select_sample_urls(pages, gold_map, sample_size=10, seed=42)
        assert first == second

    def test_sample_size_capped_by_population(self) -> None:
        pages = self._pages(3)
        sampled = select_sample_urls(pages, {"q1": []}, sample_size=10, seed=1)
        assert sorted(sampled) == sorted(page.url for page in pages)

    def test_filter_population_excludes_mega_pages(self) -> None:
        pages = [
            Page(url="https://example.com/normal", text="short body"),
            Page(url="https://example.com/mega", text="x" * 200),
        ]
        population, excluded = filter_population(pages, max_chars=100)
        assert [page.url for page in population] == ["https://example.com/normal"]
        assert excluded == ["https://example.com/mega"]


# --------------------------------------------------------------------------
# naive固定長分割（方式C）
# --------------------------------------------------------------------------


class TestNaiveChunking:
    def test_window_and_overlap(self) -> None:
        tokenizer = FakeTokenizer()
        text = " ".join(f"tok{i}" for i in range(10))
        contents = naive_chunk_page(text, tokenizer, window_tokens=4, overlap_tokens=1)
        # stride=3: [0:4], [3:7], [6:10]
        assert contents == [
            "tok0 tok1 tok2 tok3",
            "tok3 tok4 tok5 tok6",
            "tok6 tok7 tok8 tok9",
        ]

    def test_short_text_yields_single_chunk(self) -> None:
        tokenizer = FakeTokenizer()
        contents = naive_chunk_page(
            "a b c", tokenizer, window_tokens=10, overlap_tokens=2
        )
        assert contents == ["a b c"]

    def test_metadata_has_url_only(self) -> None:
        tokenizer = FakeTokenizer()
        page = Page(url="https://example.com/x", text="a b c d e f")
        records = chunk_page_naive(page, tokenizer, window_tokens=3, overlap_tokens=1)
        assert records
        for record in records:
            assert record["metadata"] == {"url": "https://example.com/x"}

    def test_naive_ignores_structure(self) -> None:
        tokenizer = FakeTokenizer()
        page = Page(
            url="https://example.com/x",
            text="# Heading\n\npara one\n\n```\ncode\n```",
        )
        records = chunk_page_naive(
            page, tokenizer, window_tokens=100, overlap_tokens=10
        )
        assert len(records) == 1
        assert "# Heading" in records[0]["content"]


# --------------------------------------------------------------------------
# 構造保持分割（方式A/B）のページメタデータ
# --------------------------------------------------------------------------


class TestStructuredChunking:
    def test_chunks_carry_page_url_and_heading_path(self) -> None:
        # HF tokenizer非依存（文字数/4 fallbackで決定的に動く）
        token_counter = TokenCounter(hf_model=None)
        pages = split_pages(SYNTHETIC_LLMS_FULL)
        records = chunk_page_structured(
            pages[0],
            token_counter,
            min_tokens=180,
            target_tokens=400,
            max_tokens=480,
            overlap_tokens=50,
        )
        assert records
        for record in records:
            assert (
                record["metadata"]["url"]
                == "https://platform.claude.com/docs/en/prompt-caching"
            )
            assert "heading_path" in record["metadata"]

    def test_chunk_id_is_content_hash_idempotent(self) -> None:
        first = chunk_id("A_structured_400", "https://x", "content")
        second = chunk_id("A_structured_400", "https://x", "content")
        assert first == second
        assert first.startswith("A_structured_400:")
        assert first != chunk_id("A_structured_400", "https://x", "other")
        assert first != chunk_id("C_naive_400", "https://x", "content")


# --------------------------------------------------------------------------
# ページ単位指標
# --------------------------------------------------------------------------


class TestPageMetrics:
    def test_dedupe_urls_keeps_first_occurrence_order(self) -> None:
        urls = ["u1", "u2", "u1", "", "u3", "u2"]
        assert dedupe_urls(urls) == ["u1", "u2", "u3"]

    def test_first_hit_rank(self) -> None:
        ranked = ["u1", "u2", "u3"]
        assert first_hit_rank(ranked, {"u2"}) == 2
        assert first_hit_rank(ranked, {"u1", "u3"}) == 1
        assert first_hit_rank(ranked, {"nope"}) is None
        assert first_hit_rank([], {"u1"}) is None

    def test_compute_metrics(self) -> None:
        # 4問: 1位・4位・圏外・2位
        metrics = compute_metrics([1, 4, None, 2])
        assert metrics["n"] == 4
        assert metrics["recall@1"] == 0.25
        assert metrics["recall@3"] == 0.5
        assert metrics["recall@5"] == 0.75
        assert metrics["recall@10"] == 0.75
        # MRR = (1 + 0.25 + 0 + 0.5) / 4
        assert metrics["mrr@10"] == round((1 + 0.25 + 0 + 0.5) / 4, 4)

    def test_compute_metrics_empty(self) -> None:
        assert compute_metrics([]) == {"n": 0}

    def test_rank_beyond_cutoff_does_not_count(self) -> None:
        metrics = compute_metrics([11])
        assert metrics["recall@10"] == 0.0
        assert metrics["mrr@10"] == 0.0


# --------------------------------------------------------------------------
# チャンク統計
# --------------------------------------------------------------------------


class TestChunkStats:
    def test_stats_counts_and_over_512(self) -> None:
        records = [
            {"metadata": {"token_count": 100}},
            {"metadata": {"token_count": 500}},
            {"metadata": {"token_count": 600}},
        ]
        stats = chunk_stats(records)
        assert stats["chunk_count"] == 3
        assert stats["avg_tokens"] == 400.0
        assert stats["max_tokens"] == 600
        assert stats["over_512_count"] == 1

    def test_stats_empty(self) -> None:
        stats = chunk_stats([])
        assert stats["chunk_count"] == 0
        assert stats["avg_tokens"] == 0

    def test_methods_definition(self) -> None:
        assert set(METHODS) == {"A_structured_400", "B_structured_200", "C_naive_400"}
        assert METHODS["A_structured_400"]["target_tokens"] == 400
        assert METHODS["B_structured_200"]["target_tokens"] == 200
        assert METHODS["C_naive_400"]["window_tokens"] == 400
