"""scripts/eval_retrieval.py の単体テスト。

実embeddingモデル・実ChromaDB・ネットワークには依存しない。
Recall/MRRの数学的正しさ、gold照合ロジック、FTS5インデックスの構築と検索を
合成データ（数件のミニコーパス）で検証する。ChromaDBの``collection``に相当する
部分はFakeCollectionでダブル化する。
"""

from __future__ import annotations

import json

from scripts import eval_retrieval as ev


class FakeCollection:
    """``collection.get(include=...)`` だけを模したテストダブル。"""

    def __init__(self, records: list[tuple[str, dict, str]]) -> None:
        self._records = records

    def get(self, include=None):  # type: ignore[no-untyped-def]
        return {
            "ids": [r[0] for r in self._records],
            "metadatas": [r[1] for r in self._records],
            "documents": [r[2] for r in self._records],
        }


# --------------------------------------------------------------------------
# gold照合ロジック
# --------------------------------------------------------------------------


def test_matches_criterion_heading_contains_is_case_insensitive():
    criterion = {"type": "heading_contains", "value": "Prompt Caching"}
    metadata = {"heading_path": "Docs > prompt caching > TTL"}
    assert ev.matches_criterion(criterion, metadata, resolved_url="") is True


def test_matches_criterion_heading_contains_no_match():
    criterion = {"type": "heading_contains", "value": "Vision"}
    metadata = {"heading_path": "Docs > Streaming"}
    assert ev.matches_criterion(criterion, metadata, resolved_url="") is False


def test_matches_criterion_url_contains_uses_resolved_url():
    criterion = {"type": "url_contains", "value": "prompt-caching"}
    metadata = {"source": "llms-full.txt"}
    resolved_url = (
        "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"
    )
    assert ev.matches_criterion(criterion, metadata, resolved_url) is True


def test_matches_criterion_url_contains_no_match_without_resolved_url():
    criterion = {"type": "url_contains", "value": "prompt-caching"}
    metadata = {"source": "llms-full.txt"}
    assert ev.matches_criterion(criterion, metadata, resolved_url="") is False


def test_is_relevant_true_if_any_criterion_matches():
    item = ev.GoldItem(
        id="q1",
        query="test",
        lang="ja",
        category="cat",
        relevant=[
            {"type": "heading_contains", "value": "nomatch"},
            {"type": "url_contains", "value": "prompt-caching"},
        ],
    )
    metadata = {
        "heading_path": "Docs > Other",
        "source": "llms-full.txt",
        "page_index": 1,
    }
    page_urls = {"1": "https://example.com/prompt-caching"}
    assert ev.is_relevant(item, metadata, page_urls) is True


def test_is_relevant_false_if_no_criterion_matches():
    item = ev.GoldItem(
        id="q1",
        query="test",
        lang="ja",
        category="cat",
        relevant=[{"type": "heading_contains", "value": "nomatch"}],
    )
    metadata = {"heading_path": "Docs > Other", "page_index": 1}
    assert ev.is_relevant(item, metadata, page_urls={}) is False


# --------------------------------------------------------------------------
# Recall@k / MRR@10 の数学的正しさ
# --------------------------------------------------------------------------


def _record(query_id: str, lang: str, first_hit_rank: int | None) -> dict:
    return {
        "id": query_id,
        "query": query_id,
        "lang": lang,
        "category": "",
        "notes": "",
        "first_hit_rank": first_hit_rank,
        "hits": [],
    }


def test_subset_metrics_recall_at_k():
    # rank1が1件、rank3が1件、rank10が1件、圏外(None)が1件 → 4件中
    records = [
        _record("a", "ja", 1),
        _record("b", "ja", 3),
        _record("c", "ja", 10),
        _record("d", "ja", None),
    ]
    metrics = ev._subset_metrics(records, top_n=10)
    assert metrics["n"] == 4
    assert metrics["recall@1"] == 1 / 4
    assert metrics["recall@3"] == 2 / 4
    assert metrics["recall@5"] == 2 / 4
    assert metrics["recall@10"] == 3 / 4


def test_subset_metrics_mrr():
    records = [
        _record("a", "ja", 1),  # 1/1
        _record("b", "ja", 2),  # 1/2
        _record("c", "ja", None),  # 0
        _record("d", "ja", 4),  # 1/4
    ]
    metrics = ev._subset_metrics(records, top_n=10)
    expected_mrr = (1 / 1 + 1 / 2 + 0 + 1 / 4) / 4
    assert metrics["mrr@10"] == round(expected_mrr, 4)


def test_subset_metrics_ignores_ranks_beyond_mrr_cutoff():
    # first_hit_rankが11（MRR_CUTOFF=10を超える）ならMRRには0として寄与する
    records = [_record("a", "ja", 11)]
    metrics = ev._subset_metrics(records, top_n=20)
    assert metrics["mrr@10"] == 0.0


def test_subset_metrics_respects_top_n_cap():
    # top_n=3なら recall@5, recall@10 は計算しない
    records = [_record("a", "ja", 1)]
    metrics = ev._subset_metrics(records, top_n=3)
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "recall@5" not in metrics
    assert "recall@10" not in metrics


def test_subset_metrics_empty_returns_none():
    assert ev._subset_metrics([], top_n=10) is None


def test_aggregate_splits_by_lang():
    records = [
        _record("a", "ja", 1),
        _record("b", "ja", None),
        _record("c", "en", 1),
    ]
    result = ev.aggregate(records, top_n=10)
    assert result["overall"]["n"] == 3
    assert result["by_lang"]["ja"]["n"] == 2
    assert result["by_lang"]["en"]["n"] == 1
    assert result["by_lang"]["ja"]["recall@1"] == 1 / 2
    assert result["by_lang"]["en"]["recall@1"] == 1.0


# --------------------------------------------------------------------------
# データセット読み込み
# --------------------------------------------------------------------------


def test_load_dataset_roundtrip(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q001",
                "query": "テスト",
                "lang": "ja",
                "category": "test",
                "relevant": [{"type": "heading_contains", "value": "X"}],
                "notes": "note",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    items = ev.load_dataset(path)
    assert len(items) == 1
    assert items[0].id == "q001"
    assert items[0].lang == "ja"
    assert items[0].relevant == [{"type": "heading_contains", "value": "X"}]


# --------------------------------------------------------------------------
# FTS5インデックスの構築・検索（合成コーパス）
# --------------------------------------------------------------------------


def _sample_records() -> list[tuple[str, dict, str]]:
    return [
        (
            "chunk-1",
            {
                "heading_path": "Prompt caching",
                "source": "llms-full.txt",
                "page_index": 1,
                "category": "docs",
            },
            "URL: https://example.com/docs/prompt-caching\n\nCache your system prompt "
            "to reduce cost and latency on repeated requests.",
        ),
        (
            "chunk-2",
            {
                "heading_path": "Vision",
                "source": "llms-full.txt",
                "page_index": 2,
                "category": "docs",
            },
            "URL: https://example.com/docs/vision\n\nSend images to Claude as base64 "
            "or a URL for visual understanding.",
        ),
        (
            "chunk-3",
            {
                "heading_path": "Vision > Limitations",
                "source": "llms-full.txt",
                "page_index": 2,
                "category": "docs",
            },
            "Claude may struggle with spatial reasoning and small text in images.",
        ),
    ]


def test_build_fts_index_and_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")
    monkeypatch.setattr(ev, "FTS_DB_PATH", tmp_path / "eval" / "cache" / "fts.sqlite")

    collection = FakeCollection(_sample_records())
    conn = ev.build_fts_index(collection, rebuild=True)

    hits = ev.fts_search(conn, "cache system prompt", top_n=5)
    assert hits, "キャッシュに関するクエリで結果が返るはず"
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].rank == 1

    hits_vision = ev.fts_search(conn, "images visual", top_n=5)
    assert hits_vision[0].chunk_id == "chunk-2"


def test_build_fts_index_reuses_existing_db_unless_rebuild(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")
    monkeypatch.setattr(ev, "FTS_DB_PATH", tmp_path / "eval" / "cache" / "fts.sqlite")

    collection = FakeCollection(_sample_records())
    ev.build_fts_index(collection, rebuild=True)
    assert ev.FTS_DB_PATH.exists()

    # 2回目はrebuild=Falseなら既存DBをそのまま使う（空コレクションでも壊れない）
    empty_collection = FakeCollection([])
    conn = ev.build_fts_index(empty_collection, rebuild=False)
    hits = ev.fts_search(conn, "cache system prompt", top_n=5)
    assert hits and hits[0].chunk_id == "chunk-1"


def test_fts_search_no_match_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")
    monkeypatch.setattr(ev, "FTS_DB_PATH", tmp_path / "eval" / "cache" / "fts.sqlite")

    conn = ev.build_fts_index(FakeCollection(_sample_records()), rebuild=True)
    hits = ev.fts_search(conn, "zzzznonexistentword", top_n=5)
    assert hits == []


# --------------------------------------------------------------------------
# ページURL復元（url_contains判定用）
# --------------------------------------------------------------------------


def test_build_page_url_index_extracts_url_from_document(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")
    monkeypatch.setattr(
        ev, "PAGE_URL_CACHE", tmp_path / "eval" / "cache" / "page_urls.json"
    )

    collection = FakeCollection(_sample_records())
    urls = ev.build_page_url_index(collection)

    assert urls["1"] == "https://example.com/docs/prompt-caching"
    assert urls["2"] == "https://example.com/docs/vision"
    assert ev.PAGE_URL_CACHE.exists()


def test_build_page_url_index_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")
    monkeypatch.setattr(
        ev, "PAGE_URL_CACHE", tmp_path / "eval" / "cache" / "page_urls.json"
    )

    ev.build_page_url_index(FakeCollection(_sample_records()))
    # 2回目は空コレクションでも、キャッシュファイルがあるのでそのまま返す
    urls = ev.build_page_url_index(FakeCollection([]))
    assert urls["1"] == "https://example.com/docs/prompt-caching"


# --------------------------------------------------------------------------
# クエリ単位の中間結果キャッシュ（再開可能設計）
# --------------------------------------------------------------------------


def test_raw_cache_append_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "eval" / "cache")

    assert ev.load_raw_cache("vector") == {}

    record = {"id": "q001", "first_hit_rank": 1, "hits": []}
    ev.append_raw_cache("vector", record)

    cached = ev.load_raw_cache("vector")
    assert "q001" in cached
    assert cached["q001"]["first_hit_rank"] == 1


def test_build_query_record_marks_first_hit_and_relevance():
    item = ev.GoldItem(
        id="q001",
        query="test",
        lang="ja",
        category="cat",
        relevant=[{"type": "heading_contains", "value": "Vision"}],
    )
    hits = [
        ev.RetrievedHit(
            rank=1, chunk_id="a", score=0.9, metadata={"heading_path": "Other"}
        ),
        ev.RetrievedHit(
            rank=2, chunk_id="b", score=0.8, metadata={"heading_path": "Vision > FAQ"}
        ),
    ]
    record = ev.build_query_record(item, hits, page_urls={})
    assert record["first_hit_rank"] == 2
    assert record["hits"][0]["is_relevant"] is False
    assert record["hits"][1]["is_relevant"] is True


# --------------------------------------------------------------------------
# FTS5用トークナイズ
# --------------------------------------------------------------------------


def test_fts_match_query_builds_or_joined_tokens():
    assert ev._fts_match_query("How to cache system-prompts?") == (
        "how OR to OR cache OR system OR prompts"
    )


def test_fts_match_query_empty_for_no_word_chars():
    assert ev._fts_match_query("???") == ""
