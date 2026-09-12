"""scripts/rag_answer.py の単体テスト。

Gemini呼び出しはモックし、実API・実ネットワークには依存しない。ChromaDBは
conftest.pyのFakeModelで決定的にembedされたテスト用コレクションを使う
（``test_chroma_client`` / ``EMBED_MODEL=paraphrase-MiniLM-L3-v2`` の
テストダブル。実モデルのダウンロードは発生しない）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts import embed
from scripts import rag_answer as ra

# --------------------------------------------------------------------------
# [n] 引用番号のパース
# --------------------------------------------------------------------------


def test_parse_citation_numbers_extracts_sorted_unique():
    text = "プロンプトキャッシュは有効です[2][1]。詳細は[2]を参照。"
    assert ra.parse_citation_numbers(text) == [1, 2]


def test_parse_citation_numbers_no_citation_returns_empty():
    assert ra.parse_citation_numbers("引用のない文章です。") == []


def test_parse_citation_numbers_ignores_non_numeric_brackets():
    text = "これは[脚注]ではなく[3]が引用です。"
    assert ra.parse_citation_numbers(text) == [3]


# --------------------------------------------------------------------------
# プロンプト構成
# --------------------------------------------------------------------------


def _make_contexts() -> list[ra.Context]:
    return [
        ra.Context(
            n=1,
            chunk_id="c1",
            url="https://example.com/prompt-caching",
            heading="Prompt caching",
            content="キャッシュは最大5分間有効です。",
            score=0.9,
        ),
        ra.Context(
            n=2,
            chunk_id="c2",
            url="https://example.com/prompt-caching-ttl",
            heading="Prompt caching > 1-hour cache duration",
            content="beta headerを付けると1時間キャッシュにできます。",
            score=0.8,
        ),
    ]


def test_build_prompt_includes_query_and_context_fields():
    contexts = _make_contexts()
    prompt = ra.build_prompt("キャッシュの保存期間は？", contexts)

    assert "キャッシュの保存期間は？" in prompt
    assert "[1]" in prompt and "[2]" in prompt
    assert "Prompt caching" in prompt
    assert "https://example.com/prompt-caching-ttl" in prompt
    assert "キャッシュは最大5分間有効です。" in prompt
    assert ra.NO_EVIDENCE_PHRASE in prompt


def test_format_context_block_handles_missing_url_and_heading():
    contexts = [
        ra.Context(
            n=1, chunk_id="c1", url="", heading="", content="本文のみ", score=0.5
        )
    ]
    block = ra.format_context_block(contexts)

    assert "(URL不明)" in block
    assert "(見出しなし)" in block
    assert "本文のみ" in block


# --------------------------------------------------------------------------
# generate_answer / call_gemini
# --------------------------------------------------------------------------


def test_generate_answer_returns_no_evidence_without_calling_gemini(monkeypatch):
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("コンテキストが空ならGeminiを呼んではいけない")

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fail)

    result = ra.generate_answer("質問", contexts=[], client=object())

    assert result == ra.NO_EVIDENCE_PHRASE


def test_generate_answer_calls_gemini_with_built_prompt(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_call(client, model, prompt):  # noqa: ANN001
        captured["model"] = model
        captured["prompt"] = prompt
        return "回答本文[1]。"

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fake_call)

    contexts = _make_contexts()
    result = ra.generate_answer(
        "質問", contexts, model="gemini-flash-lite-latest", client=object()
    )

    assert result == "回答本文[1]。"
    assert captured["model"] == "gemini-flash-lite-latest"
    assert "質問" in captured["prompt"]


def test_call_gemini_falls_back_to_next_model_on_404(monkeypatch):
    calls: list[str] = []

    def _fake_call(client, model, prompt):  # noqa: ANN001
        calls.append(model)
        if model == ra.MODEL_CANDIDATES[0]:
            raise RuntimeError("404 NOT_FOUND: model retired")
        return "フォールバック成功"

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fake_call)

    result = ra.call_gemini(object(), ra.MODEL_CANDIDATES[0], "prompt")

    assert result == "フォールバック成功"
    assert calls == ra.MODEL_CANDIDATES[:2]


def test_call_gemini_raises_immediately_on_non_404_error(monkeypatch):
    def _fake_call(client, model, prompt):  # noqa: ANN001
        raise RuntimeError("500 internal error")

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fake_call)

    with pytest.raises(RuntimeError, match="500 internal error"):
        ra.call_gemini(object(), ra.MODEL_CANDIDATES[0], "prompt")


def test_call_gemini_raises_when_all_candidates_404(monkeypatch):
    def _fake_call(client, model, prompt):  # noqa: ANN001
        raise RuntimeError("404 NOT_FOUND")

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fake_call)

    with pytest.raises(RuntimeError, match="全モデル候補で失敗"):
        ra.call_gemini(object(), ra.MODEL_CANDIDATES[0], "prompt")


# --------------------------------------------------------------------------
# answer_query（検索 + 回答生成の結合）
# --------------------------------------------------------------------------


@pytest.fixture
def seeded_collection(test_chroma_client):
    """URLマーカー付きの2ページ分（各1チャンク）をテスト用コレクションへ登録する。"""
    collection = embed.get_or_create_collection(client=test_chroma_client)
    chunks = [
        embed.Chunk(
            id="page-a-0",
            content="# Prompt caching\n\nURL: https://example.com/prompt-caching\n\nキャッシュは最大5分間有効です。",
            source="llms-full.txt",
            category="docs",
            section="Prompt caching",
            chunk_index=0,
            metadata={
                "source": "llms-full.txt",
                "category": "docs",
                "section": "Prompt caching",
                "heading_path": "Prompt caching",
                "page_index": 0,
                "chunk_index": 0,
            },
        ),
        embed.Chunk(
            id="page-b-0",
            content="# Vision\n\nURL: https://example.com/vision\n\n画像をAPIに渡す方法を説明します。",
            source="llms-full.txt",
            category="docs",
            section="Vision",
            chunk_index=0,
            metadata={
                "source": "llms-full.txt",
                "category": "docs",
                "section": "Vision",
                "heading_path": "Vision",
                "page_index": 1,
                "chunk_index": 0,
            },
        ),
    ]
    embed.embed_and_store(chunks, collection=collection)
    return collection


def test_answer_query_builds_citations_and_contexts(monkeypatch, seeded_collection):
    # get_chroma_client/get_collectionをまとめてスタブし、既存のseeded_collectionを返す
    monkeypatch.setattr(
        ra,
        "get_chroma_client",
        lambda path=None: type(
            "C", (), {"get_collection": staticmethod(lambda name: seeded_collection)}
        )(),
    )

    def _fake_call(client, model, prompt):  # noqa: ANN001
        return "キャッシュは5分間有効です[1]。"

    monkeypatch.setattr(ra, "_call_gemini_with_retry", _fake_call)

    result = ra.answer_query(
        "キャッシュの保存期間は？", db_path="unused", top_k=2, client=object()
    )

    assert result["answer"] == "キャッシュは5分間有効です[1]。"
    assert len(result["contexts"]) == 2
    assert result["citations"] == [
        {
            "n": 1,
            "url": "https://example.com/prompt-caching",
            "heading": "Prompt caching",
            "chunk_preview": result["contexts"][0]["content"][:200],
        }
    ]


def test_answer_query_citations_empty_when_no_brackets_in_answer(
    monkeypatch, seeded_collection
):
    monkeypatch.setattr(
        ra,
        "get_chroma_client",
        lambda path=None: type(
            "C", (), {"get_collection": staticmethod(lambda name: seeded_collection)}
        )(),
    )
    monkeypatch.setattr(
        ra,
        "_call_gemini_with_retry",
        lambda client, model, prompt: ra.NO_EVIDENCE_PHRASE,
    )

    result = ra.answer_query("無関係な質問", db_path="unused", top_k=2, client=object())

    assert result["answer"] == ra.NO_EVIDENCE_PHRASE
    assert result["citations"] == []
    assert len(result["contexts"]) == 2


# --------------------------------------------------------------------------
# CLI出力
# --------------------------------------------------------------------------


def test_format_human_readable_lists_citations():
    result = {
        "answer": "回答本文[1]。",
        "citations": [
            {
                "n": 1,
                "url": "https://example.com/a",
                "heading": "見出しA",
                "chunk_preview": "...",
            }
        ],
    }
    text = ra.format_human_readable(result)

    assert text.startswith("回答本文[1]。")
    assert "## 引用元" in text
    assert "[1] 見出しA — https://example.com/a" in text


def test_format_human_readable_no_citations():
    result = {"answer": ra.NO_EVIDENCE_PHRASE, "citations": []}
    text = ra.format_human_readable(result)

    assert "(引用なし)" in text


def test_main_json_output(monkeypatch, capsys):
    fake_result = {
        "query": "質問",
        "answer": "回答[1]。",
        "citations": [{"n": 1, "url": "u", "heading": "h", "chunk_preview": "p"}],
        "contexts": [],
    }
    monkeypatch.setattr(
        "sys.argv",
        ["rag_answer.py", "--db-path", "dummy", "--query", "質問", "--json"],
    )
    monkeypatch.setattr(ra, "answer_query", lambda *a, **k: fake_result)

    ra.main()

    out = capsys.readouterr().out
    assert json.loads(out) == fake_result
