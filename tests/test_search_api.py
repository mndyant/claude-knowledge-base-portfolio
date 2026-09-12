"""search_api.py の単体テスト（TC-S-001〜006）。"""

from __future__ import annotations

import json

from scripts import embed, search_api
from scripts.embed import Chunk, _make_chunk_id, get_or_create_collection


def _seed(count: int = 10, category: str = "docs") -> None:
    """アプリが参照するコレクションへチャンクを登録する。"""
    collection = get_or_create_collection()
    chunks = []
    for i in range(count):
        source = f"knowledge/{category}/doc{i}.md"
        chunks.append(
            Chunk(
                id=_make_chunk_id(source, 0),
                content=f"Claude Code のドキュメント {i} です。",
                source=source,
                category=category,
                section=f"section-{i}",
                chunk_index=0,
                metadata={
                    "source": source,
                    "category": category,
                    "section": f"section-{i}",
                    "heading_path": f"heading > section-{i}",
                    "summary_short": f"summary {i}",
                    "token_estimate": 42,
                    "has_code": False,
                    "source_url": f"https://platform.claude.com/docs/en/doc{i}",
                },
            )
        )
    embed.embed_and_store(chunks, collection=collection)


def test_search_returns_top_k_results(api_client):
    """TC-S-001: 正常検索で200・results3件・各要素にcontent/source/score。"""
    _seed(10)

    response = api_client.post("/search", json={"query": "Claude Code", "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "content" in result
        assert "source" in result
        assert "score" in result
        assert "heading_path" in result
        assert "summary_short" in result
        assert "token_estimate" in result
        assert "has_code" in result
        assert "japanese_title" in result
        assert "japanese_summary" in result
        assert result["source_url"].startswith("https://platform.claude.com/")


def test_root_serves_search_ui(api_client):
    """The root URL serves the search interface."""
    response = api_client.get("/")

    assert response.status_code == 200
    assert "Claude Knowledge" in response.text
    assert "text/html" in response.headers["content-type"]


def test_client_renders_japanese_summary(api_client):
    response = api_client.get("/app.js")

    assert response.status_code == 200
    assert "japanese_summary" in response.text
    assert "英語の原文を表示" in response.text
    assert "公式ページで詳しく読む" in response.text
    assert "RAG AGENT ANSWER" in response.text


def test_grounded_answer_uses_numbered_citations(monkeypatch):
    result = search_api.SearchResult(
        content="Streaming uses server-sent events.",
        source="llms-full.txt",
        score=0.9,
        heading_path="Streaming",
        source_url="https://platform.claude.com/docs/en/streaming",
    )
    generated = (
        '{"answer":"ストリーミングではSSEを使用します。[1]",'
        '"results":[{"title":"ストリーミング",'
        '"summary":"応答を段階的に受信します。"}]}'
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        search_api, "_call_generation_model", lambda api_key, prompt: generated
    )
    search_api._ANSWER_CACHE.clear()
    search_api._LOCALIZATION_CACHE.clear()

    answer = search_api._generate_grounded_answer([result], "実装方法は？")

    assert answer == "ストリーミングではSSEを使用します。[1]"
    assert result.japanese_title == "ストリーミング"
    assert result.japanese_summary == "応答を段階的に受信します。"


def test_grounded_answer_localization_matches_by_position_not_key_number(
    monkeypatch,
):
    """複数resultで、タイトル・要約が対応するsourceとずれないことを確認する。

    2026-07-20に実機で発見した不具合の回帰テスト: resultsが辞書形式
    （キーが0始まりindexか1始まりcitation番号か曖昧）だと、モデルの
    キー付け規約次第でタイトルが別のsourceのものと入れ替わっていた。
    配列＋位置対応にしたことでこの曖昧さ自体を排除している。
    """
    results = [
        search_api.SearchResult(
            content=f"content-{i}",
            source="llms-full.txt",
            score=0.9,
            heading_path=f"Heading {i}",
        )
        for i in range(3)
    ]
    generated = json.dumps(
        {
            "answer": "回答本文[1][2][3]",
            "results": [
                {"title": f"タイトル{i}", "summary": f"要約{i}"} for i in range(3)
            ],
        },
        ensure_ascii=False,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        search_api, "_call_generation_model", lambda api_key, prompt: generated
    )
    search_api._ANSWER_CACHE.clear()
    search_api._LOCALIZATION_CACHE.clear()

    search_api._generate_grounded_answer(results, "質問")

    for i, result in enumerate(results):
        assert result.japanese_title == f"タイトル{i}"
        assert result.japanese_summary == f"要約{i}"


def test_expand_queries_returns_multiple_english_retrieval_terms(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        search_api,
        "_call_generation_model",
        lambda api_key, prompt: (
            '{"queries":["Claude API streaming","Messages API stream true",'
            '"SSE server-sent events"]}'
        ),
    )
    search_api._QUERY_CACHE.clear()

    expanded = search_api._expand_queries("ストリーミングの実装方法")

    assert expanded == [
        "Claude API streaming",
        "Messages API stream true",
        "SSE server-sent events",
    ]


def test_unbundled_study_returns_404(api_client):
    """Unbundled learning materials return a clear 404, not a server failure."""
    for route in ("/study", "/study.css", "/study.js", "/study-data.js"):
        assert api_client.get(route).status_code == 404


def test_root_redirects_to_docs_without_ui(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(search_api, "WEB_DIR", tmp_path)
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert api_client.get("/styles.css").status_code == 404
    assert api_client.get("/docs").status_code == 200


def test_search_only_never_calls_generation(api_client, monkeypatch):
    _seed(3)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def forbidden(*args, **kwargs):
        raise AssertionError("Search-only mode must not call external generation")

    monkeypatch.setattr(search_api, "_call_generation_model", forbidden)
    monkeypatch.setattr(search_api, "_expand_queries", forbidden)
    response = api_client.post("/search", json={"query": "Claude Code", "generate_answer": False})
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["total"] == 3


def test_search_top_k_upper_bound(api_client):
    """TC-S-002: top_k=100 は 422。"""
    response = api_client.post("/search", json={"query": "test", "top_k": 100})
    assert response.status_code == 422


def test_search_empty_query(api_client):
    """TC-S-003: 空クエリは 422。"""
    response = api_client.post("/search", json={"query": "", "top_k": 5})
    assert response.status_code == 422


def test_search_category_filter(api_client):
    """TC-S-004: categoryフィルタで全resultsが該当カテゴリ配下になる。"""
    _seed(5, category="docs")
    _seed(5, category="blog")

    response = api_client.post(
        "/search", json={"query": "test", "top_k": 5, "category": "docs"}
    )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["source"].startswith("knowledge/docs/")


def test_health_check(api_client):
    """TC-S-005: /health が 200・status ok・total_documents を返す。"""
    _seed(3)

    response = api_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_documents"] == 3


def test_search_collection_not_initialized(api_client):
    """TC-S-006: コレクション未作成時は 503・案内文言を含む。"""
    response = api_client.post("/search", json={"query": "test", "top_k": 5})

    assert response.status_code == 503
    assert "embed.py" in response.json()["detail"]
