"""公開デモの実検索・境界値・回答生成の禁止を検証する。"""

from fastapi.testclient import TestClient

from public_demo.app import app

client = TestClient(app)


def test_public_demo_searches_source_notes() -> None:
    """日本語と英語の検索が、対応する公式出典を持つ要約を返す。"""
    for query, expected in [
        ("画像をAPIに渡す方法は？", "/vision"),
        ("text_stream SSE", "/streaming"),
        ("tool_use_id tool_result", "/tool-use/overview"),
    ]:
        response = client.post("/search", json={"query": query, "top_k": 1})
        assert response.status_code == 200
        payload = response.json()
        assert payload["results"][0]["source_url"].endswith(expected)
        assert payload["answer"] is None
        assert payload["retrieval"] == "bm25"


def test_public_demo_returns_no_results_for_unmatched_input() -> None:
    """固定の回答を返さず、索引との一致がなければ空にする。"""
    assert (
        client.post("/search", json={"query": "zzzzunmatched999"}).json()["results"]
        == []
    )
    assert (
        client.post("/search", json={"query": "streaming", "category": "blog"}).json()[
            "results"
        ]
        == []
    )


def test_public_demo_rejects_generation_and_invalid_input() -> None:
    """生成要求と入力上限違反を実行前に拒否する。"""
    for body in [
        {"query": "streaming", "generate_answer": True},
        {"query": " "},
        {"query": "a" * 4001},
        {"query": "vision", "top_k": 21},
    ]:
        assert client.post("/search", json=body).status_code == 422


def test_public_demo_serves_ui_and_swagger() -> None:
    """公開リンクの各経路と、全6件の索引を確認する。"""
    for route in ["/", "/styles.css", "/app.js", "/docs", "/openapi.json"]:
        assert client.get(route).status_code == 200
    assert client.get("/health").json()["total_documents"] == 6
    assert len(client.get("/sources").json()["official_urls"]) == 3
