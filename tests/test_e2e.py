"""E2Eテスト（TC-E2E-001〜002）。

GitHub Actions相当の一連実行（fetch → embed → API検索）を検証する。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from scripts import fetch
from scripts.embed import (
    embed_and_store,
    get_or_create_collection,
    load_documents,
    split_chunks,
)
from scripts.search_api import app


def _run_pipeline() -> None:
    """fetch → embed の一連を実行する。"""
    fetch.fetch_llms_full_txt()
    documents = load_documents("knowledge")
    chunks = split_chunks(documents)
    embed_and_store(chunks, collection=get_or_create_collection())


def test_pipeline_end_to_end(requests_mock, mock_llms_txt):
    """TC-E2E-001: 取得→登録→API起動→health→searchが全て成功する。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, text=mock_llms_txt)

    _run_pipeline()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["total_documents"] >= 1

    search = client.post("/search", json={"query": "Claude Code", "top_k": 5})
    assert search.status_code == 200
    assert len(search.json()["results"]) >= 1


def test_pipeline_diff_update_no_duplication(requests_mock, mock_llms_txt):
    """TC-E2E-002: 同一ソースで2回実行しても件数が増えない。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, text=mock_llms_txt)

    _run_pipeline()
    first_count = get_or_create_collection().count()

    _run_pipeline()
    second_count = get_or_create_collection().count()

    assert second_count == first_count
