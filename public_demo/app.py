"""出典付き独自要約を検索する軽量公開デモ。ローカルe5版とは別のBM25検索。"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from scripts.source_notes import CHECKED_AT, NOTES, source_for

app = FastAPI(
    title="Claude Knowledge — public source-note demo",
    description="公式資料に基づく独自要約6件のBM25検索。e5意味検索はダウンロード版で利用できます。外部LLMによる回答生成はありません。",
    version="1.1.0",
)
WEB = Path(__file__).resolve().parents[1] / "web"


class SearchRequest(BaseModel):
    """ダウンロード版と同じ形の検索リクエスト。"""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None
    generate_answer: bool = False

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """空白だけの検索を拒否する。"""
        if not value.strip():
            raise ValueError("検索語を入力してください")
        return value.strip()


def terms(text: str) -> list[str]:
    """英数字の語と日本語の連続2文字で、任意の入力をトークン化する。"""
    text = text.casefold()
    result = re.findall(r"[a-z0-9_]+", text)
    for run in re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+", text):
        result.extend(run[i : i + 2] for i in range(len(run) - 1))
    return result


DOCUMENTS = [
    Counter(terms(n["title"] + " " + n["summary"] + " " + n["content"])) for n in NOTES
]
LENGTHS = [sum(doc.values()) for doc in DOCUMENTS]
AVERAGE_LENGTH = sum(LENGTHS) / len(LENGTHS)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """検索画面を配信する。"""
    return FileResponse(WEB / "index.html")


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    """画面のCSSを配信する。"""
    return FileResponse(WEB / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def javascript() -> FileResponse:
    """画面のJavaScriptを配信する。"""
    return FileResponse(WEB / "app.js", media_type="text/javascript")


@app.get("/health")
def health() -> dict:
    """公開デモの対象と検索方式を返す。"""
    return {
        "status": "ok",
        "total_documents": len(NOTES),
        "demo_mode": "source-notes",
        "retrieval": "bm25",
        "sources_checked_at": CHECKED_AT,
        "source_pages": len({note["url"] for note in NOTES}),
    }


@app.get("/sources")
def sources() -> dict:
    """実際に索引へ登録した要約と出典URLを返す。"""
    return {
        "sources": [source_for(n) for n in NOTES],
        "total": len(NOTES),
        "official_urls": sorted({n["url"] for n in NOTES}),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    """BM25で要約を検索する。未一致の入力には空の検索結果を返す。"""
    if request.generate_answer:
        raise HTTPException(
            status_code=422,
            detail="公開デモは検索のみです。generate_answerをfalseにしてください。",
        )
    start = time.perf_counter()
    ranked = []
    query_terms = set(terms(request.query))
    for note, document, length in zip(NOTES, DOCUMENTS, LENGTHS, strict=True):
        if request.category and request.category != "docs":
            continue
        score = 0.0
        for term in query_terms:
            frequency = document[term]
            if not frequency:
                continue
            matches = sum(term in doc for doc in DOCUMENTS)
            idf = math.log(1 + (len(NOTES) - matches + 0.5) / (matches + 0.5))
            score += (
                idf
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * length / AVERAGE_LENGTH))
            )
        if score > 0:
            ranked.append(
                {
                    "content": note["content"],
                    "source": source_for(note),
                    "source_url": note["url"],
                    "score": round(score, 4),
                    "heading_path": note["title"],
                    "summary_short": note["summary"],
                    "has_code": False,
                }
            )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    results = ranked[: request.top_k]
    return {
        "results": results,
        "answer": None,
        "total": len(results),
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "retrieval": "bm25",
    }
