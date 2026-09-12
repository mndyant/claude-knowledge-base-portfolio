"""引用付き回答生成（RAG: 検索 → 番号付きコンテキスト → Gemini生成）。

``eval_retrieval.py`` と同じembedding手順（``query: `` prefix・同一コレクション）
でChromaDBからtop-k件を検索し、各チャンクのURL・見出し・本文を番号付き
コンテキストとしてプロンプトに構成した上でGemini（無料枠）に回答生成させる。
生成された回答は事実主張ごとに ``[n]`` 引用を伴うことを期待し、根拠が
コンテキストに無い場合は「ドキュメントから確認できません」と明言させる
（hallucination対策）。

モデル選定・429リトライの流儀は ``scripts/digest.py`` に合わせている
（``-latest`` エイリアスの候補リストを順に試し、404/NOT_FOUNDのみ次候補へ
フォールバックする）。

``answer_query()`` はライブラリとして ``scripts/eval_answers.py`` から
直接呼び出せるように分離してある。

使い方:
    python scripts/rag_answer.py --db-path <chroma-dataディレクトリ> --query "質問"
    python scripts/rag_answer.py --db-path <...> --query "質問" --top-k 6 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.digest import MODEL_CANDIDATES, _call_gemini_with_retry  # noqa: E402
from scripts.embed import get_chroma_client, get_collection_name  # noqa: E402
from scripts.eval_retrieval import build_page_url_index, vector_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 6
# digest.pyと同じ無料枠モデル系列・フォールバック順を流用する。
DEFAULT_MODEL = MODEL_CANDIDATES[0]

NO_EVIDENCE_PHRASE = "ドキュメントから確認できません"

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Context:
    """検索されたチャンク1件を、回答生成プロンプト用に整えたもの。"""

    n: int  # プロンプト・回答内で使う引用番号（1始まり）
    chunk_id: str
    url: str
    heading: str
    content: str
    score: float


PROMPT_TEMPLATE = """あなたはAnthropic公式ドキュメントに基づいて質問に答えるアシスタントです。
以下の「コンテキスト」だけを根拠に、質問に答えてください。

# ルール
- 質問と同じ言語で回答してください。
- 事実に関する主張には、根拠にしたコンテキスト番号を必ず [n] の形式で付けてください
  （例: 「プロンプトキャッシュは最大5分間有効です[1]。」複数の根拠がある場合は
  [1][2] のように併記してください）。
- コンテキストに書かれていないことは、推測で埋めずに「{no_evidence_phrase}」と
  明言してください。一部だけ分かる場合は、分かる範囲だけ [n] 付きで答え、
  分からない部分についてのみ確認できない旨を書いてください。
- コンテキストに無い外部知識で補完しないでください。

# コンテキスト
{context_block}

# 質問
{query}

# 回答
"""


def format_context_block(contexts: list[Context]) -> str:
    """番号付きコンテキストブロックを組み立てる（プロンプト埋め込み用）。"""
    blocks = []
    for ctx in contexts:
        heading = ctx.heading or "(見出しなし)"
        url = ctx.url or "(URL不明)"
        blocks.append(f"[{ctx.n}] 見出し: {heading}\nURL: {url}\n本文:\n{ctx.content}")
    return "\n\n".join(blocks)


def build_prompt(query: str, contexts: list[Context]) -> str:
    """質問とコンテキストから回答生成プロンプトを組み立てる。"""
    return PROMPT_TEMPLATE.format(
        no_evidence_phrase=NO_EVIDENCE_PHRASE,
        context_block=format_context_block(contexts),
        query=query,
    )


def parse_citation_numbers(text: str) -> list[int]:
    """回答本文から ``[n]`` 引用番号を重複無し・昇順で抽出する。"""
    return sorted({int(match) for match in CITATION_RE.findall(text)})


def retrieve_contexts(
    query: str, collection: Any, top_k: int, page_urls: dict[str, str]
) -> list[Context]:
    """``eval_retrieval.vector_search`` と同じ検索手順でtop-k件をContext化する。"""
    hits = vector_search(query, collection, top_k)
    contexts: list[Context] = []
    for hit in hits:
        page_index = str(hit.metadata.get("page_index", ""))
        url = page_urls.get(page_index, "")
        heading = str(
            hit.metadata.get("heading_path") or hit.metadata.get("section") or ""
        )
        contexts.append(
            Context(
                n=hit.rank,
                chunk_id=hit.chunk_id,
                url=url,
                heading=heading,
                content=hit.document,
                score=hit.score,
            )
        )
    return contexts


def call_gemini(client: Any, model: str, prompt: str) -> str:
    """指定モデルでGeminiを呼び出す。404/NOT_FOUNDのみ次候補へフォールバックする

    （digest.py の ``summarize_with_gemini`` と同じ流儀）。それ以外の例外は
    そのまま送出する。
    """
    models = [model] + [m for m in MODEL_CANDIDATES if m != model]
    last_exc: Exception | None = None
    for candidate in models:
        try:
            text = _call_gemini_with_retry(client, candidate, prompt)
            if text:
                return text
            last_exc = RuntimeError(f"{candidate} から空応答が返されました")
        except Exception as exc:  # noqa: BLE001 - 次候補へフォールバック
            if "404" in str(exc) or "NOT_FOUND" in str(exc):
                logger.warning("モデル %s は利用不可。次の候補を試します", candidate)
                last_exc = exc
                continue
            raise
    raise RuntimeError(
        f"Gemini呼び出しに失敗しました（全モデル候補で失敗）: {last_exc}"
    )


def generate_answer(
    query: str,
    contexts: list[Context],
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> str:
    """コンテキストを根拠にGeminiで回答文を生成する。

    コンテキストが空の場合はGeminiを呼ばずに確認不可の定型文を返す。
    """
    if not contexts:
        return NO_EVIDENCE_PHRASE

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY が未設定です")
        from google import genai

        client = genai.Client(api_key=api_key)

    prompt = build_prompt(query, contexts)
    return call_gemini(client, model, prompt).strip()


def answer_query(
    query: str,
    db_path: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> dict[str, Any]:
    """検索から回答生成までを行い、機械可読な結果を返す（ライブラリの中心関数）。

    ``eval_answers.py`` からはこの関数だけを呼べばよい。

    Returns:
        ``{"query", "answer", "citations", "contexts"}`` を持つdict。
        ``citations`` は回答中で実際に ``[n]`` 引用された項目のみ、
        ``contexts`` は検索されたtop-k件全件（judge評価向けに全文を含む）。
    """
    chroma_client = get_chroma_client(path=db_path)
    collection = chroma_client.get_collection(get_collection_name())
    page_urls = build_page_url_index(collection)

    contexts = retrieve_contexts(query, collection, top_k, page_urls)
    answer_text = generate_answer(query, contexts, model=model, client=client)

    cited_ns = parse_citation_numbers(answer_text)
    context_by_n = {ctx.n: ctx for ctx in contexts}
    citations = [
        {
            "n": n,
            "url": context_by_n[n].url,
            "heading": context_by_n[n].heading,
            "chunk_preview": context_by_n[n].content[:200],
        }
        for n in cited_ns
        if n in context_by_n
    ]

    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "contexts": [
            {
                "n": ctx.n,
                "chunk_id": ctx.chunk_id,
                "url": ctx.url,
                "heading": ctx.heading,
                "content": ctx.content,
                "score": ctx.score,
            }
            for ctx in contexts
        ],
    }


def format_human_readable(result: dict[str, Any]) -> str:
    """回答本文＋「引用元」セクションのテキストを組み立てる（CLI表示用）。"""
    lines = [result["answer"], "", "## 引用元"]
    if not result["citations"]:
        lines.append("(引用なし)")
    else:
        for citation in result["citations"]:
            heading = citation["heading"] or "(見出しなし)"
            url = citation["url"] or "(URL不明)"
            lines.append(f"[{citation['n']}] {heading} — {url}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="検索されたコンテキストを根拠に引用付きで回答を生成する"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="ChromaDBの永続化ディレクトリ（外部パス。リポジトリ内へはコピーしないこと）",
    )
    parser.add_argument("--query", required=True, help="質問文")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--json", action="store_true", help="機械可読なJSONで出力する")
    parser.add_argument(
        "--model", default=os.environ.get("RAG_ANSWER_MODEL", DEFAULT_MODEL)
    )
    return parser.parse_args()


def main() -> None:
    """CLIエントリポイント。"""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    args = parse_args()
    result = answer_query(
        args.query, db_path=args.db_path, top_k=args.top_k, model=args.model
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_human_readable(result))


if __name__ == "__main__":
    main()
