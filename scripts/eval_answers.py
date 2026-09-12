"""RAG回答生成のhallucination評価ハーネス（groundedness / citation precision / abstention）。

``eval/qa_dataset.jsonl``（15問: 既存goldから流用した answerable な12問 +
ドキュメントに答えが存在しないトラップ3問）に対し、2段階で評価する。

- ステージ1（generate）: ``scripts.rag_answer.answer_query`` で回答を生成し、
  質問単位で ``eval/cache/answers.jsonl`` に保存する。
- ステージ2（judge）: Gemini（LLM-judge）で回答を claim（主張）単位に分解し、
  各claimが検索コンテキストに支持されるか（groundedness）、claimに付いた
  ``[n]`` 引用が実際にそのclaimを支持しているか（citation validity）、
  回答全体として「確認できません」と正しく回答保留したか（abstention）を
  判定し、質問単位で ``eval/cache/judgements.jsonl`` に保存する。

どちらのステージも質問ID単位で中間保存し、再実行時は完了済みの質問を
スキップする（長時間プロセスが強制終了される環境を想定）。

Gemini呼び出しの流儀（モデル名フォールバック・429リトライ）は
``scripts/rag_answer.py`` 経由で ``scripts/digest.py`` に合わせている。
judgeプロンプトはJSON出力のみを強制するが、パース失敗時は1回だけ
「JSONのみを出力してください」という念押しを添えて再試行する。

使い方:
    python scripts/eval_answers.py --db-path <chroma-dataディレクトリ>
    python scripts/eval_answers.py --db-path <...> --stage generate
    python scripts/eval_answers.py --db-path <...> --stage judge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rag_answer import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    answer_query,
    call_gemini,
)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("eval/qa_dataset.jsonl")
CACHE_DIR = Path("eval/cache")
ANSWERS_CACHE = CACHE_DIR / "answers.jsonl"
JUDGEMENTS_CACHE = CACHE_DIR / "judgements.jsonl"
DEFAULT_OUT_DIR = Path("eval/results")

# Gemini無料枠のレート制限に配慮し、リクエスト間に数秒あける。
REQUEST_SLEEP_SECONDS = 4.0


@dataclass
class QAItem:
    """評価データセットの1問。"""

    id: str
    query: str
    answerable: bool
    lang: str = ""
    category: str = ""
    notes: str = ""


JUDGE_PROMPT_TEMPLATE = """あなたはRAG（検索拡張生成）システムの回答を検証するジャッジです。
以下の「質問」「検索されたコンテキスト」「生成された回答」を読み、
回答の主張を検証してください。

# 質問
{query}

# 検索されたコンテキスト（[n]は回答内の引用番号に対応します）
{context_block}

# 生成された回答
{answer}

# タスク
1. 回答を意味のある主張（claim）単位に分解してください。単なる相槌・
   質問の繰り返し・「ドキュメントから確認できません」のようなメタ発言は
   claimとして抽出しないでください。
2. 各claimについて、上記コンテキストがその主張を裏付けているかを
   判定してください:
   - "supported": コンテキストの内容から明確に裏付けられる
   - "unsupported": コンテキストに書かれていない、またはコンテキストと矛盾する
3. 各claimについて、回答文中でそのclaimに付けられていた引用番号 [n] を
   cited_ns として抽出してください（複数可、無ければ空配列 []）。
4. cited_nsが1つ以上ある場合、それらの引用先コンテキストが実際にその
   claimを支持しているかを citation_valid (true/false) として判定して
   ください（cited_nsが空の場合は citation_valid を null にしてください）。
5. 回答全体として「コンテキストから確認できない」という趣旨で回答を
   拒否・保留しているかを abstained (true/false) として判定してください
   （部分的にでも具体的な答えを述べている場合は false）。

# 出力形式
説明文やコードフェンス（```）は一切含めず、以下のJSONオブジェクトのみを
出力してください。
{{
  "claims": [
    {{"text": "claimの要約", "verdict": "supported", "cited_ns": [1], "citation_valid": true}}
  ],
  "abstained": false
}}
"""

JSON_RETRY_NOTE = (
    "前回の出力はJSONとして解析できませんでした。説明文やコードフェンスを"
    "一切含めず、JSONオブジェクトのみを出力してください。"
)


def load_dataset(path: Path = DEFAULT_DATASET) -> list[QAItem]:
    """評価データセット（JSONL）を読み込む。"""
    items: list[QAItem] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            items.append(
                QAItem(
                    id=record["id"],
                    query=record["query"],
                    answerable=bool(record["answerable"]),
                    lang=record.get("lang", ""),
                    category=record.get("category", ""),
                    notes=record.get("notes", ""),
                )
            )
    return items


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """質問ID単位の中間結果キャッシュを読み込む（存在しなければ空辞書）。"""
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["id"]] = record
    return records


def append_cache(path: Path, record: dict[str, Any]) -> None:
    """1問分の中間結果を追記する（プロセスが途中で落ちても再実行で再利用できる）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# ステージ1: 回答生成
# --------------------------------------------------------------------------


def run_generate_stage(
    dataset: list[QAItem],
    db_path: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    out_path: Path = ANSWERS_CACHE,
    sleep_seconds: float = REQUEST_SLEEP_SECONDS,
) -> list[dict[str, Any]]:
    """全問についてrag_answer.answer_queryで回答生成し、質問単位で保存する。"""
    cached = load_cache(out_path)
    todo = [item for item in dataset if item.id not in cached]
    print(f"[generate] {len(cached)} cached, {len(todo)} to run", flush=True)

    started = time.time()
    for index, item in enumerate(todo, start=1):
        result = answer_query(item.query, db_path=db_path, top_k=top_k, model=model)
        record = {
            "id": item.id,
            "query": item.query,
            "answerable": item.answerable,
            "lang": item.lang,
            "category": item.category,
            "answer": result["answer"],
            "citations": result["citations"],
            "contexts": result["contexts"],
        }
        append_cache(out_path, record)
        cached[item.id] = record
        print(
            f"[generate] {index}/{len(todo)} {item.id} "
            f"({time.time() - started:.1f}s elapsed)",
            flush=True,
        )
        if index < len(todo):
            time.sleep(sleep_seconds)

    return [cached[item.id] for item in dataset]


# --------------------------------------------------------------------------
# ステージ2: LLM-judge
# --------------------------------------------------------------------------


def _format_judge_context_block(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for ctx in contexts:
        heading = ctx.get("heading") or "(見出しなし)"
        blocks.append(f"[{ctx['n']}] 見出し: {heading}\n本文:\n{ctx['content']}")
    return "\n\n".join(blocks)


def build_judge_prompt(
    query: str, contexts: list[dict[str, Any]], answer: str, retry_note: str = ""
) -> str:
    """judge用プロンプトを組み立てる。"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        context_block=_format_judge_context_block(contexts),
        answer=answer,
    )
    if retry_note:
        prompt = f"{prompt}\n\n{retry_note}"
    return prompt


def parse_judge_response(text: str) -> dict[str, Any]:
    """judge応答のJSONをパースする（コードフェンス付きにも対応）。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(stripped)
    if "claims" not in parsed or "abstained" not in parsed:
        raise ValueError("judge応答に claims / abstained が含まれていません")
    return parsed


def judge_answer(
    client: Any,
    model: str,
    query: str,
    contexts: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """1問分の回答をLLM-judgeで評価する。パース失敗時は1回だけ再試行する。"""
    prompt = build_judge_prompt(query, contexts, answer)
    text = call_gemini(client, model, prompt)
    try:
        return parse_judge_response(text)
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        logger.warning("judge応答のJSON解析に失敗。1回だけ再試行します: %s", exc)
        retry_prompt = build_judge_prompt(
            query, contexts, answer, retry_note=JSON_RETRY_NOTE
        )
        retry_text = call_gemini(client, model, retry_prompt)
        return parse_judge_response(retry_text)


def run_judge_stage(
    answer_records: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    out_path: Path = JUDGEMENTS_CACHE,
    client: Any = None,
    sleep_seconds: float = REQUEST_SLEEP_SECONDS,
) -> list[dict[str, Any]]:
    """全問についてjudge_answerを実行し、質問単位で保存する。"""
    cached = load_cache(out_path)
    todo = [r for r in answer_records if r["id"] not in cached]
    print(f"[judge] {len(cached)} cached, {len(todo)} to run", flush=True)

    if todo and client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY が未設定です")
        from google import genai

        client = genai.Client(api_key=api_key)

    started = time.time()
    for index, record in enumerate(todo, start=1):
        try:
            judgement = judge_answer(
                client, model, record["query"], record["contexts"], record["answer"]
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - judge失敗は記録して評価を継続する
            logger.warning(
                "judge失敗（%s）。この問題は集計から除外します: %s", record["id"], exc
            )
            judgement = {"claims": [], "abstained": False}
            error = str(exc)

        out_record: dict[str, Any] = {
            "id": record["id"],
            "answerable": record["answerable"],
            **judgement,
        }
        if error:
            out_record["error"] = error
        append_cache(out_path, out_record)
        cached[record["id"]] = out_record
        print(
            f"[judge] {index}/{len(todo)} {record['id']} "
            f"({time.time() - started:.1f}s elapsed)",
            flush=True,
        )
        if index < len(todo):
            time.sleep(sleep_seconds)

    return [cached[r["id"]] for r in answer_records]


# --------------------------------------------------------------------------
# 指標集計
# --------------------------------------------------------------------------


def aggregate(
    answer_records: list[dict[str, Any]], judgements: list[dict[str, Any]]
) -> dict[str, Any]:
    """groundedness率・citation precision・abstention accuracyを集計する。"""
    judgement_by_id = {j["id"]: j for j in judgements}

    total_supported = 0
    total_unsupported = 0
    total_citation_valid = 0
    total_citation_checked = 0
    trap_total = 0
    trap_correct = 0
    answerable_wrongly_abstained = 0
    judge_errors = 0
    per_question: list[dict[str, Any]] = []

    for record in answer_records:
        judgement = judgement_by_id.get(record["id"], {})
        claims = judgement.get("claims", [])

        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        unsupported = sum(1 for c in claims if c.get("verdict") == "unsupported")
        total_supported += supported
        total_unsupported += unsupported

        cited_claims = [c for c in claims if c.get("cited_ns")]
        valid = sum(1 for c in cited_claims if c.get("citation_valid") is True)
        total_citation_valid += valid
        total_citation_checked += len(cited_claims)

        answerable = bool(record["answerable"])
        abstained = bool(judgement.get("abstained", False))
        abstain_correct: bool | None = None
        if not answerable:
            trap_total += 1
            abstain_correct = abstained
            if abstained:
                trap_correct += 1
        else:
            if abstained:
                answerable_wrongly_abstained += 1
            abstain_correct = not abstained

        if judgement.get("error"):
            judge_errors += 1

        per_question.append(
            {
                "id": record["id"],
                "query": record["query"],
                "answerable": answerable,
                "n_claims": len(claims),
                "n_supported": supported,
                "n_unsupported": unsupported,
                "n_citations_checked": len(cited_claims),
                "n_citations_valid": valid,
                "abstained": abstained,
                "abstain_correct": abstain_correct,
                "judge_error": judgement.get("error"),
            }
        )

    total_claims = total_supported + total_unsupported
    groundedness_rate = (
        round(total_supported / total_claims, 4) if total_claims else None
    )
    citation_precision = (
        round(total_citation_valid / total_citation_checked, 4)
        if total_citation_checked
        else None
    )
    trap_abstention_accuracy = (
        round(trap_correct / trap_total, 4) if trap_total else None
    )
    n_answerable = sum(1 for r in answer_records if r["answerable"])

    return {
        "n_questions": len(answer_records),
        "n_answerable": n_answerable,
        "n_trap": trap_total,
        "groundedness_rate": groundedness_rate,
        "citation_precision": citation_precision,
        "trap_abstention_accuracy": trap_abstention_accuracy,
        "answerable_wrongly_abstained": answerable_wrongly_abstained,
        "total_claims": total_claims,
        "total_supported_claims": total_supported,
        "total_unsupported_claims": total_unsupported,
        "total_citations_checked": total_citation_checked,
        "total_citations_valid": total_citation_valid,
        "judge_errors": judge_errors,
        "per_question": per_question,
    }


# --------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------


def write_result_json(out_dir: Path, result: dict[str, Any]) -> Path:
    """集計結果をJSONへ書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "answer_eval.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary_md(out_dir: Path, result: dict[str, Any]) -> Path:
    """人間が読むサマリーMarkdownを書き出す。"""
    lines = [
        "# 回答生成 hallucination評価サマリー",
        "",
        f"- 問題数: {result['n_questions']}"
        f"（answerable {result['n_answerable']} / trap {result['n_trap']}）",
        f"- groundedness率: {result['groundedness_rate']}"
        f"（supported {result['total_supported_claims']} / "
        f"unsupported {result['total_unsupported_claims']} / "
        f"claims合計 {result['total_claims']}）",
        f"- citation precision: {result['citation_precision']}"
        f"（valid {result['total_citations_valid']} / "
        f"checked合計 {result['total_citations_checked']}）",
        f"- trap abstention accuracy: {result['trap_abstention_accuracy']}"
        f"（{result['n_trap']}問中の正答率）",
        f"- answerable問での不必要な回答拒否: {result['answerable_wrongly_abstained']}件",
    ]
    if result.get("judge_errors"):
        lines.append(f"- judge解析失敗（集計除外）: {result['judge_errors']}件")

    lines += [
        "",
        "## 質問別内訳",
        "",
        "| id | answerable | claims | supported | unsupported | "
        "citations(valid/checked) | abstained | abstain_correct |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["per_question"]:
        lines.append(
            f"| {row['id']} | {row['answerable']} | {row['n_claims']} | "
            f"{row['n_supported']} | {row['n_unsupported']} | "
            f"{row['n_citations_valid']}/{row['n_citations_checked']} | "
            f"{row['abstained']} | {row['abstain_correct']} |"
        )
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "answer_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="RAG回答生成のhallucination評価ハーネス"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="ChromaDBの永続化ディレクトリ（外部パス。リポジトリ内へはコピーしないこと）",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--model", default=os.environ.get("RAG_ANSWER_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("RAG_JUDGE_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument("--stage", choices=["all", "generate", "judge"], default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help="Geminiリクエスト間のsleep秒数（レート制限対策）",
    )
    return parser.parse_args()


def main() -> None:
    """CLIエントリポイント。"""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    args = parse_args()
    dataset = load_dataset(args.dataset)
    print(f"{len(dataset)} questions loaded from {args.dataset}", flush=True)

    if args.stage in ("all", "generate"):
        run_generate_stage(
            dataset,
            args.db_path,
            top_k=args.top_k,
            model=args.model,
            sleep_seconds=args.sleep_seconds,
        )
    if args.stage == "generate":
        return

    answers_cached = load_cache(ANSWERS_CACHE)
    missing = [item.id for item in dataset if item.id not in answers_cached]
    if missing:
        raise SystemExit(
            f"回答未生成の質問があります: {missing}。"
            "先に --stage generate を実行してください"
        )
    answer_records = [answers_cached[item.id] for item in dataset]

    if args.stage in ("all", "judge"):
        run_judge_stage(
            answer_records,
            model=args.judge_model,
            sleep_seconds=args.sleep_seconds,
        )
    if args.stage == "judge":
        return

    judgements_cached = load_cache(JUDGEMENTS_CACHE)
    missing_judge = [
        r["id"] for r in answer_records if r["id"] not in judgements_cached
    ]
    if missing_judge:
        raise SystemExit(
            f"judge未実行の質問があります: {missing_judge}。"
            "先に --stage judge を実行してください"
        )
    judgements = [judgements_cached[r["id"]] for r in answer_records]

    result = aggregate(answer_records, judgements)
    result_path = write_result_json(args.out, result)
    summary_path = write_summary_md(args.out, result)
    print(f"wrote {result_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(
        f"groundedness_rate={result['groundedness_rate']} "
        f"citation_precision={result['citation_precision']} "
        f"trap_abstention_accuracy={result['trap_abstention_accuracy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
