"""scripts/eval_answers.py の単体テスト。

Gemini呼び出し（生成・judge）は全てモックし、実API・実DB・ネットワークには
依存しない。中間キャッシュの再開可能性・judgeのJSON解析リトライ・指標集計
（groundedness率・citation precision・abstention accuracy）を検証する。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts import eval_answers as ea

# --------------------------------------------------------------------------
# データセット読み込み
# --------------------------------------------------------------------------


def test_load_dataset_parses_answerable_and_trap(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "q1", "query": "質問1", "answerable": True, "lang": "ja"}
                ),
                json.dumps(
                    {"id": "t1", "query": "架空機能", "answerable": False, "lang": "ja"}
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    items = ea.load_dataset(path)

    assert len(items) == 2
    assert items[0].answerable is True
    assert items[1].answerable is False


# --------------------------------------------------------------------------
# キャッシュ（質問単位・再開可能）
# --------------------------------------------------------------------------


def test_append_and_load_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.jsonl"
    ea.append_cache(path, {"id": "a", "value": 1})
    ea.append_cache(path, {"id": "b", "value": 2})

    cached = ea.load_cache(path)

    assert cached["a"]["value"] == 1
    assert cached["b"]["value"] == 2


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert ea.load_cache(tmp_path / "missing.jsonl") == {}


# --------------------------------------------------------------------------
# ステージ1: 回答生成（再開可能・レート制限sleep）
# --------------------------------------------------------------------------


def test_run_generate_stage_resumable(monkeypatch, tmp_path):
    dataset = [
        ea.QAItem(id="q1", query="質問1", answerable=True),
        ea.QAItem(id="q2", query="質問2", answerable=False),
    ]
    out_path = tmp_path / "answers.jsonl"
    calls: list[str] = []

    def _fake_answer_query(query, db_path, top_k, model):  # noqa: ANN001
        calls.append(query)
        return {"answer": f"回答: {query}", "citations": [], "contexts": []}

    monkeypatch.setattr(ea, "answer_query", _fake_answer_query)

    records = ea.run_generate_stage(
        dataset, db_path="dummy", out_path=out_path, sleep_seconds=0
    )

    assert len(records) == 2
    assert calls == ["質問1", "質問2"]

    # 2回目の実行はキャッシュ済みなのでanswer_queryを呼ばない
    calls.clear()
    records_again = ea.run_generate_stage(
        dataset, db_path="dummy", out_path=out_path, sleep_seconds=0
    )
    assert calls == []
    assert len(records_again) == 2


def test_run_generate_stage_partial_resume(monkeypatch, tmp_path):
    dataset = [
        ea.QAItem(id="q1", query="質問1", answerable=True),
        ea.QAItem(id="q2", query="質問2", answerable=True),
    ]
    out_path = tmp_path / "answers.jsonl"
    ea.append_cache(
        out_path,
        {
            "id": "q1",
            "query": "質問1",
            "answerable": True,
            "lang": "",
            "category": "",
            "answer": "既存の回答",
            "citations": [],
            "contexts": [],
        },
    )

    calls: list[str] = []

    def _fake_answer_query(query, db_path, top_k, model):  # noqa: ANN001
        calls.append(query)
        return {"answer": "新規回答", "citations": [], "contexts": []}

    monkeypatch.setattr(ea, "answer_query", _fake_answer_query)

    records = ea.run_generate_stage(
        dataset, db_path="dummy", out_path=out_path, sleep_seconds=0
    )

    assert calls == ["質問2"]  # q1はキャッシュ済みなので呼ばれない
    assert records[0]["answer"] == "既存の回答"
    assert records[1]["answer"] == "新規回答"


# --------------------------------------------------------------------------
# judge応答のJSONパース・リトライ
# --------------------------------------------------------------------------


def test_parse_judge_response_valid_json():
    text = json.dumps(
        {
            "claims": [
                {
                    "text": "x",
                    "verdict": "supported",
                    "cited_ns": [1],
                    "citation_valid": True,
                }
            ],
            "abstained": False,
        }
    )
    parsed = ea.parse_judge_response(text)
    assert parsed["abstained"] is False
    assert parsed["claims"][0]["verdict"] == "supported"


def test_parse_judge_response_strips_code_fence():
    text = "```json\n" + json.dumps({"claims": [], "abstained": True}) + "\n```"
    parsed = ea.parse_judge_response(text)
    assert parsed["abstained"] is True


def test_parse_judge_response_missing_keys_raises_value_error():
    with pytest.raises(ValueError):
        ea.parse_judge_response(json.dumps({"foo": "bar"}))


def test_judge_answer_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    responses = iter(
        ["これはJSONではありません", json.dumps({"claims": [], "abstained": False})]
    )

    def _fake_call_gemini(client, model, prompt):  # noqa: ANN001
        return next(responses)

    monkeypatch.setattr(ea, "call_gemini", _fake_call_gemini)

    result = ea.judge_answer(object(), "model", "質問", contexts=[], answer="回答")

    assert result == {"claims": [], "abstained": False}


def test_judge_answer_raises_after_retry_still_invalid(monkeypatch):
    monkeypatch.setattr(ea, "call_gemini", lambda client, model, prompt: "invalid json")

    with pytest.raises(json.JSONDecodeError):
        ea.judge_answer(object(), "model", "質問", contexts=[], answer="回答")


def test_run_judge_stage_resumable(monkeypatch, tmp_path):
    answer_records = [
        {
            "id": "q1",
            "query": "質問1",
            "answerable": True,
            "answer": "回答1",
            "contexts": [],
        },
        {
            "id": "q2",
            "query": "質問2",
            "answerable": False,
            "answer": "回答2",
            "contexts": [],
        },
    ]
    out_path = tmp_path / "judgements.jsonl"
    calls: list[str] = []

    def _fake_judge_answer(client, model, query, contexts, answer):  # noqa: ANN001
        calls.append(query)
        return {"claims": [], "abstained": False}

    monkeypatch.setattr(ea, "judge_answer", _fake_judge_answer)

    results = ea.run_judge_stage(
        answer_records, out_path=out_path, client=object(), sleep_seconds=0
    )

    assert calls == ["質問1", "質問2"]
    assert len(results) == 2

    calls.clear()
    ea.run_judge_stage(
        answer_records, out_path=out_path, client=object(), sleep_seconds=0
    )
    assert calls == []  # 既にキャッシュ済みなので再実行しない


def test_run_judge_stage_records_error_and_continues(monkeypatch, tmp_path):
    answer_records = [
        {
            "id": "q1",
            "query": "質問1",
            "answerable": True,
            "answer": "回答1",
            "contexts": [],
        },
    ]
    out_path = tmp_path / "judgements.jsonl"

    def _fake_judge_answer(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("judge API失敗")

    monkeypatch.setattr(ea, "judge_answer", _fake_judge_answer)

    results = ea.run_judge_stage(
        answer_records, out_path=out_path, client=object(), sleep_seconds=0
    )

    assert results[0]["error"] == "judge API失敗"
    assert results[0]["claims"] == []


# --------------------------------------------------------------------------
# 指標集計
# --------------------------------------------------------------------------


def _answer_record(qid: str, answerable: bool, query: str = "q") -> dict[str, Any]:
    return {"id": qid, "query": query, "answerable": answerable}


def test_aggregate_groundedness_rate():
    answer_records = [_answer_record("q1", True), _answer_record("q2", True)]
    judgements = [
        {
            "id": "q1",
            "claims": [
                {
                    "text": "a",
                    "verdict": "supported",
                    "cited_ns": [1],
                    "citation_valid": True,
                },
                {
                    "text": "b",
                    "verdict": "unsupported",
                    "cited_ns": [],
                    "citation_valid": None,
                },
            ],
            "abstained": False,
        },
        {
            "id": "q2",
            "claims": [
                {
                    "text": "c",
                    "verdict": "supported",
                    "cited_ns": [],
                    "citation_valid": None,
                },
            ],
            "abstained": False,
        },
    ]

    result = ea.aggregate(answer_records, judgements)

    # supported=2, unsupported=1 -> 2/3
    assert result["groundedness_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["total_claims"] == 3


def test_aggregate_citation_precision():
    answer_records = [_answer_record("q1", True)]
    judgements = [
        {
            "id": "q1",
            "claims": [
                {
                    "text": "a",
                    "verdict": "supported",
                    "cited_ns": [1],
                    "citation_valid": True,
                },
                {
                    "text": "b",
                    "verdict": "supported",
                    "cited_ns": [2],
                    "citation_valid": False,
                },
                {
                    "text": "c",
                    "verdict": "supported",
                    "cited_ns": [],
                    "citation_valid": None,
                },
            ],
            "abstained": False,
        }
    ]

    result = ea.aggregate(answer_records, judgements)

    # 引用ありのclaimは2件、うちvalidは1件
    assert result["citation_precision"] == pytest.approx(0.5)
    assert result["total_citations_checked"] == 2


def test_aggregate_trap_abstention_accuracy():
    answer_records = [
        _answer_record("t1", False),
        _answer_record("t2", False),
        _answer_record("t3", False),
    ]
    judgements = [
        {"id": "t1", "claims": [], "abstained": True},
        {"id": "t2", "claims": [], "abstained": True},
        {"id": "t3", "claims": [], "abstained": False},  # 誤って答えてしまった
    ]

    result = ea.aggregate(answer_records, judgements)

    assert result["n_trap"] == 3
    assert result["trap_abstention_accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_aggregate_answerable_wrongly_abstained():
    answer_records = [_answer_record("q1", True), _answer_record("q2", True)]
    judgements = [
        {"id": "q1", "claims": [], "abstained": True},  # 答えられるのに拒否した
        {"id": "q2", "claims": [], "abstained": False},
    ]

    result = ea.aggregate(answer_records, judgements)

    assert result["answerable_wrongly_abstained"] == 1
    per_q1 = next(r for r in result["per_question"] if r["id"] == "q1")
    assert per_q1["abstain_correct"] is False


def test_aggregate_handles_missing_judgement_gracefully():
    answer_records = [_answer_record("q1", True)]
    result = ea.aggregate(answer_records, judgements=[])

    assert result["total_claims"] == 0
    assert result["groundedness_rate"] is None
    assert result["citation_precision"] is None


def test_aggregate_counts_judge_errors():
    answer_records = [_answer_record("q1", True)]
    judgements = [{"id": "q1", "claims": [], "abstained": False, "error": "judge失敗"}]

    result = ea.aggregate(answer_records, judgements)

    assert result["judge_errors"] == 1


# --------------------------------------------------------------------------
# サマリー出力
# --------------------------------------------------------------------------


def test_write_summary_md_contains_key_metrics(tmp_path):
    answer_records = [_answer_record("q1", True), _answer_record("t1", False)]
    judgements = [
        {
            "id": "q1",
            "claims": [
                {
                    "text": "a",
                    "verdict": "supported",
                    "cited_ns": [1],
                    "citation_valid": True,
                }
            ],
            "abstained": False,
        },
        {"id": "t1", "claims": [], "abstained": True},
    ]
    result = ea.aggregate(answer_records, judgements)

    path = ea.write_summary_md(tmp_path, result)
    text = path.read_text(encoding="utf-8")

    assert "groundedness率" in text
    assert "citation precision" in text
    assert "trap abstention accuracy" in text
    assert "q1" in text and "t1" in text


def test_write_result_json_roundtrip(tmp_path):
    result = {"groundedness_rate": 1.0, "per_question": []}
    path = ea.write_result_json(tmp_path, result)

    assert json.loads(path.read_text(encoding="utf-8")) == result
