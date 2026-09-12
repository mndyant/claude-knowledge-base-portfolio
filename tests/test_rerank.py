"""scripts/rerank.py の単体テスト。

実CrossEncoderモデル・ネットワークには依存しない。``rerank._get_model``を
モックへ差し替え、並べ替えロジック・top_kへの切り詰め・空入力の扱いだけを
検証する。
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts import rerank as rr


class FakeCrossEncoder:
    """``model.predict(pairs)``だけを模したテストダブル。

    ``query``に応じてスコアを返す関数を注入できる。既定では
    candidateの ``text`` に含まれる数字をスコアとして扱う（並べ替えの
    検証をしやすくするため）。
    """

    def __init__(self, score_fn) -> None:  # type: ignore[no-untyped-def]
        self._score_fn = score_fn
        self.predict_calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.predict_calls.append(pairs)
        return [self._score_fn(query, text) for query, text in pairs]


@pytest.fixture(autouse=True)
def _reset_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """各テストの前後でモジュールレベルのモデルキャッシュをリセットする。"""
    monkeypatch.setattr(rr, "_model", None)
    yield
    rr._model = None


def _install_fake_model(monkeypatch: pytest.MonkeyPatch, score_fn) -> FakeCrossEncoder:  # type: ignore[no-untyped-def]
    fake = FakeCrossEncoder(score_fn)
    monkeypatch.setattr(rr, "_get_model", lambda: fake)
    return fake


# --------------------------------------------------------------------------
# 並べ替えロジック
# --------------------------------------------------------------------------


def test_rerank_sorts_by_score_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    def score_fn(query: str, text: str) -> float:
        # textの末尾の数字をそのままスコアにする
        return float(text.split("-")[-1])

    _install_fake_model(monkeypatch, score_fn)

    candidates: list[dict[str, Any]] = [
        {"chunk_id": "a", "text": "low-score-1"},
        {"chunk_id": "b", "text": "high-score-9"},
        {"chunk_id": "c", "text": "mid-score-5"},
    ]
    result = rr.rerank("query", candidates, top_k=10)

    assert [item["chunk_id"] for item in result] == ["b", "c", "a"]
    assert result[0]["rerank_score"] == 9.0
    assert result[1]["rerank_score"] == 5.0
    assert result[2]["rerank_score"] == 1.0


def test_rerank_preserves_original_candidate_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model(monkeypatch, lambda query, text: 1.0)

    candidates = [{"chunk_id": "a", "text": "hello", "metadata": {"k": "v"}}]
    result = rr.rerank("query", candidates, top_k=10)

    assert result[0]["chunk_id"] == "a"
    assert result[0]["metadata"] == {"k": "v"}
    assert result[0]["rerank_score"] == 1.0


def test_rerank_does_not_mutate_input_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model(monkeypatch, lambda query, text: 1.0)

    candidates = [{"chunk_id": "a", "text": "hello"}]
    rr.rerank("query", candidates, top_k=10)

    assert "rerank_score" not in candidates[0]


# --------------------------------------------------------------------------
# top_kへの切り詰め
# --------------------------------------------------------------------------


def test_rerank_truncates_to_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    def score_fn(query: str, text: str) -> float:
        return float(text)

    _install_fake_model(monkeypatch, score_fn)

    candidates = [{"chunk_id": str(i), "text": str(i)} for i in range(5)]
    result = rr.rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert [item["chunk_id"] for item in result] == ["4", "3"]


def test_rerank_top_k_larger_than_candidates_returns_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model(monkeypatch, lambda query, text: 1.0)

    candidates = [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}]
    result = rr.rerank("query", candidates, top_k=10)

    assert len(result) == 2


# --------------------------------------------------------------------------
# 空入力
# --------------------------------------------------------------------------


def test_rerank_empty_candidates_returns_empty_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_get_model() -> Any:
        raise AssertionError("空入力なのにモデルをロードしてはいけない")

    monkeypatch.setattr(rr, "_get_model", _fail_get_model)

    result = rr.rerank("query", [], top_k=10)

    assert result == []


# --------------------------------------------------------------------------
# 遅延ロード・モジュールレベルキャッシュ
# --------------------------------------------------------------------------


def test_get_model_is_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class _StubCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            created.append(model_name)

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder", _StubCrossEncoder, raising=False
    )

    assert rr._model is None  # 呼び出し前はロードされていない

    first = rr._get_model()
    second = rr._get_model()

    assert first is second  # 2回目はキャッシュを再利用する
    assert created == [rr.MODEL_NAME]  # コンストラクタは1回しか呼ばれない
