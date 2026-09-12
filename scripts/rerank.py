"""多言語CrossEncoderによる検索結果の再ランキング（reranker）。

ChromaDBのベクトル検索（バイエンコーダ）はtop-N候補への粗い絞り込みには
強いが、クエリとチャンク本文を独立にベクトル化するため細かい語彙対応の
精度に限界がある。CrossEncoderはクエリとチャンク本文のペアを直接モデルへ
入力して関連度スコアを出すため精度は高いが、コーパス全件に対して使うには
遅すぎる。そのため vector検索で粗くtop-30程度に絞り込んだ候補だけを
CrossEncoderでrerankする2段構成にする（``scripts/eval_retrieval.py`` の
``vector_rerank`` バックエンド参照）。

モデル: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``
（mMARCO=多言語MS MARCOで学習済みのCrossEncoder。日本語・英語どちらの
クエリにも対応する）。CPUで動作する。初回呼び出し時にHugging Faceから
モデルをダウンロードする（数百MB。低速ディスク環境では数分かかることが
あるが異常ではない）。以降はプロセス内でモジュールレベルにキャッシュし
再ロードしない。

使い方:
    from scripts.rerank import rerank

    candidates = [{"text": "...", "chunk_id": "..."}, ...]
    reranked = rerank(query, candidates, top_k=10)
    # 戻り値: rerank_scoreの降順に並べ替えたtop_k件（各dictに
    # "rerank_score" キーが追加される。元のdictは変更しない）。
"""

from __future__ import annotations

from typing import Any

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model: Any = None


def _get_model() -> Any:
    """CrossEncoderを遅延ロードし、モジュールレベルにキャッシュして返す。

    呼び出しのたびにモデルを読み直さないよう、プロセス内で使い回す。
    """
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, device="cpu")
    return _model


def rerank(
    query: str, candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """CrossEncoderでcandidatesをrerankし、スコア降順でtop_k件を返す。

    各candidateは ``text``キー（rerank対象の本文）を持つ辞書である必要が
    ある。戻り値の各dictには ``rerank_score`` キーが追加される（元の
    dictそのものは変更せず、コピーへ追加する）。candidatesが空の場合は
    モデルをロードせずに空リストを返す。
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, str(candidate.get("text", ""))] for candidate in candidates]
    scores = model.predict(pairs)

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores, strict=False)
    ]
    scored.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
    return scored[:top_k]
