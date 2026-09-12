"""pytest共有フィクスチャとテスト用ダブル。

外部HTTPはrequests-mockでモックし、ChromaDBはテスト用一時コレクション
（``test_anthropic_docs``）を使う。エンベディングモデルはネットワーク非依存の
決定的なテストダブル（``FakeModel`` / ``FakeTokenizer``）へ差し替える
（TEST.md §1.4「sentence-transformersモデルは軽量モデルに差し替え」の方針）。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest
from tenacity import wait_fixed

TEST_MODEL = "paraphrase-MiniLM-L3-v2"
TEST_COLLECTION = "test_anthropic_docs"
EMBED_DIM = 64


class FakeTokenizer:
    """空白区切りで決定的にトークン化するテスト用tokenizer。

    ``encode`` / ``decode`` がラウンドトリップ可能なため、チャンク分割の
    トークン境界・オーバーラップ検証に利用できる。
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            new_id = len(self._vocab) + 1
            self._vocab[token] = new_id
            self._inv[new_id] = token
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(t) for t in re.findall(r"\S+", text)]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self._inv.get(i, "") for i in ids).strip()


class FakeModel:
    """ハッシュベースの決定的なベクトルを返すテスト用モデル。

    同一テキストは常に同一ベクトルになるため、検索の関連度テストが安定する。
    """

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    @staticmethod
    def _bucket(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % EMBED_DIM

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs):  # type: ignore[no-untyped-def]
        vectors = []
        for text in texts:
            vec = np.zeros(EMBED_DIM, dtype="float32")
            for token in re.findall(r"\S+", text.lower()):
                vec[self._bucket(token)] += 1.0
            norm = np.linalg.norm(vec)
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


_FAKE_MODEL = FakeModel()


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
    """各テストで環境を分離する。

    - 作業ディレクトリをtmpへ移し、knowledge/・chroma-data/を汚染しない
    - エンベディングモデルをFakeModelへ差し替える（ネットワーク非依存）
    - tenacityのリトライ待機を0にしてテストを高速化する
    """
    monkeypatch.setenv("EMBED_MODEL", TEST_MODEL)
    monkeypatch.setenv("CHROMA_COLLECTION", TEST_COLLECTION)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    # エンベディングモデルをテストダブルへ差し替え
    monkeypatch.setattr("scripts.embed.get_model", lambda *a, **k: _FAKE_MODEL)

    # リトライ待機を0にする
    import scripts.fetch as fetch

    fetch.fetch_with_retry.retry.wait = wait_fixed(0)
    yield


@pytest.fixture
def mock_llms_txt() -> str:
    """llms-full.txtのモックレスポンス本文を返す。"""
    return (
        "# Claude Code\n\n"
        "Claude Codeはターミナルで動作するエージェント型のコーディングツールです。\n\n"
        "## サブエージェント\n\n"
        "Task toolを使ってサブエージェントを起動し、並列にタスクを処理できます。\n"
    )


@pytest.fixture
def sample_chunks():
    """テスト用チャンク（3件）を返す。"""
    from scripts.embed import Chunk, _make_chunk_id

    chunks = []
    for i in range(3):
        source = f"knowledge/docs/doc{i}.md"
        chunks.append(
            Chunk(
                id=_make_chunk_id(source, 0),
                content=f"これはサンプルチャンク {i} の本文です。Claude Code について説明します。",
                source=source,
                category="docs",
                section=f"section-{i}",
                chunk_index=0,
                metadata={
                    "source": source,
                    "category": "docs",
                    "section": f"section-{i}",
                    "chunk_index": 0,
                },
            )
        )
    return chunks


@pytest.fixture
def test_chroma_client():
    """テスト用ChromaDBクライアントを生成し、テスト後にコレクションを削除する。"""
    from scripts.embed import get_chroma_client

    client = get_chroma_client()
    yield client
    try:
        client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


@pytest.fixture
def api_client():
    """FastAPIテストクライアントを返す。"""
    from fastapi.testclient import TestClient

    from scripts.search_api import app

    return TestClient(app)
