"""Markdownファイルのチャンク分割・ベクトル化・ChromaDB登録を行うモジュール。

``knowledge/`` 配下のMarkdownを読み込み、512トークン（overlap 50）でチャンク分割し、
sentence-transformers でベクトル化して ChromaDB（``./chroma-data/``）にupsertする。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = "knowledge"
DEFAULT_CHUNKS_JSONL = "data/processed/chunks.jsonl"

# 環境変数で差し替え可能な設定
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_COLLECTION = "anthropic_docs"
DEFAULT_CHROMA_DIR = "./chroma-data"
DEFAULT_LOCAL_FILES_ONLY = "1"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 128
DEFAULT_ENCODE_BATCH_SIZE = 8


@dataclass
class Document:
    """1つのMarkdownファイルを表すデータ構造。"""

    content: str
    source: str
    category: str = ""
    section: str = ""


@dataclass
class Chunk:
    """チャンク分割後の1断片を表すデータ構造。"""

    id: str
    content: str
    source: str
    category: str = ""
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    embed_text: str = ""
    """embedding計算に使うテキスト（空文字なら``content``をそのまま使う）。

    split_llms_full.pyが列挙定数の羅列など埋め込みに不要な定型部分を
    間引いた場合に別値が入る。表示用の``content``はこの影響を受けない。
    """

    def text_for_embedding(self) -> str:
        return self.embed_text or self.content


def get_model_name() -> str:
    """使用するエンベディングモデル名を返す（``EMBED_MODEL`` で差し替え可能）。"""
    return os.environ.get("EMBED_MODEL", DEFAULT_MODEL)


def get_chroma_dir() -> str:
    """ChromaDBの永続化パスを返す（``CHROMA_DIR`` で差し替え可能）。"""
    return os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)


def get_collection_name() -> str:
    """ChromaDBコレクション名を返す（``CHROMA_COLLECTION`` で差し替え可能）。"""
    return os.environ.get("CHROMA_COLLECTION", DEFAULT_COLLECTION)


def _env_flag(name: str, default: str = "0") -> bool:
    """環境変数の真偽値を読む。"""
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def get_local_files_only() -> bool:
    """モデルロード時にローカルキャッシュだけを使うか返す。"""
    return _env_flag("EMBED_LOCAL_FILES_ONLY", DEFAULT_LOCAL_FILES_ONLY)


@lru_cache(maxsize=2)
def get_model(model_name: str | None = None):  # type: ignore[no-untyped-def]
    """SentenceTransformerモデルをロードする（同一名はキャッシュ）。

    Raises:
        Exception: モデルのロードに失敗した場合は送出してプロセスを止める。
    """
    name = model_name or get_model_name()
    logger.info("モデルをロード中: %s", name)
    local_files_only = get_local_files_only()
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    logger.info("local_files_only=%s", local_files_only)
    return SentenceTransformer(
        name,
        local_files_only=local_files_only,
        tokenizer_kwargs={"fix_mistral_regex": False},
    )


def _is_e5(model_name: str) -> bool:
    """e5系モデルか判定する（passage/queryプレフィックスの要否判定に使う）。"""
    return "e5" in model_name.lower()


def get_encode_batch_size() -> int:
    """モデル推論のバッチサイズを返す。CPUでは大きすぎる値を避ける。"""
    raw = os.environ.get("EMBED_ENCODE_BATCH_SIZE", str(DEFAULT_ENCODE_BATCH_SIZE))
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "invalid EMBED_ENCODE_BATCH_SIZE=%r; using %d",
            raw,
            DEFAULT_ENCODE_BATCH_SIZE,
        )
        return DEFAULT_ENCODE_BATCH_SIZE


def embed_texts(
    texts: list[str],
    is_query: bool = False,
    model_name: str | None = None,
    encode_batch_size: int | None = None,
):  # type: ignore[no-untyped-def]
    """テキスト群をベクトル化する。e5系は ``passage:``/``query:`` を付与する。

    Args:
        texts: ベクトル化対象のテキスト群。
        is_query: 検索クエリの場合 ``True``（``query:`` を付与）。
        model_name: 使用するモデル名（省略時は環境変数）。

    Returns:
        ベクトルのリスト（list[list[float]]）。
    """
    name = model_name or get_model_name()
    model = get_model(name)
    if _is_e5(name):
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=max(1, encode_batch_size or get_encode_batch_size()),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """YAML Frontmatterを簡易パースし、(メタデータ, 本文) を返す。"""
    metadata: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()
            body = parts[2].lstrip("\n")
    return metadata, body


def load_documents(knowledge_dir: str = KNOWLEDGE_DIR) -> list[Document]:
    """``knowledge_dir`` 配下の全Markdownを読み込みDocumentのリストを返す。

    Args:
        knowledge_dir: 走査対象ディレクトリ。

    Returns:
        Documentオブジェクトのリスト（本文が空のファイルは除外）。
    """
    documents: list[Document] = []
    root = Path(knowledge_dir)
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(text)
        if not body.strip():
            continue
        documents.append(
            Document(
                content=body,
                source=str(path).replace("\\", "/"),
                category=metadata.get("category", ""),
                section=metadata.get("section", ""),
            )
        )
    return documents


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """source と chunk_index から決定的なチャンクIDを生成する。"""
    return hashlib.md5(f"{source}::{chunk_index}".encode()).hexdigest()


def _metadata_value(value: object) -> str | int | float | bool:
    """ChromaDBに保存できるmetadata値へ正規化する。"""
    if isinstance(value, str | int | float | bool):
        return value
    if value is None:
        return ""
    return str(value)


def _normalize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    """JSONL由来のmetadataをChromaDB互換の平坦なdictへ整える。"""
    return {str(key): _metadata_value(value) for key, value in metadata.items()}


def load_chunks_jsonl(input_path: str | Path = DEFAULT_CHUNKS_JSONL) -> list[Chunk]:
    """split_llms_full.pyが生成したJSONLをChunkリストとして読み込む。

    JSONLのmetadataはそのまま残しつつ、既存の検索APIが参照する
    ``source`` / ``section`` / ``category`` も補う。
    """
    path = Path(input_path)
    chunks: list[Chunk] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content = str(record.get("content", "")).strip()
            if not content:
                continue

            raw_metadata = record.get("metadata") or {}
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"{path}:{line_number} metadata must be an object")

            metadata = _normalize_metadata(raw_metadata)
            source = str(metadata.get("source_file") or metadata.get("source") or path)
            heading_path = str(metadata.get("heading_path") or "")
            raw_chunk_index = metadata.get("chunk_index")
            chunk_index = (
                int(raw_chunk_index)
                if raw_chunk_index not in (None, "")
                else len(chunks)
            )

            # search_api.pyとの互換性を保つため、検索用の別名を足す。
            metadata.setdefault("source", source)
            metadata.setdefault("section", heading_path)
            metadata.setdefault("category", "docs")

            chunk_id = str(record.get("id") or _make_chunk_id(source, chunk_index))
            embed_text = str(record.get("embed_text") or "").strip()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=content,
                    source=source,
                    category=str(metadata.get("category") or ""),
                    section=heading_path,
                    chunk_index=chunk_index,
                    metadata=metadata,
                    embed_text=embed_text,
                )
            )
    return chunks


def split_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    model_name: str | None = None,
) -> list[Chunk]:
    """Documentをトークン単位（overlapあり）でチャンク分割する。

    モデルのtokenizerでトークン化し、``chunk_size`` トークンの窓を
    ``chunk_size - overlap`` トークンずつスライドさせて分割する。
    これにより各チャンクは ``chunk_size`` トークン以内・隣接チャンク間に
    ``overlap`` トークンの重複を持つ。

    Args:
        documents: 分割対象のDocumentリスト。
        chunk_size: 1チャンクの最大トークン数。
        overlap: 隣接チャンク間の重複トークン数。
        model_name: tokenizer取得に使うモデル名（省略時は環境変数）。

    Returns:
        Chunkのリスト。
    """
    name = model_name or get_model_name()
    tokenizer = get_model(name).tokenizer
    stride = max(chunk_size - overlap, 1)

    chunks: list[Chunk] = []
    for doc in documents:
        token_ids = tokenizer.encode(doc.content, add_special_tokens=False)
        if not token_ids:
            continue
        index = 0
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + chunk_size]
            content = tokenizer.decode(window, skip_special_tokens=True).strip()
            if content:
                chunks.append(
                    Chunk(
                        id=_make_chunk_id(doc.source, index),
                        content=content,
                        source=doc.source,
                        category=doc.category,
                        section=doc.section,
                        chunk_index=index,
                        metadata={
                            "source": doc.source,
                            "category": doc.category,
                            "section": doc.section,
                            "chunk_index": index,
                        },
                    )
                )
                index += 1
            if start + chunk_size >= len(token_ids):
                break
            start += stride
    return chunks


def get_chroma_client(path: str | None = None):  # type: ignore[no-untyped-def]
    """ChromaDBのPersistentClientを生成する。

    Raises:
        Exception: 接続に失敗した場合は送出してプロセスを止める。
    """
    import chromadb

    return chromadb.PersistentClient(path=path or get_chroma_dir())


def get_or_create_collection(client=None, name: str | None = None):  # type: ignore[no-untyped-def]
    """コレクションを取得（なければ作成）する。距離関数はcosine。"""
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=name or get_collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(
    chunks: list[Chunk],
    collection=None,
    model_name: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    encode_batch_size: int | None = None,
) -> None:  # type: ignore[no-untyped-def]
    """チャンクをベクトル化しChromaDBにupsertする。

    Args:
        chunks: 登録対象のChunkリスト。
        collection: 登録先コレクション（省略時は既定コレクション）。
        model_name: 使用するモデル名（省略時は環境変数）。
    """
    if not chunks:
        logger.info("登録対象のチャンクがありません")
        return

    collection = collection if collection is not None else get_or_create_collection()
    batch_size = max(1, batch_size)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embed_texts(
            [c.text_for_embedding() for c in batch],
            is_query=False,
            model_name=model_name,
            encode_batch_size=encode_batch_size,
        )
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.content for c in batch],
            metadatas=[_normalize_metadata(c.metadata) for c in batch],
            embeddings=embeddings,
        )
    logger.info("%d 件のチャンクをupsertしました", len(chunks))


def get_collection_stats(collection=None) -> dict:  # type: ignore[no-untyped-def]
    """コレクションの統計情報を返す。"""
    collection = collection if collection is not None else get_or_create_collection()
    return {
        "collection": collection.name,
        "total_documents": collection.count(),
    }


def _legacy_markdown_main() -> None:
    """knowledge/配下を読み込み、チャンク化してChromaDBへ登録する。"""
    documents = load_documents()
    logger.info("%d 件のドキュメントを読み込みました", len(documents))
    chunks = split_chunks(documents)
    logger.info("%d 件のチャンクに分割しました", len(chunks))
    embed_and_store(chunks)
    logger.info("統計: %s", get_collection_stats())


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="Markdownまたは分割済みJSONLをembeddingしてChromaDBへ登録する"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="split_llms_full.pyが生成したchunks.jsonl",
    )
    parser.add_argument(
        "--from-markdown",
        action="store_true",
        help="JSONLではなくknowledge/配下のMarkdownを従来方式で分割する",
    )
    parser.add_argument("--knowledge-dir", default=KNOWLEDGE_DIR)
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="モデル推論バッチサイズ（既定: CPU向け8、EMBED_ENCODE_BATCH_SIZEでも指定可）",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="登録済みIDも含めて全チャンクを再embeddingする",
    )
    return parser.parse_args()


def main() -> None:
    """分割済みJSONLを優先してChromaDBへ登録する。"""
    args = parse_args()
    input_path = args.input or Path(DEFAULT_CHUNKS_JSONL)

    uses_jsonl = False
    if args.from_markdown:
        documents = load_documents(args.knowledge_dir)
        logger.info("%d markdown documents loaded", len(documents))
        chunks = split_chunks(documents, model_name=args.model)
    elif input_path.exists():
        chunks = load_chunks_jsonl(input_path)
        uses_jsonl = True
        logger.info("%d jsonl chunks loaded from %s", len(chunks), input_path)
    elif args.input:
        raise SystemExit(f"input file not found: {input_path}")
    else:
        documents = load_documents(args.knowledge_dir)
        logger.info(
            "%s not found, fallback to %d markdown documents",
            input_path,
            len(documents),
        )
        chunks = split_chunks(documents, model_name=args.model)

    logger.info("%d chunks ready for embedding", len(chunks))
    collection = get_or_create_collection(name=args.collection)
    if uses_jsonl and not args.force:
        existing_ids = set(collection.get(include=[])["ids"])
        before = len(chunks)
        chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
        logger.info(
            "%d registered chunks skipped; %d chunks need embedding",
            before - len(chunks),
            len(chunks),
        )
    embed_and_store(
        chunks,
        collection=collection,
        model_name=args.model,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
    )
    logger.info("stats: %s", get_collection_stats(collection))


if __name__ == "__main__":
    main()
