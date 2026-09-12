"""自作サンプルを実モデルで検索する、個人データ・外部LLM不要のデモ。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def prepare_demo() -> int:
    """専用DBへ固定サンプルをupsertし、登録件数を返す。"""
    # User-provided production settings are deliberately not used by this demo.
    root = Path(__file__).resolve().parents[1]
    os.environ['CHROMA_DIR'] = str(root / '.portfolio-demo' / 'chroma')
    os.environ['CHROMA_COLLECTION'] = 'portfolio_sample_docs'
    os.environ['EMBED_MODEL'] = 'intfloat/multilingual-e5-base'
    os.environ['EMBED_LOCAL_FILES_ONLY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    os.environ.pop('GEMINI_API_KEY', None)

    from scripts.embed import Chunk, embed_and_store, get_model, get_or_create_collection

    samples = [
        ('chunking', 'Token limits and structure', 'This sample RAG pipeline preserves Markdown headings, code fences and tables when splitting documents. It counts tokens with the same e5 tokenizer used for embedding. Each passage must fit within 512 tokens including the passage prefix and special tokens.'),
        ('resume', 'Resumable indexing', 'This sample indexing pipeline assigns a stable ID derived from document content. Re-running an indexing job upserts the same IDs instead of duplicating records. Unchanged chunks can be skipped. A changed document is indexed incrementally.'),
        ('search', 'Local semantic search', 'This sample search service uses multilingual-e5-base embeddings and a local ChromaDB collection. A Japanese query can retrieve an English passage by meaning. Search-only mode uses no external language model API. Optional answer generation is separate.'),
        ('testing', 'Isolated regression tests', 'This sample test suite mocks external HTTP calls and replaces the embedding model with a deterministic test double. Each test uses a separate temporary database. Production collections and notification destinations are never used in tests.'),
    ]
    tokenizer = get_model('intfloat/multilingual-e5-base').tokenizer
    chunks = []
    for key, heading, content in samples:
        tokens = len(tokenizer.encode('passage: ' + content, add_special_tokens=True))
        if tokens > 512:
            raise ValueError('Sample exceeds the embedding token limit')
        source = f'sample/{key}.md'
        chunks.append(Chunk(
            id=f'portfolio-sample-v1:{key}', content=content, source=source,
            metadata={'source': source, 'category': 'docs', 'heading_path': heading,
                      'token_estimate': tokens, 'summary_short': '自作の動作確認用サンプル。公式文書の転載ではありません。'},
        ))
    collection = get_or_create_collection()
    embed_and_store(chunks, collection=collection)
    return collection.count()


def main() -> None:
    """独立したサンプルDBを作成し、必要に応じてローカルAPIを起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true', help='検索画面・Swagger UIを起動')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    count = prepare_demo()
    print(f'Sample index ready: {count} documents. Real e5 embeddings; no external LLM.', flush=True)
    if args.serve:
        import uvicorn
        uvicorn.run('scripts.search_api:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
