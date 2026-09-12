"""公開サンプルのDB分離・再実行・トークン上限を検証する。"""
import os
from types import SimpleNamespace

import pytest

from scripts import embed, portfolio_demo


def test_demo_is_isolated_and_idempotent(monkeypatch, tmp_path):
    root = tmp_path / 'public-copy'
    monkeypatch.setattr(portfolio_demo, '__file__', str(root / 'scripts/portfolio_demo.py'))
    monkeypatch.setenv('CHROMA_DIR', str(tmp_path / 'private-db'))
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    # Track demo-owned environment mutations for restoration by pytest.
    for name in ('CHROMA_COLLECTION', 'EMBED_MODEL', 'EMBED_LOCAL_FILES_ONLY',
                 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'ANONYMIZED_TELEMETRY'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    assert portfolio_demo.prepare_demo() == 4
    assert portfolio_demo.prepare_demo() == 4
    assert not (tmp_path / 'private-db').exists()
    assert os.environ['CHROMA_COLLECTION'] == 'portfolio_sample_docs'
    assert 'GEMINI_API_KEY' not in os.environ


def test_demo_rejects_oversized_passage_before_db_write(monkeypatch, tmp_path):
    monkeypatch.setattr(portfolio_demo, '__file__', str(tmp_path / 'scripts/portfolio_demo.py'))
    for name in ('CHROMA_DIR', 'CHROMA_COLLECTION', 'EMBED_MODEL', 'EMBED_LOCAL_FILES_ONLY',
                 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'ANONYMIZED_TELEMETRY', 'GEMINI_API_KEY'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1] * 513))
    monkeypatch.setattr(embed, 'get_model', lambda *a, **k: model)
    with pytest.raises(ValueError, match='token limit'):
        portfolio_demo.prepare_demo()
    assert not (tmp_path / '.portfolio-demo').exists()
