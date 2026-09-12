"""digest.pyの差分検知・ページ分解・メッセージ整形を検証する。"""

from __future__ import annotations

import json

from scripts.digest import (
    DISCORD_CHUNK,
    DigestItem,
    build_messages,
    collect_docs_pages,
    diff_items,
    load_state,
    save_state,
)

# 2026年7月版llms-full.txtの形式A（通常docs: H1がURLの前）と
# 形式B（APIリファレンス: 見出しがURLの後）を混在させたサンプル
LLMS_FULL_JULY = """---
source_url: https://platform.claude.com/llms-full.txt
fetched_at: 2026-07-10T04:00:00+09:00
category: docs
---

# Anthropic Developer Documentation - Full Content

This file provides comprehensive documentation.

---

# Get started with Claude

**URL:** https://platform.claude.com/docs/en/get-started

Make your first API call to Claude.

## Prerequisites

* An Anthropic Console account

---

**URL:** https://platform.claude.com/docs/en/api/beta/environments/work/ack

---

## Acknowledge Work

Acknowledge a work item so it can be processed.
"""

# 2026年6月版形式（URL: プレーン表記）
LLMS_FULL_JUNE = """# Get started with Claude

URL: https://platform.claude.com/docs/en/get-started

Make your first API call to Claude.
"""


def _write_llms(tmp_path, content):
    path = tmp_path / "llms-full.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_collect_docs_pages_july_format(tmp_path):
    """7月版の形式A/B混在ファイルから両方のページを検出する。"""
    path = _write_llms(tmp_path, LLMS_FULL_JULY)
    items = collect_docs_pages(path)

    keys = {item.key for item in items}
    assert "docs:https://platform.claude.com/docs/en/get-started" in keys
    assert (
        "docs:https://platform.claude.com/docs/en/api/beta/environments/work/ack"
        in keys
    )

    by_key = {item.key: item for item in items}
    assert by_key["docs:https://platform.claude.com/docs/en/get-started"].title == (
        "Get started with Claude"
    )
    # 形式B: 見出しはURLの後ろにある
    assert (
        by_key[
            "docs:https://platform.claude.com/docs/en/api/beta/environments/work/ack"
        ].title
        == "Acknowledge Work"
    )


def test_collect_docs_pages_june_format(tmp_path):
    """6月版のプレーンURL表記でもページを検出する。"""
    path = _write_llms(tmp_path, LLMS_FULL_JUNE)
    items = collect_docs_pages(path)

    assert len(items) == 1
    assert items[0].url == "https://platform.claude.com/docs/en/get-started"
    assert items[0].title == "Get started with Claude"


def test_diff_items_detects_new_and_updated():
    """前回状態にないキーはnew、ハッシュが変わったキーはupdatedになる。"""
    items = [
        DigestItem("docs:a", "docs", "new", "A", "https://a", "", "hash-a"),
        DigestItem("docs:b", "docs", "new", "B", "https://b", "", "hash-b2"),
        DigestItem("docs:c", "docs", "new", "C", "https://c", "", "hash-c"),
    ]
    state = {"hashes": {"docs:b": "hash-b1", "docs:c": "hash-c"}}

    changed = diff_items(items, state)

    kinds = {item.key: item.kind for item in changed}
    assert kinds == {"docs:a": "new", "docs:b": "updated"}


def test_diff_items_orders_status_before_blog_and_docs():
    """要約枠の割り当て順: status → blog → release-notes → docs。"""
    items = [
        DigestItem("docs:a", "docs", "new", "A", "", "", "h1"),
        DigestItem("blog:b", "blog", "new", "B", "", "", "h2"),
        DigestItem("release-notes:c", "release-notes", "new", "C", "", "", "h3"),
        DigestItem("status:d", "status", "new", "D", "", "", "h4"),
    ]
    # 各ソースが既に前回状態に存在する（新規追加ソースではない）ことを示すダミーキー
    state = {
        "hashes": {
            "docs:existing": "x",
            "blog:existing": "x",
            "release-notes:existing": "x",
            "status:existing": "x",
        }
    }
    changed = diff_items(items, state)

    assert [item.source for item in changed] == ["status", "blog", "release-notes", "docs"]


def test_diff_items_baselines_brand_new_source_silently():
    """初めて登場したソースの全件は通知対象から除外される（過去分の一斉通知を防ぐ）。

    既存運用にstatusのような新カテゴリを追加した初回、過去の全インシデントが
    "新着"として一気に通知されるのを防ぐための挙動。
    """
    items = [
        DigestItem("docs:a", "docs", "new", "A", "", "", "hash-a"),
        DigestItem("status:x", "status", "new", "X", "", "", "hash-x"),
        DigestItem("status:y", "status", "new", "Y", "", "", "hash-y"),
    ]
    # 前回状態にdocsは存在するが、statusはまだ一度も記録されたことがない
    state = {"hashes": {"docs:a": "hash-a-old"}}

    changed = diff_items(items, state)

    assert [item.key for item in changed] == ["docs:a"]


def test_build_messages_respects_discord_limit():
    """長文でも各メッセージがDiscordの文字数制限内に収まる。"""
    items = [
        DigestItem(
            key=f"docs:{i}",
            source="docs",
            kind="new",
            title=f"Page {i} " + "x" * 80,
            url=f"https://example.com/{i}",
            excerpt="",
            digest="h",
            summary="要約 " * 50,
        )
        for i in range(20)
    ]
    messages = build_messages(items)

    assert len(messages) > 1
    assert all(len(message) <= DISCORD_CHUNK for message in messages)
    assert "ダイジェスト" in messages[0]


def test_state_roundtrip(tmp_path):
    """save_state → load_state で全キーとハッシュが保持される。"""
    path = tmp_path / "state.json"
    items = [
        DigestItem("docs:a", "docs", "new", "A", "https://a", "", "hash-a"),
        DigestItem("blog:b.md", "blog", "new", "B", "https://b", "", "hash-b"),
    ]
    save_state(items, path)
    state = load_state(path)

    assert state["hashes"] == {"docs:a": "hash-a", "blog:b.md": "hash-b"}
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
