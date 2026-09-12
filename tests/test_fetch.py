"""fetch.py の単体テスト（TC-F-001〜006）。"""

from __future__ import annotations

from pathlib import Path

import requests

from scripts import fetch


def test_fetch_llms_full_txt_success(requests_mock, mock_llms_txt):
    """TC-F-001: llms-full.txt 正常取得でファイルが生成される。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, text=mock_llms_txt)

    fetch.fetch_llms_full_txt()

    path = Path("knowledge/docs/llms-full.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()


def test_fetch_llms_full_txt_timeout_retry(requests_mock, mock_llms_txt):
    """TC-F-002: タイムアウト2回→3回目成功でファイル生成・呼び出し3回。"""
    requests_mock.get(
        fetch.LLMS_FULL_TXT_URL,
        [
            {"exc": requests.exceptions.Timeout},
            {"exc": requests.exceptions.Timeout},
            {"text": mock_llms_txt},
        ],
    )

    fetch.fetch_llms_full_txt()

    assert Path("knowledge/docs/llms-full.md").exists()
    assert requests_mock.call_count == 3


def test_fetch_llms_full_txt_http_500_skip(requests_mock):
    """TC-F-003: HTTP 500 は例外を送出せずスキップ・ファイル非生成。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, status_code=500)

    fetch.fetch_llms_full_txt()  # 例外が出ないこと

    assert not Path("knowledge/docs/llms-full.md").exists()


def test_is_updated_returns_false_when_same():
    """TC-F-004: 既存内容と同一なら False。"""
    body = "# タイトル\n\n本文です。"
    path = Path("knowledge/docs/sample.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        fetch.add_frontmatter(body, url="https://example.com", category="docs"),
        encoding="utf-8",
    )

    assert fetch.is_updated(path, body) is False


def test_is_updated_returns_true_when_different():
    """TC-F-005: 既存内容と異なれば True。"""
    path = Path("knowledge/docs/sample.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        fetch.add_frontmatter("古い本文", url="https://example.com", category="docs"),
        encoding="utf-8",
    )

    assert fetch.is_updated(path, "新しい本文") is True


def test_fetch_blog_posts_creates_three_files(requests_mock):
    """TC-F-006: RSS 3件からブログ3ファイルがFrontmatter付きで生成される。"""
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <title>Anthropic News</title>
      <item><title>記事A</title><link>https://www.anthropic.com/news/a</link>
        <description>記事Aの概要です。</description></item>
      <item><title>記事B</title><link>https://www.anthropic.com/news/b</link>
        <description>記事Bの概要です。</description></item>
      <item><title>記事C</title><link>https://www.anthropic.com/news/c</link>
        <description>記事Cの概要です。</description></item>
    </channel></rss>"""
    requests_mock.get(fetch.BLOG_RSS_URL, text=rss)

    fetch.fetch_blog_posts()

    blog_files = list(Path("knowledge/blog").glob("*.md"))
    assert len(blog_files) == 3
    for f in blog_files:
        assert f.read_text(encoding="utf-8").startswith("---")


# --- 補助テスト（カバレッジ向上・各ソース関数とmainの正常系/異常系） ---


def test_fetch_release_notes_success(requests_mock):
    """リリースノートJSONから release-notes ファイルが生成される。"""
    requests_mock.get(
        fetch.RELEASE_NOTES_URL,
        json=[
            {
                "tag_name": "v1.2.0",
                "name": "Release 1.2.0",
                "body": "変更点の一覧",
                "html_url": "https://github.com/anthropics/claude-code/releases/v1.2.0",
            }
        ],
    )

    fetch.fetch_release_notes()

    files = list(Path("knowledge/release-notes").glob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").startswith("---")


def test_fetch_cookbooks_success(requests_mock):
    """Cookbooks一覧から index.md が生成される。"""
    requests_mock.get(
        fetch.COOKBOOKS_URL,
        json=[
            {"name": "rag.ipynb", "type": "file"},
            {"name": "tools", "type": "dir"},
            {"name": "summarization.ipynb", "type": "file"},
        ],
    )

    fetch.fetch_cookbooks()

    index = Path("knowledge/cookbooks/index.md")
    assert index.exists()
    text = index.read_text(encoding="utf-8")
    assert "rag.ipynb" in text
    assert "tools" not in text  # ディレクトリは含めない


def test_fetch_release_notes_http_error_skips(requests_mock):
    """リリースノートが500でも例外を送出せずスキップする。"""
    requests_mock.get(fetch.RELEASE_NOTES_URL, status_code=500)

    fetch.fetch_release_notes()  # 例外が出ないこと

    assert not Path("knowledge/release-notes").exists()


def test_main_runs_all_sources(requests_mock, mock_llms_txt):
    """main() が全ソースを順に実行し、各ファイルが生成される。"""
    requests_mock.get(fetch.LLMS_FULL_TXT_URL, text=mock_llms_txt)
    requests_mock.get(
        fetch.BLOG_RSS_URL,
        text=(
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item><title>記事</title><link>https://www.anthropic.com/news/x</link>"
            "<description>概要</description></item></channel></rss>"
        ),
    )
    requests_mock.get(
        fetch.RELEASE_NOTES_URL,
        json=[{"tag_name": "v1", "name": "v1", "body": "notes", "html_url": "u"}],
    )
    requests_mock.get(fetch.COOKBOOKS_URL, json=[{"name": "a.ipynb", "type": "file"}])

    fetch.main()

    assert Path("knowledge/docs/llms-full.md").exists()
    assert list(Path("knowledge/blog").glob("*.md"))
    assert list(Path("knowledge/release-notes").glob("*.md"))
    assert Path("knowledge/cookbooks/index.md").exists()
