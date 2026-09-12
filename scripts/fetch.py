"""Anthropic公式情報の取得・Markdown整形・保存を行うモジュール。

各ソース（llms-full.txt / 公式ブログ / リリースノート / Cookbooks）から情報を取得し、
Frontmatter付きMarkdownとして ``knowledge/{category}/`` 配下に保存する。
差分がない場合はスキップし、HTTPエラーは握りつぶさずログ出力して他ソースの処理を継続する。
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# 保存先のルートディレクトリ
KNOWLEDGE_DIR = Path("knowledge")

# 情報ソース定義
LLMS_FULL_TXT_URL = "https://platform.claude.com/llms-full.txt"
# 公式RSS（anthropic.com/news/rss.xml）は2026年に廃止され404。
# コミュニティ維持のミラーを使う（tech-news-botも同じソースで毎日稼働実績あり）
BLOG_RSS_URL = "https://tim-hilde.github.io/anthropic-rss/rss.xml"
RELEASE_NOTES_URL = "https://api.github.com/repos/anthropics/claude-code/releases"
COOKBOOKS_URL = "https://api.github.com/repos/anthropics/claude-cookbooks/contents"
# インシデント・メンテナンス・モデル別の稼働状況（旧status.anthropic.comは
# status.claude.comへ302リダイレクト済み）。docs/blogとは別系統の一次情報
STATUS_RSS_URL = "https://status.claude.com/history.rss"

# HTTPタイムアウト（秒）
TIMEOUT = 30


def is_updated(filepath: Path, new_content: str) -> bool:
    """前回保存内容とのMD5ハッシュ比較で差分を検出する。

    Args:
        filepath: 比較対象の既存ファイルパス。
        new_content: 新たに取得したコンテンツ（Frontmatterを除く本文）。

    Returns:
        差分がある（または既存ファイルが存在しない）場合 ``True``。
    """
    if not filepath.exists():
        return True
    existing = filepath.read_text(encoding="utf-8")
    # 既存ファイルからFrontmatterを除いた本文を比較対象にする
    existing_body = _strip_frontmatter(existing)
    return (
        hashlib.md5(existing_body.encode("utf-8")).hexdigest()
        != hashlib.md5(new_content.encode("utf-8")).hexdigest()
    )


def _strip_frontmatter(text: str) -> str:
    """先頭のYAML Frontmatter（``---`` で囲まれたブロック）を除去する。"""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return text


def add_frontmatter(
    content: str, url: str, category: str, section: str | None = None
) -> str:
    """Markdown本文の先頭にFrontmatterを付与する。

    Args:
        content: Markdown本文。
        url: 取得元URL。
        category: カテゴリ（docs / blog / release-notes など）。
        section: サブセクション名（任意）。

    Returns:
        Frontmatterを付与したMarkdown文字列。
    """
    fetched_at = datetime.now(JST).isoformat()
    lines = [
        "---",
        f"source_url: {url}",
        f"fetched_at: {fetched_at}",
        f"category: {category}",
    ]
    if section:
        lines.append(f"section: {section}")
    lines.append("---")
    lines.append("")
    lines.append("")
    return "\n".join(lines) + content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(requests.Timeout),
    reraise=True,
)
def fetch_with_retry(url: str, **kwargs: object) -> requests.Response:
    """タイムアウト時に最大2回リトライしながらHTTP GETを実行する。

    Args:
        url: 取得対象URL。
        **kwargs: ``requests.get`` に渡す追加引数。

    Returns:
        ``requests.Response`` オブジェクト。
    """
    response = requests.get(url, timeout=TIMEOUT, **kwargs)  # type: ignore[arg-type]
    response.raise_for_status()
    return response


def _save(
    filepath: Path, body: str, url: str, category: str, section: str | None
) -> None:
    """差分がある場合のみFrontmatter付きで保存する。"""
    if is_updated(filepath, body):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(
            add_frontmatter(body, url=url, category=category, section=section),
            encoding="utf-8",
        )
        logger.info("%s 更新", filepath)
    else:
        logger.info("%s 差分なし スキップ", filepath)


def _slugify(text: str) -> str:
    """ファイル名向けに文字列を安全なスラッグへ変換する。"""
    slug = re.sub(r"[^\w\-]+", "-", text.strip().lower())
    return slug.strip("-")[:80] or "untitled"


def fetch_llms_full_txt() -> None:
    """llms-full.txtを取得し ``knowledge/docs/`` に保存する。"""
    try:
        response = fetch_with_retry(LLMS_FULL_TXT_URL)
        body = response.text
        filepath = KNOWLEDGE_DIR / "docs" / "llms-full.md"
        _save(filepath, body, url=LLMS_FULL_TXT_URL, category="docs", section=None)
    except requests.Timeout:
        logger.warning(
            "タイムアウト（リトライ上限到達）: %s スキップ", LLMS_FULL_TXT_URL
        )
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error("HTTPエラー %s: %s スキップ", status, LLMS_FULL_TXT_URL)
    except Exception as e:  # noqa: BLE001 - 他ソースの処理を止めないため
        logger.error("予期しないエラー: %s スキップ (%s)", e, LLMS_FULL_TXT_URL)


def fetch_blog_posts() -> None:
    """公式ブログのRSSフィードを取得し ``knowledge/blog/`` に各記事を保存する。"""
    try:
        response = fetch_with_retry(BLOG_RSS_URL)
        feed = feedparser.parse(response.content)
        for entry in feed.entries:
            title = getattr(entry, "title", "untitled")
            link = getattr(entry, "link", BLOG_RSS_URL)
            summary = getattr(entry, "summary", "")
            # HTMLタグを除去してプレーンな本文にする
            body_text = BeautifulSoup(summary, "html.parser").get_text("\n").strip()
            body = f"# {title}\n\n{body_text}\n"
            filepath = KNOWLEDGE_DIR / "blog" / f"{_slugify(title)}.md"
            _save(filepath, body, url=link, category="blog", section=title)
    except requests.Timeout:
        logger.warning("タイムアウト（リトライ上限到達）: %s スキップ", BLOG_RSS_URL)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error("HTTPエラー %s: %s スキップ", status, BLOG_RSS_URL)
    except Exception as e:  # noqa: BLE001
        logger.error("予期しないエラー: %s スキップ (%s)", e, BLOG_RSS_URL)


def fetch_release_notes() -> None:
    """GitHubリリースノートを取得し ``knowledge/release-notes/`` に保存する。"""
    try:
        response = fetch_with_retry(
            RELEASE_NOTES_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        releases = response.json()
        for release in releases:
            tag = release.get("tag_name") or release.get("name") or "release"
            name = release.get("name") or tag
            notes = release.get("body") or ""
            url = release.get("html_url", RELEASE_NOTES_URL)
            body = f"# {name}\n\n{notes}\n"
            filepath = KNOWLEDGE_DIR / "release-notes" / f"{_slugify(tag)}.md"
            _save(filepath, body, url=url, category="release-notes", section=tag)
    except requests.Timeout:
        logger.warning(
            "タイムアウト（リトライ上限到達）: %s スキップ", RELEASE_NOTES_URL
        )
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error("HTTPエラー %s: %s スキップ", status, RELEASE_NOTES_URL)
    except Exception as e:  # noqa: BLE001
        logger.error("予期しないエラー: %s スキップ (%s)", e, RELEASE_NOTES_URL)


def fetch_status_updates() -> None:
    """稼働状況ページのRSSを取得し ``knowledge/status/`` に各更新を保存する。

    インシデント・メンテナンス・特定モデルの障害告知など、docsやblogには
    載らない運用上のお知らせを拾う（例: モデルのエラーレート上昇、一時停止）。
    """
    try:
        response = fetch_with_retry(STATUS_RSS_URL)
        feed = feedparser.parse(response.content)
        for entry in feed.entries:
            title = getattr(entry, "title", "untitled")
            link = getattr(entry, "link", STATUS_RSS_URL)
            summary = getattr(entry, "summary", "")
            body_text = BeautifulSoup(summary, "html.parser").get_text("\n").strip()
            body = f"# {title}\n\n{body_text}\n"
            filepath = KNOWLEDGE_DIR / "status" / f"{_slugify(title)}.md"
            _save(filepath, body, url=link, category="status", section=title)
    except requests.Timeout:
        logger.warning("タイムアウト（リトライ上限到達）: %s スキップ", STATUS_RSS_URL)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error("HTTPエラー %s: %s スキップ", status, STATUS_RSS_URL)
    except Exception as e:  # noqa: BLE001
        logger.error("予期しないエラー: %s スキップ (%s)", e, STATUS_RSS_URL)


def fetch_cookbooks() -> None:
    """Cookbooksリポジトリのトップレベルファイル一覧を取得し保存する。"""
    try:
        response = fetch_with_retry(
            COOKBOOKS_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        items = response.json()
        names = [item.get("name", "") for item in items if item.get("type") == "file"]
        body = "# Claude Cookbooks 一覧\n\n" + "\n".join(
            f"- {name}" for name in names if name
        )
        filepath = KNOWLEDGE_DIR / "cookbooks" / "index.md"
        _save(filepath, body, url=COOKBOOKS_URL, category="cookbooks", section=None)
    except requests.Timeout:
        logger.warning("タイムアウト（リトライ上限到達）: %s スキップ", COOKBOOKS_URL)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error("HTTPエラー %s: %s スキップ", status, COOKBOOKS_URL)
    except Exception as e:  # noqa: BLE001
        logger.error("予期しないエラー: %s スキップ (%s)", e, COOKBOOKS_URL)


def main() -> None:
    """全ソースの取得を順に実行する。"""
    fetch_llms_full_txt()
    fetch_blog_posts()
    fetch_release_notes()
    fetch_status_updates()
    fetch_cookbooks()


if __name__ == "__main__":
    main()
