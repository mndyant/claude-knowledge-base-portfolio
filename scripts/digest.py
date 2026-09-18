"""Anthropic公式情報の差分を日本語ダイジェストとしてDiscordへ配信する。

fetch.py実行後のknowledge/配下を前回状態（data/digest_state.json）と比較し、
新規・更新されたページだけをGemini（無料枠）で日本語要約してDiscord webhookへ
投稿する。RAG（検索=プル型）を補完するプッシュ型の通知レイヤー。

設計方針:
- Gemini呼び出しはバッチ方式で1実行あたり最大2回（無料枠のごく一部しか使わない）
- Gemini失敗時は要約なし（タイトル+URLのみ）で投稿を続行する（fail-soft）
- 初回実行は状態の初期化のみ行い、投稿しない（全1,400ページの通知爆発を防ぐ）
- 差分ハッシュはFrontmatter（fetched_atが毎回変わる）を除いた本文で取る

使い方:
    python scripts/fetch.py && python scripts/digest.py
    python scripts/digest.py --dry-run   # 投稿せず内容を表示（状態も保存しない）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

KNOWLEDGE_DIR = Path("knowledge")
LLMS_FULL_MD = KNOWLEDGE_DIR / "docs" / "llms-full.md"
STATE_PATH = Path("data/digest_state.json")

# -latestエイリアスは最新版へ自動追従する（固定名はモデル廃止時に404で壊れる。
# 2026-07-10にgemini-2.5-flash-lite廃止で実際に発生）。404時は後続へフォールバック。
MODEL_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
]
DEFAULT_MODEL = MODEL_CANDIDATES[0]
MAX_SUMMARIZE = 30  # これを超えた分はタイトルのみ列挙する
SUMMARIZE_BATCH = 15  # 1回のGemini呼び出しに詰める件数（最大2回/実行）
EXCERPT_CHARS = 1000
DISCORD_CHUNK = 1900  # Discordの2000文字制限へのマージン

# ソース種別ごとの表示（優先度順。要約枠はこの順で割り当てる）
# statusはインシデント・障害告知で即応性が要るため最優先にする
KIND_LABEL = {"new": "🆕", "updated": "♻️"}
SOURCE_ORDER = {"status": 0, "blog": 1, "release-notes": 2, "docs": 3, "cookbooks": 4}


@dataclass
class DigestItem:
    """差分検知された1件（docsの1ページ or blog等の1ファイル）。"""

    key: str
    source: str  # docs / blog / release-notes / status / cookbooks
    kind: str  # new / updated
    title: str
    url: str
    excerpt: str
    digest: str  # 本文ハッシュ
    summary: str = ""


def _strip_frontmatter(text: str) -> str:
    """先頭のYAML Frontmatterを除去する（fetched_atをハッシュに含めない）。"""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return text


def _body_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


# llms-full.txt内の各ページを一意に示すURLマーカー行。
# 2026年6月版は「URL: https://...」、7月版は「**URL:** https://...」。両対応。
URL_MARKER_RE = re.compile(r"^\*{0,2}URL:\*{0,2}\s*(https?://\S+)")

# ページタイトル検出に使う見出し行（H1〜H3）
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


def _find_page_title(lines: list[str], url_index: int) -> str:
    """URLマーカー行の位置からページタイトルを探す。

    形式A（通常docs）: 見出しがURLの直前にある
        # Get started with Claude
        **URL:** https://...
    形式B（APIリファレンス）: 見出しがURLの後ろ（--- を挟んで）にある
        ---
        **URL:** https://...
        ---
        ## Acknowledge Work
    """
    # 直前方向（最大4行さかのぼる。空行はスキップ、それ以外の行が出たら打ち切り）
    for index in range(url_index - 1, max(-1, url_index - 5), -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        match = HEADING_RE.match(stripped)
        return match.group(2).strip() if match else ""
    return ""


def _find_title_after(lines: list[str], url_index: int) -> str:
    """形式B用: URLマーカーの後方から最初の見出しを探す。"""
    for index in range(url_index + 1, min(len(lines), url_index + 8)):
        stripped = lines[index].strip()
        if not stripped or stripped == "---":
            continue
        match = HEADING_RE.match(stripped)
        return match.group(2).strip() if match else ""
    return ""


def _page_excerpt(lines: list[str]) -> str:
    body_lines = [
        line
        for line in lines
        if not HEADING_RE.match(line.strip())
        and not URL_MARKER_RE.match(line.strip())
        and line.strip() != "---"
    ]
    text = re.sub(r"\s+", " ", "\n".join(body_lines)).strip()
    return text[:EXCERPT_CHARS]


def collect_docs_pages(llms_full_path: Path = LLMS_FULL_MD) -> list[DigestItem]:
    """llms-full.mdをページ単位に分解しDigestItem化する。

    ページ区切りの水平線はコードブロックや本文中と紛らわしいため、
    各ページに必ず1つあるURLマーカー行を境界の基準にする。
    ページ範囲は「URLマーカーの少し手前（形式Aのタイトルを含む）」から
    「次のURLマーカーの手前」まで。
    """
    if not llms_full_path.exists():
        logger.warning(
            "%s が見つかりません（docsの差分検知をスキップ）", llms_full_path
        )
        return []

    text = _strip_frontmatter(llms_full_path.read_text(encoding="utf-8"))
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    url_indices = [
        index for index, line in enumerate(lines) if URL_MARKER_RE.match(line.strip())
    ]
    if not url_indices:
        logger.warning("URLマーカーが見つかりません。llms-full.txtの形式変更の可能性")
        return []

    items: list[DigestItem] = []
    seen_keys: set[str] = set()
    for position, url_index in enumerate(url_indices):
        url = URL_MARKER_RE.match(lines[url_index].strip()).group(1)  # type: ignore[union-attr]
        title = _find_page_title(lines, url_index) or _find_title_after(
            lines, url_index
        )

        # ページ範囲: タイトル行（あれば）〜次ページのタイトル/URLの手前
        start = max(0, url_index - 4)
        if position + 1 < len(url_indices):
            end = max(0, url_indices[position + 1] - 4)
        else:
            end = len(lines)
        page_lines = lines[start:end]
        body = "\n".join(page_lines).strip()

        key = f"docs:{url}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        items.append(
            DigestItem(
                key=key,
                source="docs",
                kind="new",  # diff時に確定する
                title=title or url.rsplit("/", 1)[-1],
                url=url,
                excerpt=_page_excerpt(page_lines),
                digest=_body_hash(body),
            )
        )
    return items


def collect_knowledge_files() -> list[DigestItem]:
    """blog / release-notes / status / cookbooks のファイルをDigestItem化する。"""
    items: list[DigestItem] = []
    for source in ("blog", "release-notes", "status", "cookbooks"):
        directory = KNOWLEDGE_DIR / source
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            body = _strip_frontmatter(raw)
            if not body.strip():
                continue
            metadata = _parse_frontmatter_meta(raw)
            title = metadata.get("section") or path.stem
            items.append(
                DigestItem(
                    key=f"{source}:{path.name}",
                    source=source,
                    kind="new",
                    title=title,
                    url=metadata.get("source_url", ""),
                    excerpt=re.sub(r"\s+", " ", body).strip()[:EXCERPT_CHARS],
                    digest=_body_hash(body),
                )
            )
    return items


def _parse_frontmatter_meta(text: str) -> dict[str, str]:
    """Frontmatterのkey: valueを辞書で返す。"""
    metadata: dict[str, str] = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()
    return metadata


def load_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(items: list[DigestItem], path: Path = STATE_PATH) -> None:
    state = {
        "version": 1,
        "updated_at": datetime.now(JST).isoformat(),
        "hashes": {item.key: item.digest for item in items},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def diff_items(items: list[DigestItem], state: dict) -> list[DigestItem]:
    """前回状態と比較し、新規・更新のみ返す（削除は通知しない）。

    ソース（status等）が今回初めて登場した場合は、そのソースの全件を
    「新規追加ソースの後追い」とみなして通知せず静かにベースライン化する
    （既存の運用にstatusのような新カテゴリを足した初回に、過去の全件が
    "新着"として一気に通知されるのを防ぐ。初回実行時の全件初期化と同じ考え方）。
    """
    previous: dict[str, str] = state.get("hashes", {})
    known_sources = {key.split(":", 1)[0] for key in previous}

    changed: list[DigestItem] = []
    for item in items:
        if item.source not in known_sources:
            continue
        old = previous.get(item.key)
        if old is None:
            item.kind = "new"
            changed.append(item)
        elif old != item.digest:
            item.kind = "updated"
            changed.append(item)
    changed.sort(key=lambda i: (SOURCE_ORDER.get(i.source, 9), i.kind != "new"))
    return changed


def summarize_with_gemini(items: list[DigestItem], model: str) -> None:
    """変更点をバッチでGeminiに渡し日本語要約を付与する。失敗しても投稿は続行。"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY未設定。要約なしで投稿します")
        return

    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai未インストール。要約なしで投稿します")
        return

    client = genai.Client(api_key=api_key)
    targets = items[:MAX_SUMMARIZE]

    # 指定モデルが廃止済み（404）の場合は候補リストへフォールバックする
    models = [model] + [m for m in MODEL_CANDIDATES if m != model]

    for start in range(0, len(targets), SUMMARIZE_BATCH):
        batch = targets[start : start + SUMMARIZE_BATCH]
        payload = [
            {
                "i": index,
                "kind": "新規ページ" if item.kind == "new" else "更新されたページ",
                "source": item.source,
                "title": item.title,
                "excerpt": item.excerpt,
            }
            for index, item in enumerate(batch)
        ]
        prompt = (
            "あなたはAnthropic（Claude）公式ドキュメントの更新を日本のエンジニアに"
            "伝える編集者です。以下のJSONの各項目について、日本語1〜2文で"
            "「何についての内容か」「開発者にとって何が嬉しい/影響するか」を要約してください。\n"
            '出力は {"0": "要約", "1": "要約", ...} のJSONのみ。前置きや```は不要。\n'
            "技術用語・API名は英語のまま残すこと。\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        text = ""
        for candidate in models:
            try:
                text = _call_gemini_with_retry(client, candidate, prompt)
                break
            except Exception as exc:  # noqa: BLE001 - 次候補へフォールバック
                if "404" in str(exc) or "NOT_FOUND" in str(exc):
                    logger.warning(
                        "モデル %s は利用不可。次の候補を試します", candidate
                    )
                    continue
                logger.warning("Gemini要約に失敗（このバッチは要約なし）: %s", exc)
                break
        if not text:
            continue

        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            summaries = json.loads(text)
            for index, item in enumerate(batch):
                item.summary = str(summaries.get(str(index), "")).strip()
        except Exception as exc:  # noqa: BLE001 - 要約失敗は投稿を止めない
            logger.warning("Gemini応答のJSON解析に失敗（要約なし）: %s", exc)


def _call_gemini_with_retry(client, model: str, prompt: str, max_retries: int = 3):  # type: ignore[no-untyped-def]
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text or ""
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries:
                wait = 2**attempt * 5
                logger.warning("Geminiレート制限。%d秒待って再試行", wait)
                time.sleep(wait)
                continue
            raise
    return ""


def build_messages(changed: list[DigestItem]) -> list[str]:
    """変更一覧をDiscordの2000文字制限に収まるメッセージ群へ整形する。"""
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    header = f"**📰 Anthropic公式更新ダイジェスト** — {now} JST（{len(changed)}件）"

    lines: list[str] = []
    for item in changed[:MAX_SUMMARIZE]:
        label = KIND_LABEL.get(item.kind, "")
        entry = f"{label} **{item.title}**"
        if item.url:
            entry += f"\n<{item.url}>"
        if item.summary:
            entry += f"\n{item.summary}"
        lines.append(entry)

    remainder = changed[MAX_SUMMARIZE:]
    if remainder:
        titles = "、".join(item.title for item in remainder[:15])
        suffix = f" ほか{len(remainder) - 15}件" if len(remainder) > 15 else ""
        lines.append(f"📎 その他の更新: {titles}{suffix}")

    messages: list[str] = []
    current = header
    for line in lines:
        candidate = f"{current}\n\n{line}"
        if len(candidate) > DISCORD_CHUNK:
            messages.append(current)
            current = line
        else:
            current = candidate
    messages.append(current)
    return messages


def post_discord(messages: list[str], webhook_url: str) -> None:
    """webhookへ投稿する。429はretry_afterに従って再試行する。"""
    for message in messages:
        for attempt in range(4):
            response = requests.post(webhook_url, json={"content": message}, timeout=30)
            if response.status_code == 429:
                retry_after = float(response.json().get("retry_after", 2**attempt))
                logger.warning("Discordレート制限。%.1f秒待機", retry_after)
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            break
        else:
            # 429のまま再試行上限に達した。ここで黙ってreturnすると
            # 呼び出し元のsave_stateが未配信の差分を配信済みとして記録する。
            response.raise_for_status()
        time.sleep(1)  # 連投時のレート制限予防


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="knowledge/の差分を日本語要約してDiscordへ投稿する"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="投稿・状態保存を行わず、検知内容と生成メッセージを表示する",
    )
    parser.add_argument(
        "--model", default=os.environ.get("DIGEST_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    return parser.parse_args()


def main() -> None:
    # Windowsコンソール（cp932）でも絵文字入りメッセージを表示できるようにする
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    args = parse_args()

    items = collect_docs_pages() + collect_knowledge_files()
    if not items:
        raise SystemExit(
            "knowledge/配下に対象がありません。先にfetch.pyを実行してください"
        )
    logger.info("%d 件（docsページ+ファイル）を収集", len(items))

    state = load_state(args.state)
    if not state:
        if args.dry_run:
            logger.info("[dry-run] 初回実行: 状態を初期化して終了（投稿なし）")
            return
        save_state(items, args.state)
        logger.info("初回実行: %d 件の状態を初期化しました（投稿なし）", len(items))
        return

    previous_sources = {key.split(":", 1)[0] for key in state.get("hashes", {})}
    new_sources = {item.source for item in items} - previous_sources
    if new_sources:
        logger.info(
            "新規ソースを検知（%s）: 通知せずベースライン化します", ", ".join(sorted(new_sources))
        )

    changed = diff_items(items, state)
    if not changed:
        logger.info("差分なし")
        if not args.dry_run:
            save_state(items, args.state)
        return

    logger.info(
        "差分 %d 件（new=%d / updated=%d）",
        len(changed),
        sum(1 for i in changed if i.kind == "new"),
        sum(1 for i in changed if i.kind == "updated"),
    )

    summarize_with_gemini(changed, args.model)
    messages = build_messages(changed)

    if args.dry_run:
        for index, message in enumerate(messages, start=1):
            print(f"----- message {index}/{len(messages)} -----")
            print(message)
        return

    webhook_url = os.environ.get("DISCORD_WEBHOOK_CLAUDE")
    if not webhook_url:
        raise SystemExit("DISCORD_WEBHOOK_CLAUDE が未設定です")
    post_discord(messages, webhook_url)
    save_state(items, args.state)
    logger.info("投稿完了（%d メッセージ）", len(messages))


if __name__ == "__main__":
    main()
