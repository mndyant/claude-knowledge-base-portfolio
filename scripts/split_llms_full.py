"""llms-full.txtをRAG向けにMarkdown構造を保って分割するスクリプト。

外部APIは使わず、標準ライブラリだけで動作する。tiktokenがインストール済みの
環境ではtoken数を実測し、未導入の場合は文字数/4で概算する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_INPUT = Path("llms-full.txt")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_STATS = Path("data/processed/stats.json")

# multilingual-e5-base の最大入力512トークンに、"passage: " prefixと
# special tokens分の余裕を持たせて収める。
DEFAULT_HF_TOKENIZER = "intfloat/multilingual-e5-base"
MIN_TOKENS = 180
TARGET_TOKENS = 400
MAX_TOKENS = 480
OVERLAP_TOKENS = 50

PAGE_SEPARATOR_RE = re.compile(r"^\s*---\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-+*]\s+|\d+[.)]\s+)")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
# 2026年6月版は「URL: https://...」、7月版は「**URL:** https://...」。両対応
# （digest.pyのURL_MARKER_REと同じ方針。HANDOFF.md §4のフォーマット変更を参照）。
PAGE_URL_RE = re.compile(r"^\*{0,2}URL:\*{0,2}\s*(https?://\S+)\s*$", re.IGNORECASE)

# SDKリファレンスの列挙定数（例: `- \`MESSAGE_BATCHES_2024_09_24("message-batches-2024-09-24")\``)。
# 複数の異なるAPIエンドポイントページにバイト同一で繰り返されるため、埋め込み対象から
# 除外する（docs/EVAL.md「既知の限界」の"定型的なパラメータ一覧チャンクによるノイズ"参照）。
ENUM_BULLET_RE = re.compile(r'^-\s+`[A-Z][A-Z0-9_]*\("[a-z0-9\-]+"\)`$')
BOILERPLATE_MIN_RUN = 5


@dataclass(frozen=True)
class Block:
    """Markdown上の壊したくない最小単位。"""

    text: str
    kind: str
    page_index: int
    heading_path: tuple[str, ...]
    has_code: bool = False


@dataclass(frozen=True)
class ChunkDraft:
    """metadataを付ける前のチャンク候補。"""

    content: str
    page_index: int
    heading_path: tuple[str, ...]
    has_code: bool


@dataclass(frozen=True)
class AtomicUnit:
    """最終ガードで使う、コードブロックを壊さない単位。"""

    text: str
    has_code: bool = False


class TokenCounter:
    """embeddingモデルのtokenizerで実測する。なければtiktoken、最後は文字数/4。

    チャンクサイズの上限はembeddingモデル（multilingual-e5-base = 512トークン）の
    入力上限に合わせる必要があるため、同じtokenizerで数えるのが最も正確。
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        hf_model: str | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.name = "char_count/4"
        self._encoding = None
        self._hf_tokenizer = None

        if hf_model:
            self._hf_tokenizer = _load_hf_tokenizer(hf_model)
            if self._hf_tokenizer is not None:
                self.name = f"hf:{hf_model}"
                return

        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoding = tiktoken.get_encoding(encoding_name)
            self.name = f"tiktoken:{encoding_name}"
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        """token数を返す。fallback時はOpenAI系tokenizerの雑な目安として文字数/4。"""
        if not text:
            return 0
        if self._hf_tokenizer is not None:
            return len(self._hf_tokenizer.encode(text, add_special_tokens=False))
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, math.ceil(len(text) / 4))

    def tail(self, text: str, token_limit: int) -> str:
        """overlap用に末尾token相当の文字列を返す。"""
        if token_limit <= 0 or not text:
            return ""
        if self._hf_tokenizer is not None:
            token_ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
            return self._hf_tokenizer.decode(
                token_ids[-token_limit:], skip_special_tokens=True
            ).strip()
        if self._encoding is not None:
            token_ids = self._encoding.encode(text)
            return self._encoding.decode(token_ids[-token_limit:]).strip()
        return text[-token_limit * 4 :].strip()

    def split(self, text: str, token_limit: int) -> list[str]:
        """巨大な1単位をtoken上限に収まる程度へ分割する。"""
        if token_limit <= 0:
            return [text]
        if self._hf_tokenizer is not None:
            token_ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
            return [
                part
                for start in range(0, len(token_ids), token_limit)
                if (
                    part := self._hf_tokenizer.decode(
                        token_ids[start : start + token_limit],
                        skip_special_tokens=True,
                    ).strip()
                )
            ]
        if self._encoding is not None:
            token_ids = self._encoding.encode(text)
            return [
                self._encoding.decode(token_ids[start : start + token_limit]).strip()
                for start in range(0, len(token_ids), token_limit)
                if token_ids[start : start + token_limit]
            ]

        char_limit = max(1, token_limit * 4)
        chunks: list[str] = []
        remaining = text.strip()
        while len(remaining) > char_limit:
            split_at = _find_split_position(remaining, char_limit)
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks


def _load_hf_tokenizer(model_name: str):  # type: ignore[no-untyped-def]
    """HFのtokenizerをロードする。ローカルキャッシュ優先・失敗時はNone。"""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception:
        return None

    for local_files_only in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        except Exception:
            continue
    return None


def _find_split_position(text: str, limit: int) -> int:
    """指定位置付近の空白・改行で切る。見つからない場合だけ固定長で切る。"""
    window_start = max(1, int(limit * 0.7))
    for marker in ("\n\n", "\n", "。", ". ", " "):
        pos = text.rfind(marker, window_start, limit)
        if pos > 0:
            return pos + len(marker)
    return limit


def _strip_frontmatter(text: str) -> str:
    """fetch.py由来のYAML Frontmatterが先頭にあれば除外する。"""
    if not text.startswith("---"):
        return text

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text

    # ページ区切りとの誤判定を避けるため、先頭50行以内のkey: value形式だけを対象にする。
    for index, line in enumerate(lines[1:50], start=1):
        if line.strip() != "---":
            continue
        header_lines = [item.strip() for item in lines[1:index] if item.strip()]
        if header_lines and all(
            ":" in item and not item.startswith("#") for item in header_lines
        ):
            return "".join(lines[index + 1 :]).lstrip("\n")
        return text
    return text


def _clean_heading(raw: str) -> str:
    """Markdown見出しの装飾をmetadata向けに落とす。"""
    title = raw.strip()
    title = re.sub(r"\s+#+\s*$", "", title)
    title = re.sub(r"\s*\{#[^}]+\}\s*$", "", title)
    return re.sub(r"\s+", " ", title).strip()


def _heading_path(stack: list[str]) -> tuple[str, ...]:
    return tuple(item for item in stack if item)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.count("|") >= 2 or bool(TABLE_SEPARATOR_RE.match(stripped))


def _is_list_line(line: str) -> bool:
    return bool(LIST_RE.match(line))


def _is_indented_continuation(line: str) -> bool:
    return line.startswith((" ", "\t"))


def _next_non_empty_line(lines: list[str], start: int) -> str:
    """start以降の最初の非空行を返す。

    ``lines[start:]`` のスライスは残り全行のコピーを作るため、
    93MB級の入力（約200万行）で区切り行ごとに呼ばれるとO(n^2)の
    メモリ確保になりMemoryErrorを起こす（2026-07-20実測）。
    indexアクセスでコピーせずに走査する。
    """
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped:
            return stripped
    return ""


def _is_page_separator(lines: list[str], index: int) -> bool:
    """llms-full.txtのページ境界らしい水平線だけを判定する。

    2026年6月版は「--- → # 見出し」、7月版のAPIリファレンス系は
    「--- → **URL:** → --- → ## 見出し」の構造。直後の非空行がH1または
    URLマーカーの水平線だけをページ境界とみなす（本文中の---を誤検出しない）。
    """
    if not PAGE_SEPARATOR_RE.match(lines[index]):
        return False
    next_line = _next_non_empty_line(lines, index + 1)
    return next_line.startswith("# ") or bool(PAGE_URL_RE.match(next_line))


def extract_page_urls(text: str) -> dict[int, str]:
    """llms-full.txtのページ番号と公式URLの対応を抽出する。

    ページURLは必ずページ先頭付近（6月版は見出しの直後、7月版は見出しの前）に
    現れるため、各ページで最初にマッチしたURLだけを採用する。本文中の
    「URL: ...」行（画像URLの例示など）が後から上書きするのを防ぐ。
    """
    text = _strip_frontmatter(text)
    lines = text.splitlines()
    page_index = 0
    urls: dict[int, str] = {}
    for index, line in enumerate(lines):
        if _is_page_separator(lines, index):
            page_index += 1
            continue
        match = PAGE_URL_RE.match(line.strip())
        if match:
            urls.setdefault(page_index, match.group(1))
    return urls


def parse_markdown_blocks(text: str) -> list[Block]:
    """Markdownをコード・表・リスト・段落のブロックへ分解する。"""
    normalized = _strip_frontmatter(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    blocks: list[Block] = []
    heading_stack = ["", "", ""]
    page_index = 0
    buffer: list[str] = []
    buffer_kind = ""

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if not buffer:
            return
        body = "\n".join(buffer).strip("\n")
        if body.strip():
            blocks.append(
                Block(
                    text=body,
                    kind=buffer_kind or "paragraph",
                    page_index=page_index,
                    heading_path=_heading_path(heading_stack),
                )
            )
        buffer = []
        buffer_kind = ""

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(1)
            code_lines = [line]
            line_index += 1
            while line_index < len(lines):
                code_line = lines[line_index]
                code_lines.append(code_line)
                if code_line.strip().startswith(fence):
                    line_index += 1
                    break
                line_index += 1
            blocks.append(
                Block(
                    text="\n".join(code_lines).strip("\n"),
                    kind="code",
                    page_index=page_index,
                    heading_path=_heading_path(heading_stack),
                    has_code=True,
                )
            )
            continue

        if PAGE_SEPARATOR_RE.match(line):
            flush()
            if _is_page_separator(lines, line_index):
                page_index += 1
                heading_stack = ["", "", ""]
            line_index += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match and len(heading_match.group(1)) <= 3:
            flush()
            level = len(heading_match.group(1))
            heading_stack[level - 1] = _clean_heading(heading_match.group(2))
            for child_index in range(level, len(heading_stack)):
                heading_stack[child_index] = ""
            line_index += 1
            continue

        if not line.strip():
            if buffer_kind == "list":
                buffer.append(line)
            else:
                flush()
            line_index += 1
            continue

        line_kind = "paragraph"
        if _is_table_line(line):
            line_kind = "table"
        elif _is_list_line(line):
            line_kind = "list"

        if not buffer:
            buffer = [line]
            buffer_kind = line_kind
            line_index += 1
            continue

        if buffer_kind == "list":
            if line_kind == "list" or _is_indented_continuation(line):
                buffer.append(line)
            else:
                flush()
                buffer = [line]
                buffer_kind = line_kind
            line_index += 1
            continue

        if buffer_kind == "table":
            if line_kind == "table":
                buffer.append(line)
            else:
                flush()
                buffer = [line]
                buffer_kind = line_kind
            line_index += 1
            continue

        if buffer_kind == line_kind == "paragraph":
            buffer.append(line)
        else:
            flush()
            buffer = [line]
            buffer_kind = line_kind
        line_index += 1

    flush()
    return blocks


def _format_heading_path(heading_path: tuple[str, ...]) -> str:
    """チャンク本文の先頭に入れる見出しコンテキストを作る。"""
    return "\n\n".join(
        f"{'#' * level} {title}" for level, title in enumerate(heading_path, start=1)
    )


def _render_chunk(prefix: str, blocks: list[Block]) -> str:
    body = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
    if prefix and body:
        return f"{prefix}\n\n{body}".strip()
    return (body or prefix).strip()


def _pack_units(
    units: list[str],
    joiner: str,
    block: Block,
    token_counter: TokenCounter,
    token_budget: int,
) -> list[Block]:
    """行・段落・文などの単位をbudget内へ詰める。"""
    packed: list[Block] = []
    current: list[str] = []

    for unit in [item.strip() for item in units if item.strip()]:
        if token_counter.count(unit) > token_budget:
            if current:
                packed.append(replace(block, text=joiner.join(current).strip()))
                current = []
            packed.extend(
                replace(block, text=part)
                for part in token_counter.split(unit, token_budget)
                if part.strip()
            )
            continue

        candidate = joiner.join([*current, unit]).strip()
        if current and token_counter.count(candidate) > token_budget:
            packed.append(replace(block, text=joiner.join(current).strip()))
            current = [unit]
        else:
            current.append(unit)

    if current:
        packed.append(replace(block, text=joiner.join(current).strip()))
    return packed


def _split_table_block(
    block: Block,
    token_counter: TokenCounter,
    token_budget: int,
) -> list[Block]:
    """大きな表はヘッダーを繰り返しながら行単位で分割する。"""
    lines = [line for line in block.text.splitlines() if line.strip()]
    if len(lines) <= 2:
        return _pack_units(lines, "\n", block, token_counter, token_budget)

    header = lines[:2] if TABLE_SEPARATOR_RE.match(lines[1].strip()) else lines[:1]
    rows = lines[len(header) :]
    chunks: list[Block] = []
    current = header.copy()

    for row in rows:
        candidate = "\n".join([*current, row])
        if current != header and token_counter.count(candidate) > token_budget:
            chunks.append(replace(block, text="\n".join(current).strip()))
            current = [*header, row]
        else:
            current.append(row)

    if current:
        chunks.append(replace(block, text="\n".join(current).strip()))
    return chunks


def _split_paragraph_block(
    block: Block,
    token_counter: TokenCounter,
    token_budget: int,
) -> list[Block]:
    """大きな段落は段落、文、最後にtoken単位の順で分割する。"""
    paragraphs = re.split(r"\n{2,}", block.text)
    if len([item for item in paragraphs if item.strip()]) > 1:
        return _pack_units(paragraphs, "\n\n", block, token_counter, token_budget)

    sentences = re.split(r"(?<=[。！？.!?])\s+", block.text)
    if len([item for item in sentences if item.strip()]) > 1:
        return _pack_units(sentences, " ", block, token_counter, token_budget)

    return [
        replace(block, text=part)
        for part in token_counter.split(block.text, token_budget)
        if part.strip()
    ]


def split_oversized_blocks(
    blocks: list[Block],
    prefix: str,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[Block]:
    """巨大ブロックを、Markdown構造をできるだけ保つ範囲で細分化する。"""
    prefix_tokens = token_counter.count(prefix)
    token_budget = max(100, max_tokens - prefix_tokens)
    split_blocks: list[Block] = []

    for block in blocks:
        if block.has_code:
            split_blocks.append(block)
            continue

        if token_counter.count(_render_chunk(prefix, [block])) <= max_tokens:
            split_blocks.append(block)
            continue

        if block.kind == "table":
            split_blocks.extend(_split_table_block(block, token_counter, token_budget))
        elif block.kind == "list":
            split_blocks.extend(
                _pack_units(
                    block.text.splitlines(), "\n", block, token_counter, token_budget
                )
            )
        else:
            split_blocks.extend(
                _split_paragraph_block(block, token_counter, token_budget)
            )

    return split_blocks


def _select_overlap_blocks(
    blocks: list[Block],
    token_counter: TokenCounter,
    overlap_tokens: int,
) -> list[Block]:
    """次チャンクへ持ち越す末尾コンテキストを選ぶ。コードは重複させない。"""
    if overlap_tokens <= 0:
        return []

    selected: list[Block] = []
    selected_tokens = 0
    for block in reversed(blocks):
        if block.has_code or block.kind == "table":
            continue

        block_tokens = token_counter.count(block.text)
        if block_tokens > overlap_tokens:
            tail = token_counter.tail(block.text, overlap_tokens)
            if tail:
                selected.insert(0, replace(block, text=tail, kind="overlap"))
            break

        selected.insert(0, replace(block, kind="overlap"))
        selected_tokens += block_tokens
        if selected_tokens >= overlap_tokens:
            break
    return selected


def _iter_segments(blocks: list[Block]) -> list[list[Block]]:
    """page_indexとheading_pathが同じ連続ブロックを1セグメントにまとめる。"""
    segments: list[list[Block]] = []
    current: list[Block] = []
    current_key: tuple[int, tuple[str, ...]] | None = None

    for block in blocks:
        key = (block.page_index, block.heading_path)
        if current and key != current_key:
            segments.append(current)
            current = []
        current.append(block)
        current_key = key

    if current:
        segments.append(current)
    return segments


def split_segment(
    blocks: list[Block],
    token_counter: TokenCounter,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    """1つの見出し配下をチャンクへ分割する。"""
    if not blocks:
        return []

    page_index = blocks[0].page_index
    heading_path = blocks[0].heading_path
    prefix = _format_heading_path(heading_path)
    prepared_blocks = split_oversized_blocks(blocks, prefix, token_counter, max_tokens)

    chunks: list[ChunkDraft] = []
    current: list[Block] = []

    def emit() -> None:
        nonlocal current
        content = _render_chunk(prefix, current)
        if not content:
            current = []
            return
        chunks.append(
            ChunkDraft(
                content=content,
                page_index=page_index,
                heading_path=heading_path,
                has_code=any(block.has_code for block in current),
            )
        )
        current = _select_overlap_blocks(current, token_counter, overlap_tokens)

    for block in prepared_blocks:
        if not block.text.strip():
            continue

        candidate = [*current, block]
        candidate_tokens = token_counter.count(_render_chunk(prefix, candidate))
        current_tokens = (
            token_counter.count(_render_chunk(prefix, current)) if current else 0
        )

        should_flush = bool(current) and (
            candidate_tokens > max_tokens
            or (current_tokens >= min_tokens and candidate_tokens > target_tokens)
        )

        if should_flush:
            emit()
            candidate = [*current, block]
            if (
                current
                and token_counter.count(_render_chunk(prefix, candidate)) > max_tokens
            ):
                current = []

        current.append(block)

    if current:
        emit()
    return chunks


def _common_heading_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    """複数セクションをまとめたチャンク用に共通の見出しprefixを返す。"""
    if not paths:
        return ()
    shortest = min(len(path) for path in paths)
    common: list[str] = []
    for index in range(shortest):
        value = paths[0][index]
        if all(path[index] == value for path in paths):
            common.append(value)
        else:
            break
    return tuple(common) or paths[0]


def _merge_drafts(buffer: list[ChunkDraft]) -> ChunkDraft:
    """複数の短い候補を1チャンクへまとめる。"""
    common_path = _common_heading_path([item.heading_path for item in buffer])
    contents: list[str] = []
    for index, item in enumerate(buffer):
        content = item.content.strip()
        if index > 0:
            common_prefix = _format_heading_path(common_path)
            if common_prefix and content.startswith(common_prefix):
                content = content[len(common_prefix) :].lstrip()
        if content:
            contents.append(content)

    return ChunkDraft(
        content="\n\n".join(contents),
        page_index=buffer[0].page_index,
        heading_path=common_path,
        has_code=any(item.has_code for item in buffer),
    )


def _split_plain_units(
    text: str, token_counter: TokenCounter, max_tokens: int
) -> list[AtomicUnit]:
    """通常テキストを段落単位、必要ならtoken単位に分ける。"""
    units: list[AtomicUnit] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if token_counter.count(paragraph) <= max_tokens:
            units.append(AtomicUnit(paragraph))
            continue
        units.extend(
            AtomicUnit(part) for part in token_counter.split(paragraph, max_tokens)
        )
    return units


def _content_units_preserving_code(
    content: str,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[AtomicUnit]:
    """チャンク本文をコードフェンスを壊さない単位へ分ける。"""
    lines = content.splitlines()
    units: list[AtomicUnit] = []
    plain_buffer: list[str] = []
    line_index = 0

    def flush_plain() -> None:
        nonlocal plain_buffer
        if plain_buffer:
            units.extend(
                _split_plain_units("\n".join(plain_buffer), token_counter, max_tokens)
            )
            plain_buffer = []

    while line_index < len(lines):
        line = lines[line_index]
        fence_match = FENCE_RE.match(line)
        if not fence_match:
            plain_buffer.append(line)
            line_index += 1
            continue

        flush_plain()
        fence = fence_match.group(1)
        code_lines = [line]
        line_index += 1
        while line_index < len(lines):
            code_line = lines[line_index]
            code_lines.append(code_line)
            line_index += 1
            if code_line.strip().startswith(fence):
                break
        units.append(AtomicUnit("\n".join(code_lines).strip(), has_code=True))

    flush_plain()
    return [unit for unit in units if unit.text.strip()]


def _split_code_unit(
    unit: AtomicUnit,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[AtomicUnit]:
    """巨大コードブロックをフェンスを保ったまま行単位で分割する。"""
    lines = unit.text.splitlines()
    if len(lines) < 3 or not FENCE_RE.match(lines[0]):
        return [
            AtomicUnit(part, has_code=unit.has_code)
            for part in token_counter.split(unit.text, max_tokens)
            if part.strip()
        ]

    opening = lines[0]
    closing = lines[-1] if FENCE_RE.match(lines[-1]) else "```"
    body_lines = lines[1:-1] if FENCE_RE.match(lines[-1]) else lines[1:]
    fence_overhead = token_counter.count(f"{opening}\n{closing}")
    token_budget = max(50, max_tokens - fence_overhead)

    parts: list[AtomicUnit] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            body = "\n".join(current)
            parts.append(AtomicUnit(f"{opening}\n{body}\n{closing}", has_code=True))
            current = []
            current_tokens = 0

    for line in body_lines:
        line_tokens = token_counter.count(line) + 1
        if line_tokens > token_budget:
            flush()
            for piece in token_counter.split(line, token_budget):
                parts.append(
                    AtomicUnit(f"{opening}\n{piece}\n{closing}", has_code=True)
                )
            continue
        if current and current_tokens + line_tokens > token_budget:
            flush()
        current.append(line)
        current_tokens += line_tokens

    flush()
    return parts


def _pack_atomic_units(
    units: list[AtomicUnit],
    draft: ChunkDraft,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[ChunkDraft]:
    """AtomicUnitをmax_tokens以下へ詰める。巨大unitはコードでも細分化する。"""
    packed: list[ChunkDraft] = []
    current: list[AtomicUnit] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        packed.append(
            ChunkDraft(
                content="\n\n".join(unit.text.strip() for unit in current),
                page_index=draft.page_index,
                heading_path=draft.heading_path,
                has_code=any(unit.has_code for unit in current),
            )
        )
        current = []

    for unit in units:
        unit_tokens = token_counter.count(unit.text)
        if unit_tokens > max_tokens:
            flush()
            packed.extend(
                ChunkDraft(
                    content=part.text,
                    page_index=draft.page_index,
                    heading_path=draft.heading_path,
                    has_code=part.has_code,
                )
                for part in _split_code_unit(unit, token_counter, max_tokens)
                if part.text.strip()
            )
            continue

        candidate = "\n\n".join(
            [*(item.text.strip() for item in current), unit.text.strip()]
        )
        if current and token_counter.count(candidate) > max_tokens:
            flush()
        current.append(unit)

    flush()
    return packed


def enforce_max_tokens(
    drafts: list[ChunkDraft],
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[ChunkDraft]:
    """最終出力前に、コードフェンスを壊さずmax_tokensへ収める。"""
    enforced: list[ChunkDraft] = []
    for draft in drafts:
        if token_counter.count(draft.content) <= max_tokens:
            enforced.append(draft)
            continue
        units = _content_units_preserving_code(draft.content, token_counter, max_tokens)
        enforced.extend(_pack_atomic_units(units, draft, token_counter, max_tokens))
    return enforced


def merge_adjacent_drafts(
    drafts: list[ChunkDraft],
    token_counter: TokenCounter,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
) -> list[ChunkDraft]:
    """同一ページ内の短いチャンク候補を目標サイズまで結合する。"""
    merged: list[ChunkDraft] = []
    buffer: list[ChunkDraft] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            merged.append(_merge_drafts(buffer))
            buffer = []

    for draft in drafts:
        if not buffer:
            buffer = [draft]
            continue

        current = _merge_drafts(buffer)
        candidate = _merge_drafts([*buffer, draft])
        current_tokens = token_counter.count(current.content)
        candidate_tokens = token_counter.count(candidate.content)
        same_page = draft.page_index == buffer[-1].page_index

        should_merge = (
            same_page
            and candidate_tokens <= max_tokens
            and (current_tokens < min_tokens or candidate_tokens <= target_tokens)
        )

        if should_merge:
            buffer.append(draft)
        else:
            flush()
            buffer = [draft]

    flush()
    return merged


def build_chunk_drafts(
    blocks: list[Block],
    token_counter: TokenCounter,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    """全Markdownブロックをチャンク候補へ変換する。"""
    drafts: list[ChunkDraft] = []
    for segment in _iter_segments(blocks):
        drafts.extend(
            split_segment(
                segment,
                token_counter=token_counter,
                min_tokens=min_tokens,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    merged = merge_adjacent_drafts(
        drafts,
        token_counter=token_counter,
        min_tokens=min_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )
    return enforce_max_tokens(
        merged, token_counter=token_counter, max_tokens=max_tokens
    )


def _normalize_for_dedupe(text: str) -> str:
    """完全重複に近いチャンクを検出するため、空白だけ正規化する。"""
    return re.sub(r"\s+", " ", text).strip().lower()


def _heading_path_text(heading_path: tuple[str, ...]) -> str:
    return " > ".join(heading_path)


def _summary_short(content: str, heading_path: tuple[str, ...]) -> str:
    """外部モデルなしで短い説明文を作る。"""
    without_code = re.sub(r"```.*?```", " code block ", content, flags=re.DOTALL)
    lines = []
    for line in without_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if TABLE_SEPARATOR_RE.match(stripped):
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)

    head = _heading_path_text(heading_path)
    first = re.sub(r"\s+", " ", lines[0]).strip() if lines else ""
    if head and first:
        summary = f"{head}: {first}"
    else:
        summary = head or first
    return summary[:220].rstrip()


def strip_boilerplate_enum_runs(content: str, min_run: int = BOILERPLATE_MIN_RUN) -> str:
    """埋め込み用テキストから、連続する列挙定数の羅列を間引く。

    SDKリファレンスの`betas`パラメータのような列挙定数リストは、Java/Ruby/
    Python等の異なるエンドポイントページにバイト同一で繰り返される。
    このため各チャンクの埋め込みベクトルがこの定型部分に支配され、
    エンドポイント固有の内容（本来の識別情報）が埋没する。結果として、
    無関係な自然文クエリに対しても異常に高いコサイン類似度を示す
    「ハブ」的挙動を引き起こす（2026-07-18、q001で実測・
    docs/EVAL.md「既知の限界」参照）。

    表示用の``content``自体は変更せず、埋め込み計算にのみこの関数を通す
    （呼び出し側で``embed_text``として別に保持する）。
    """
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if ENUM_BULLET_RE.match(lines[i]):
            j = i
            while j < n and ENUM_BULLET_RE.match(lines[j]):
                j += 1
            run_len = j - i
            if run_len >= min_run:
                out.append(f"[{run_len}件の列挙定数を省略]")
            else:
                out.extend(lines[i:j])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _chunk_id(source_file: str, content: str) -> str:
    """内容だけで決まる安定ID。chunk_index（並び順）に依存させない。

    通し番号を含めると、ファイルのどこか1箇所が変わっただけで後続の
    全チャンクのchunk_indexがズレてIDが変わり、内容が同一のチャンクまで
    再embedding対象になってしまう。重複除外後は内容が全チャンクで一意な
    ので、source_fileとcontentのハッシュだけで安定・一意なIDになる。
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{source_file}:{digest}"


def materialize_records(
    drafts: list[ChunkDraft],
    source_file: str,
    token_counter: TokenCounter,
    page_urls: dict[int, str] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """重複除外後、JSONLに書けるdictへ変換する。"""
    seen: set[str] = set()
    unique: list[ChunkDraft] = []
    duplicate_removed = 0

    for draft in drafts:
        normalized = _normalize_for_dedupe(draft.content)
        if not normalized:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicate_removed += 1
            continue
        seen.add(digest)
        unique.append(draft)

    records: list[dict[str, object]] = []
    for chunk_index, draft in enumerate(unique):
        token_estimate = token_counter.count(draft.content)
        metadata = {
            "source_file": source_file,
            "page_index": draft.page_index,
            "heading_path": _heading_path_text(draft.heading_path),
            "chunk_index": chunk_index,
            "char_count": len(draft.content),
            "token_estimate": token_estimate,
            "has_code": draft.has_code,
            "summary_short": _summary_short(draft.content, draft.heading_path),
        }
        if page_urls and draft.page_index in page_urls:
            metadata["source_url"] = page_urls[draft.page_index]
        record: dict[str, object] = {
            "id": _chunk_id(source_file, draft.content),
            "content": draft.content,
            "metadata": metadata,
        }
        embed_text = strip_boilerplate_enum_runs(draft.content)
        if embed_text != draft.content:
            record["embed_text"] = embed_text
        records.append(record)

    return records, duplicate_removed


def write_jsonl(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stats_path_value(path: Path) -> str:
    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError:
        return resolved.name


def reuse_existing_ids(records: list[dict[str, object]], output_path: Path) -> int:
    """再分割前と本文が同じチャンクへ既存IDを引き継ぐ。"""
    if not output_path.exists():
        return 0
    existing_by_content: dict[str, str] = {}
    with output_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            existing = json.loads(line)
            normalized = _normalize_for_dedupe(str(existing.get("content", "")))
            if normalized and existing.get("id"):
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                existing_by_content[digest] = str(existing["id"])

    reused = 0
    for record in records:
        normalized = _normalize_for_dedupe(str(record.get("content", "")))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        existing_id = existing_by_content.get(digest)
        if existing_id:
            record["id"] = existing_id
            reused += 1
    return reused


def write_stats(
    records: list[dict[str, object]],
    stats_path: Path,
    source_file: str,
    input_char_count: int,
    block_count: int,
    page_count: int,
    duplicate_removed: int,
    token_counter: TokenCounter,
    args: argparse.Namespace,
) -> dict[str, object]:
    token_counts = [
        int(record["metadata"]["token_estimate"])  # type: ignore[index]
        for record in records
    ]
    code_chunk_count = sum(
        1 for record in records if bool(record["metadata"]["has_code"])  # type: ignore[index]
    )

    stats: dict[str, object] = {
        "source_file": source_file,
        "output_file": _stats_path_value(args.output),
        "stats_file": _stats_path_value(stats_path),
        "token_counter": token_counter.name,
        "input_char_count": input_char_count,
        "block_count": block_count,
        "page_count": page_count,
        "chunk_count": len(records),
        "average_token_count": (
            round(statistics.mean(token_counts), 2) if token_counts else 0
        ),
        "max_token_count": max(token_counts) if token_counts else 0,
        "code_chunk_count": code_chunk_count,
        "duplicate_removed_count": duplicate_removed,
        "oversize_chunk_count": sum(
            1 for count in token_counts if count > args.max_tokens
        ),
        "parameters": {
            "min_tokens": args.min_tokens,
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
        },
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def require_hf_token_counter(counter: TokenCounter, requested_model: str) -> None:
    """HF tokenizerが要求されたのにロードできなかった場合は即座に失敗させる。

    黙ってtiktoken/文字数4分の1へ劣化すると、チャンクがembeddingモデルの
    入力上限512トークンを超えても気づけず、超過分が無警告で切り捨てられて
    検索精度が壊れる（HANDOFF.md §1の再発。実際に2026-07-13の実行が
    char_count/4へ黙って劣化し、30,470チャンク中59.2%が実測512超過の
    汚染データを生成した）。概算で分割したい場合は ``--hf-tokenizer ""``
    で明示的にHF計測を無効化すること。
    """
    if requested_model and not counter.name.startswith("hf:"):
        raise SystemExit(
            f"HF tokenizer '{requested_model}' がロードできませんでした"
            f"（現在: {counter.name}）。このまま進むとチャンクの実トークン数を"
            "計測できず、embedding時に512トークン超過分が黙って切り捨てられます。"
            "transformersのインストールとモデルキャッシュ"
            "（EMBED_LOCAL_FILES_ONLY=0で初回ダウンロード）を確認してください。"
            '概算での分割を明示的に許可する場合のみ --hf-tokenizer "" を指定してください。'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="llms-full.txtをMarkdown構造を保ってRAG向けJSONLへ分割する"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="入力Markdown"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="出力JSONL")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS, help="統計JSON")
    parser.add_argument(
        "--reuse-ids-from",
        type=Path,
        default=None,
        help="本文が同じチャンクのIDを引き継ぐ既存JSONL（既定: 出力先）",
    )
    parser.add_argument("--encoding", default="utf-8", help="入力ファイルの文字コード")
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS)
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument(
        "--hf-tokenizer",
        default=DEFAULT_HF_TOKENIZER,
        help="token数の実測に使うHF tokenizer名（空文字で無効化しfallbackを使う）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"入力ファイルが見つかりません: {args.input}")

    raw_text = args.input.read_text(encoding=args.encoding)
    token_counter = TokenCounter(
        args.tiktoken_encoding, hf_model=args.hf_tokenizer or None
    )
    require_hf_token_counter(token_counter, args.hf_tokenizer)
    print(f"token counter: {token_counter.name}")
    blocks = parse_markdown_blocks(raw_text)
    drafts = build_chunk_drafts(
        blocks,
        token_counter=token_counter,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    source_file = (
        "llms-full.txt"
        if args.input.stem == "llms-full"
        else str(args.input).replace("\\", "/")
    )
    records, duplicate_removed = materialize_records(
        drafts,
        source_file,
        token_counter,
        page_urls=extract_page_urls(raw_text),
    )
    reused_ids = reuse_existing_ids(records, args.reuse_ids_from or args.output)
    write_jsonl(records, args.output)
    page_count = (
        (max((block.page_index for block in blocks), default=-1) + 1) if blocks else 0
    )
    stats = write_stats(
        records=records,
        stats_path=args.stats,
        source_file=source_file,
        input_char_count=len(raw_text),
        block_count=len(blocks),
        page_count=page_count,
        duplicate_removed=duplicate_removed,
        token_counter=token_counter,
        args=args,
    )

    print(
        "完了: "
        f"{stats['chunk_count']} chunks, "
        f"avg={stats['average_token_count']} tokens, "
        f"max={stats['max_token_count']} tokens, "
        f"duplicates_removed={stats['duplicate_removed_count']}"
        f", reused_ids={reused_ids}"
    )


if __name__ == "__main__":
    main()
