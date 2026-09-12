"""llms-full.txtの分割を、中断・再開可能なパート単位で実行するドライバー。

split_llms_full.pyの一括実行が（メモリ・強制終了などで）完走しない環境向け。

1. llms-full.txtをページ境界で約20パートへ分けて作業ディレクトリに保存
2. 未処理のパートだけを分割ロジックへかけて中間JSONを保存
3. 全パート完了後、パート横断で重複除外・採番し直して data/processed/ へマージ

途中で強制終了されても、再実行すれば処理済みパートはスキップされる。
min/target/max/overlap-tokensはsplit_llms_full.pyと同じCLI引数で調整できる。
前回と異なるパラメータで実行すると、中間ファイルが古い基準のままなのを防ぐため
自動的に作り直す（--cleanと同じ扱い）。

使い方:
    python scripts/shard_split.py                     # 実行（killされたら再実行で再開）
    python scripts/shard_split.py --target-tokens 300 # パラメータを変えて再分割
    python scripts/shard_split.py --clean             # 中間ファイルを消して最初から
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import split_llms_full as sp

DEFAULT_INPUT = Path("llms-full.txt")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_STATS = Path("data/processed/stats.json")
DEFAULT_WORK = Path("data/processed/split-parts")
PART_COUNT = 20


def _token_params(args: argparse.Namespace) -> dict:
    """中間キャッシュの有効性判定に使うパラメータの指紋。"""
    return {
        "min_tokens": args.min_tokens,
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "hf_tokenizer": args.hf_tokenizer,
    }


def make_parts(input_path: Path, work: Path, token_params: dict) -> dict:
    """ページ境界でパートファイルを作る。

    前回と同じtoken_paramsならmanifestを返すだけ。パラメータが変わっていたら
    （チャンクサイズ調整などで）中間ファイルは古い基準のままなので作り直す。
    """
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("token_params") == token_params:
            return manifest
        print(
            "パラメータ変更を検知したため中間ファイルを再構築します: "
            f"{manifest.get('token_params')} -> {token_params}",
            flush=True,
        )
        shutil.rmtree(work)

    raw = input_path.read_text(encoding="utf-8")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    boundaries = [0]
    for index in range(len(lines)):
        if sp.PAGE_SEPARATOR_RE.match(lines[index]) and sp._is_page_separator(
            lines, index
        ):
            boundaries.append(index)
    boundaries.append(len(lines))

    total_pages = len(boundaries) - 1
    per_part = max(1, total_pages // PART_COUNT)
    work.mkdir(parents=True, exist_ok=True)

    parts = []
    page_offset = 0
    for part_index in range(0, total_pages, per_part):
        start_line = boundaries[part_index]
        end_line = boundaries[min(part_index + per_part, total_pages)]
        name = f"part_{len(parts):03d}"
        (work / f"{name}.txt").write_text(
            "\n".join(lines[start_line:end_line]), encoding="utf-8"
        )
        parts.append({"name": name, "page_offset": page_offset})
        page_offset += min(per_part, total_pages - part_index)

    manifest = {
        "parts": parts,
        "input_char_count": len(raw),
        "token_params": token_params,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"created {len(parts)} parts ({total_pages} pages)", flush=True)
    return manifest


def process_parts(
    manifest: dict,
    work: Path,
    counter: sp.TokenCounter,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    """未処理パートを分割して中間JSONへ保存する（処理済みはスキップ）。"""
    for part in manifest["parts"]:
        out_path = work / f"{part['name']}.records.json"
        if out_path.exists():
            continue
        started = time.time()
        text = (work / f"{part['name']}.txt").read_text(encoding="utf-8")
        blocks = sp.parse_markdown_blocks(text)
        drafts = sp.build_chunk_drafts(
            blocks,
            token_counter=counter,
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        # パート内の相対page_indexでURLを対応付ける（draftsも同じ相対値を持つ）。
        # metadata["page_index"]への絶対値オフセットはこの後で適用される。
        page_urls = sp.extract_page_urls(text)
        records, dup = sp.materialize_records(
            drafts, "llms-full.txt", counter, page_urls=page_urls
        )
        for record in records:
            record["metadata"]["page_index"] += part["page_offset"]
        payload = {
            "records": records,
            "duplicate_removed": dup,
            "block_count": len(blocks),
            "page_count": (max((b.page_index for b in blocks), default=-1) + 1),
        }
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)
        print(
            f"{part['name']}: {len(records)} records, dup={dup}, "
            f"{time.time() - started:.0f}s",
            flush=True,
        )


def merge(
    manifest: dict,
    work: Path,
    counter: sp.TokenCounter,
    output: Path,
    stats_path: Path,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    """全パートの中間JSONを重複除外・採番し直してマージする。"""
    seen: set[str] = set()
    merged: list[dict] = []
    duplicate_removed = 0
    block_count = 0
    page_count = 0

    for part in manifest["parts"]:
        payload = json.loads(
            (work / f"{part['name']}.records.json").read_text(encoding="utf-8")
        )
        duplicate_removed += payload["duplicate_removed"]
        block_count += payload["block_count"]
        page_count += payload["page_count"]
        for record in payload["records"]:
            normalized = sp._normalize_for_dedupe(record["content"])
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicate_removed += 1
                continue
            seen.add(digest)
            merged.append(record)

    for chunk_index, record in enumerate(merged):
        record["metadata"]["chunk_index"] = chunk_index
        record["id"] = sp._chunk_id("llms-full.txt", record["content"])

    sp.write_jsonl(merged, output)
    stats = sp.write_stats(
        records=merged,
        stats_path=stats_path,
        source_file="llms-full.txt",
        input_char_count=manifest["input_char_count"],
        block_count=block_count,
        page_count=page_count,
        duplicate_removed=duplicate_removed,
        token_counter=counter,
        args=SimpleNamespace(
            output=output,
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
    )
    print(
        f"merged: {stats['chunk_count']} chunks, avg={stats['average_token_count']}, "
        f"max={stats['max_token_count']}, oversize={stats['oversize_chunk_count']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="llms-full.txtを再開可能に分割する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--min-tokens", type=int, default=sp.MIN_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=sp.TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=sp.MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=sp.OVERLAP_TOKENS)
    parser.add_argument("--hf-tokenizer", default=sp.DEFAULT_HF_TOKENIZER)
    parser.add_argument(
        "--clean", action="store_true", help="中間ファイルを削除して最初からやり直す"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
        print(f"cleaned: {args.work_dir}", flush=True)

    counter = sp.TokenCounter(hf_model=args.hf_tokenizer or None)
    sp.require_hf_token_counter(counter, args.hf_tokenizer)
    print(f"token counter: {counter.name}", flush=True)

    token_params = _token_params(args)
    manifest = make_parts(args.input, args.work_dir, token_params)
    process_parts(
        manifest,
        args.work_dir,
        counter,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    if all(
        (args.work_dir / f"{p['name']}.records.json").exists()
        for p in manifest["parts"]
    ):
        merge(
            manifest,
            args.work_dir,
            counter,
            args.output,
            args.stats,
            min_tokens=args.min_tokens,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
        )
    else:
        print("incomplete: rerun to resume", flush=True)


if __name__ == "__main__":
    main()
