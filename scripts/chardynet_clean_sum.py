#!/usr/bin/env python3
"""
Clean raw summary files and produce canonical per-book text files.
Steps:
  1. For each raw .txt file, extract author and title from headers
  2. Identify valid section headers and paragraphs, drop boilerplate
  3. Write cleaned file <Author>_<Title>_clean.txt
  4. Remove any lines starting w 'Summary ' from clean files
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

WINDOW_LOOKBACK = 10

AUTHOR_HDR_RE = re.compile(r"^##\s+(.+)$")
TITLE_HDR_RE = re.compile(
    r"^#\s*\[\s*(.+?)\s*\]\("
    r"https?://[^/]+/(?:lit|drama)/[^/]+/+\)"
)
HEADER_RE = re.compile(r"^###\s+(.+)$")
VALID_HDR_PREFIXES = (
    "Summary",
    "Chapter",
    "Prologue",
    "Book",
    "Part",
    "Act",
    "Scene",
    "Introduction",
)

NAV_LINK_RE = re.compile(r"\[ *Previous.*?\]\(.*?\)|\[ *Next.*?\]\(.*?\)")
BLOCKQUOTE_LINK_RE = re.compile(r"^>\s*\[See.*\]")
IMAGE_RE = re.compile(r"^!\[.*\]\(.*\)")
UNWANTED_STARTS = [
    "See Important Quotations Explained",
    "Read important quotes about",
    "Read more about",
    "Read other important quotes by and about",
    "Content Warning:",
    "Read a translation of",
    "PLUSTest your comprehension of",
    "Read a full summary of",
    "Read the (",
    "This is an abridged summary of",
]

INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")


# Convert a string to a safe filename component
def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9\-]", "", s)

# Extract author and title from the top of the markdown file.
def extract_metadata(lines: List[str]) -> Tuple[str | None, str | None]:
    author = None
    title = None

    for i, line in enumerate(lines):
        if line.strip() == "Study Guide" and i > 0:
            candidate = lines[i - 1].strip()
            candidate = INLINE_LINK_RE.sub(r"\1", candidate)
            m = AUTHOR_HDR_RE.match(candidate)
            if m:
                author = m.group(1).strip()

        if title is None:
            m2 = TITLE_HDR_RE.match(line)
            if m2:
                title = m2.group(1).strip()

        if author and title:
            break

    return author, title

#  Find valid section headers and collect paragraphs under then, filtering boilerplate and deduplicating.
def extract_summary_blocks(lines: List[str]) -> List[str]:
    blocks: List[str] = []
    seen = set()
    i = 0
    n = len(lines)

    while i < n:
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue

        hdr = m.group(1).strip()
        if not any(hdr.startswith(pref) for pref in VALID_HDR_PREFIXES):
            i += 1
            continue

        pre_summary = None
        for j in range(max(0, i - WINDOW_LOOKBACK), i):
            txt = lines[j].strip()
            if txt.startswith("Summary "):
                pre_summary = txt
                break

        headers = [f"### {hdr}"]
        if hdr.startswith("Summary") and i + 1 < n:
            m2 = HEADER_RE.match(lines[i + 1])
            if m2:
                next_hdr = m2.group(1).strip()
                if any(next_hdr.startswith(pref) for pref in VALID_HDR_PREFIXES):
                    headers.append(f"### {next_hdr}")
                    i += 1

        i += 1
        paras: List[str] = []

        while i < n and not HEADER_RE.match(lines[i]):
            text = lines[i].rstrip("\n")
            stripped = text.strip()
            if (
                not stripped
                or NAV_LINK_RE.search(text)
                or BLOCKQUOTE_LINK_RE.match(text)
                or IMAGE_RE.match(text)
                or any(stripped.startswith(ws) for ws in UNWANTED_STARTS)
            ):
                i += 1
                continue

            text = INLINE_LINK_RE.sub(r"\1", text)
            first_sent = text.split(". ", 1)[0]
            key = (tuple(headers), first_sent)
            if key not in seen:
                paras.append(text)
                seen.add(key)
            i += 1

        if paras:
            block_lines: List[str] = []
            if pre_summary:
                block_lines.append(pre_summary)
            block_lines.extend(headers)
            block_lines.extend(paras)
            blocks.append("\n".join(block_lines))

    return blocks

# Create the <Author>_<Title>_clean.txt file for one raw summary
def write_clean_file(original_path: Path, lines: List[str]) -> Path:
    author, title = extract_metadata(lines)
    if not author or not title:
        print(f"Skipping {original_path.name}: missing metadata")
        return Path()

    blocks = extract_summary_blocks(lines)
    if not blocks:
        print(f"No summary blocks in {original_path.name}")
        return Path()

    cleaned: List[str] = [f"Author: {author}", f"Book Title: {title}", ""]
    for blk in blocks:
        cleaned.append(blk)
        cleaned.append("")

    safe_author = sanitize_filename(author)
    safe_title = sanitize_filename(title)
    out_fn = f"{safe_author}_{safe_title}_clean.txt"
    out_path = original_path.with_name(out_fn)

    out_path.write_text("\n".join(cleaned), encoding="utf-8")
    print(f"Created {out_fn}")
    return out_path

# Remove lines starting w 'Summary ' from all *_clean.txt files.
def strip_summary_leading_lines(root_dir: Path) -> None:
    for txt in root_dir.rglob("*_clean.txt"):
        if txt.name.endswith("_characters_clean.txt"):
            continue
        lines = txt.read_text(encoding="utf-8").splitlines(keepends=True)
        filtered = [ln for ln in lines if not ln.startswith("Summary ")]
        txt.write_text("".join(filtered), encoding="utf-8")
        print(f"Cleaned Summary markers in {txt}")


def run(base_dir: Path) -> None:
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith(".txt") or fn.endswith("_clean.txt"):
                continue
            path = Path(root) / fn
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            write_clean_file(path, lines)

    strip_summary_leading_lines(base_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw summary files into canonical per-book files.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("raw_summaries"),
        help="Root directory containing per-book raw summary files.",
    )
    args = parser.parse_args()

    if not args.base_dir.is_dir():
        print(f"Error: {args.base_dir} is not a directory.")
        raise SystemExit(1)

    run(args.base_dir)


if __name__ == "__main__":
    main()
