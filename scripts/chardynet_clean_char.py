#!/usr/bin/env python3
"""
Clean character-list pages into canonical per-book character files.
For each *_characters.txt file this script:
  - loads author and title from the corresponding *_clean.txt summary file
  - trims note sections and navigation links
  - parses each character heading and its description paragraphs
  - writes <Author>_<Title>_characters_clean.txt
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
NAV_LINK_RE = re.compile(r"\[ *Next.*?\]\(.*?\)")
IMAGE_RE = re.compile(r"^!\[.*\]\(.*\)")
ANALYSIS_LINK_RE = re.compile(r"^\[Read an in-depth analysis")
CHAR_HDR_RE = re.compile(r"^(?:\*\*(.+?)\*\*|###\s*(.+))\s*$")
CHAR_LIST_HDR_RE = re.compile(r"^Characters\s+(.+)\s+Character List$")
NOTE_MARKER_RE = re.compile(r"^Did you know you can highlight text to take a note\?")


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9\-]", "", s)


def load_metadata_from_summary(root: Path) -> Tuple[str | None, str | None]:
    for fn in root.iterdir():
        if fn.name.endswith("_clean.txt") and "characters" not in fn.name:
            with fn.open(encoding="utf-8") as f:
                author = f.readline().strip().removeprefix("Author: ").strip()
                title = f.readline().strip().removeprefix("Book Title: ").strip()
            return author, title
    return None, None


def clean_characters_file(path: Path) -> None:
    root = path.parent
    author, title = load_metadata_from_summary(root)
    if not author or not title:
        print(f"Skipping {path.name}: no summary_clean file for metadata")
        return

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines: List[str] = []
    for line in raw_lines:
        if NOTE_MARKER_RE.match(line):
            break
        lines.append(line)

    char_list_hdr = None
    for line in lines:
        m = CHAR_LIST_HDR_RE.match(line.strip())
        if m:
            char_list_hdr = INLINE_LINK_RE.sub(r"\1", line.strip())
            break

    if not char_list_hdr:
        char_list_hdr = f"Characters {title} Character List"

    cleaned: List[str] = [
        f"Author: {author}",
        f"Book Title: {title}",
        "",
        char_list_hdr,
        "",
    ]

    entries: List[Tuple[str, List[str]]] = []
    current_name: str | None = None
    current_para: List[str] = []

    for line in lines:
        s = line.strip()
        if NAV_LINK_RE.search(s) or IMAGE_RE.match(s) or ANALYSIS_LINK_RE.match(s):
            continue

        m = CHAR_HDR_RE.match(s)
        if m:
            if current_name and current_para:
                entries.append((current_name, current_para))
            name = m.group(1) or m.group(2)
            current_name = name.strip()
            current_para = []
            continue

        if current_name and s:
            txt = INLINE_LINK_RE.sub(r"\1", s)
            current_para.append(txt)

    if current_name and current_para:
        entries.append((current_name, current_para))

    for name, paras in entries:
        cleaned.append(f"### {name}")
        for p in paras:
            cleaned.append(p)
        cleaned.append("")

    safe_author = sanitize_filename(author)
    safe_title = sanitize_filename(title)
    out_fn = f"{safe_author}_{safe_title}_characters_clean.txt"
    out_path = root / out_fn

    out_path.write_text("\n".join(cleaned), encoding="utf-8")
    print(f"Created {out_fn}")


def run(base_dir: Path) -> None:
    for root, _, files in os.walk(base_dir):
        root_path = Path(root)
        for fn in files:
            if fn.endswith("_characters.txt") and not fn.endswith("_characters_clean.txt"):
                clean_characters_file(root_path / fn)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean character list files into canonical per-book format.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("raw_summaries"),
        help="Root directory containing per-book summary and character files.",
    )
    args = parser.parse_args()

    if not args.base_dir.is_dir():
        print(f"Error: {args.base_dir} is not a directory.")
        raise SystemExit(1)

    run(args.base_dir)


if __name__ == "__main__":
    main()
