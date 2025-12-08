#!/usr/bin/env python3
# Scan per-book relationship statistics to find books w large character networks.

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def find_large_character_networks(base_path: Path, threshold: int = 20):
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Directory not found: {base_path}")
        return [], []

    print(f"Analyzing corpus for books w more than {threshold} characters")
    print(f"Base directory: {base_path}")
    print("=" * 60)

    large_networks: List[Tuple[str, int, str]] = []
    small_networks: List[Tuple[str, int]] = []
    no_stats: List[str] = []
    error_books: List[str] = []

    book_dirs = sorted(d for d in base_path.iterdir() if d.is_dir())
    print(f"Found {len(book_dirs)} book directories")
    print("\nAnalyzing character counts...")

    for book_dir in book_dirs:
        book_name = book_dir.name
        stats_files = list(book_dir.glob("*_relationship_stats.csv"))
        if not stats_files:
            no_stats.append(book_name)
            continue

        stats_file = stats_files[0]
        try:
            df = pd.read_csv(stats_file)
            row = df[df["Metric"] == "Unique Characters"]
            if row.empty:
                error_books.append(f"{book_name}: 'Unique Characters' metric not found")
                continue
            char_count = int(row["Value"].iloc[0])
            if char_count > threshold:
                large_networks.append((book_name, char_count, stats_file.stem))
                print(f"{book_name}: {char_count} characters")
            else:
                small_networks.append((book_name, char_count))
        except Exception as exc:
            error_books.append(f"{book_name}: error reading stats - {exc}")

    print("\n" + "=" * 60)
    print("Analysis summary")
    print("=" * 60)
    print(f"Books above threshold ({threshold}): {len(large_networks)}")
    print(f"Books at or below threshold: {len(small_networks)}")
    print(f"Books without stats files: {len(no_stats)}")
    print(f"Books with errors: {len(error_books)}")

    all_counts = [c for _, c in small_networks] + [c for _, c, _ in large_networks]
    if all_counts:
        print("\nCharacter count statistics:")
        print(f"  Total books analyzed: {len(all_counts)}")
        print(f"  Average characters: {sum(all_counts) / len(all_counts):.1f}")
        print(f"  Range: {min(all_counts)} - {max(all_counts)}")

    return large_networks, small_networks


def main() -> None:
    base_path = Path("processed_books")
    threshold = 20
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        threshold = int(sys.argv[2])

    find_large_character_networks(base_path, threshold)


if __name__ == "__main__":
    main()
