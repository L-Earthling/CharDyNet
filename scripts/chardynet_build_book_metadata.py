#!/usr/bin/env python3
"""
Build a corpus-level CSV of books and relationship statistics.
For each book directory it expects: a *_network.csv file w the canonical book filename, a *_relationship_stats.csv file w metrics.
The output CSV has one row per book w author, title and aggregated metrics.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def parse_author_and_title(network_filename: str) -> Tuple[Optional[str], str, Optional[str]]:
    #  Parse author name and book title from network CSV filename.
    #  Ex: "Jason-Reynolds_All-American-Boys_network.csv" -> ("Jason", "Reynolds", "All American Boys")
    base_name = network_filename.replace("_network.csv", "")

    try:
        parts = base_name.split("_", 1)
        if len(parts) != 2:
            return None, base_name, None

        author_part, title_part = parts
        author_components = author_part.split("-")

        if len(author_components) == 1:
            first_name = ""
            last_name = author_components[0]
        else:
            last_name = author_components[-1]
            first_name_parts = author_components[:-1]
            formatted_parts: List[str] = []

            for part in first_name_parts:
                if len(part) == 1:
                    formatted_parts.append(part)
                else:
                    formatted_parts.append(part)

            first_name = " ".join(formatted_parts)

        book_title = title_part.replace("-", " ")
        book_title = book_title.replace(" s ", "'s ")
        book_title = book_title.replace("Winters Tale", "Winter's Tale")
        book_title = book_title.replace("Gullivers Travels", "Gulliver's Travels")

        return first_name, last_name, book_title

    except Exception:
        return None, network_filename, None


def load_relationship_stats(stats_file_path: Path) -> Dict[str, float]:
    # Load relationship statistics from a CSV into a dictionary.
    df = pd.read_csv(stats_file_path)
    stats_dict: Dict[str, float] = {}
    for _, row in df.iterrows():
        metric = row["Metric"]
        value = row["Value"]
        stats_dict[metric] = value
    return stats_dict


def process_corpus(base_path: Path, output_path: Path | None = None):
    # Process all book directories under base_path and create a CSV w author/title and relationship statistics.
    if not base_path.exists():
        print(f"Directory not found: {base_path}")
        return

    if output_path is None:
        output_path = base_path / "books_metadata.csv"

    error_log_path = base_path / "books_metadata_errors.txt"

    print(f"Processing corpus")
    print(f"Base directory: {base_path}")
    print(f"Output CSV: {output_path}")
    print(f"Error log: {error_log_path}")
    print("=" * 60)

    book_dirs = sorted(d for d in base_path.iterdir() if d.is_dir())
    print(f"Found {len(book_dirs)} book directories")

    books_data: List[Dict[str, object]] = []
    error_log: List[str] = []

    columns = [
        "Author First Name",
        "Author Last Name",
        "Book Title",
        "Total Relationship Observations",
        "Unique Relationships",
        "New Relationships",
        "Changed Relationships",
        "Positive Relationships",
        "Negative Relationships",
        "Neutral Relationships",
        "Average Confidence",
        "Total Chapters",
        "Unique Characters",
        "Avg Character Span",
    ]

    for i, book_dir in enumerate(book_dirs, 1):
        book_name = book_dir.name

        if i % 50 == 0:
            print(f"Processed {i}/{len(book_dirs)} books...")

        try:
            network_files = list(book_dir.glob("*_network.csv"))
            if not network_files:
                error_log.append(f"No network CSV found in {book_name}")
                book_data = {
                    "Author First Name": "",
                    "Author Last Name": book_name,
                    "Book Title": "",
                }
                for col in columns[3:]:
                    book_data[col] = "NA"
                books_data.append(book_data)
                continue

            network_file = network_files[0]
            first_name, last_name, book_title = parse_author_and_title(network_file.name)
            if first_name is None:
                error_log.append(f"Failed to parse author/title from {network_file.name}")

            book_data = {
                "Author First Name": first_name or "",
                "Author Last Name": last_name or book_name,
                "Book Title": book_title or "",
            }

            stats_files = list(book_dir.glob("*_relationship_stats.csv"))
            if not stats_files:
                error_log.append(f"No relationship stats CSV found in {book_name}")
                for col in columns[3:]:
                    book_data[col] = "NA"
            else:
                stats_file = stats_files[0]
                try:
                    stats_dict = load_relationship_stats(stats_file)
                except Exception as exc:
                    error_log.append(f"Error reading stats from {book_name}: {exc}")
                    for col in columns[3:]:
                        book_data[col] = "NA"
                else:
                    stats_mapping = {
                        "Total Relationship Observations": "Total Relationship Observations",
                        "Unique Relationships": "Unique Relationships",
                        "New Relationships": "New Relationships",
                        "Changed Relationships": "Changed Relationships",
                        "Positive Relationships": "Positive Relationships",
                        "Negative Relationships": "Negative Relationships",
                        "Neutral Relationships": "Neutral Relationships",
                        "Average Confidence": "Average Confidence",
                        "Total Chapters": "Total Chapters",
                        "Unique Characters": "Unique Characters",
                        "Avg Character Span": "Avg Character Span",
                    }

                    for csv_col, stats_key in stats_mapping.items():
                        if stats_key in stats_dict:
                            value = stats_dict[stats_key]
                            if csv_col == "Average Confidence" and isinstance(value, (int, float)):
                                book_data[csv_col] = f"{value:.3f}"
                            else:
                                book_data[csv_col] = value
                        else:
                            book_data[csv_col] = "NA"
                            error_log.append(f"Missing metric '{stats_key}' in {book_name}")

            books_data.append(book_data)

        except Exception as exc:
            error_msg = f"Unexpected error processing {book_name}: {exc}"
            error_log.append(error_msg)
            book_data = {
                "Author First Name": "",
                "Author Last Name": book_name,
                "Book Title": f"Error: {exc}",
            }
            for col in columns[3:]:
                book_data[col] = "NA"
            books_data.append(book_data)

    df = pd.DataFrame(books_data)
    for col in columns:
        if col not in df.columns:
            df[col] = "NA"
    df = df[columns]
    df.to_csv(output_path, index=False)

    if error_log:
        with error_log_path.open("w", encoding="utf-8") as f:
            f.write("Processing error log\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Total errors: {len(error_log)}\n")
            f.write("=" * 50 + "\n\n")
            for error in error_log:
                f.write(f"{error}\n")

    valid_entries = sum(1 for book in books_data if book["Author Last Name"] != "NA")
    print("=" * 60)
    print("Processing complete")
    print(f"Total books processed: {len(books_data)}")
    print(f"Books with author information: {valid_entries}")
    print(f"CSV saved to: {output_path}")
    if error_log:
        print(f"Error log saved to: {error_log_path}")
    print("Sample rows:")
    print(df.head(3).to_string(index=False))

    return df, error_log


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build corpus-level book metadata CSV.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("processed_books"),
        help="Directory containing one subdirectory per book.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output CSV. Defaults to <base-dir>/books_metadata.csv",
    )
    args = parser.parse_args()

    process_corpus(args.base_dir, args.output)


if __name__ == "__main__":
    main()
