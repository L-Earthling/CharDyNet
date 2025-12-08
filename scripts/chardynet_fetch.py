#!/usr/bin/env python3
"""
Fetch chapter or section summaries from a literature index page.
The script:
  - crawls a single index page that lists literary works
  - extracts canonical work URLs
  - for each work, walks section1, section2, ... pages
  - writes one markdown text file per work w all sections
Dependencies: crawl4ai
"""

import asyncio
import re
import string
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Generate a safe slug from a title for use in file and folder names.
def slugify(text: str) -> str:
    valid = "-_()%s%s" % (string.ascii_letters, string.digits)
    return "".join(c if c in valid else "_" for c in text).strip("_")

# Return a sorted list of canonical work URLs extracted from an index page.
async def get_work_links(index_url: str, crawler) -> list[str]:
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    result = await crawler.arun(url=index_url, config=run_conf)
    md = result.markdown.raw_markdown

    # Grab any /lit/... link-like pattern
    rels = re.findall(r"\[[^\]]+\]\((?:https?://[^)]+)?(/lit/[^)]+)\)", md)
    candidates = {urlparse(index_url)._replace(path=r, query="", fragment="").geturl() for r in rels}

    def is_canonical(url: str) -> bool:
        p = urlparse(url)
        return (
            p.scheme in ("http", "https")
            and re.fullmatch(r"/lit/[A-Za-z0-9\-_]+/", p.path)
            and not p.query
        )

    works = sorted(u for u in candidates if is_canonical(u))
    print(f"Extracted {len(works)} work URLs.")
    return works

# Try /section1/, /section2/, ... until a page is missing or an analysis page is hit. Returns a list of (section_number, raw_markdown).
async def fetch_section_pages(crawler, work_url: str, max_sections: int = 100) -> list[tuple[int, str]]:
    sections: list[tuple[int, str]] = []
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    base = work_url.rstrip("/")

    for i in range(1, max_sections + 1):
        url = f"{base}/section{i}/"
        print(f"  Trying {url} ... ", end="")
        try:
            result = await crawler.arun(url=url, config=run_conf)
        except Exception:
            print("request failed.")
            break

        if not result.success:
            print("no page.")
            break

        md = result.markdown.raw_markdown
        # stop if this looks like an analysis page rather than a section
        if re.search(r"^#\s*(Analysis|Study Questions)", md, re.MULTILINE):
            print("hit analysis page, stopping.")
            break

        print("ok.")
        sections.append((i, md))

    return sections

# Main scraping orchestration.
async def run(index_url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    browser_conf = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        work_urls = await get_work_links(index_url, crawler)
        if not work_urls:
            print("No works found, aborting.")
            return

        print(f"{len(work_urls)} works found. Processing them in order.")

        for work_url in work_urls:
            slug = slugify(Path(work_url.rstrip("/")).name)
            out_dir = output_dir / slug
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{slug}.txt"

            print(f"\nFetching sections for '{slug}'")
            sections = await fetch_section_pages(crawler, work_url)
            if not sections:
                print(f"No section pages for {slug}, skipping.")
                continue

            with out_path.open("w", encoding="utf-8") as f:
                for sec_num, md in sections:
                    f.write(f"# Section {sec_num}\n\n")
                    f.write(md.strip() + "\n\n")

            print(f"Saved {len(sections)} sections to {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch section summaries from a literature index page.")
    parser.add_argument(
        "--index-url",
        type=str,
        required=True,
        help="URL of the index page listing the works.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw_summaries"),
        help="Directory where per-work text files will be stored.",
    )
    args = parser.parse_args()

    asyncio.run(run(args.index_url, args.output_dir))


if __name__ == "__main__":
    main()
