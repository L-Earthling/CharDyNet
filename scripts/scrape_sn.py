#!/usr/bin/env python3
# coding: utf-8

import os
import string
import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

BASE_INDEX_URL = "https://www.sparknotes.com/lit/?scr=1&genreSelected=novels"
OUTPUT_DIR = Path("sparknotes_summaries")
OUTPUT_DIR.mkdir(exist_ok=True)

# Helper function to slugify novel titles for directory names
def slugify(text: str) -> str:
    valid = "-_()%s%s" % (string.ascii_letters, string.digits)
    return "".join(c if c in valid else "_" for c in text).strip("_")

async def get_novel_links(crawler) -> list[str]:
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    result = await crawler.arun(url=BASE_INDEX_URL, config=run_conf)
    md = result.markdown.raw_markdown

    # Grab any /lit/... link
    rels = re.findall(
        r"\[[^\]]+\]\((?:https?://www\.sparknotes\.com)?(/lit/[^)]+)\)",
        md
    )
    candidates = { "https://www.sparknotes.com" + r for r in rels }

    # Keep only URLs whose path is exactly /lit/<slug>/ with no query
    def is_novel(url: str) -> bool:
        p = urlparse(url)
        return (
            p.scheme in ("http", "https")
            and p.netloc == "www.sparknotes.com"
            and re.fullmatch(r"/lit/[A-Za-z0-9\-_]+/", p.path)
            and p.query == ""
        )

    novels = sorted(u for u in candidates if is_novel(u))
    print(f"[DEBUG] Extracted {len(novels)} novel URLs, sample: {novels[:5]}")
    return novels

async def fetch_chapter_sections(crawler, novel_url: str) -> list[tuple[int,str]]:
    """
    Try /section1/, /section2/, … until we either 404 or encounter an Analysis page.
    Return list of (section_number, raw_markdown).
    """
    sections = []
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    base = novel_url.rstrip("/")
    for i in range(1, 101):                # assume <100 chapters
        url = f"{base}/section{i}/"
        print(f"  → Trying {url}", end=" … ")
        result = await crawler.arun(url=url, config=run_conf)
        if not result.success:
            print("no page.")
            break

        md = result.markdown.raw_markdown
        # if this page is actually an “Analysis” or the full summary, stop.
        if re.search(r"^#\s*(Analysis|Study Questions)", md, re.MULTILINE):
            print("hit Analysis, stopping.")
            break

        print("OK.")
        sections.append((i, md))

    return sections

#  Main orchestration — loop novels, fetch sections, save one .txt each
async def main():
    browser_conf = BrowserConfig(headless=True)
    
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # 1) get all novel index URLs
        novel_urls = await get_novel_links(crawler)
        if not novel_urls:
            print("‼️ No novels found, aborting.")
            return
        print(f" {len(novel_urls)} novels found; first three:", novel_urls[:3])

        # 2) for each novel, fetch numbered sections
        for novel_url in novel_urls:
            slug = slugify(Path(novel_url.rstrip("/")).name)
            out_dir = OUTPUT_DIR / slug
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{slug}.txt"

            print(f"\n Crawling '{slug}'")
            secs = await fetch_chapter_sections(crawler, novel_url)
            if not secs:
                print(f"  No true section pages for {slug}, skipping.")
                continue

            # 3) write them all into one file
            with open(out_path, "w", encoding="utf-8") as f:
                for sec_num, md in secs:
                    f.write(f"# Section {sec_num}\n\n")
                    f.write(md.strip() + "\n\n")

            print(f" Saved {len(secs)} sections to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())

