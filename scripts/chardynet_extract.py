"""
CharDyNet – Relationship Extraction Module

This module implements the network / relation extraction phase of the CharDyNet pipeline. It faithfully preserves the behaviour of the prototype:
- Character parsing (with alias extraction)
- Partition parsing (### markers)
- Multi-API LLM-based relationship extraction
- Relationship history tracking
- Temporal metrics calculation
- Output CSV generation

Dependencies: pandas, numpy, networkx, openai, sqlite3
"""

from __future__ import annotations
import re
import os
import csv
import json
import time
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
from openai import OpenAI


# Logging

logger = logging.getLogger("CharDyNetExtraction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


# Data classes

@dataclass
class Character:
    name: str
    description: str
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

@dataclass
class Partition:
    number: int
    title: str
    content: str

@dataclass
class RelationshipHistory:
    chapter: int
    char_a: str
    char_b: str
    relationship: str
    previous_relationship: Optional[str] = None
    change_type: Optional[str] = None
    confidence: float = 1.0


# Progress tracking db

# Tracks per-book and per-partition processing state.
class ProgressTracker:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_status (
                book_name TEXT PRIMARY KEY,
                status TEXT,
                total_partitions INTEGER,
                completed_partitions INTEGER,
                start_time REAL,
                last_updated REAL,
                error_message TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS partition_status (
                book_name TEXT,
                partition_number INTEGER,
                status TEXT,
                llm_calls INTEGER,
                relationships_extracted INTEGER,
                processing_time REAL,
                PRIMARY KEY (book_name, partition_number)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_name TEXT,
                partition_number INTEGER,
                timestamp REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                processing_time REAL,
                model_name TEXT,
                success INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def get_book_status(self, book_name: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT status FROM book_status WHERE book_name = ?", (book_name,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def set_book_status(self, book_name: str, status: str, total_partitions: int = 0, error_msg: str = None):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        now = time.time()

        cur.execute("""
            INSERT OR REPLACE INTO book_status
            (book_name, status, total_partitions, completed_partitions, start_time, last_updated, error_message)
            VALUES (
                ?, ?, ?, 
                COALESCE((SELECT completed_partitions FROM book_status WHERE book_name=?), 0),
                COALESCE((SELECT start_time FROM book_status WHERE book_name=?), ?),
                ?, ?
            )
        """, (book_name, status, total_partitions, book_name, book_name, now, now, error_msg))

        conn.commit()
        conn.close()

    def mark_partition_completed(self, book_name: str, partition_number: int, llm_calls: int, relationships_extracted: int, proc_time: float):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO partition_status
            (book_name, partition_number, status, llm_calls, relationships_extracted, processing_time)
            VALUES (?, ?, 'completed', ?, ?, ?)
        """, (book_name, partition_number, llm_calls, relationships_extracted, proc_time))

        cur.execute("""
            UPDATE book_status
            SET completed_partitions = (
                SELECT COUNT(*) FROM partition_status
                WHERE book_name = ? AND status = 'completed'
            ), last_updated = ?
            WHERE book_name = ?
        """, (book_name, time.time(), book_name))

        conn.commit()
        conn.close()


# API client w rate imiting

@dataclass
class APIConfig:
    name: str
    api_key: str
    base_url: str
    model_name: str
    max_requests_per_minute: int
    max_requests_per_day: int
    max_requests_per_hour: Optional[int] = None


class MultiAPIRateLimiter:
    def __init__(self, config: APIConfig):
        self.config = config
        self.request_times_minute = []
        self.request_times_hour = []
        self.request_times_day = []
        self.lock = time

    def enforce(self):
        now = time.time()

        self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
        if len(self.request_times_minute) >= self.config.max_requests_per_minute:
            wait = 60 - (now - self.request_times_minute[0])
            time.sleep(wait)

        if self.config.max_requests_per_hour:
            self.request_times_hour = [t for t in self.request_times_hour if now - t < 3600]
            if len(self.request_times_hour) >= self.config.max_requests_per_hour:
                wait = 3600 - (now - self.request_times_hour[0])
                time.sleep(wait)

        self.request_times_day = [t for t in self.request_times_day if now - t < 86400]
        if len(self.request_times_day) >= self.config.max_requests_per_day:
            wait = 86400 - (now - self.request_times_day[0])
            time.sleep(wait)

    def record(self):
        now = time.time()
        self.request_times_minute.append(now)
        self.request_times_hour.append(now)
        self.request_times_day.append(now)


class MultiAPILLMClient:
    def __init__(self, config: APIConfig, progress: ProgressTracker):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.ratelimiter = MultiAPIRateLimiter(config)
        self.progress = progress
        self.call_count = 0

    def call(self, messages, book_name, partition_number):
        for attempt in range(3):
            try:
                self.ratelimiter.enforce()
                start = time.time()
                self.call_count += 1

                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.model_name,
                    temperature=0.5,
                    max_tokens=1000
                )

                duration = time.time() - start
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)

                self.progress.track_llm_usage(
                    book_name, partition_number,
                    input_tokens, output_tokens, duration, True
                )

                self.ratelimiter.record()
                content = response.choices[0].message.content.strip()
                return content

            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


# Character alias extraction 

def extract_character_aliases(description: str) -> List[str]:
    aliases = []
    patterns = [
        r"also known as ([^.,]+)",
        r"nicknamed ([^.,]+)",
        r"called ([^.,]+)",
        r"\(([^)]+)\)",
        r"goes by ([^.,]+)",
        r"known as ([^.,]+)",
        r"referred to as ([^.,]+)",
        r"sometimes called ([^.,]+)",
        r"aka ([^.,]+)",
    ]

    for pat in patterns:
        matches = re.findall(pat, description, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"[^\w\s'-]", "", m).strip()
            if cleaned and 1 < len(cleaned) < 25:
                aliases.append(cleaned)
    return aliases[:5]


# Character File Parsing

def parse_characters(filepath: Path) -> List[Character]:
    text = filepath.read_text(encoding="utf-8")
    sections = re.split(r"###\s+", text)

    characters = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        lines = sec.split("\n", 1)
        if len(lines) < 2:
            continue

        name = lines[0].strip()
        desc = lines[1].strip()

        aliases = extract_character_aliases(desc)
        characters.append(Character(name=name, description=desc, aliases=aliases))

    logger.info(f"Parsed {len(characters)} characters")
    return characters


# Partition Parsing

def parse_partitions(filepath: Path) -> List[Partition]:
    text = filepath.read_text(encoding="utf-8")
    raw_parts = text.split("### ")
    if raw_parts and not raw_parts[0].strip():
        raw_parts.pop(0)

    partitions = []
    for sec in raw_parts:
        sec = sec.strip()
        if not sec:
            continue

        title, *rest = sec.split("\n", 1)
        content = rest[0].strip() if rest else ""

        if title.startswith("Author:") or title.startswith("Book Title:"):
            continue
        if len(content) <= 50:
            continue

        partitions.append(Partition(
            number=len(partitions) + 1,
            title=title,
            content=content
        ))

    logger.info(f"Parsed {len(partitions)} partitions")
    return partitions


# Name resolution (faithful)

def resolve_name(mentioned: str, characters: List[Character]) -> Tuple[str, float]:
    mentioned = mentioned.strip()

    for c in characters:
        if mentioned == c.name:
            return c.name, 1.0
        if mentioned in c.aliases:
            return c.name, 0.9

    lower = mentioned.lower()
    for c in characters:
        if lower == c.name.lower():
            return c.name, 0.85
        if any(lower == a.lower() for a in c.aliases):
            return c.name, 0.8

    # partial matching
    parts = mentioned.split()
    for c in characters:
        for m in parts:
            if len(m) > 2 and m.lower() in c.name.lower():
                return c.name, 0.7

    return mentioned, 0.0


# Relationship Tracking

class RelationshipTracker:
    def __init__(self):
        self.history = defaultdict(list)
        self.latest = {}

    def _pair(self, a, b):
        return tuple(sorted([a, b]))

    def add(self, chapter, a, b, rel, conf):
        pair = self._pair(a, b)
        prev = self.latest.get(pair)

        if prev is None:
            change = "new"
            prev_rel = None
        elif prev.relationship == rel:
            change = "maintained"
            prev_rel = prev.relationship
        else:
            change = "changed"
            prev_rel = prev.relationship

        entry = RelationshipHistory(
            chapter=chapter,
            char_a=pair[0],
            char_b=pair[1],
            relationship=rel,
            previous_relationship=prev_rel,
            change_type=change,
            confidence=conf
        )

        self.history[pair].append(entry)
        self.latest[pair] = entry
        return entry


# Relationship extraction

def extract_relationships(partition: Partition, characters: List[Character],
                          book_name: str, tracker: RelationshipTracker,
                          llm: MultiAPILLMClient) -> List[RelationshipHistory]:

    prev = tracker.latest.values()
    prev_context = "\n".join(
        f"- {h.char_a} ↔ {h.char_b}: {h.relationship} (chapter {h.chapter})"
        for h in list(prev)[-10:]
    )

    character_context = "\n".join(
        f"- {c.name} (aliases: {', '.join(c.aliases)})"
        for c in characters
    )

    prompt = f"""
You are analyzing character relationships in "{book_name.replace('-', ' ').title()}" with a focus on PRECISION and CONTEXT.

CHARACTERS IN THIS STORY:
{character_context}

CURRENT PARTITION: {partition.title} (Chapter {partition.number})
PARTITION CONTENT:
{partition.content}
{prev_context}

TASK: Extract character relationships considering:
1. Direct interactions in THIS partition
2. Relationship changes from previous chapters
3. Both explicit and implicit relationship indicators

RELATIONSHIP TYPES:
- POSITIVE: cooperation, friendship, support, alliance, shared goals
- NEGATIVE: conflict, hostility, opposition, betrayal, competing interests  
- NEUTRAL: professional interaction, acquaintance, unclear sentiment

IMPORTANT RULES:
1. Include relationships even if they don't change from previous chapters
2. Consider subtle relationship indicators (tone, cooperation level)
3. Use canonical character names from the character list
4. If uncertain about relationship type, choose neutral
5. Include relationships for major characters even if interaction is brief

OUTPUT FORMAT: Return ONLY a JSON array of relationships:
[
  {{
    "chapter": {partition.number},
    "char_a": "Character Name",
    "char_b": "Character Name", 
    "relationship": "positive|negative|neutral"
  }}
]

If no clear relationships are found, return [].
"""

    response = llm.call(
        [
            {"role": "system", "content": "You extract character relationships with high precision."},
            {"role": "user", "content": prompt}
        ],
        book_name,
        partition.number
    )

    try:
        data = json.loads(response)
    except Exception:
        logger.warning("JSON parsing failed; skipping partition")
        return []

    results = []
    for item in data:
        raw_a = item.get("char_a", "").strip()
        raw_b = item.get("char_b", "").strip()
        rel = item.get("relationship", "").strip().lower()

        if rel not in ("positive", "negative", "neutral"):
            continue

        a, conf_a = resolve_name(raw_a, characters)
        b, conf_b = resolve_name(raw_b, characters)

        if a == b:
            continue
        if min(conf_a, conf_b) < 0.6:
            continue

        entry = tracker.add(partition.number, a, b, rel, min(conf_a, conf_b))
        results.append(entry)

    return results



# Temporal Metrics 

def compute_temporal_metrics(all_history: List[RelationshipHistory]) -> Dict[int, Dict]:
    chapters = sorted({h.chapter for h in all_history})
    out = {}

    for ch in chapters:
        rels = [h for h in all_history if h.chapter == ch]
        G = nx.Graph()

        for h in rels:
            G.add_edge(h.char_a, h.char_b, relationship=h.relationship)

        density = nx.density(G) if G.number_of_nodes() > 1 else 0
        degs = list(dict(G.degree()).values())
        avg_deg = np.mean(degs) if degs else 0

        pos = sum(1 for h in rels if h.relationship == "positive")
        neg = sum(1 for h in rels if h.relationship == "negative")
        neu = sum(1 for h in rels if h.relationship == "neutral")
        total = max(len(rels), 1)

        out[ch] = dict(
            network_density=density,
            active_characters=G.number_of_nodes(),
            avg_degree=avg_deg,
            positive_ratio=pos / total,
            negative_ratio=neg / total,
            neutral_ratio=neu / total,
        )

    return out


# Save Outputs

def save_outputs(history: List[RelationshipHistory], outdir: Path, base: str):
    outdir = Path(outdir)

    # network.csv (deduplicated)
    net_path = outdir / f"{base}_network.csv"
    dedup = {(h.chapter, h.char_a, h.char_b): h for h in history}

    with open(net_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Chapter", "Character A", "Character B", "Relationship"])
        for h in sorted(dedup.values(), key=lambda x: (x.chapter, x.char_a, x.char_b)):
            w.writerow([h.chapter, h.char_a, h.char_b, h.relationship])

    # full history
    hist_path = outdir / f"{base}_relationship_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Chapter", "Character A", "Character B", "Relationship",
            "Previous", "ChangeType", "Confidence"
        ])
        for h in sorted(history, key=lambda x: (x.chapter, x.char_a, x.char_b)):
            w.writerow([
                h.chapter, h.char_a, h.char_b, h.relationship,
                h.previous_relationship or "N/A",
                h.change_type,
                f"{h.confidence:.2f}"
            ])

    # temporal metrics
    metrics = compute_temporal_metrics(history)
    met_path = outdir / f"{base}_temporal_metrics.csv"

    with open(met_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Chapter", "NetworkDensity", "ActiveCharacters", "AvgDegree",
            "PositiveRatio", "NegativeRatio", "NeutralRatio"
        ])
        for ch in sorted(metrics):
            m = metrics[ch]
            w.writerow([
                ch, m["network_density"], m["active_characters"],
                m["avg_degree"], m["positive_ratio"],
                m["negative_ratio"], m["neutral_ratio"]
            ])

    logger.info(f"Saved outputs for {base}")


# Main book processing function

def process_book(book_dir: Path, base_name: str,
                 llm_client: MultiAPILLMClient,
                 progress: ProgressTracker) -> bool:

    book_dir = Path(book_dir)

    characters_file = next(book_dir.glob("*_characters_clean.txt"), None)
    summary_file = next((f for f in book_dir.glob("*_clean.txt")
                         if "characters" not in f.name), None)

    if not characters_file or not summary_file:
        progress.set_book_status(base_name, "failed", error_msg="Missing input files")
        return False

    progress.set_book_status(base_name, "processing")

    characters = parse_characters(characters_file)
    partitions = parse_partitions(summary_file)
    tracker = RelationshipTracker()

    all_history = []

    for part in partitions:
        start = time.time()
        rels = extract_relationships(part, characters, base_name, tracker, llm_client)
        all_history.extend(rels)
        duration = time.time() - start

        progress.mark_partition_completed(
            base_name, part.number,
            llm_calls=1,
            relationships_extracted=len(rels),
            proc_time=duration
        )

    save_outputs(all_history, book_dir, base_name)
    progress.set_book_status(base_name, "completed")

    return True
