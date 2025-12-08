#!/usr/bin/env python3
"""
CharDyNet – Visualisation Utilities
-----------------------------------

This module implements the visualisation phase of the CharDyNet pipeline:

1. Per-book temporal panels (9 panels):
   - Cast size (Active_Characters)
   - Interaction load (Edges)
   - Emotional composition (Positive/Negative/Neutral edge proportions)
   - Structural balance (BalanceFrac)
   - Structural tension (FrustrationEdgeFrac_t)
   - Relationship volatility (ChangedEdges_t + flip_rate)
   - Protagonist trajectory (Bet / Deg)
   - Community fragmentation (Community_Count)
   - Cohesion (LCC_pct)

2. Optional per-book extras:
   - Dyad stability heatmap (top-K dyads, sign -1/0/+1 by chapter)

3. Corpus-level plots:
   - Mean±SE curves across normalized narrative time, faceted by Epoch / Genre:
       * Active_Characters, Edges, Density
       * PosFrac, NeuFrac, NegFrac
       * LCC_pct
       * BalanceZ or FrustrationEdgeFrac_t
   - Balance state diagram (BalanceZ_mean vs BalanceZ_var)
   - Climax alignment histogram (x_at_max_betweenness)
   - Cumulative unbalanced triads (tension accumulation)

All paths are configured via CLI arguments. No absolute paths or
project-specific names are hard-coded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap

# Global style

POS_COLOR = "#2ca02c"   # green
NEG_COLOR = "#d62728"   # red
NEU_COLOR = "#7f7f7f"   # grey
LINE_COLOR = "#1f77b4"  # default line color for non-sign panels

YLABEL_KW = dict(fontsize=13, fontweight="bold")
TITLE_KW  = dict(fontsize=15)
SMALL_KW  = dict(fontsize=9)


# Safe figure saving

def safe_savefig(fig: plt.Figure, stem: Path, overwrite: bool = True) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")

    if png.exists() and not overwrite:
        print(f"[SKIP exists] {png}")
    else:
        fig.savefig(png, dpi=300, bbox_inches="tight")
        print(f"[WRITE] {png}")

    if pdf.exists() and not overwrite:
        print(f"[SKIP exists] {pdf}")
    else:
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        print(f"[WRITE] {pdf}")

    plt.close(fig)


# Generic helpers

def slugify(s: str) -> str:
    """ASCII-only slug, used for filenames."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s

def parse_author_title_from_basename(basename: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"^(.+?)_(.+?)(?:_temporal_metrics.*)?$", basename)
    if not m:
        return None, None
    author = m.group(1).replace("-", " ")
    title = m.group(2).replace("-", " ")
    return author, title

# Light smoothing for redability, not used to compute values, only to draw lines.
def smooth_series(y: np.ndarray, window_frac: float = 0.12, poly: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return y
    win = max(5, int(round(window_frac * n)) | 1)  # odd, >=5
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, win, poly)
    except Exception:
        k = max(3, int(round(0.5 * win)))
        k = k + 1 if k % 2 == 0 else k
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        kern = np.ones(k) / k
        ysm = np.convolve(ypad, kern, mode="valid")
        return ysm[:n]

# Simple local maxima finder, return indices of top n peaks separated by at least min_separation indices
def local_maxima(x: np.ndarray, y: np.ndarray, n_peaks: int = 2, min_separation: int = 2) -> List[int]:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
    selected = []
    for p in peaks:
        if all(abs(p - q) >= min_separation for q in selected):
            selected.append(p)
        if len(selected) >= n_peaks:
            break
    return selected


# Book directory indexing

def find_book_dirs_with_enriched(root: Path) -> List[Path]:

    root = Path(root)
    return sorted({p.parent for p in root.rglob("*_temporal_metrics_enriched.csv")})


# Metadata helpers

# Find the metadata row for a title
def load_meta_row(meta_csv_path: Optional[Path], title_guess: str) -> Dict[str, Optional[str]]:
    if meta_csv_path is None:
        return {}

    meta_csv_path = Path(meta_csv_path)
    if not meta_csv_path.exists():
        return {}

    df = None
    try:
        df = pd.read_csv(meta_csv_path)
    except Exception:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(meta_csv_path, sep=sep)
                break
            except Exception:
                df = None
    if df is None or df.empty:
        return {}

    tcol = next((c for c in df.columns if "title" in c.lower()), None)
    acol = next((c for c in df.columns if "author" in c.lower()), None)
    gcol = next((c for c in df.columns if "genre" in c.lower()), None)
    ycol = next((c for c in df.columns if "year" in c.lower()), None)

    if not tcol:
        return {}

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

    df = df.copy()
    df["_norm"] = df[tcol].astype(str).apply(norm)
    tnorm = norm(title_guess)

    hit = df[df["_norm"] == tnorm]
    if hit.empty and len(tnorm) >= 6:
        hit = df[df["_norm"].str.contains(tnorm[: max(5, len(tnorm) // 2)], na=False)]

    if hit.empty:
        return {}

    row = hit.iloc[0]
    return {
        "author_full": row[acol] if acol else None,
        "title": row[tcol] if tcol else None,
        "year": row[ycol] if ycol else None,
        "genres": row[gcol] if gcol else None,
    }


# Protagonist detection helpers

# Look ofr a 'protagonist' label in *_characters_clean.txt. Returns the display name string if found, else None
def extract_labeled_protagonist(book_dir: Path) -> Optional[str]:
    book_dir = Path(book_dir)
    try:
        char_file = next(iter(book_dir.glob("*_characters_clean.txt")))
    except StopIteration:
        return None

    text = char_file.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"(?m)^###\s*", text)
    for blk in blocks:
        if not blk.strip():
            continue
        lines = blk.splitlines()
        name = lines[0].strip()
        body = "\n".join(lines[1:]).lower()
        if "protagonist" in body or re.search(r"\bmain protagonist\b", body):
            return name
    return None

# match btw a target name string and a list of node labels usiing Jaccard similarity over tokens
def best_node_match(target_name: str, node_names: List[str]) -> Optional[str]:
    if not target_name or not node_names:
        return None
    t = set(re.sub(r"[^a-z0-9]+", " ", target_name.lower()).split())
    best, score = None, 0.0
    for n in node_names:
        nt = set(re.sub(r"[^a-z0-9]+", " ", str(n).lower()).split())
        s = len(t & nt) / max(1, len(t | nt))
        if s > score:
            best, score = n, s
    return best if score >= 0.30 else None

# Choose protagonist trajectory series. Preference order: manual_override (string, or dict keyed by folder name) fuzzy match, labeled protagonist in *characters_clean.txt fuzzy match, 
# node with highest median Betweenness. 
def choose_protagonist_traj(
    df_traj: pd.DataFrame,
    book_dir: Path,
    manual_override: Optional[str | Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[Tuple[np.ndarray, np.ndarray]]]:
    if "Node" not in df_traj.columns:
        return None, None

    if isinstance(manual_override, dict):
        override_str = manual_override.get(Path(book_dir).name)
    else:
        override_str = manual_override

    nodes = list(df_traj["Node"].unique())

    for cand in [override_str, extract_labeled_protagonist(book_dir)]:
        if cand:
            m = best_node_match(cand, nodes)
            if m:
                g = df_traj[df_traj["Node"] == m].sort_values("x")
                if "Bet" in g.columns and g["Bet"].notna().any():
                    y = g["Bet"].values
                elif "Deg" in g.columns:
                    y = g["Deg"].values
                else:
                    y = None
                if y is not None:
                    return m, (g["x"].values, y)

    score_col = None
    if "Bet" in df_traj.columns and df_traj["Bet"].notna().any():
        score_col = "Bet"
    elif "Deg" in df_traj.columns:
        score_col = "Deg"

    if score_col is None:
        return None, None

    med = df_traj.groupby("Node")[score_col].median().sort_values(ascending=False)
    node = med.index[0]
    g = df_traj[df_traj["Node"] == node].sort_values("x")
    y = g[score_col].values
    return node, (g["x"].values, y)


# Dyad stability heatmap

# Rows = top-K dyads by number of observed chapters; columns = Chapter; cell = -1/0/+1 sign.
# Requires *_relationship_history.csv with columns: Chapter, Character A, Character B, Relationship
def dyad_stability_heatmap(
    book_dir: Path,
    chapters: pd.Series,
    save_path: Path,
    top_k: int = 15,
) -> Optional[Path]:
    book_dir = Path(book_dir)
    rel_file = next(iter(book_dir.glob("*_relationship_history.csv")), None)
    if rel_file is None:
        return None

    df = pd.read_csv(rel_file)
    required = {"Chapter", "Character A", "Character B", "Relationship"}
    if not required.issubset(df.columns):
        return None

    def sign_map(s: str) -> int:
        s = str(s).strip().lower()
        if s.startswith("pos"):
            return 1
        if s.startswith("neg"):
            return -1
        return 0

    df["sign"] = df["Relationship"].map(sign_map)
    df["Dyad"] = df["Character A"].astype(str) + " ↔ " + df["Character B"].astype(str)

    counts = df.groupby("Dyad")["Chapter"].nunique().sort_values(ascending=False)
    keep = list(counts.head(top_k).index)
    top = df[df["Dyad"].isin(keep)]

    all_ch = sorted(pd.unique(chapters))
    heat = []
    labels = []
    for dyad, g in top.groupby("Dyad"):
        s = pd.Series(index=all_ch, dtype=float)
        s.loc[g["Chapter"].values] = g["sign"].values
        heat.append(s.values)
        labels.append(dyad)

    if not heat:
        return None

    H = np.array(heat)
    fig, ax = plt.subplots(
        figsize=(min(12, 1.2 + 0.5 * len(all_ch)), 0.44 * len(keep) + 1.6)
    )
    cmap = ListedColormap([NEG_COLOR, NEU_COLOR, POS_COLOR])
    ax.imshow(H, aspect="auto", vmin=-1, vmax=1, cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(all_ch)))
    ax.set_xticklabels(all_ch, rotation=0, fontsize=8)
    ax.set_xlabel("Chapter")
    ax.set_title("Dyad Stability (−1/0/+1 = Neg/Neu/Pos)")

    patches = [
        mpatches.Patch(color=POS_COLOR, label="Positive"),
        mpatches.Patch(color=NEG_COLOR, label="Negative"),
        mpatches.Patch(color=NEU_COLOR, label="Neutral"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {save_path}")
    return save_path


# Per-book 9-panel temporal figure

def plot_per_book_panels(
    book_dir: Path,
    meta_csv_path: Optional[Path] = None,
    dpi: int = 160,
    smooth: bool = True,
    smooth_window_frac: float = 0.12,
    fade_edges_threshold: int = 3,
    triad_thresh_mode: str = "adaptive",   # "adaptive" or "fixed"
    triad_fixed_thresh: int = 12,
    triad_min: int = 10,
    triad_percentile: int = 30,
    ffill_discrete: bool = True,
    shade_climax: bool = True,
    climax_sources: Tuple[str, ...] = ("frustration", "flip_rate", "protagonist"),
    climax_n_peaks: int = 2,
    manual_protagonist: Optional[str | Dict[str, str]] = None,
    dyad_heatmap_flag: bool = False,
    dyad_top_k: int = 15,
    overwrite: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Create the 9-panel figure and (optionally) a dyad stability heatmap.
    Saves outputs in the book folder.

    Parameters
    book_dir : Path
        Directory containing per-book CSVs, including:
            *_temporal_metrics_enriched.csv
            (optional) *_protagonist_traj.csv
            (optional for heatmap) *_relationship_history.csv
    meta_csv_path : Path or None
        Metadata CSV with author/title/year/genres (optional).
    dpi : int
        Resolution for PNG outputs.
    smooth : bool
        If True, smooth continuous series for plotting.
    smooth_window_frac : float
        Fractional window size for smoothing.
    fade_edges_threshold : int
        Chapters with Edges < threshold are lightly shaded in the
        emotional composition panel.
    triad_thresh_mode : {"adaptive","fixed"}
        If "adaptive", triad shading threshold is chosen based on a
        percentile of observed triad counts (with minimum triad_min).
    triad_fixed_thresh : int
        Used if triad_thresh_mode == "fixed".
    triad_min : int
        Minimum triad threshold for adaptive mode.
    triad_percentile : int
        Percentile for adaptive triad threshold.
    ffill_discrete : bool
        Forward-fill missing values for discrete series like Community_Count
        and LCC_pct before smoothing/plotting.
    shade_climax : bool
        If True, shade narrow bands around inferred climax peaks.
    climax_sources : tuple
        Subset of {"frustration","flip_rate","protagonist"} used as peak sources.
    climax_n_peaks : int
        Number of peaks per source considered.
    manual_protagonist : str or dict or None
        Optional manual override for protagonist name.
    dyad_heatmap_flag : bool
        If True, also builds a dyad stability heatmap.
    dyad_top_k : int
        Number of dyads to include in the heatmap.
    overwrite : bool
        Whether to overwrite existing PNG/PDF files.

    Returns
    panel_path : Path
        Path to the 9-panel figure (PNG).
    heatmap_path : Path or None
        Path to the dyad stability heatmap (PNG), or None if not created.
    """
    book_dir = Path(book_dir)

    enriched_files = list(book_dir.glob("*_temporal_metrics_enriched.csv"))
    if not enriched_files:
        raise FileNotFoundError(f"No *_temporal_metrics_enriched.csv in {book_dir}")
    enriched = enriched_files[0]

    base = enriched.name.replace("_temporal_metrics_enriched.csv", "")
    author_guess, title_guess = parse_author_title_from_basename(enriched.name)

    df = pd.read_csv(enriched)

    if "x" not in df.columns and "Chapter" in df.columns:
        df["x"] = (df["Chapter"] - 1) / max(1, df["Chapter"].max() - 1)
    x = df["x"].values

    pe = df["Positive_Edges"]
    ne = df["Negative_Edges"]
    ue = df["Neutral_Edges"]
    total = (pe + ne + ue).replace(0, np.nan)
    df["pos_prop"] = pe / total
    df["neg_prop"] = ne / total
    df["neu_prop"] = ue / total

    df["flip_rate"] = df["ChangedEdges_t"] / df["Edges"].clip(lower=1)

    if "TotalSignedTriads" not in df.columns:
        if {"Balanced_Triads", "Unbalanced_Triads"}.issubset(df.columns):
            df["TotalSignedTriads"] = (
                df["Balanced_Triads"].fillna(0) + df["Unbalanced_Triads"].fillna(0)
            )
        else:
            df["TotalSignedTriads"] = np.nan

    def maybe_smooth(series: pd.Series) -> np.ndarray:
        arr = np.asarray(series.values, dtype=float)
        return smooth_series(arr, window_frac=smooth_window_frac, poly=2) if smooth else arr

    prot_node: Optional[str] = None
    prot_series: Optional[Tuple[np.ndarray, np.ndarray]] = None
    traj_file = next(iter(book_dir.glob("*_protagonist_traj.csv")), None)
    if traj_file is not None:
        traj = pd.read_csv(traj_file)
        if "x" not in traj.columns and "Chapter" in traj.columns:
            traj["x"] = (traj["Chapter"] - 1) / max(1, df["Chapter"].max() - 1)
        prot_node, prot_series = choose_protagonist_traj(traj, book_dir, manual_override=manual_protagonist)

    author_full = author_guess or ""
    title = title_guess or base.replace("_", " ")
    year = ""
    genres = ""

    meta = load_meta_row(meta_csv_path, title)
    author_full = meta.get("author_full") or author_full
    title = meta.get("title") or title
    year = meta.get("year") or year
    genres = meta.get("genres") or genres

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=9,
        ncols=1,
        figsize=(12, 21),
        sharex=True,
        constrained_layout=True,
    )

    # Panel A1: Cast size
    axes[0].plot(x, maybe_smooth(df["Active_Characters"]), color=LINE_COLOR, lw=2)
    axes[0].set_ylabel("Active\nCharacters", **YLABEL_KW)
    axes[0].set_title("Cast Size Over Time", **TITLE_KW)
    axes[0].grid(axis="y", alpha=0.2)

    # Panel A2: Interactions (Edges)
    axes[1].plot(x, maybe_smooth(df["Edges"]), color=LINE_COLOR, lw=2)
    axes[1].set_ylabel("Edges", **YLABEL_KW)
    axes[1].set_title("Interactions Over Time", **TITLE_KW)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel B: Emotional composition
    axes[2].stackplot(
        x,
        df["pos_prop"].fillna(0),
        df["neg_prop"].fillna(0),
        df["neu_prop"].fillna(0),
        labels=["Positive", "Negative", "Neutral"],
        colors=[POS_COLOR, NEG_COLOR, NEU_COLOR],
        alpha=0.75,
    )
    axes[2].legend(
        loc="center left",
        bbox_to_anchor=(1.002, 0.5),
        frameon=False,
        fontsize=9,
        ncols=1,
    )
    axes[2].set_ylabel("Proportion", **YLABEL_KW)
    axes[2].set_title("Emotional Composition", **TITLE_KW)
    axes[2].grid(axis="y", alpha=0.2)

    low_edges = df["Edges"].fillna(0) < fade_edges_threshold
    if low_edges.any():
        for i in np.where(low_edges)[0]:
            if i < len(x) - 1:
                axes[2].axvspan(x[i], x[i + 1], color="k", alpha=0.05, lw=0)
        axes[2].text(
            0.01,
            0.06,
            f"Shaded: Edges < {fade_edges_threshold}",
            transform=axes[2].transAxes,
            fontsize=8,
        )

    # Panel C: Structural balance
    axes[3].plot(x, maybe_smooth(df["BalanceFrac"].fillna(0)), color=LINE_COLOR, lw=2)
    axes[3].set_ylabel("BalanceFrac", **YLABEL_KW)
    axes[3].set_title("Structural Balance (fraction of balanced triads)", **TITLE_KW)
    axes[3].grid(axis="y", alpha=0.2)

    # Panel D: Structural tension (frustration)
    axes[4].plot(
        x,
        maybe_smooth(df["FrustrationEdgeFrac_t"].fillna(0)),
        color=LINE_COLOR,
        lw=2,
    )
    axes[4].set_ylabel("Frustration", **YLABEL_KW)
    axes[4].set_title("Structural Tension (frustrated edges)", **TITLE_KW)
    axes[4].grid(axis="y", alpha=0.2)

    triads = df["TotalSignedTriads"].replace([np.inf, -np.inf], np.nan)
    if triad_thresh_mode == "adaptive":
        valid = triads.dropna()
        if len(valid):
            thr = max(triad_min, np.percentile(valid, triad_percentile))
        else:
            thr = triad_min
    else:
        thr = triad_fixed_thresh

    low_triads = triads.fillna(0) < thr
    if low_triads.any():
        for ax in (axes[3], axes[4]):
            for i in np.where(low_triads)[0]:
                if i < len(x) - 1:
                    ax.axvspan(x[i], x[i + 1], color="grey", alpha=0.12, lw=0)
        patch = mpatches.Patch(color="grey", alpha=0.12, label=f"Triads < {int(thr)}")
        handles, labels = axes[3].get_legend_handles_labels()
        axes[3].legend(
            handles + [patch],
            labels + [f"Triads < {int(thr)}"],
            loc="lower left",
            fontsize=8,
            frameon=False,
        )

    # Panel E: Relationship volatility
    axes[5].bar(
        x,
        df["ChangedEdges_t"].fillna(0),
        width=(1.0 / max(10, len(x)) * 0.9),
        alpha=0.85,
        label="Flips (count)",
    )
    axes[5].plot(x, df["flip_rate"].fillna(0).values, lw=2, label="Flip rate (per edge)")
    axes[5].legend(loc="upper right", fontsize=9, frameon=False)
    axes[5].set_ylabel("Flips / Rate", **YLABEL_KW)
    axes[5].set_title("Relationship Volatility", **TITLE_KW)
    axes[5].grid(axis="y", alpha=0.2)

    # Panel F: Protagonist trajectory
    if prot_series is not None:
        px, py = prot_series
        axes[6].plot(px, smooth_series(py) if smooth else py, color=LINE_COLOR, lw=2)
        axes[6].set_title(f"Protagonist Trajectory ({prot_node})", **TITLE_KW)
    else:
        axes[6].text(
            0.5,
            0.5,
            "No protagonist trajectory available",
            ha="center",
            va="center",
            fontsize=11,
        )
    axes[6].set_ylabel("Centrality", **YLABEL_KW)
    axes[6].grid(axis="y", alpha=0.2)

    # Panel G: Community fragmentation
    comm_series = df["Community_Count"]
    if ffill_discrete:
        comm_series = comm_series.fillna(method="ffill")
    axes[7].plot(x, maybe_smooth(comm_series), color=LINE_COLOR, lw=2)
    axes[7].set_ylabel("Communities", **YLABEL_KW)
    axes[7].set_title("Community Fragmentation", **TITLE_KW)
    axes[7].grid(axis="y", alpha=0.2)

    # Panel H: Cohesion (LCC %)
    if "LCC_pct" in df.columns:
        lcc_series = df["LCC_pct"]
        if ffill_discrete:
            lcc_series = lcc_series.fillna(method="ffill")
        y_vals = np.clip(maybe_smooth(lcc_series), 0, 1)
        axes[8].plot(x, y_vals, color=LINE_COLOR, lw=2)
        axes[8].set_ylabel("LCC %", **YLABEL_KW)
        axes[8].set_title("Cohesion (Largest Component % of cast)", **TITLE_KW)
    else:
        axes[8].text(
            0.5,
            0.5,
            "No LCC_pct available",
            ha="center",
            va="center",
            fontsize=11,
        )
        axes[8].set_ylabel("LCC %", **YLABEL_KW)

    axes[8].set_xlabel("Narrative Time (normalized)", fontsize=11, fontweight="bold")
    axes[8].grid(axis="y", alpha=0.2)

    # Climax shading
    if shade_climax:
        candidates_x: List[float] = []

        if "frustration" in climax_sources:
            idx = local_maxima(x, df["FrustrationEdgeFrac_t"].fillna(0).values, n_peaks=climax_n_peaks)
            candidates_x += [x[i] for i in idx]

        if "flip_rate" in climax_sources:
            idx = local_maxima(x, df["flip_rate"].fillna(0).values, n_peaks=climax_n_peaks)
            candidates_x += [x[i] for i in idx]

        if prot_series is not None and "protagonist" in climax_sources:
            px, py = prot_series
            idx = local_maxima(px, py, n_peaks=climax_n_peaks)
            candidates_x += [px[i] for i in idx]

        band_half = 0.5 / max(10, len(x))
        for cx in sorted(set(candidates_x)):
            for ax in axes:
                ax.axvspan(max(0, cx - band_half), min(1, cx + band_half), color="#f0e68c", alpha=0.28, lw=0)

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0)

    year_str = str(year).strip()
    year_int = None
    if re.search(r"\d", year_str):
        try:
            year_int = int(re.sub(r"[^0-9]", "", year_str))
        except Exception:
            year_int = None

    fig_title = f"{author_full} — {title}" + (f" ({year_int})" if year_int is not None else "")
    fig.suptitle(fig_title, fontsize=20, y=1.04)

    if genres:
        subtitle = textwrap.fill(str(genres), width=70)
        fig.text(0.5, 0.99, subtitle, ha="center", va="top", fontsize=12)

    author_for_file = slugify(author_full) if author_full else slugify(base.split("_")[0])
    title_for_file = slugify(title)
    year_for_file = f"_{year_int}" if year_int is not None else ""
    out_name = f"{author_for_file}_{title_for_file}{year_for_file}_temporal_9panel"
    out_panel_png = (book_dir / out_name).with_suffix(".png")

    # use safe_savefig to get png + pdf
    safe_savefig(fig, book_dir / out_name, overwrite=overwrite)

    heatmap_path: Optional[Path] = None
    if dyad_heatmap_flag and "Chapter" in df.columns:
        heatmap_path = (book_dir / f"{author_for_file}_{title_for_file}{year_for_file}_dyad_stability_topK{dyad_top_k}").with_suffix(".png")
        try:
            dyad_stability_heatmap(book_dir, df["Chapter"], heatmap_path, top_k=dyad_top_k)
        except Exception as e:
            warnings.warn(f"Dyad heatmap failed for {book_dir.name}: {e}")
            heatmap_path = None

    return out_panel_png, heatmap_path


# Corpus-level helpers and plots

def density_series(g: pd.DataFrame) -> np.ndarray:
    denom = g["Active_Characters"].apply(
        lambda n: n * (n - 1) / 2 if pd.notna(n) and n >= 2 else np.nan
    )
    return (g["Edges"] / denom).values

def binned_mean(df: pd.DataFrame, yvals: np.ndarray) -> Optional[np.ndarray]:
    x = df["x"].values
    if len(x) < 2:
        return None
    bins = np.linspace(0, 1, 51)
    digit = np.digitize(x, bins) - 1
    mu = [
        np.nanmean(yvals[digit == i]) if np.any(digit == i) else np.nan
        for i in range(50)
    ]
    return np.array(mu, dtype=float)

# Return dict facet_value -> (xs, mean, se), where xs has 50 points in [0,1).
def facet_mean_curves(df: pd.DataFrame, metric: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for val, grp in df.groupby("facet_value"):
        curves: List[np.ndarray] = []
        for key, g in grp.groupby("Book_Key"):
            if metric == "Density":
                y = density_series(g)
            elif metric in ("PosFrac", "NeuFrac", "NegFrac"):
                total = (g["Positive_Edges"] + g["Negative_Edges"] + g["Neutral_Edges"]).clip(lower=1)
                if metric == "PosFrac":
                    y = (g["Positive_Edges"] / total).values
                elif metric == "NeuFrac":
                    y = (g["Neutral_Edges"] / total).values
                else:
                    y = (g["Negative_Edges"] / total).values
            else:
                if metric not in g.columns:
                    continue
                y = g[metric].values
            mu = binned_mean(g, y)
            if mu is not None:
                curves.append(mu)
        if curves:
            A = np.vstack(curves)
            m = np.nanmean(A, axis=0)
            n_eff = np.maximum(1, np.sum(np.isfinite(A), axis=0))
            se = np.nanstd(A, axis=0) / np.sqrt(n_eff)
            result[str(val)] = (np.linspace(0, 1, 50, endpoint=False), m, se)
    return result


def plot_facet_curves(panel: pd.DataFrame, facet: str, metric: str, title: str, outdir: Path, overwrite: bool = True) -> None:
    df = panel.copy()
    df["facet_value"] = df[facet].fillna("Unknown").astype(str)
    res = facet_mean_curves(df, metric)
    if not res:
        print(f"[WARN] No curves for {metric} by {facet}.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for val, (xs, m, se) in res.items():
        ax.plot(xs, m, label=str(val))
        ax.fill_between(xs, m - se, m + se, alpha=0.2)
    ax.set_title(f"{title} by {facet}")
    ax.set_xlabel("Normalized narrative time")
    ax.legend(ncol=2, fontsize=8)

    stem = outdir / f"corpus_mean_{metric}_{facet}"
    safe_savefig(fig, stem, overwrite=overwrite)

# compute mean±SE of cumulative metric over 50 bins in x.
def cumulative_binned_mean(df: pd.DataFrame, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves: List[np.ndarray] = []
    for key, g in df.groupby("Book_Key"):
        g = g.sort_values("x")
        y = g[metric].fillna(0).values
        c = np.cumsum(y)
        x = g["x"].values
        bins = np.linspace(0, 1, 51)
        digit = np.digitize(x, bins) - 1
        mu = [
            np.nanmean(c[digit == i]) if np.any(digit == i) else np.nan
            for i in range(50)
        ]
        curves.append(mu)
    if not curves:
        return None
    A = np.array(curves)
    m = np.nanmean(A, axis=0)
    n_eff = np.maximum(1, np.sum(np.isfinite(A), axis=0))
    se = np.nanstd(A, axis=0) / np.sqrt(n_eff)
    xs = np.linspace(0, 1, 50, endpoint=False)
    return xs, m, se

# Generate corpus-level vis
def run_corpus_plots(analysis_dir: Path, overwrite: bool = True) -> None:
    analysis_dir = Path(analysis_dir)
    vis_dir = analysis_dir / "vis"
    log_dir = analysis_dir / "logs"
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    panel_path = analysis_dir / "corpus_chapter_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}. Run analysis aggregation first.")

    panel = pd.read_csv(panel_path)

    # Facets by Epoch and Genre
    for facet in ["Epoch", "Genre"]:
        plot_facet_curves(panel, facet, "Active_Characters", "Active_Characters (mean±SE)", vis_dir, overwrite)
        plot_facet_curves(panel, facet, "Edges", "Edges (mean±SE)", vis_dir, overwrite)
        plot_facet_curves(panel, facet, "Density", "Density (mean±SE)", vis_dir, overwrite)

        plot_facet_curves(panel, facet, "PosFrac", "Positive fraction (mean±SE)", vis_dir, overwrite)
        plot_facet_curves(panel, facet, "NeuFrac", "Neutral fraction (mean±SE)", vis_dir, overwrite)
        plot_facet_curves(panel, facet, "NegFrac", "Negative fraction (mean±SE)", vis_dir, overwrite)

        plot_facet_curves(panel, facet, "LCC_pct", "LCC% (mean±SE)", vis_dir, overwrite)

        if "BalanceZ" in panel.columns and panel["BalanceZ"].notna().sum() > 0:
            plot_facet_curves(panel, facet, "BalanceZ", "BalanceZ (mean±SE)", vis_dir, overwrite)
        elif "FrustrationEdgeFrac_t" in panel.columns and panel["FrustrationEdgeFrac_t"].notna().sum() > 0:
            plot_facet_curves(panel, facet, "FrustrationEdgeFrac_t", "Frustration (edge fraction, mean±SE)", vis_dir, overwrite)

    # Balance state diagram
    bf_path = analysis_dir / "corpus_book_features.csv"
    if not bf_path.exists():
        raise FileNotFoundError(f"Missing {bf_path}. Run analysis aggregation first.")

    bf = pd.read_csv(bf_path)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    genres = bf["Genre"].fillna("Unknown").astype(str)
    for gval in sorted(genres.unique()):
        m = genres == gval
        ax.scatter(
            bf.loc[m, "BalanceZ_mean"],
            bf.loc[m, "BalanceZ_var"],
            s=18,
            alpha=0.75,
            label=gval,
        )

    ax.axvline(0, ls="--", alpha=0.3)
    ax.set_xlabel("Mean BalanceZ")
    ax.set_ylabel("Var BalanceZ")
    ax.set_title("Balance state diagram")
    ax.legend(title="Genre", fontsize=8)

    if bf["BalanceZ_mean"].isna().mean() > 0.3:
        ax.text(
            0.02,
            0.02,
            "Note: Many titles lack signed triads.\nSee frustration-based plots.",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
    safe_savefig(fig, vis_dir / "corpus_balance_state_diagram", overwrite)

    # Climax alignment
    xvals = bf["x_at_max_betweenness"].astype(float)
    xvals = xvals[np.isfinite(xvals)]

    if len(xvals) == 0:
        print("[WARN] No finite x_at_max_betweenness values to plot.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(xvals, bins=20, alpha=0.85)
        ax.axvspan(0.6, 0.9, color="gray", alpha=0.2, label="expected window")
        ax.set_xlabel("x at max betweenness (climax proxy)")
        ax.set_ylabel("Count")
        ax.set_title("Climax alignment")
        ax.legend()
        safe_savefig(fig, vis_dir / "corpus_climax_alignment", overwrite)

    # Tension accumulation
    res = cumulative_binned_mean(panel, "Unbalanced_Triads")
    if res is None:
        print("[WARN] No data for Unbalanced_Triads.")
    else:
        xs, m, se = res
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, m)
        ax.fill_between(xs, m - se, m + se, alpha=0.2)
        ax.set_xlabel("Normalized narrative time")
        ax.set_ylabel("Cumulative unbalanced triads (mean±SE)")
        ax.set_title("Tension accumulation")
        safe_savefig(fig, vis_dir / "corpus_tension_accumulation", overwrite)

    print("[INFO] Corpus-level plots written to", vis_dir)



# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="CharDyNet visualisation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Per-book panels
    p_book = subparsers.add_parser("book-panels", help="Generate per-book 9-panel temporal figures")
    p_book.add_argument("--summaries-dir", type=Path, required=True,
                        help="Root directory with per-book subfolders (each containing *_temporal_metrics_enriched.csv).")
    p_book.add_argument("--metadata-csv", type=Path, default=None,
                        help="Metadata CSV with author/title/year/genres (optional).")
    p_book.add_argument("--dyad-heatmap", action="store_true",
                        help="Also generate dyad stability heatmaps per book.")
    p_book.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing figures.")
    p_book.add_argument("--manual-protagonist-json", type=Path, default=None,
                        help="Optional JSON mapping {folder_name: protagonist_name} for manual overrides.")

    # Corpus-level plots
    p_corpus = subparsers.add_parser("corpus-plots", help="Generate corpus-level plots from analysis outputs")
    p_corpus.add_argument("--analysis-dir", type=Path, required=True,
                          help="Directory containing corpus_chapter_panel.csv and corpus_book_features.csv.")
    p_corpus.add_argument("--overwrite", action="store_true",
                          help="Overwrite existing figures.")

    args = parser.parse_args()

    if args.command == "book-panels":
        root = args.summaries_dir
        meta_csv = args.metadata_csv
        overwrite = args.overwrite
        dyad_flag = args.dyad_heatmap

        manual_map = None
        if args.manual_protagonist_json and args.manual_protagonist_json.exists():
            manual_map = json.loads(args.manual_protagonist_json.read_text(encoding="utf-8"))

        book_dirs = find_book_dirs_with_enriched(root)
        print(f"[INFO] Found {len(book_dirs)} book directories with enriched metrics under {root}")

        ok, fail = 0, 0
        for d in book_dirs:
            try:
                panel_path, heatmap_path = plot_per_book_panels(
                    d,
                    meta_csv_path=meta_csv,
                    dyad_heatmap_flag=dyad_flag,
                    manual_protagonist=manual_map,
                    overwrite=overwrite,
                )
                print(f"[OK] {d.name} -> {panel_path.name}"
                      + (f", heatmap {heatmap_path.name}" if heatmap_path else ""))
                ok += 1
            except Exception as e:
                print(f"[FAIL] {d.name} — {e.__class__.__name__}: {e}")
                fail += 1

        print(f"[INFO] Done. OK={ok}, FAIL={fail}")

    elif args.command == "corpus-plots":
        run_corpus_plots(args.analysis_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
