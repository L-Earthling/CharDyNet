#!/usr/bin/env python3
"""
(1) For each processed book directory:
      - Load *_temporal_metrics.csv and *_relationship_history.csv
      - Build chapter-level signed graphs
      - Compute temporal metrics, structural balance, flip rates, protagonist trajectory
      - Save:
            {Book_Key}_temporal_metrics_enriched.csv
            {Book_Key}_dyad_flips.csv
            {Book_Key}_features.csv
            {Book_Key}_nullmeta.json

(2) Aggregate all enriched per-book outputs into:
            corpus_chapter_panel.csv
            corpus_book_features.csv
"""

import argparse
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# Optional community detection
try:
    import community as community_louvain
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False

try:
    import igraph as ig
    import leidenalg as la
    HAVE_LEIDEN = True
except Exception:
    HAVE_LEIDEN = False

warnings.filterwarnings("ignore")


# Utility helpers

def normalize_time(ch: int, total: int) -> float:
    return float(ch) / float(total) if total > 0 else np.nan


def moving_average(x: pd.Series, k: int = 3) -> pd.Series:
    if k <= 1 or x.isna().all():
        return x
    return x.rolling(window=k, center=True, min_periods=1).mean()


def first_derivative(x: pd.Series) -> pd.Series:
    return x.diff()


def zscore_series(x: pd.Series) -> pd.Series:
    v = x.dropna()
    if len(v) < 2 or v.std(ddof=0) == 0:
        return pd.Series(index=x.index, dtype=float)
    return (x - v.mean()) / v.std(ddof=0)


def gini(arr: List[float]) -> float:
    v = np.array([a for a in arr if a is not None], dtype=float)
    v = v[v >= 0]
    if v.size == 0:
        return np.nan
    if np.allclose(v.sum(), 0.0):
        return 0.0
    v_sorted = np.sort(v)
    n = v_sorted.size
    cum = np.cumsum(v_sorted)
    g = (n + 1 - 2*np.sum(cum) / cum[-1]) / n
    return g


def comb3(n: int) -> int:
    return int(n * (n - 1) * (n - 2) // 6) if n >= 3 else 0


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# Loaders

def load_temporal_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common variants of columns
    cmap = {
        "chapter": "Chapter",
        "network_density": "Network_Density",
    }
    df = df.rename(columns={c: cmap.get(c.lower(), c) for c in df.columns})
    df["Chapter"] = df["Chapter"].astype(int)
    return df


def load_relationship_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    low = {c.lower().strip(): c for c in df.columns}

    # Identify necessary fields
    src = next((low[c] for c in low if c in ["source","character a","char a","char_a","from","node1"]), None)
    tgt = next((low[c] for c in low if c in ["target","character b","char b","char_b","to","node2"]), None)
    sgn = next((low[c] for c in low if c in ["sign","relationship","edge_sign","valence"]), None)

    if src is None or tgt is None or sgn is None:
        raise RuntimeError(f"Cannot identify source/target/sign columns in {path.name}")

    out = pd.DataFrame({
        "Chapter": pd.to_numeric(df[low.get("chapter","Chapter")], errors="coerce"),
        "Source":  df[src].astype(str),
        "Target":  df[tgt].astype(str),
    })

    # map sign values to {-1,0,1}
    raw = df[sgn].astype(str).str.lower().str.strip()
    smap = {"positive":1,"pos":1,"1":1,"negative":-1,"neg":-1,"-1":-1,"neutral":0,"neu":0,"0":0}
    def to_sign(x):
        if x in smap:
            return smap[x]
        try:
            xi = int(float(x))
            return 1 if xi>0 else (-1 if xi<0 else 0)
        except:
            return 0

    out["Sign"] = raw.apply(to_sign).astype(int)

    # confidence optional
    confcol = next((low[c] for c in low if c in ["confidence","conf"]), None)
    if confcol:
        out["Confidence"] = pd.to_numeric(df[confcol], errors="coerce").fillna(1.0).clip(0,1)
    else:
        out["Confidence"] = 1.0

    out = out.dropna(subset=["Chapter"])
    out["Chapter"] = out["Chapter"].astype(int)

    # canonical undirected ordering
    u = out["Source"].where(out["Source"] <= out["Target"], out["Target"])
    v = out["Target"].where(out["Source"] <= out["Target"], out["Source"])
    out["Source"], out["Target"] = u, v
    return out


# Chapter graph construction and SBT helpers

def build_chapter_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df.iterrows():
        u, v = r["Source"], r["Target"]
        if u == v:
            continue
        s = int(r["Sign"])
        c = float(r["Confidence"])
        if G.has_edge(u, v):
            G[u][v]["confidence"] = max(G[u][v]["confidence"], c)
            prev_s = G[u][v]["sign"]
            if abs(s) > abs(prev_s):
                G[u][v]["sign"] = s
            elif abs(s) == abs(prev_s) and s != prev_s:
                G[u][v]["sign"] = 0
        else:
            G.add_edge(u, v, sign=s, confidence=c)
    return G


def community_count(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    if HAVE_LEIDEN:
        ig_g = ig.Graph()
        ig_g.add_vertices(list(G.nodes()))
        ig_g.add_edges(list(G.edges()))
        part = la.find_partition(ig_g, la.RBConfigurationVertexPartition)
        return len(part)
    if HAVE_LOUVAIN:
        part = community_louvain.best_partition(G)
        return len(set(part.values()))
    return nx.number_connected_components(G)


def count_signed_triads(G: nx.Graph) -> Tuple[int,int,int,float,float]:
    if G.number_of_nodes() < 3:
        return 0,0,0,0.0,0.0

    nodes = list(G.nodes())
    adj = {n: set(G.neighbors(n)) for n in nodes}

    bal = unbal = total = 0
    bal_w = unbal_w = 0.0

    for i, u in enumerate(nodes):
        Nu = adj[u]
        for j in range(i+1, len(nodes)):
            v = nodes[j]
            if v not in Nu:
                continue
            Nv = adj[v]
            common = [w for w in Nv if w in Nu and nodes.index(w) > j]
            for w in common:
                if not (G.has_edge(v,w) and G.has_edge(w,u)):
                    continue
                s_uv = G[u][v]["sign"]
                s_vw = G[v][w]["sign"]
                s_wu = G[w][u]["sign"]
                if 0 in (s_uv, s_vw, s_wu):
                    continue
                total += 1
                prod = s_uv * s_vw * s_wu
                c_uv = G[u][v]["confidence"]
                c_vw = G[v][w]["confidence"]
                c_wu = G[w][u]["confidence"]
                tri_w = c_uv * c_vw * c_wu
                if prod == 1:
                    bal += 1
                    bal_w += tri_w
                else:
                    unbal += 1
                    unbal_w += tri_w
    return bal, unbal, total, bal_w, unbal_w


# Per-book processing

def process_book(
    book_dir: Path,
    book_key: str,
    n_shuffles: int = 200,
    smooth_k: int = 3,
    betweenness_sample_nodes: Optional[int] = None
) -> Dict[str, Path]:
    """
    Compute temporal metrics for a single book.
    """
    files = {
        "temporal": next(book_dir.glob("*_temporal_metrics.csv"), None),
        "history": next(book_dir.glob("*_relationship_history.csv"), None),
    }
    if files["temporal"] is None or files["history"] is None:
        raise FileNotFoundError(f"Missing temporal/history files in {book_dir}")

    df_temp = load_temporal_metrics(files["temporal"])
    df_hist = load_relationship_history(files["history"])

    T = int(df_temp["Chapter"].max())
    rows = []
    dyad_records = []
    prev_signs = {}

    # Loop chapters
    for ch in range(1, T+1):
        x = normalize_time(ch, T)
        df_rel_ch = df_hist[df_hist["Chapter"] == ch]
        G = build_chapter_graph(df_rel_ch)

        active = G.number_of_nodes()
        edges = G.number_of_edges()

        # connectedness
        if edges > 0:
            lcc = max(nx.connected_components(G), key=len)
            lcc_pct = len(lcc) / active if active > 0 else 0
        else:
            lcc_pct = 1.0 if active > 0 else 0.0

        # communities
        n_comm = community_count(G)

        # degree metrics
        degs = dict(G.degree())
        sorted_deg = sorted(degs.values())
        top1deg = max(sorted_deg) if sorted_deg else 0
        if sorted_deg:
            k_eff = min(5, len(sorted_deg))
            topkmean = float(np.mean(sorted_deg[-k_eff:]))
            gini_deg = gini(sorted_deg)
        else:
            topkmean, gini_deg = 0.0, 0.0

        # betweenness
        if active > 1 and edges > 0:
            if betweenness_sample_nodes and betweenness_sample_nodes < active:
                sample = np.random.choice(list(G.nodes()), betweenness_sample_nodes, replace=False)
                bet = nx.betweenness_centrality_subset(G, sources=sample, targets=sample)
            else:
                bet = nx.betweenness_centrality(G)
            top1bet = max(bet.values()) if bet else 0.0
        else:
            top1bet = 0.0

        # sign composition
        pos_edges = int((df_rel_ch["Sign"] == 1).sum())
        neg_edges = int((df_rel_ch["Sign"] == -1).sum())
        neu_edges = int((df_rel_ch["Sign"] == 0).sum())
        total_edges = pos_edges + neg_edges + neu_edges
        negfrac = (neg_edges/total_edges) if total_edges > 0 else np.nan
        posminusneg = (pos_edges - neg_edges) / total_edges if total_edges > 0 else np.nan

        pos_w = float(df_rel_ch.loc[df_rel_ch["Sign"]==1, "Confidence"].sum())
        neg_w = float(df_rel_ch.loc[df_rel_ch["Sign"]==-1,"Confidence"].sum())
        neu_w = float(df_rel_ch.loc[df_rel_ch["Sign"]==0, "Confidence"].sum())
        total_w = pos_w + neg_w + neu_w
        negfrac_w = neg_w / total_w if total_w > 0 else np.nan
        posminusneg_w = (pos_w - neg_w) / total_w if total_w > 0 else np.nan

        # structural balance
        if active >= 3 and edges >= 3:
            bal, unbal, total_tri, bal_w, unbal_w = count_signed_triads(G)
            if total_tri > 0:
                balance_frac = bal / total_tri
                tension_frac = unbal / total_tri
                balance_frac_w = bal_w / (bal_w + unbal_w) if (bal_w+unbal_w) > 0 else np.nan
                tension_frac_w = unbal_w / (bal_w + unbal_w) if (bal_w+unbal_w) > 0 else np.nan
            else:
                balance_frac = tension_frac = np.nan
                balance_frac_w = tension_frac_w = np.nan
        else:
            balance_frac = tension_frac = np.nan
            balance_frac_w = tension_frac_w = np.nan
            total_tri = 0

        # dyad flips
        curr_signs = {}
        for _, r in df_rel_ch.iterrows():
            u, v, s = r["Source"], r["Target"], int(r["Sign"])
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            curr_signs[key] = s

        changed_edges_t = 0
        for key, s in curr_signs.items():
            prev = prev_signs.get(key)
            if prev is not None and s != prev:
                changed_edges_t += 1

        for key, s in curr_signs.items():
            prev = prev_signs.get(key)
            dyad_records.append({
                "Book_Key": book_key,
                "Chapter": ch,
                "Dyad_u": key[0],
                "Dyad_v": key[1],
                "prev_sign": prev if prev is not None else np.nan,
                "sign_t": s,
                "flip_01": 1 if prev is not None and s != prev else 0,
            })

        prev_signs = curr_signs

        rows.append({
            "Book_Key": book_key,
            "Chapter": ch,
            "x": x,
            "Active_Characters": active,
            "Edges": edges,
            "LCC_pct": lcc_pct,
            "Community_Count": n_comm,
            "Top1Deg": top1deg,
            "Top1Bet": top1bet,
            "TopKDegMean": topkmean,
            "GiniDegree": gini_deg,
            "Positive_Edges": pos_edges,
            "Negative_Edges": neg_edges,
            "Neutral_Edges": neu_edges,
            "NegFrac": negfrac,
            "PosMinusNeg": posminusneg,
            "NegFrac_w": negfrac_w,
            "PosMinusNeg_w": posminusneg_w,
            "Balanced_Triads": bal,
            "Unbalanced_Triads": unbal,
            "TotalSignedTriads": total_tri,
            "BalanceFrac": balance_frac,
            "TensionFrac": tension_frac,
            "BalanceFrac_w": balance_frac_w,
            "TensionFrac_w": tension_frac_w,
            "ChangedEdges_t": changed_edges_t,
        })

    df_ch = pd.DataFrame(rows).sort_values("Chapter").reset_index(drop=True)

    # smoothing + derivatives + z-scores
    for col in ["LCC_pct","Community_Count","Top1Deg","Top1Bet","TopKDegMean",
                "GiniDegree","NegFrac","PosMinusNeg","BalanceFrac","TensionFrac",
                "ChangedEdges_t"]:
        df_ch[f"{col}_sm"] = moving_average(df_ch[col], k=smooth_k)
        df_ch[f"d_{col}"] = first_derivative(df_ch[col])
        df_ch[f"z_{col}"] = zscore_series(df_ch[col])

    # save outputs
    out_ch = book_dir / f"{book_key}_temporal_metrics_enriched.csv"
    df_ch.to_csv(out_ch, index=False)

    out_dyad = book_dir / f"{book_key}_dyad_flips.csv"
    pd.DataFrame(dyad_records).to_csv(out_dyad, index=False)

    return {
        "chapter_panel": out_ch,
        "dyad_flips": out_dyad,
    }


# Corpus aggregation

def aggregate_corpus(book_dirs: List[Path], meta: pd.DataFrame, analysis_dir: Path):
    rows_ch = []
    rows_feat = []

    for bdir in book_dirs:
        # find enriched temporal file
        cands = list(bdir.glob("*_temporal_metrics_enriched.csv"))
        if not cands:
            continue
        f = cands[0]
        key = f.name.replace("_temporal_metrics_enriched.csv","")

        df = pd.read_csv(f)
        df["Book_Key"] = key
        meta_row = meta.loc[meta["Book_Key"] == key]
        if not meta_row.empty:
            for col in meta_row.columns:
                df[col] = meta_row.iloc[0][col]
        rows_ch.append(df)

        # features
        feat_file = bdir / f"{key}_features.csv"
        if feat_file.exists():
            rows_feat.append(pd.read_csv(feat_file))

    if rows_ch:
        pd.concat(rows_ch, ignore_index=True).to_csv(analysis_dir/"corpus_chapter_panel.csv", index=False)

    if rows_feat:
        pd.concat(rows_feat, ignore_index=True).to_csv(analysis_dir/"corpus_book_features.csv", index=False)


# CLI

def main():
    p = argparse.ArgumentParser(description="CharDyNet temporal metrics & aggregation")
    p.add_argument("--summaries-dir", type=Path, required=True,
                   help="Directory containing per-book folders with *_temporal_metrics.csv and *_relationship_history.csv")
    p.add_argument("--metadata-csv", type=Path, required=True,
                   help="CSV containing Book_Key and metadata fields")
    p.add_argument("--analysis-dir", type=Path, required=True,
                   help="Directory to write corpus-level outputs")
    p.add_argument("--n-shuffles", type=int, default=200)
    p.add_argument("--smooth-k", type=int, default=3)
    p.add_argument("--betweenness-sample-nodes", type=int, default=None)
    args = p.parse_args()

    summaries_dir = args.summaries_dir
    analysis_dir = args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    book_dirs = sorted([p for p in summaries_dir.iterdir() if p.is_dir()])

    for bdir in book_dirs:
        # detect Book_Key via filename
        cands = list(bdir.glob("*_temporal_metrics.csv"))
        if not cands:
            continue
        key = cands[0].name.replace("_temporal_metrics.csv","")

        try:
            process_book(
                bdir,
                key,
                n_shuffles=args.n_shuffles,
                smooth_k=args.smooth_k,
                betweenness_sample_nodes=args.betweenness_sample_nodes
            )
        except Exception as e:
            print(f"Error in {bdir}: {e}")

    aggregate_corpus(book_dirs, meta, analysis_dir)


if __name__ == "__main__":
    main()
