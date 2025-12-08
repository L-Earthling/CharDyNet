import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import imageio


# Configuration

def load_visualisation_config():
    return {
        "figure_size": (16, 10),
        "small_network": {
            "node_size_real": 2000,
            "font_size": 12,
            "edge_width": 2.0
        },
        "large_network": {
            "node_size_real": 1200,
            "font_size": 9,
            "edge_width": 1.5
        },
        "node_size_dummy": 500,
        "node_alpha_real": 0.3,
        "node_alpha_dummy": 0.01,

        "edge_color_map": {
            "neutral": "gray",
            "positive": "green",
            "negative": "red"
        },
        "title_font_size": 16,
        "gif_duration": 5.0,
        "video_fps": 1,
        "layout_seed": 42,
        "dummy_margin": 0.15,
        "large_network_threshold": 20,
        "large_network_layout": "fruchterman_reingold",
        "layout_weights": {
            "positive": 3.0,
            "neutral": 1.0,
            "negative": 0.3
        }
    }


# I/O utilities

def find_book_dirs(base_path: Path):
    return sorted([d for d in base_path.iterdir() if d.is_dir()])

def find_network_csv(book_dir: Path):
    files = list(book_dir.glob("*_network.csv"))
    return files[0] if files else None

def find_stats_csv(book_dir: Path):
    files = list(book_dir.glob("*_relationship_stats.csv"))
    return files[0] if files else None

def load_stats(stats_csv: Path):
    try:
        df = pd.read_csv(stats_csv)
        return dict(zip(df["Metric"], df["Value"]))
    except Exception:
        return None


# Data loading and cumulative networks

def load_network_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Unable to read {csv_path}: {e}")

    expected = ["Chapter", "Character A", "Character B", "Relationship"]
    if list(df.columns) != expected:
        # Preserve original recovery attempt
        df.columns = expected[:len(df.columns)]

    df["Chapter"] = df["Chapter"].astype(int)
    df["Relationship"] = df["Relationship"].str.lower().str.strip()
    df = df.dropna()

    return df

# Construct chapter → edge list mapping, with edges persisting across chapters
def build_cumulative(df):
    chapters = {}
    latest_edges = {}

    grouped = df.sort_values(by="Chapter").groupby("Chapter")
    for chap, g in grouped:
        for _, row in g.iterrows():
            a, b = row["Character A"], row["Character B"]
            rel = row["Relationship"]
            key = tuple(sorted([a, b]))
            latest_edges[key] = (a, b, rel)
        chapters[chap] = list(latest_edges.values())

    return chapters


# Layout

# Compute stable layout for final chapter. Implements identical behaviour to the prototype.
def compute_layout(final_edges, config, algorithm="spring", use_weighted=False):
    G = nx.Graph()
    for a, b, rel in final_edges:
        G.add_edge(a, b, relationship=rel)

    # Optional weighted layout
    if algorithm == "spring" and use_weighted:
        weight_map = config["layout_weights"]
        for (u, v, d) in G.edges(data=True):
            rel = d.get("relationship", "neutral")
            G[u][v]["weight"] = weight_map.get(rel, 1.0)
        pos = nx.spring_layout(G, seed=config["layout_seed"], weight="weight")
    else:
        if algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, seed=config["layout_seed"])
        elif algorithm == "fruchterman_reingold":
            pos = nx.fruchterman_reingold_layout(G, seed=config["layout_seed"])
        elif algorithm == "sfdp":
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
            except Exception:
                pos = nx.fruchterman_reingold_layout(G, seed=config["layout_seed"])
        else:
            pos = nx.spring_layout(G, seed=config["layout_seed"])

    # Dummy node stabilization
    xs, ys = zip(*pos.values())
    margin = config["dummy_margin"]

    dummy = {
        "dummy1": (min(xs)-margin, min(ys)-margin),
        "dummy2": (min(xs)-margin, max(ys)+margin),
        "dummy3": (max(xs)+margin, min(ys)-margin),
        "dummy4": (max(xs)+margin, max(ys)+margin),
    }

    pos.update(dummy)
    return pos, dummy.keys()


# Rendering

def render_chapter_png(G, pos, dummy_nodes, config, params, chap, title, out_path, character_count):
    plt.figure(figsize=config["figure_size"])

    # Edge colours
    ec = []
    for u, v, d in G.edges(data=True):
        rel = d.get("relationship", "neutral")
        ec.append(config["edge_color_map"].get(rel, "gray"))
    nx.draw_networkx_edges(G, pos, edge_color=ec, width=params["edge_width"])

    # Nodes
    for node in G.nodes():
        if node in dummy_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node],
                node_color="white",
                edgecolors="black",
                node_size=config["node_size_dummy"],
                alpha=config["node_alpha_dummy"]
            )
        else:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node],
                node_color="white",
                edgecolors="black",
                node_size=params["node_size_real"],
                alpha=config["node_alpha_real"]
            )

    # Labels
    labels = {n: n for n in G.nodes() if n not in dummy_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=params["font_size"])

    plt.axis("off")

    title_txt = f"{title}\nPartition {chap}"
    if isinstance(character_count, int):
        title_txt += f" ({character_count} characters)"

    plt.title(title_txt, fontsize=config["title_font_size"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def build_animation(png_paths, gif_path, mp4_path, config):
    images = []
    for p in png_paths:
        try:
            images.append(imageio.v2.imread(p))
        except Exception:
            pass

    if images:
        imageio.mimsave(gif_path, images, duration=config["gif_duration"], loop=0)
        with imageio.get_writer(mp4_path, fps=config["video_fps"]) as w:
            for img in images:
                w.append_data(img)


# Per-book pipeline

def process_book(book_dir: Path, config, layout_override=None):
    net_csv = find_network_csv(book_dir)
    if net_csv is None:
        return False, "No network CSV found"

    df = load_network_csv(net_csv)
    chapters = build_cumulative(df)

    # Character count detection
    stats_csv = find_stats_csv(book_dir)
    character_count = None
    if stats_csv:
        stats = load_stats(stats_csv)
        if stats and "Unique Characters" in stats:
            character_count = int(stats["Unique Characters"])

    # Decide small vs large
    params = config["small_network"]
    is_large = False
    if isinstance(character_count, int) and character_count > config["large_network_threshold"]:
        params = config["large_network"]
        is_large = True

    # Layout selection
    algorithm = layout_override
    if algorithm is None:
        algorithm = config["large_network_layout"] if is_large else "spring"

    final_edges = chapters[max(chapters.keys())]
    pos, dummy_nodes = compute_layout(final_edges, config, algorithm)

    # Title (same formatting rules preserved)
    stem = net_csv.stem.replace("_network", "")
    if "_" in stem:
        a, t = stem.split("_", 1)
        title = f"{a.replace('-', ' ')} – {t.replace('-', ' ')}"
    else:
        title = stem.replace("-", " ")

    out_dir = book_dir / f"vis_{stem}_{algorithm}"
    out_dir.mkdir(exist_ok=True)

    png_paths = []
    for chap in sorted(chapters.keys()):
        G = nx.Graph()
        for a, b, rel in chapters[chap]:
            G.add_edge(a, b, relationship=rel)
        for d in dummy_nodes:
            G.add_node(d)

        out_png = out_dir / f"partition_{chap:02d}.png"
        render_chapter_png(G, pos, dummy_nodes, config, params, chap, title, out_png, character_count)
        png_paths.append(out_png)

    gif_path = out_dir / f"{stem}_network.gif"
    mp4_path = out_dir / f"{stem}_network.mp4"
    build_animation(png_paths, gif_path, mp4_path, config)

    return True, None


# Batch driver

def run_all(base_path: Path, layout=None):
    config = load_visualisation_config()
    dirs = find_book_dirs(base_path)

    for d in dirs:
        ok, err = process_book(d, config, layout_override=layout)
        if ok:
            print(f"Processed {d.name}")
        else:
            print(f"Failed {d.name}: {err}")


# CLI

def main():
    parser = argparse.ArgumentParser(description="Temporal network visualisation")
    parser.add_argument("--base", type=str, required=True,
                        help="Path to directory containing per-book folders")
    parser.add_argument("--layout", type=str, default=None,
                        help="Force specific layout algorithm")
    parser.add_argument("--single", type=str, default=None,
                        help="Process only this book folder name")

    args = parser.parse_args()
    base = Path(args.base)

    if args.single:
        book_dir = base / args.single
        ok, err = process_book(book_dir, load_visualisation_config(),
                               layout_override=args.layout)
        if ok:
            print(f"Processed {args.single}")
        else:
            print(f"Failed: {err}")
    else:
        run_all(base, layout=args.layout)


if __name__ == "__main__":
    main()
