<p align="center"> 

  <img src="figures/CharDyNet_logo.png" alt="CharDyNet logo" width="420"> 

</p> 

  

<h1 align="center">CharDyNet: Temporal Character Networks and Social Dynamics in Literary Texts</h1> 

  

<p align="center"> 

  <div align="center">
    <a href="https://raw.githubusercontent.com/L-Earthling/CharDyNet/refs/heads/main/dataset/CharDyNet_Sparknotes_all_networks.csv">
      <img
        alt="Download CharDyNet-v1"
        src="https://img.shields.io/badge/Download-CharDyNet__prototype__all__networks-blue?style=for-the-badge"
      >
    </a>
  </div> 

</p> 

  

--- 

  

## Table of Contents 

| [TL;DR](#-tldr) | [Features](#-features) | [Pipeline](#-pipeline-overview) | [Description](#-description) | [Installation](#-installation) | 

  

--- 

  

## 🧭 TL;DR 

**CharDyNet** transforms literary texts into evolving social graphs.   

It combines **LLMs**, **temporal network analysis**, and **structural balance theory** to study how alliances, conflicts, and communities evolve through narrative time.   

Explore the **CharDyNet-SN sandbox** to see analysis and chapter-by-chapter visualisations for 600+ web-scraped-based networks. 

  

<p align="center"> 

  <img src="figures/pipeline_overview.png" alt="Pipeline Overview" width="780"> 

</p> 

  

--- 

  

## ✨ Features 

- **End-to-end pipeline** – from raw or summarised text to temporal, signed graphs 

- **LLM-based relationship extraction** – identifies characters and assigns tie polarity (+ / – / 0) 

- **Temporal updates** – tracks evolving edges, sign flips, and network volatility 

- **Structural balance analysis** – computes balanced/unbalanced triads, frustration, tension 

- **Network metrics** – density, centrality, cohesion, community fragmentation, protagonist trajectory 

- **Corpus-level aggregation** – genre-, author-, and period-level comparisons 

- **Visual outputs** – multi-panel per-book time series and animated GIFs 

- **Reproducible setup** – consistent seeds, metadata manifests, and modular Python scripts 

  

--- 

  

## 🔬 Pipeline Overview 

1. **Text Segmentation** – parse chapters/partitions (SparkNotes/Gutenberg or custom). 

2. **Character Extraction** – identify entities and unify aliases (e.g., “Aragorn” → “Strider”). 

3. **Relation Extraction** – LLM prompting to classify pairwise relations (positive/negative/neutral). 

4. **Temporal Graph Construction** – per-chapter signed networks; track updates & flips. 

5. **Metric Computation** – density, balance, frustration, flip-rate, etc. 

6. **Visualisation & Export** – temporal multi-panels, protagonist trajectories, GIFs. 

7. **Corpus-Level Analysis** – aggregate by genre/author/epoch; clustering & stats. 

  

--- 

  

## 📊 Example Outputs 

Each processed book includes: 

- `*_network.csv` – extracted signed relationships 

- `*_temporal_metrics_enriched.csv` – per-chapter metrics 

- `*_relationship_stats.csv` – per-book metric stats 

- `*_protagonist_traj.csv` – protagonist centrality trajectory 

- `*_temporal_9panel.png` – per-book visual summary 

- `*_dyad_stability_topK15.png` – stability of key relationships 

  

### Visual Examples 

  

<p align="center"> 

  <img src="figures/CharDyNet_network_example.gif" alt="Animated evolving character network" width="720"><br> 

  <em>Animated evolving character network (chapter-by-chapter)</em> 

</p> 



<p align="center"> 

  <img src="figures/CharDyNet_network_example.png" alt="Character relationship network" width="720"><br> 

  <em>Static snapshot of a signed character network</em> 

</p> 

  

<p align="center"> 

  <img src="figures/temporal_9panel_example.png" alt="Per-book nine-panel analysis" width="720"><br> 

  <em>Nine-panel temporal summary: balance, density, centralities, and more</em> 

</p> 

  

<p align="center"> 

  <img src="figures/CharDyNet_relationship_stats_example.png" alt="Per-book relationship statistics" width="720"><br> 

  <em>Per-book relationship statistics</em> 

</p> 

  

<p align="center"> 

  <img src="figures/CharDyNet_dyad_stability_topk15_example.png" alt="Stability of key relationships" width="720"><br> 

  <em>Top-15 dyad stability across the narrative</em> 

</p> 

  



 

--- 

  

## ⚙️ Installation 

```bash 

# 1) (optional) create env 

python3 -m venv .venv && source .venv/bin/activate 

  

# 2) install requirements 

pip install -r requirements.txt 
