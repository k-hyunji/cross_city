"""
Visualize region embeddings with t-SNE.
Generates 2-row × N-col figure:
  Row 1: colored by city
  Row 2: colored by log1p(pop_density)

Usage:
  python scripts/visualize_embeddings.py \
      --emb_dirs embeddings/nyc_seoul embeddings/nyc_seoul_0501 \
      --labels "tau=0.15 (pre-Fix2)" "Fix2 UnifiedCityCon" \
      --output figures/tsne_comparison.png
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CITIES = [
    ("nyc",       "BoroCT2020", "NYC",       "#4C8BE2"),
    ("seoul",     "ADM_CD",     "Seoul",     "#E2774C"),
    ("singapore", "row_index",  "Singapore", "#4CE27A"),
]

def load_emb(path):
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0:
        x = x.item()
        if isinstance(x, dict):
            x = x.get("embeddings", list(x.values())[0])
    return x.astype(np.float32)

def load_pop(city, id_col, data_root="data"):
    csv = os.path.join(data_root, city, "pop_gt.csv")
    df  = pd.read_csv(csv)
    if id_col == "row_index":
        df[id_col] = df[id_col].astype(int).astype(str)
    else:
        try:
            df[id_col] = df[id_col].astype(float).astype(int).astype(str)
        except (ValueError, TypeError):
            df[id_col] = df[id_col].astype(str)
    return df.set_index(id_col)["population_density"]

def load_region_ids(city, data_root="data"):
    """Read region IDs in the same order as the embedding file."""
    import sys
    from data.dataset import UrbanRegionDataset
    ds = UrbanRegionDataset(city, data_root)
    return [str(ds[i]["region_id"]) for i in range(len(ds))]

def run_tsne(embs_list, perplexity=40, seed=42):
    all_embs = np.concatenate(embs_list, axis=0)
    all_embs = StandardScaler().fit_transform(all_embs)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, n_iter=1000, init="pca")
    return tsne.fit_transform(all_embs)

def split_coords(coords, sizes):
    out, start = [], 0
    for s in sizes:
        out.append(coords[start:start+s])
        start += s
    return out

def plot_panel_city(ax, coords_list, title):
    for (city_id, id_col, label, color), coords in zip(CITIES, coords_list):
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=4, alpha=0.55, color=color, label=label, linewidths=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])

def plot_panel_pop(ax, coords_list, pop_list, title):
    all_c = np.concatenate(pop_list)
    vmin, vmax = np.nanpercentile(all_c, 2), np.nanpercentile(all_c, 98)
    all_xy = np.concatenate(coords_list, axis=0)
    sc = ax.scatter(all_xy[:, 0], all_xy[:, 1],
                    c=all_c, cmap="plasma", s=4, alpha=0.55,
                    vmin=vmin, vmax=vmax, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(pop density)")
    ax.set_title(title + "\n(colored by pop density)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--labels",   nargs="+", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output",    default="figures/tsne_comparison.png")
    parser.add_argument("--perplexity", type=int, default=40)
    args = parser.parse_args()

    assert len(args.emb_dirs) == len(args.labels), "need one label per emb_dir"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Loading region IDs and pop densities...")
    pop_by_city = {}
    ids_by_city = {}
    for city, id_col, label, _ in CITIES:
        ids_by_city[city] = load_region_ids(city, args.data_root)
        pop_ser = load_pop(city, id_col, args.data_root)
        pop_by_city[city] = np.array([
            np.log1p(float(pop_ser.get(rid, np.nan)))
            for rid in ids_by_city[city]
        ])

    ncols = len(args.emb_dirs)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 11))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, (emb_dir, run_label) in enumerate(zip(args.emb_dirs, args.labels)):
        print(f"\nProcessing: {run_label}  ({emb_dir})")
        embs_list, pop_list, sizes = [], [], []
        for city, id_col, label, color in CITIES:
            path = os.path.join(emb_dir, f"{city}_region_emb.npy")
            emb  = load_emb(path)
            embs_list.append(emb)
            pop_list.append(pop_by_city[city])
            sizes.append(len(emb))
            print(f"  {city}: {len(emb)} regions")

        print("  Running t-SNE...")
        coords  = run_tsne(embs_list, perplexity=args.perplexity)
        c_list  = split_coords(coords, sizes)

        plot_panel_city(axes[0, col], c_list, run_label)
        plot_panel_pop (axes[1, col], c_list, pop_list, run_label)

    fig.suptitle("Region Embedding t-SNE: City vs Functional Structure", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
