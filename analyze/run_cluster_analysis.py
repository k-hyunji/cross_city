"""
Cross-city urban region clustering analysis.
Part A: City-mean subtraction + cross-city K-Means
Part B: Within-city clustering vs population density + land use
"""

import sys
import os
import ast
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/lab05/hyunji/cross_city6')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE

from data.dataset import UrbanRegionDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT        = '/home/lab05/hyunji/cross_city6/data'
LANDUSE_ROOT     = '/home/lab05/hyunji/cross_city5_gpt/data'
FIGURES_DIR      = '/home/lab05/hyunji/cross_city6/figures'
CITIES           = ['nyc', 'seoul', 'singapore']

CITY_COLORS = {'nyc': '#4C8BE2', 'seoul': '#E2774C', 'singapore': '#4CE27A'}

LANDUSE_CONFIG = {
    'nyc':       {'id_col': 'BoroCT2020', 'dims': 5},
    'seoul':     {'id_col': 'ADM_CD',     'dims': 11},
    'singapore': {'id_col': 'region_id',  'dims': 5},
}

LANDUSE_NAMES = {
    'nyc':       ['Residential', 'Commercial/Mixed', 'Industrial', 'Transport & Public', 'Open Space/Vacant'],
    'seoul':     ['Green & Open Space', 'Public Use', 'Rivers & Lakes', 'Special Area',
                  'Residential', 'Commercial & Business', 'Urban Support', 'Transportation',
                  'Vacant/Bare Land', 'Res/Comm Mixed', 'Industrial'],
    'singapore': ['Residential', 'Commercial', 'Industrial', 'Public/Transport/Inst', 'Parks/Other'],
}

POP_CONFIG = {
    'nyc':       {'id_col': 'BoroCT2020', 'val_col': 'population_density'},
    'seoul':     {'id_col': 'ADM_CD',     'val_col': 'population_density'},
    'singapore': {'id_col': 'row_index',  'val_col': 'population_density'},
}

K_VALUES = [3, 5, 8]

os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: nanmean over valid BG rows; fill remaining NaN with column mean
# ---------------------------------------------------------------------------

def region_mean(mat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """mat: [M, D], valid_mask: [M] bool → [D] float"""
    if valid_mask.sum() > 0:
        sub = mat[valid_mask].astype(float)
    else:
        sub = mat.astype(float)
    mean_vec = np.nanmean(sub, axis=0)
    # fill any remaining NaN (all-NaN column) with 0
    mean_vec = np.where(np.isnan(mean_vec), 0.0, mean_vec)
    return mean_vec


def fill_col_nan(X: np.ndarray) -> np.ndarray:
    """Fill column NaNs with column mean (or 0 if all NaN)."""
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col_mean = np.nanmean(col)
            col[nan_mask] = 0.0 if np.isnan(col_mean) else col_mean
            X[:, j] = col
    return X

# ---------------------------------------------------------------------------
# Load landuse for a city → dict {region_id_str: np.array([dims])}
# ---------------------------------------------------------------------------

def load_landuse(city: str) -> dict:
    lu_path = os.path.join(LANDUSE_ROOT, city, 'landuse_gt.csv')
    cfg = LANDUSE_CONFIG[city]
    id_col = cfg['id_col']

    df = pd.read_csv(lu_path)

    if city == 'seoul':
        # columns: ZoneID, population_density, ADM_NM, name_norm, ADM_CD, ...
        id_col = 'ADM_CD'
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
    elif city == 'nyc':
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
    elif city == 'singapore':
        # region_id is string e.g. "MARINA EAST"
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()

    lu_dict = {}
    for _, row in df.iterrows():
        rid = str(row[id_col]).strip()
        if city == 'singapore':
            rid = rid.upper()
        lu_arr = np.array(ast.literal_eval(row['land_use']), dtype=float)
        lu_dict[rid] = lu_arr

    print(f"  [Landuse] {city}: {len(lu_dict)} regions loaded from {lu_path}")
    return lu_dict


# ---------------------------------------------------------------------------
# Load population density for a city → dict {region_id_str: float}
# ---------------------------------------------------------------------------

def load_pop(city: str) -> dict:
    pop_path = os.path.join(DATA_ROOT, city, 'pop_gt.csv')
    cfg = POP_CONFIG[city]
    id_col  = cfg['id_col']
    val_col = cfg['val_col']

    df = pd.read_csv(pop_path)

    if city == 'singapore':
        # row_index is integer index; region_id is the name
        df['row_index'] = df['row_index'].astype(int).astype(str)
        key_col = 'row_index'
    elif city == 'nyc':
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
        key_col = id_col
    else:
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
        key_col = id_col

    pop_dict = {}
    for _, row in df.iterrows():
        rid = str(row[key_col]).strip()
        pop_dict[rid] = float(row[val_col])

    print(f"  [Pop] {city}: {len(pop_dict)} regions loaded")
    return pop_dict


# ---------------------------------------------------------------------------
# Singapore: dataset region_id is row_index (str int), but landuse uses name
# We need a mapping row_index → region_name
# ---------------------------------------------------------------------------

def load_singapore_id_map() -> dict:
    """Returns {row_index_str: region_name_upper}"""
    pop_path = os.path.join(DATA_ROOT, 'singapore', 'pop_gt.csv')
    df = pd.read_csv(pop_path)
    mapping = {}
    for _, row in df.iterrows():
        ri = str(int(row['row_index']))
        name = str(row['region_id']).strip().upper()
        mapping[ri] = name
    return mapping


# ===========================================================================
# MAIN DATA LOADING
# ===========================================================================

print("=" * 70)
print("Loading datasets...")
print("=" * 70)

datasets = {}
for city in CITIES:
    ds = UrbanRegionDataset(city, DATA_ROOT, satellite_mode='npy')
    datasets[city] = ds

# Singapore row_index → region_name map (for landuse lookup)
sg_id_map = load_singapore_id_map()

# Load landuse and pop dicts
landuse_dicts = {city: load_landuse(city) for city in CITIES}
pop_dicts     = {city: load_pop(city)     for city in CITIES}

# ---------------------------------------------------------------------------
# Build region-level feature matrices for each city
# ---------------------------------------------------------------------------

print("\nBuilding region-level feature matrices...")

city_data = {}  # city → dict with sat_feats, poi_feats, region_ids, pop, landuse, etc.

for city in CITIES:
    ds  = datasets[city]
    lu_dict  = landuse_dicts[city]
    pop_dict = pop_dicts[city]
    lu_dims  = LANDUSE_CONFIG[city]['dims']
    lu_names = LANDUSE_NAMES[city]

    n_regions = len(ds)
    print(f"\n  {city.upper()}: {n_regions} regions")

    sat_list  = []
    poi_list  = []
    region_ids = []
    pop_vals   = []
    lu_vecs    = []

    for i in range(n_regions):
        item      = ds[i]
        rid_raw   = item['region_id']           # str
        sat_data  = item['sat_data'].numpy()    # [M, sat_dim]
        poi_data  = item['poi_emb'].numpy()     # [M, poi_dim]
        valid_mask = item['valid_mask'].numpy() # [M] bool

        # Compute region-level mean
        sat_vec = region_mean(sat_data, valid_mask)
        poi_vec = region_mean(poi_data, valid_mask)  # all BGs valid for POI

        sat_list.append(sat_vec)
        poi_list.append(poi_vec)
        region_ids.append(rid_raw)

        # Population
        pop_key = rid_raw
        pop_vals.append(pop_dict.get(pop_key, np.nan))

        # Landuse
        if city == 'singapore':
            lu_key = sg_id_map.get(rid_raw, '').upper()
        else:
            lu_key = rid_raw

        if lu_key in lu_dict:
            lu_vecs.append(lu_dict[lu_key])
        else:
            lu_vecs.append(np.full(lu_dims, np.nan))

    sat_feats  = np.array(sat_list)   # [N, sat_dim]
    poi_feats  = np.array(poi_list)   # [N, poi_dim]
    pop_arr    = np.array(pop_vals, dtype=float)
    lu_arr     = np.array(lu_vecs, dtype=float)  # [N, lu_dims]

    # Fill NaN in features with column mean
    sat_feats = fill_col_nan(sat_feats)
    poi_feats = fill_col_nan(poi_feats)

    # Dominant land use class
    lu_valid_mask = ~np.all(np.isnan(lu_arr), axis=1)  # [N] bool
    lu_dominant   = np.full(n_regions, -1, dtype=int)
    for j in range(n_regions):
        if lu_valid_mask[j]:
            lu_dominant[j] = int(np.argmax(lu_arr[j]))

    city_data[city] = {
        'sat':        sat_feats,
        'poi':        poi_feats,
        'region_ids': region_ids,
        'pop':        pop_arr,
        'lu_arr':     lu_arr,
        'lu_dominant': lu_dominant,
        'lu_valid':   lu_valid_mask,
        'lu_dims':    lu_dims,
        'lu_names':   lu_names,
        'n':          n_regions,
    }

    n_lu_found = lu_valid_mask.sum()
    n_pop_found = np.sum(~np.isnan(pop_arr))
    print(f"    sat_feats: {sat_feats.shape}, poi_feats: {poi_feats.shape}")
    print(f"    pop found: {n_pop_found}/{n_regions}, landuse found: {n_lu_found}/{n_regions}")


# ===========================================================================
# PART A — City-mean subtraction + cross-city K-Means
# ===========================================================================

print("\n" + "=" * 70)
print("PART A — City-mean subtraction + cross-city K-Means")
print("=" * 70)

# Concatenate all cities
all_sat  = np.concatenate([city_data[c]['sat'] for c in CITIES], axis=0)
all_poi  = np.concatenate([city_data[c]['poi'] for c in CITIES], axis=0)
all_city_labels = np.concatenate([
    np.full(city_data[c]['n'], i) for i, c in enumerate(CITIES)
])  # 0=NYC, 1=Seoul, 2=Singapore

# City means
city_sat_means = {}
city_poi_means = {}
offsets = {}
start = 0
for c in CITIES:
    n = city_data[c]['n']
    offsets[c] = (start, start + n)
    city_sat_means[c] = all_sat[start:start+n].mean(axis=0)
    city_poi_means[c] = all_poi[start:start+n].mean(axis=0)
    start += n

print("\nCity sat/poi means computed:")
for c in CITIES:
    print(f"  {c}: sat_mean norm={np.linalg.norm(city_sat_means[c]):.3f}, "
          f"poi_mean norm={np.linalg.norm(city_poi_means[c]):.3f}")

# Demeaned features
all_sat_dm = all_sat.copy()
all_poi_dm = all_poi.copy()
for c in CITIES:
    s, e = offsets[c]
    all_sat_dm[s:e] -= city_sat_means[c]
    all_poi_dm[s:e] -= city_poi_means[c]

# t-SNE for Figure 1 (compute once each for original and demeaned)
print("\nRunning t-SNE for Part A visualizations...")
n_total = len(all_city_labels)
perp = min(30, n_total // 5)

tsne_kwargs = dict(perplexity=perp, random_state=42, n_iter=1000, init='pca')

print(f"  t-SNE sat original  (n={n_total}, perp={perp})...")
tsne_sat_orig = TSNE(n_components=2, **tsne_kwargs).fit_transform(
    StandardScaler().fit_transform(all_sat))

print(f"  t-SNE poi original  (n={n_total}, perp={perp})...")
tsne_poi_orig = TSNE(n_components=2, **tsne_kwargs).fit_transform(
    StandardScaler().fit_transform(all_poi))

print(f"  t-SNE sat demeaned  (n={n_total}, perp={perp})...")
tsne_sat_dm = TSNE(n_components=2, **tsne_kwargs).fit_transform(
    StandardScaler().fit_transform(all_sat_dm))

print(f"  t-SNE poi demeaned  (n={n_total}, perp={perp})...")
tsne_poi_dm = TSNE(n_components=2, **tsne_kwargs).fit_transform(
    StandardScaler().fit_transform(all_poi_dm))

# K-Means on original vs demeaned
print("\nK-Means clustering (original vs demeaned):")
print(f"  {'':25s} | {'K':>3s} | {'ARI':>6s} | {'AvgPurity':>9s} | City composition (%)")
print("  " + "-" * 80)

ari_results = {}

for feat_name, X_orig, X_dm in [('SAT', all_sat, all_sat_dm),
                                  ('POI', all_poi, all_poi_dm)]:
    for tag, X in [('Original', X_orig), ('Demeaned', X_dm)]:
        Xs = StandardScaler().fit_transform(X)
        for k in K_VALUES:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)

            ari = adjusted_rand_score(all_city_labels, labels)

            # AvgCityPurity: for each cluster, dominant city fraction; average
            purities = []
            city_comp_rows = []
            for cl in range(k):
                mask = labels == cl
                if mask.sum() == 0:
                    continue
                city_counts = np.bincount(all_city_labels[mask], minlength=3)
                purity = city_counts.max() / city_counts.sum()
                purities.append(purity)
                pct = city_counts / city_counts.sum() * 100
                city_comp_rows.append(pct)

            avg_purity = np.mean(purities)
            comp_str = ' | '.join(
                f"C{cl}:[NYC={city_comp_rows[cl][0]:.0f}% SEO={city_comp_rows[cl][1]:.0f}% SG={city_comp_rows[cl][2]:.0f}%]"
                for cl in range(len(city_comp_rows))
            )
            key = f"{feat_name}_{tag}_K{k}"
            ari_results[key] = {'ari': ari, 'avg_purity': avg_purity}

            print(f"  {feat_name} {tag:8s} K={k} | ARI={ari:+.3f} | AvgPurity={avg_purity:.3f}")
            for cl_i, pct in enumerate(city_comp_rows):
                print(f"    Cluster {cl_i}: NYC={pct[0]:.1f}%  Seoul={pct[1]:.1f}%  SG={pct[2]:.1f}%")

print("\nSummary comparison (Demeaned vs Original, ARI):")
for feat in ['SAT', 'POI']:
    for k in K_VALUES:
        orig_ari = ari_results[f"{feat}_Original_K{k}"]['ari']
        dm_ari   = ari_results[f"{feat}_Demeaned_K{k}"]['ari']
        delta    = dm_ari - orig_ari
        print(f"  {feat} K={k}: Original ARI={orig_ari:+.3f}  →  Demeaned ARI={dm_ari:+.3f}  "
              f"(Δ={delta:+.3f})")


# ===========================================================================
# PART B — Within-city clustering with landuse
# ===========================================================================

print("\n" + "=" * 70)
print("PART B — Within-city clustering with population + land use")
print("=" * 70)

within_city_results = {}

for city in CITIES:
    cd = city_data[city]
    lu_names = cd['lu_names']
    lu_dims  = cd['lu_dims']
    within_city_results[city] = {}

    for feat_name, X in [('sat', cd['sat']), ('poi', cd['poi'])]:
        Xs = StandardScaler().fit_transform(X)
        within_city_results[city][feat_name] = {}

        for k in K_VALUES:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(Xs)

            # Silhouette (skip if k >= n)
            if k < len(X):
                sil = silhouette_score(Xs, cluster_labels)
            else:
                sil = np.nan

            # Per-cluster stats
            cluster_stats = []
            for cl in range(k):
                mask = cluster_labels == cl
                n_cl = mask.sum()

                # Mean pop density
                pop_vals_cl = cd['pop'][mask]
                mean_pop = np.nanmean(pop_vals_cl)

                # Land use distribution (only valid regions)
                lu_valid_cl = cd['lu_valid'][mask]
                if lu_valid_cl.sum() > 0:
                    lu_cl = cd['lu_arr'][mask][lu_valid_cl]  # [n_valid, lu_dims]
                    lu_dist = np.nanmean(lu_cl, axis=0)
                    dom_cls = int(np.argmax(lu_dist))
                    lu_pct  = lu_dist / (lu_dist.sum() + 1e-9) * 100
                else:
                    lu_dist  = np.zeros(lu_dims)
                    dom_cls  = -1
                    lu_pct   = np.zeros(lu_dims)

                cluster_stats.append({
                    'n': int(n_cl),
                    'mean_pop': mean_pop,
                    'dom_lu': dom_cls,
                    'lu_pct': lu_pct,
                })

            within_city_results[city][feat_name][k] = {
                'labels': cluster_labels,
                'sil':    sil,
                'stats':  cluster_stats,
            }

# Print clean table
print("\n{:<12s} {:<6s} {:<4s} {:>6s} | {}".format(
    'City', 'Feat', 'K', 'Sil', 'Cluster stats'))
print("-" * 100)

for city in CITIES:
    cd       = city_data[city]
    lu_names = cd['lu_names']
    for feat_name in ['sat', 'poi']:
        for k in K_VALUES:
            res    = within_city_results[city][feat_name][k]
            sil    = res['sil']
            stats  = res['stats']
            sil_s  = f"{sil:.3f}" if not np.isnan(sil) else "  N/A"
            print(f"\n{city:<12s} {feat_name:<6s} K={k} Sil={sil_s}")
            for ci, st in enumerate(stats):
                dom = st['dom_lu']
                dom_name = lu_names[dom] if 0 <= dom < len(lu_names) else 'N/A'
                lu_str = ' '.join(f"{lu_names[j][:6]}:{st['lu_pct'][j]:.0f}%"
                                  for j in range(len(lu_names)))
                print(f"  Cluster {ci} (n={st['n']:3d}): "
                      f"pop={st['mean_pop']:8.1f}  dom={dom_name:<18s}  [{lu_str}]")


# ===========================================================================
# FIGURE 1 — City-mean subtracted t-SNE
# ===========================================================================

print("\nSaving Figure 1: cluster_citymean_subtracted.png ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
city_labels_list = [all_city_labels == i for i in range(3)]

plot_configs = [
    (0, 0, 'Satellite — Original',          tsne_sat_orig),
    (0, 1, 'Satellite — City-mean Subtracted', tsne_sat_dm),
    (1, 0, 'POI — Original',                 tsne_poi_orig),
    (1, 1, 'POI — City-mean Subtracted',      tsne_poi_dm),
]

for row, col, title, emb in plot_configs:
    ax = axes[row, col]
    for i, c in enumerate(CITIES):
        mask = all_city_labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CITY_COLORS[c], label=c.upper(), s=15, alpha=0.7,
                   linewidths=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    if row == 0 and col == 0:
        ax.legend(fontsize=9, markerscale=2)

plt.suptitle('t-SNE: Cross-city Feature Distribution\n(Original vs City-mean Subtracted)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cluster_citymean_subtracted.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: cluster_citymean_subtracted.png")


# ===========================================================================
# Helper: within-city t-SNE + K=5 cluster / pop / landuse figure
# ===========================================================================

def make_withincity_figure(feat_name: str, save_name: str):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    for row_i, city in enumerate(CITIES):
        cd       = city_data[city]
        lu_names = cd['lu_names']
        lu_dims  = cd['lu_dims']
        X        = cd['sat'] if feat_name == 'sat' else cd['poi']
        n        = cd['n']

        perp_c = min(30, n // 5)
        print(f"  t-SNE {feat_name} {city} (n={n}, perp={perp_c})...")
        Xs   = StandardScaler().fit_transform(X)
        tsne_c = TSNE(n_components=2, perplexity=perp_c, random_state=42,
                      n_iter=1000, init='pca').fit_transform(Xs)

        k5_labels = within_city_results[city][feat_name][5]['labels']

        # Col 0: t-SNE colored by K=5 cluster
        ax = axes[row_i, 0]
        scatter = ax.scatter(tsne_c[:, 0], tsne_c[:, 1],
                             c=k5_labels, cmap='tab10', vmin=0, vmax=9,
                             s=20, alpha=0.8, linewidths=0)
        ax.set_title(f'{city.upper()} — K=5 Clusters', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

        # Col 1: t-SNE colored by log1p(pop_density)
        ax = axes[row_i, 1]
        pop_vals_c = cd['pop'].copy()
        pop_vals_log = np.log1p(np.where(np.isnan(pop_vals_c), 0.0, pop_vals_c))
        sc = ax.scatter(tsne_c[:, 0], tsne_c[:, 1],
                        c=pop_vals_log, cmap='plasma',
                        s=20, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='log1p(pop density)')
        ax.set_title(f'{city.upper()} — Pop Density', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

        # Col 2: t-SNE colored by dominant land use class
        ax = axes[row_i, 2]
        lu_dom = cd['lu_dominant'].astype(float)
        lu_dom_plot = lu_dom.copy()
        lu_dom_plot[lu_dom < 0] = np.nan  # mask invalid

        cmap_lu = plt.get_cmap('Set2', lu_dims)
        colors_arr = np.array([cmap_lu(lu_dom_plot[j]) if not np.isnan(lu_dom_plot[j])
                                else (0.7, 0.7, 0.7, 0.5)
                                for j in range(n)])
        ax.scatter(tsne_c[:, 0], tsne_c[:, 1],
                   c=colors_arr, s=20, alpha=0.8, linewidths=0)

        # Legend
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=cmap_lu(i), label=lu_names[i])
                          for i in range(lu_dims)]
        ax.legend(handles=legend_handles, fontsize=6, loc='best',
                  framealpha=0.7, markerscale=1)
        ax.set_title(f'{city.upper()} — Land Use', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

    feat_label = 'Satellite' if feat_name == 'sat' else 'POI'
    plt.suptitle(f'Within-city Clustering Analysis — {feat_label} Features',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_name}")


# ===========================================================================
# FIGURE 2 — Within-city Satellite
# ===========================================================================

print("\nSaving Figure 2: cluster_withincity_sat.png ...")
make_withincity_figure('sat', 'cluster_withincity_sat.png')

# ===========================================================================
# FIGURE 3 — Within-city POI
# ===========================================================================

print("\nSaving Figure 3: cluster_withincity_poi.png ...")
make_withincity_figure('poi', 'cluster_withincity_poi.png')


# ===========================================================================
# FIGURE 4 — Land use composition stacked bar (K=5 POI clusters, per city)
# ===========================================================================

print("\nSaving Figure 4: cluster_landuse_composition.png ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for col_i, city in enumerate(CITIES):
    ax       = axes[col_i]
    cd       = city_data[city]
    lu_names = cd['lu_names']
    lu_dims  = cd['lu_dims']
    stats    = within_city_results[city]['poi'][5]['stats']
    k        = 5

    # Build matrix [k, lu_dims]
    comp_mat = np.zeros((k, lu_dims))
    for ci, st in enumerate(stats):
        comp_mat[ci] = st['lu_pct']

    cmap_lu = plt.get_cmap('Set2', lu_dims)
    bottoms = np.zeros(k)
    for j in range(lu_dims):
        vals = comp_mat[:, j]
        ax.bar(range(k), vals, bottom=bottoms,
               color=cmap_lu(j), label=lu_names[j], width=0.6)
        bottoms += vals

    ax.set_title(f'{city.upper()} — POI K=5 Clusters\nLand Use Composition',
                 fontsize=10)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Land Use %')
    ax.set_xticks(range(k))
    ax.set_xticklabels([f'C{i}' for i in range(k)])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)

plt.suptitle('Land Use Composition per POI Cluster (K=5)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cluster_landuse_composition.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: cluster_landuse_composition.png")


# ===========================================================================
# Final summary
# ===========================================================================

print("\n" + "=" * 70)
print("DONE. Figures saved to:", FIGURES_DIR)
print("=" * 70)
saved_figs = [
    'cluster_citymean_subtracted.png',
    'cluster_withincity_sat.png',
    'cluster_withincity_poi.png',
    'cluster_landuse_composition.png',
]
for f in saved_figs:
    path = os.path.join(FIGURES_DIR, f)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"  {'OK' if exists else 'MISSING':6s}  {f}  ({size/1024:.1f} KB)")
