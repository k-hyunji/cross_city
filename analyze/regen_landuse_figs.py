"""Regenerate land use composition and within-city figures with corrected category names."""

import sys
import os
import ast
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/lab05/hyunji/cross_city6')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from data.dataset import UrbanRegionDataset

DATA_ROOT    = '/home/lab05/hyunji/cross_city6/data'
LANDUSE_ROOT = '/home/lab05/hyunji/cross_city5_gpt/data'
FIGURES_DIR  = '/home/lab05/hyunji/cross_city6/figures'
CITIES       = ['nyc', 'seoul', 'singapore']

CITY_COLORS = {'nyc': '#4C8BE2', 'seoul': '#E2774C', 'singapore': '#4CE27A'}

LANDUSE_CONFIG = {
    'nyc':       {'id_col': 'BoroCT2020', 'dims': 5},
    'seoul':     {'id_col': 'ADM_CD',     'dims': 11},
    'singapore': {'id_col': 'region_id',  'dims': 5},
}

# NYC: cat0=Residential(Low-density/One-Two Family), cat1=Residential(High-density/Multi-Family),
#       cat2=Industrial, cat3=Commercial/Mixed, cat4=Parks/Other
# Verified: cat3 pure → Financial District (Commercial), cat1 pure → Upper West Side (Residential)
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


def region_mean(mat, valid_mask):
    sub = mat[valid_mask].astype(float) if valid_mask.sum() > 0 else mat.astype(float)
    mean_vec = np.nanmean(sub, axis=0)
    return np.where(np.isnan(mean_vec), 0.0, mean_vec)


def fill_col_nan(X):
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col_mean = np.nanmean(col)
            col[nan_mask] = 0.0 if np.isnan(col_mean) else col_mean
            X[:, j] = col
    return X


def load_landuse(city):
    lu_path = os.path.join(LANDUSE_ROOT, city, 'landuse_gt.csv')
    cfg = LANDUSE_CONFIG[city]
    df = pd.read_csv(lu_path)
    id_col = cfg['id_col']
    if city == 'seoul':
        id_col = 'ADM_CD'
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
    elif city == 'nyc':
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
    elif city == 'singapore':
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    lu_dict = {}
    for _, row in df.iterrows():
        rid = str(row[id_col]).strip()
        if city == 'singapore':
            rid = rid.upper()
        lu_dict[rid] = np.array(ast.literal_eval(row['land_use']), dtype=float)
    print(f"  [Landuse] {city}: {len(lu_dict)} regions")
    return lu_dict


def load_pop(city):
    pop_path = os.path.join(DATA_ROOT, city, 'pop_gt.csv')
    cfg = POP_CONFIG[city]
    df = pd.read_csv(pop_path)
    id_col = cfg['id_col']
    val_col = cfg['val_col']
    if city == 'singapore':
        df['row_index'] = df['row_index'].astype(int).astype(str)
        key_col = 'row_index'
    elif city == 'nyc':
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
        key_col = id_col
    else:
        df[id_col] = df[id_col].astype(float).astype(int).astype(str)
        key_col = id_col
    pop_dict = {str(row[key_col]).strip(): float(row[val_col]) for _, row in df.iterrows()}
    print(f"  [Pop] {city}: {len(pop_dict)} regions")
    return pop_dict


def load_sg_id_map():
    df = pd.read_csv(os.path.join(DATA_ROOT, 'singapore', 'pop_gt.csv'))
    return {str(int(row['row_index'])): str(row['region_id']).strip().upper()
            for _, row in df.iterrows()}


print("Loading datasets...")
datasets = {city: UrbanRegionDataset(city, DATA_ROOT, satellite_mode='npy') for city in CITIES}
sg_id_map = load_sg_id_map()
landuse_dicts = {city: load_landuse(city) for city in CITIES}
pop_dicts = {city: load_pop(city) for city in CITIES}

print("\nBuilding feature matrices...")
city_data = {}
for city in CITIES:
    ds = datasets[city]
    lu_dict = landuse_dicts[city]
    pop_dict = pop_dicts[city]
    lu_dims = LANDUSE_CONFIG[city]['dims']
    lu_names = LANDUSE_NAMES[city]
    n = len(ds)
    sat_list, poi_list, region_ids, pop_vals, lu_vecs = [], [], [], [], []
    for i in range(n):
        item = ds[i]
        rid = item['region_id']
        sat_vec = region_mean(item['sat_data'].numpy(), item['valid_mask'].numpy())
        poi_vec = region_mean(item['poi_emb'].numpy(), item['valid_mask'].numpy())
        sat_list.append(sat_vec)
        poi_list.append(poi_vec)
        region_ids.append(rid)
        pop_vals.append(pop_dict.get(rid, np.nan))
        lu_key = sg_id_map.get(rid, '').upper() if city == 'singapore' else rid
        lu_vecs.append(lu_dict[lu_key] if lu_key in lu_dict else np.full(lu_dims, np.nan))

    sat_feats = fill_col_nan(np.array(sat_list))
    poi_feats = fill_col_nan(np.array(poi_list))
    pop_arr = np.array(pop_vals, dtype=float)
    lu_arr = np.array(lu_vecs, dtype=float)
    lu_valid_mask = ~np.all(np.isnan(lu_arr), axis=1)
    lu_dominant = np.array([int(np.argmax(lu_arr[j])) if lu_valid_mask[j] else -1 for j in range(n)])
    city_data[city] = dict(sat=sat_feats, poi=poi_feats, region_ids=region_ids, pop=pop_arr,
                           lu_arr=lu_arr, lu_dominant=lu_dominant, lu_valid=lu_valid_mask,
                           lu_dims=lu_dims, lu_names=lu_names, n=n)
    print(f"  {city}: sat={sat_feats.shape}, lu_found={lu_valid_mask.sum()}/{n}")


print("\nRunning K-Means within-city clustering...")
within_city_results = {}
for city in CITIES:
    cd = city_data[city]
    within_city_results[city] = {}
    for feat_name, X in [('sat', cd['sat']), ('poi', cd['poi'])]:
        Xs = StandardScaler().fit_transform(X)
        within_city_results[city][feat_name] = {}
        for k in K_VALUES:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Xs)
            sil = silhouette_score(Xs, labels) if k < len(X) else np.nan
            cluster_stats = []
            for cl in range(k):
                mask = labels == cl
                pop_cl = cd['pop'][mask]
                lu_valid_cl = cd['lu_valid'][mask]
                if lu_valid_cl.sum() > 0:
                    lu_cl = cd['lu_arr'][mask][lu_valid_cl]
                    lu_dist = np.nanmean(lu_cl, axis=0)
                    lu_pct = lu_dist / (lu_dist.sum() + 1e-9) * 100
                else:
                    lu_dist = np.zeros(cd['lu_dims'])
                    lu_pct = np.zeros(cd['lu_dims'])
                cluster_stats.append(dict(n=int(mask.sum()), mean_pop=np.nanmean(pop_cl),
                                          dom_lu=int(np.argmax(lu_dist)), lu_pct=lu_pct))
            within_city_results[city][feat_name][k] = dict(labels=labels, sil=sil, stats=cluster_stats)


# --- FIGURE: Land use composition stacked bar (K=5 POI clusters, per city) ---
print("\nSaving cluster_landuse_composition.png ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for col_i, city in enumerate(CITIES):
    ax = axes[col_i]
    cd = city_data[city]
    lu_names = cd['lu_names']
    lu_dims = cd['lu_dims']
    stats = within_city_results[city]['poi'][5]['stats']
    comp_mat = np.array([st['lu_pct'] for st in stats])  # [5, lu_dims]

    cmap_lu = plt.get_cmap('tab10', lu_dims)
    bottoms = np.zeros(5)
    for j in range(lu_dims):
        vals = comp_mat[:, j]
        ax.bar(range(5), vals, bottom=bottoms, color=cmap_lu(j),
               label=lu_names[j], width=0.6)
        bottoms += vals

    ax.set_title(f'{city.upper()} — POI K=5 Clusters\nLand Use Composition', fontsize=10)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Land Use %')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'C{i}\n(n={stats[i]["n"]})' for i in range(5)], fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

plt.suptitle('Land Use Composition per POI Cluster (K=5)\n'
             'NYC: cat0=Residential, cat1=Commercial/Mixed(04+05), cat2=Industrial, cat3=Transport&Public(07+08+10), cat4=OpenSpace/Vacant\n'
             'SG: cat0=Residential, cat1=Commercial, cat2=Industrial(Business), cat3=Public/Transport/Inst, cat4=Parks/Other',
             fontsize=8, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'cluster_landuse_composition.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


# --- FIGURE: Within-city t-SNE + clusters + pop + landuse ---
def make_withincity_figure(feat_name, save_name):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row_i, city in enumerate(CITIES):
        cd = city_data[city]
        lu_names = cd['lu_names']
        lu_dims = cd['lu_dims']
        X = cd['sat'] if feat_name == 'sat' else cd['poi']
        n = cd['n']
        perp_c = min(30, n // 5)
        print(f"  t-SNE {feat_name} {city} (n={n}, perp={perp_c})...")
        Xs = StandardScaler().fit_transform(X)
        tsne_c = TSNE(n_components=2, perplexity=perp_c, random_state=42,
                      n_iter=1000, init='pca').fit_transform(Xs)
        k5_labels = within_city_results[city][feat_name][5]['labels']

        # Col 0: t-SNE by cluster
        ax = axes[row_i, 0]
        ax.scatter(tsne_c[:, 0], tsne_c[:, 1], c=k5_labels, cmap='tab10',
                   vmin=0, vmax=9, s=20, alpha=0.8, linewidths=0)
        ax.set_title(f'{city.upper()} — K=5 Clusters', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

        # Col 1: t-SNE by pop density
        ax = axes[row_i, 1]
        pop_log = np.log1p(np.where(np.isnan(cd['pop']), 0.0, cd['pop']))
        sc = ax.scatter(tsne_c[:, 0], tsne_c[:, 1], c=pop_log, cmap='plasma',
                        s=20, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='log1p(pop)')
        ax.set_title(f'{city.upper()} — Pop Density', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

        # Col 2: t-SNE by dominant land use
        ax = axes[row_i, 2]
        lu_dom = cd['lu_dominant'].astype(float)
        cmap_lu = plt.get_cmap('tab10', lu_dims)
        colors_arr = [cmap_lu(lu_dom[j]) if lu_dom[j] >= 0 else (0.7, 0.7, 0.7, 0.5)
                      for j in range(n)]
        ax.scatter(tsne_c[:, 0], tsne_c[:, 1], c=colors_arr, s=20, alpha=0.8, linewidths=0)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=cmap_lu(i), label=lu_names[i]) for i in range(lu_dims)]
        ax.legend(handles=handles, fontsize=6, loc='best', framealpha=0.7)
        ax.set_title(f'{city.upper()} — Land Use', fontsize=10)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

    feat_label = 'Satellite' if feat_name == 'sat' else 'POI'
    plt.suptitle(f'Within-city Clustering — {feat_label} Features\n'
                 f'NYC: Residential / Commercial&Mixed / Industrial / Transport&Public / OpenSpace\n'
                 f'SG: Residential / Commercial / Industrial / Public&Transport&Inst / Parks',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")


print("\nSaving cluster_withincity_sat.png ...")
make_withincity_figure('sat', 'cluster_withincity_sat.png')

print("\nSaving cluster_withincity_poi.png ...")
make_withincity_figure('poi', 'cluster_withincity_poi.png')

print("\nAll done.")
