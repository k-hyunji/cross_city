import numpy as np
from numpy.linalg import norm
import sys, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cos_sim_matrix(A, B):
    A = A / (norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T

nyc_emb   = np.load('embeddings/nyc_seoul/nyc_region_emb.npy',        allow_pickle=True).item()['embeddings']
seoul_emb = np.load('embeddings/nyc_seoul/seoul_region_emb.npy',      allow_pickle=True).item()['embeddings']
sg_emb    = np.load('embeddings/nyc_seoul/singapore_region_emb.npy',  allow_pickle=True).item()['embeddings']

nyc_nyc     = cos_sim_matrix(nyc_emb, nyc_emb);   np.fill_diagonal(nyc_nyc, np.nan)
seoul_seoul = cos_sim_matrix(seoul_emb, seoul_emb); np.fill_diagonal(seoul_seoul, np.nan)
nyc_seoul   = cos_sim_matrix(nyc_emb, seoul_emb)
nyc_sg      = cos_sim_matrix(nyc_emb, sg_emb)
seoul_sg    = cos_sim_matrix(seoul_emb, sg_emb)

print("=== Embedding cosine similarity ===")
print(f"  Within NYC:    mean={np.nanmean(nyc_nyc):.4f}  std={np.nanstd(nyc_nyc):.4f}")
print(f"  Within Seoul:  mean={np.nanmean(seoul_seoul):.4f}  std={np.nanstd(seoul_seoul):.4f}")
print(f"  NYC<->Seoul:   mean={nyc_seoul.mean():.4f}  std={nyc_seoul.std():.4f}")
print(f"  NYC<->SG:      mean={nyc_sg.mean():.4f}  std={nyc_sg.std():.4f}")
print(f"  Seoul<->SG:    mean={seoul_sg.mean():.4f}  std={seoul_sg.std():.4f}")

print()
nyc_c   = nyc_emb.mean(0);   nyc_c   /= norm(nyc_c)
seoul_c = seoul_emb.mean(0); seoul_c /= norm(seoul_c)
sg_c    = sg_emb.mean(0);    sg_c    /= norm(sg_c)
print("=== City centroid cosine sim (city-domain clustering indicator) ===")
print(f"  NYC<->Seoul centroids:  {np.dot(nyc_c, seoul_c):.4f}   (near 1 = city blobs collapsed together)")
print(f"  NYC<->SG    centroids:  {np.dot(nyc_c, sg_c):.4f}")
print(f"  Seoul<->SG  centroids:  {np.dot(seoul_c, sg_c):.4f}")

print()
print("=== SG coverage by train-city embeddings ===")
for t in [0.3, 0.5, 0.7, 0.9]:
    m = ((nyc_sg > t).any(0) | (seoul_sg > t).any(0)).mean()
    print(f"  threshold={t}: {m*100:.1f}% of SG regions match any train region")

print()
print("=== Embedding norm stats ===")
print(f"  NYC   norm: mean={norm(nyc_emb, axis=1).mean():.4f}  std={norm(nyc_emb, axis=1).std():.4f}")
print(f"  Seoul norm: mean={norm(seoul_emb, axis=1).mean():.4f}  std={norm(seoul_emb, axis=1).std():.4f}")
print(f"  SG    norm: mean={norm(sg_emb, axis=1).mean():.4f}  std={norm(sg_emb, axis=1).std():.4f}")

print()
print("=== sat_mean pseudo-positive hit rate (region-level) ===")
poi_nyc   = np.load('data/nyc/poi_emb.npy',        allow_pickle=True).item()
poi_seoul = np.load('data/seoul/poi_emb.npy',       allow_pickle=True).item()
sat_nyc   = np.load('data/nyc/satellite_emb.npy')
sat_seoul = np.load('data/seoul/satellite_emb.npy')

# BG-level sat_emb global stats
sat_nyc_n   = sat_nyc   / (norm(sat_nyc,   axis=1, keepdims=True) + 1e-8)
sat_seoul_n = sat_seoul / (norm(sat_seoul, axis=1, keepdims=True) + 1e-8)
# sample 500 BGs from each for cross-city sim estimate
rng = np.random.default_rng(42)
idx_nyc   = rng.choice(len(sat_nyc_n),   min(500, len(sat_nyc_n)),   replace=False)
idx_seoul = rng.choice(len(sat_seoul_n), min(500, len(sat_seoul_n)), replace=False)
bg_sim = sat_nyc_n[idx_nyc] @ sat_seoul_n[idx_seoul].T
print(f"  BG-level sat cosine NYC<->Seoul: mean={bg_sim.mean():.4f}  std={bg_sim.std():.4f}")
for t in [0.5, 0.6, 0.7, 0.8]:
    hit = (bg_sim > t).mean()
    print(f"    threshold={t}: {hit*100:.2f}% of BG pairs are pseudo-positives")
