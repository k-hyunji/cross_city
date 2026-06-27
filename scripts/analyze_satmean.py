"""Analyze region-level sat_mean similarity distribution."""
import numpy as np
from numpy.linalg import norm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import UrbanRegionDataset

ds_nyc   = UrbanRegionDataset('nyc',   'data')
ds_seoul = UrbanRegionDataset('seoul', 'data')

nyc_sat_means   = np.stack([ds_nyc[i]['sat_mean'].numpy()   for i in range(len(ds_nyc))])
seoul_sat_means = np.stack([ds_seoul[i]['sat_mean'].numpy() for i in range(len(ds_seoul))])

nyc_n   = nyc_sat_means   / (norm(nyc_sat_means,   axis=1, keepdims=True) + 1e-8)
seoul_n = seoul_sat_means / (norm(seoul_sat_means, axis=1, keepdims=True) + 1e-8)

sim = nyc_n @ seoul_n.T   # [2312, 426]
print(f"Region-level sat_mean NYC<->Seoul sim: mean={sim.mean():.4f}  std={sim.std():.4f}  min={sim.min():.4f}  max={sim.max():.4f}")
print(f"Total region pairs: {sim.size:,}")
print()
print("Hit rate at align_threshold:")
for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    n = (sim > t).sum()
    print(f"  threshold={t}: {n:>7,} pairs  ({n/sim.size*100:.1f}% of all cross-city pairs)")

print()
print("Distribution of sim values:")
for lo, hi in [(0.0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
    count = ((sim >= lo) & (sim < hi)).sum()
    print(f"  [{lo:.1f}, {hi:.1f}): {count:>8,} pairs  ({count/sim.size*100:.1f}%)")

print()
# Per-anchor (NYC region): how many Seoul positives at threshold=0.7?
per_row = (sim > 0.7).sum(axis=1)
print(f"Per NYC region at threshold=0.7: mean positives={per_row.mean():.1f}, max={per_row.max()}, min={per_row.min()}, zeros={( per_row==0).sum()}")
per_row_85 = (sim > 0.85).sum(axis=1)
print(f"Per NYC region at threshold=0.85: mean positives={per_row_85.mean():.1f}, max={per_row_85.max()}, min={per_row_85.min()}, zeros={(per_row_85==0).sum()}")
