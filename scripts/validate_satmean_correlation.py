"""
4a: Validate sat_mean -> functional similarity hypothesis.
For each pseudo-positive pair (NYC region i, Seoul region j, sat_sim > 0.9),
check whether their pop densities are correlated.
"""
import numpy as np, pandas as pd
from numpy.linalg import norm
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import UrbanRegionDataset

ds_nyc   = UrbanRegionDataset('nyc',   'data')
ds_seoul = UrbanRegionDataset('seoul', 'data')

# Region-level sat_mean
nyc_sat   = np.stack([ds_nyc[i]['sat_mean'].numpy()   for i in range(len(ds_nyc))])
seoul_sat = np.stack([ds_seoul[i]['sat_mean'].numpy() for i in range(len(ds_seoul))])
nyc_sat_n   = nyc_sat   / (norm(nyc_sat,   axis=1, keepdims=True) + 1e-8)
seoul_sat_n = seoul_sat / (norm(seoul_sat, axis=1, keepdims=True) + 1e-8)
sim = nyc_sat_n @ seoul_sat_n.T  # [2312, 426]

# Region IDs
nyc_ids   = [ds_nyc[i]['region_id']   for i in range(len(ds_nyc))]
seoul_ids = [ds_seoul[i]['region_id'] for i in range(len(ds_seoul))]

# Pop densities
pop_nyc_df   = pd.read_csv('data/nyc/pop_gt.csv')
pop_seoul_df = pd.read_csv('data/seoul/pop_gt.csv')
# cast index to str to match region_id strings from dataset
pop_nyc   = pop_nyc_df.set_index(pop_nyc_df.iloc[:,0].astype(str))['population_density']
pop_seoul = pop_seoul_df.set_index(pop_seoul_df['ADM_CD'].astype(str))['population_density']

nyc_pop   = np.array([pop_nyc.get(str(rid), np.nan) for rid in nyc_ids])
seoul_pop = np.array([pop_seoul.get(str(rid), np.nan) for rid in seoul_ids])

print("=== 4a: sat_mean -> functional (pop density) correlation ===\n")

for threshold in [0.85, 0.90, 0.92, 0.95]:
    rows, cols = np.where(sim > threshold)
    valid = ~(np.isnan(nyc_pop[rows]) | np.isnan(seoul_pop[cols]))
    rows, cols = rows[valid], cols[valid]

    if len(rows) < 10:
        print(f"threshold={threshold}: too few pairs ({len(rows)}) to compute")
        continue

    nyc_d   = nyc_pop[rows]
    seoul_d = seoul_pop[cols]

    r, p       = stats.pearsonr(nyc_d, seoul_d)
    rho, p_rho = stats.spearmanr(nyc_d, seoul_d)

    # log-space
    log_nyc   = np.log1p(nyc_d)
    log_seoul = np.log1p(seoul_d)
    r_log, p_log = stats.pearsonr(log_nyc, log_seoul)

    # absolute density difference
    abs_diff = np.abs(nyc_d - seoul_d)

    print(f"threshold={threshold}  n_pairs={len(rows):>6}")
    print(f"  Pearson  r={r:>+.3f}  p={p:.3e}")
    print(f"  Spearman r={rho:>+.3f}  p={p_rho:.3e}")
    print(f"  Pearson (log1p) r={r_log:>+.3f}  p={p_log:.3e}")
    print(f"  |density diff|: mean={abs_diff.mean():.0f}  median={np.median(abs_diff):.0f}  std={abs_diff.std():.0f}")
    print()

# Baseline: random pairs
print("--- Baseline: random cross-city pairs (n=10000) ---")
rng = np.random.default_rng(42)
ridx = rng.integers(0, len(nyc_ids), 10000)
cidx = rng.integers(0, len(seoul_ids), 10000)
valid = ~(np.isnan(nyc_pop[ridx]) | np.isnan(seoul_pop[cidx]))
r_rand, _ = stats.pearsonr(nyc_pop[ridx[valid]], seoul_pop[cidx[valid]])
rho_rand, _ = stats.spearmanr(nyc_pop[ridx[valid]], seoul_pop[cidx[valid]])
print(f"  Pearson r={r_rand:+.3f}  Spearman r={rho_rand:+.3f}  (expected ~0 if random)")
print()
print("Interpretation:")
print("  If pseudo-positive r >> random r → sat_mean is a valid functional proxy")
print("  If pseudo-positive r ≈ random r  → sat_mean is NOT capturing function")
