# Cross-City Urban Region Representation Learning

## Goal

Learn transferable region embeddings from source cities (e.g., NYC + Seoul) that generalize to unseen target cities (e.g., Singapore) via domain generalization — without any labeled data from the target city. The primary downstream task is population density prediction.

---

## Pipeline Overview

```
Train on source cities
        ↓
DualModalNet (satellite + POI → region_emb [128-d])
        ↓
Extract embeddings for all cities (including target)
        ↓
Linear probe (Ridge regression) on source → evaluate on target
```

**Three-step workflow:**

1. **Train** — `train.py` trains `DualModalNet` on source cities using three self-supervised losses.
2. **Extract** — `scripts/extract_embeddings.py` runs the trained model over all cities (source + target) to produce `{city}_region_emb.npy`.
3. **Evaluate** — `scripts/linear_probe.py` fits a Ridge regressor on source embeddings and predicts population density for the target city. Reports MAE, RMSE, R².

---

## Data

### Layout (per city)

```
data/{city}/
    satellite_emb.npy    # [N_bg, 64] pre-computed satellite embeddings (per block group)
    poi_emb.npy          # dict with 'morph_emb' [N_bg, 64] and 'centroid' [N_bg, 2]
    region.shp           # region polygon boundaries
    pop_gt.csv           # population density ground truth
```

Singapore additionally has `data/singapore/landuse_gt_list.csv` (land use labels per region — unused during training).

### Supported Cities

| City        | Region ID column | Notes                                |
|-------------|-----------------|--------------------------------------|
| NYC         | `BoroCT2020`    | Census tract level (n=2,312)         |
| Seoul       | `ADM_CD`        | Administrative district (n=426)      |
| Singapore   | `row_index`     | Polygon index (n=346)                |

### Raw Feature Distribution Warning

K-means analysis on raw features (K=3) shows that both satellite and POI embeddings are almost perfectly city-separated in raw space:

| Feature | ARI (K=3) | Meaning |
|---------|-----------|---------|
| Satellite | 0.866 | Cities occupy distinct clusters |
| POI | 0.943 | Stronger city separation than satellite |

With K=16 (current prototype count), AvgCityPurity = 0.999 — every sub-cluster is 100% city-pure in raw feature space. **This is the fundamental root cause of city-domain clustering** in the learned embedding space.

### Granularity: Block Group (BG) vs. Region

- **Block Group (BG)**: the atomic unit. Each BG has one satellite embedding [64] and one POI embedding [64]. POI embeddings also carry a `centroid` [2] (lat/lon) — currently loaded but unused by the model.
- **Region**: a polygon (e.g., census tract) that contains multiple BGs. The model aggregates all BGs in a region into a single region embedding [128].

### Spatial Join

`build_region_groups()` uses GeoPandas `sjoin` to map each BG centroid (from `poi_emb.npy`) into the region polygon it falls within. Unmatched BGs are assigned to the nearest region via `sjoin_nearest`.

---

## Model — `DualModalNet`

### Dimensions

| Symbol       | Value | Meaning                                      |
|-------------|-------|----------------------------------------------|
| `sat_dim`   | 64    | Input satellite embedding dim                |
| `poi_dim`   | 64    | Input POI embedding dim                      |
| `shared_dim`| 32    | Shared (modal-invariant) subspace            |
| `spec_dim`  | 32    | Specific (modal-specific) subspace           |
| `region_emb`| 128   | Final region embedding (sat_region + poi_region) |

### BG-Level Projections

Each BG's satellite and POI embedding is projected into two subspaces, then reconstructed before aggregation:

```python
sat_shared = sat_shared_proj(sat_emb)     # [M, 32]
sat_spec   = sat_spec_proj(sat_emb)       # [M, 32]
sat_recon  = cat([sat_shared, sat_spec])  # [M, 64] → fed to aggregator
```

The same applies to POI. Both `sat_recon` and `poi_recon` are 64-d (same as inputs). The aggregator therefore operates on reconstructed 64-d features, not the raw inputs.

> ⚠️ **Disentanglement note**: Because reconstruction simply concatenates shared + spec back to 64-d, the aggregator sees no explicit shared/spec boundary and cannot leverage the disentanglement structure. `BGDisLoss` regularizes the projections but contributes <1% of the total gradient (0.1 × ~0.05 = 0.005). Disentanglement is currently near-cosmetic.

### Region Aggregation

```python
sat_region = self.sat_agg(sat_recon, valid_mask)   # [64]
poi_region = self.poi_agg(poi_recon, valid_mask)   # [64]  (same mask for both)
region_emb = torch.cat([sat_region, poi_region], dim=-1)  # [128]
```

**`AttentionAggregator`** (default):
1. Self-attention over all valid BGs `[M, 64]`.
2. Cross-attention pooling via a learnable query vector → single vector.
3. FFN + LayerNorm.
4. Edge cases: zero vector if `M == 0`; single BG vector if `M == 1`.

**`MeanAggregator`** (fallback): simple mean of valid BG features.

### Frozen `sat_mean`

Each region pre-computes `sat_mean = mean(sat_emb[valid BGs])` at dataset load time. This frozen vector is used by the contrastive losses as a city-agnostic "visual fingerprint" for cross-city pseudo-positive mining.

---

## Losses (Current)

```
L_total = 1.0 · L_clip  +  0.1 · L_dis  +  0.3 · L_proto
```

`L_align = 0` (not used when `use_clip=true`).

See `losses.md` for full documentation of each loss.

### 1. `CLIPRegionCon` (L_clip, λ=1.0)

CLIP-style B×B cross-modal contrastive loss. `logits[i, j] = sat_i · poi_j / τ`. Within-modality pairs are never explicit negatives. Pseudo-positives require:
- `sat_mean_i · sat_mean_j > 0.9` (frozen satellite similarity)
- `poi_region_i · poi_region_j > 0.5` (learned POI similarity)

> ⚠️ Raw POI cross-city similarity max=0.271 (K-means analysis). The `poi_threshold=0.5` condition is **never satisfied early in training** — effectively making this satellite-only mining until embedding collapse inflates POI sim spuriously.

### 2. `BGDisLoss` (L_dis, λ=0.1)

Minimizes `|cos(spec, shared.detach())|` at BG level. Gradient contribution ~0.5% of total. Near-cosmetic.

### 3. `FunctionalPrototypeLoss` (L_proto, λ=0.3)

K=16 learnable prototype vectors. Sinkhorn-Knopp balanced assignment. Cross-entropy between soft assignment and log_softmax of prototype scores.

> ⚠️ With raw features being 100% city-pure (AvgCityPurity=0.999 at K=16), prototypes cannot learn functional archetypes from city-biased embeddings alone. Prototypes learn "city sub-regions", not "functional types across cities."

---

## Training

- **Optimizer**: Adam (`lr=1e-4`, `weight_decay=1e-4`)
- **Scheduler**: CosineAnnealingLR over 100 epochs
- **Batch size**: 32 regions (16 NYC + 16 Seoul with `balanced_sampler=true`)
- **Train/val split**: 80/20 from source cities
- **Gradient clipping**: max norm 1.0
- **Checkpointing**: `epoch_NNN_best.pt` when val loss improves; periodic every `save_every` epochs

---

## Downstream Evaluation

`scripts/linear_probe.py`:
1. Load source embeddings; apply `log1p` to population density targets.
2. Fit `StandardScaler` on source embeddings.
3. Grid-search Ridge (`α ∈ {0.01, 0.1, 1, 10, 100}`) by source train R².
4. Predict on target. Report MAE, RMSE (de-log1p scale), R² (log-space).

> ⚠️ Alpha selected by source train R² — no held-out validation. Prone to overfitting source distribution.

### Current Results (Best Run)

| Run | Config | MAE | RMSE | R² (log) |
|-----|--------|-----|------|----------|
| Baseline | broken arch | 21,400 | 24,907 | -4.59 |
| Fix 0+1+thr+bal | wire + InfoNCE + thr=0.9 + balanced | 9,656 | 12,079 | -0.31 |
| Fix 2 | UnifiedCityCon 2B×2B | 11,715 | 16,581 | -0.17 |
| Fix 2+4+8 corrected | + FunctionalPrototypeLoss | 10,611 | 18,934 | **-0.07** |
| CLIP-like | CLIPRegionCon B×B + L_proto | 8,416 | 11,141 | -0.11 |

R² remains negative — the model is still worse than predicting the mean on Singapore.

---

## Configuration (`configs/base.yaml`)

| Key                  | Current    | Description                                      |
|----------------------|-----------|--------------------------------------------------|
| `satellite_mode`     | `"npy"`   | Pre-extracted embeddings                         |
| `model.aggregator`   | `"attention"` | `"attention"` or `"mean"`                   |
| `training.epochs`    | 100       | Total epochs                                     |
| `training.batch_size`| 32        | Regions per batch                                |
| `training.lr`        | 1e-4      | Adam learning rate                               |
| `use_clip`           | `true`    | CLIPRegionCon (B×B cross-modal)                  |
| `use_unified_con`    | `false`   | UnifiedCityCon (2B×2B) — disabled               |
| `use_proto`          | `true`    | FunctionalPrototypeLoss                          |
| `lambda_contrast`    | 1.0       | Weight for L_clip                                |
| `lambda_dis`         | 0.1       | Weight for BGDisLoss                             |
| `lambda_proto`       | 0.3       | Weight for L_proto                               |
| `contrast_temp`      | 0.07      | NT-Xent temperature                              |
| `align_threshold`    | 0.9       | Satellite cosine sim threshold for pseudo-pairs  |
| `poi_threshold`      | 0.5       | POI cosine sim threshold (effectively unused early) |
| `n_prototypes`       | 16        | Functional archetypes                            |
| `balanced_sampler`   | `true`    | 16 NYC + 16 Seoul per batch                     |

---

## Key Design Decisions

- **Two-level architecture**: BG-level disentanglement + region-level aggregation. *In practice, the spec/shared split is not load-bearing because the aggregator sees reconstructed 64-d features, not separate branches.*
- **Frozen satellite mean as cross-city anchor**: avoids labeled target data. Satellite visual space serves as an alignment proxy.
- **No target city data during training**: true zero-shot transfer setup.
- **`collate_fn` returns list, not tensor batch**: variable-length BG sets processed independently by the aggregator.
- **Ridge regression probe**: isolates representation quality from downstream model complexity.

---

## Known Issues

| Issue | Code Location | Severity | Fix |
|-------|-----------|----|-----|
| Raw features are city-specific (ARI=0.87–0.94) | data preprocessing | 🔴 Root cause | solution.md §1 |
| poi_threshold=0.5 never satisfied in raw space (max sim=0.271) | `configs/base.yaml` | 🔴 Mining broken | Set to 0 or use satellite-only |
| L_dis gradient <1% of total | `models/losses.py` | 🟡 Cosmetic | solution.md §3 |
| POI centroid [2] loaded but unused | `data/dataset.py:195` | 🟡 Free signal wasted | Inject spatial features |
| Ridge alpha selected by source train R² | `scripts/linear_probe.py:33-36` | 🟡 Overfitting source | Use CV on held-out split |
| Singapore land use labels unused in evaluation | `data/singapore/landuse_gt_list.csv` | 🟡 Richer eval possible | Use for downstream task |
| `cfg["model"]["n_cities"]` set but unused | `train.py:51` | 🟢 Cleanup | Remove |
| Prototype collapse to city sub-regions | `models/losses.py:199-231` | 🔴 Functional cluster blocked | solution.md §2 |
