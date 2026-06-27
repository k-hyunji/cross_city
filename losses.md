# Loss Functions — Current Status

Current config: `configs/base.yaml` with `use_clip: true`, `use_proto: true`.

---

## Total Loss Formula

```
L_total = 1.0 · L_clip      (CLIPRegionCon — B×B cross-modal)
        + 0.1 · L_dis        (BGDisLoss)
        + 0.3 · L_proto      (FunctionalPrototypeLoss — K=16 prototypes)
```

`L_align = 0` (unused when `use_clip=true`). `L_unified` (UnifiedCityCon 2B×2B) superseded.

### Loss History

| Loss | Status | Reason |
|------|--------|--------|
| RegionContrastiveLoss (NT-Xent) | Superseded | Treated cross-city pairs as negatives — conflicted with L_align |
| SatAlignLoss (attraction-only) | Superseded | Gradient 10× weaker than L_contrast |
| SatAlignLoss (InfoNCE, Fix 1) | Superseded | Gradient conflict persisted (separate losses) |
| UnifiedCityCon (2B×2B, Fix 2+4) | Superseded | Solved city-domain clustering but caused over-alignment (within-city sim=0.81, Seoul norm std=0.028) |
| **CLIPRegionCon (B×B, current)** | **Active** | Cross-modal only — spec branch shielded from within-modality negative pressure |

---

## 1. L_clip — CLIPRegionCon (λ = 1.0)

**Replaces**: UnifiedCityCon (2B×2B). Key difference: B×B cross-modal only.

**What it does**: SupCon loss over a B×B similarity matrix where rows = sat embeddings, cols = poi embeddings. Only cross-modal pairs are computed — within-modality (sat_i vs sat_j) pairs are **never explicit negatives**.

### Similarity matrix

```
logits[i, j] = sat_i · poi_j / τ      τ = 0.07
shape: [B, B]
```

Both directions (sat→poi and poi→sat) computed symmetrically.

### Positive mask — two types of positives

**(a) Same-region cross-modal** (diagonal, always positive):
```
pos_mask[i, i] = 1   # sat_i ↔ poi_i
```

**(b) Cross-city pseudo-positives — dual-signal:**
A cross-city pair (region i from city A, region j from city B) where:
- `sat_mean_i · sat_mean_j > 0.9` (frozen satellite similarity)
- `poi_region_i · poi_region_j > 0.5` (learned POI similarity)

> ⚠️ **Critical: poi_threshold=0.5 is non-functional.** Raw POI cross-city similarity: mean=0.092, max=0.271. The `poi_threshold=0.5` condition is never satisfied during early training, making this effectively satellite-only mining. After embedding collapse, 76% of pairs exceed 0.5 — but this is a collapse artifact, not genuine functional similarity. **Recommended fix: set `poi_threshold: 0.0`.**

### Loss computation (SupCon-style, both directions)

```
# sat → poi direction
for each sat anchor i:
    log_denom = logsumexp(logits[i, :])          # all B poi
    log_numer = logsumexp(logits[i, pos_idx])    # positive poi only
    loss_i    = log_denom - log_numer

# poi → sat direction (symmetric)
for each poi anchor j:
    log_denom = logsumexp(logits[:, j])
    log_numer = logsumexp(logits[pos_idx, j])
    loss_j    = log_denom - log_numer

L_clip = mean over all anchors with at least one positive
```

### Why CLIPRegionCon fixes over-alignment (and what it breaks)

**Fix**: In UnifiedCityCon (2B×2B), `sat_i` and `sat_j` from different cities were in the same row as negatives, forcing the spec branch toward alignment. In CLIPRegionCon (B×B), within-modality pairs are never negatives — the spec branch only receives gradient from L_proto and L_dis.

**New problem**: CLIPRegionCon also removes cross-city repulsion pressure from the denominator. Without this pressure, cities drift back to opposite hemispheres (NYC↔Seoul centroid cos: +0.816 → −0.449). The fix for within-city collapse introduced city-domain clustering.

### What CLIPRegionCon does NOT do

It does not prevent city-domain clustering caused by the pre-trained feature city bias (see problems.md Problem 6). The loss operates on top of already city-separated features.

---

## 1b. L_unified — UnifiedCityCon (λ = 1.0) [SUPERSEDED]

Superseded by CLIPRegionCon. Key difference: 2B×2B matrix with within-modality negatives.

```
all_embs = [sat_0..sat_{B-1}, poi_0..poi_{B-1}]   # [2B, 64]
logits[i, j] = all_embs[i] · all_embs[j] / τ       # [2B, 2B]
```

Four positive types: sat_i↔poi_i, sat_i↔sat_j (cross-city), poi_i↔poi_j (cross-city), sat_i↔poi_j (cross-city cross-modal). The within-modality positives (sat_i↔sat_j) caused the spec branch to be pulled into alignment, leading to over-alignment (within-city sim=0.81, Seoul norm std=0.027). Kept here for reference.

---

## 2. L_dis — BGDisLoss (λ = 0.1)

**What it does**: at BG level, forces the specific branch to be orthogonal to the shared branch:

```
L_dis = 0.5 · ( mean(|cos(sat_spec, sat_shared.detach())|)
              + mean(|cos(poi_spec, poi_shared.detach())|) )
```

`shared` is detached so gradients only flow through `spec`.

**Gradient budget**: `0.1 × 0.05 ≈ 0.005` — about **0.2% of L_clip's contribution**. Near-cosmetic.

**Deeper problem**: The shared/spec architecture is not load-bearing because the aggregator sees reconstructed 64-d features (shared + spec concatenated), not separate branches. The projections are regularized to be orthogonal but the aggregator mixes them freely afterward.

**Potential replacement**: VICReg variance term — penalizes embedding collapse directly without requiring the shared/spec architecture. Barlow Twins off-diagonal term achieves similar decorrelation at the output level.

---

## 3. L_proto — FunctionalPrototypeLoss (λ = 0.3)

**What it does**: K=16 learnable prototype vectors shared across all cities. Each prototype represents a functional archetype. Every region is assigned to prototypes via Sinkhorn-Knopp balanced assignment.

### Forward pass

```python
embs    = normalize(region_emb)          # [B, 128]
proto_w = normalize(prototypes.weight)   # [K, 128]  gradient flows here
scores  = embs @ proto_w.T / τ_proto    # [B, K]    τ_proto = 0.1

Q       = Sinkhorn(scores.detach().exp())  # [B, K]  soft balanced assignment
L_proto = -mean_B( sum_K( Q * log_softmax(scores) ) )
```

**Baseline loss**: log(K) = log(16) ≈ 2.77 (random uniform assignment). As prototypes learn, L_proto should decrease toward 0.

### Sinkhorn-Knopp Balanced Assignment

```
M = scores.detach().exp().T    # [K, B]
M /= M.sum()                   # global normalize
for 3 iterations:
    M /= M.sum(dim=1) + ε; M /= K   # each prototype row → 1/K
    M /= M.sum(dim=0) + ε; M /= B   # each sample col → 1/B
M *= B                               # rescale: each col sums to 1
Q = M.T                              # [B, K], rows sum to 1
```

Forces each prototype to be used by ~B/K regions per batch, preventing collapse.

### Why L_proto cannot create functional clusters in the current setup

K-means K=16 on raw input features produces AvgCityPurity=0.999 (satellite) and 0.974 (POI). Prototypes initialized randomly and trained with Sinkhorn on top of these features will converge to city-sub-region archetypes (e.g., "Seoul high-density", "NYC commercial"), not cross-city functional archetypes (e.g., "Dense residential across all cities").

The city-agnostic assumption of L_proto requires city-agnostic input features. **Until the raw feature city bias is removed (Problem 6), L_proto provides marginal functional alignment benefit.**

### Positive contribution of L_proto (despite limitations)

Across-run comparison shows that adding L_proto improves Singapore geometry:
- NYC↔SG centroid cos: +0.400 → +0.525 (Fix 2 → Fix 2+4+8 corrected)
- Seoul↔SG centroid cos: +0.346 → +0.409

This suggests L_proto is creating some pull of Singapore toward the source city prototypes, even if those prototypes are not purely functional. This is a useful signal — Singapore lands closer to the source embedding space.

---

## Current Config Values

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `use_clip` | `true` | CLIPRegionCon (B×B cross-modal) |
| `use_unified_con` | `false` | UnifiedCityCon disabled |
| `use_proto` | `true` | FunctionalPrototypeLoss |
| `lambda_contrast` | 1.0 | Weight for L_clip |
| `lambda_dis` | 0.1 | Weight for L_dis (near-cosmetic) |
| `lambda_proto` | 0.3 | Weight for L_proto |
| `contrast_temp` | 0.07 | τ for CLIPRegionCon |
| `align_threshold` | 0.9 | sat_mean threshold for pseudo-positives |
| `poi_threshold` | 0.5 | ⚠️ Non-functional — set to 0.0 |
| `n_prototypes` | 16 | Functional archetypes |
| `proto_temp` | 0.1 | τ for prototype score |
| `sinkhorn_iters` | 3 | Sinkhorn iterations |
| `balanced_sampler` | true | 16 NYC + 16 Seoul per batch |

---

## Gradient Budget (approximate, at convergence)

| Loss term | Typical value | × λ | Effective contribution |
|-----------|--------------|-----|----------------------|
| L_clip | ~2.0–2.5 | 1.0 | **~2.0–2.5** (~75%) |
| L_dis | ~0.05 | 0.1 | **~0.005** (<1%) |
| L_proto | 2.77→converged | 0.3 | **~0.83** (~23%) at start, decreasing |

---

## Monitoring Checklist

| Signal | Healthy | Problem |
|--------|---------|---------|
| `val_contrast` (= L_clip) | Steadily decreasing | Rising → L_proto interfering |
| `val_proto` | Decreasing from ~2.77 toward 0 | Stuck at ~2.77 → prototypes not learning |
| `val_dis` | Stable ~0.04–0.06 | Collapsed to 0 → spec branch degenerate |
| NYC↔Seoul centroid cos | Positive (≥ +0.5) | Negative → city-domain clustering |
| Within-city sim | **0.4–0.6** | > 0.7 → over-aligned; < 0.2 → collapsed |
| Seoul norm std | > 0.05 | < 0.03 → Seoul collapsed |
| K-means K=16 purity | < 0.7 (mixed cities) | ~1.0 → prototypes = city sub-regions |

---

## What Was Tried and Removed

| Component | Status | Why removed |
|-----------|--------|-------------|
| RegionContrastiveLoss (NT-Xent) | Superseded | Cross-city pairs as negatives — conflict with L_align |
| SatAlignLoss (attraction-only MSE) | Superseded | Gradient 10× weaker; no repulsion |
| SatAlignLoss (InfoNCE, Fix 1) | Superseded | Separate from L_contrast → conflict persisted |
| τ=0.15 | Rejected | City-domain centroid: -0.397 → -0.594 (worse) |
| L_proto with bugged Sinkhorn | Fixed | Rows of Q summed to 2.0 → loss 5.4 instead of 2.77 |
| UnifiedCityCon (2B×2B) | Superseded | Solved clustering but caused over-alignment (sim=0.81, Seoul std=0.028) |
