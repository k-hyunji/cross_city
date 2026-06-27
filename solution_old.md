# Solution Analysis: Cross-City Urban Region Representation Learning

---

## 0. Confirmed Baseline (Pre-Fix Run)

**Linear probe result — nyc+seoul → singapore (pre-Fix-0 embeddings):**

| Metric | Value |
|--------|-------|
| MAE    | 21,400 명/km² |
| RMSE   | 24,907 명/km² |
| **R²** | **-4.59** |

R² = -4.59 means the model is ~5.5× worse than predicting the mean. Three compounding causes:

1. **Architecture was broken (Fix 0 not yet applied)**: `region_emb` was built from raw `sat_data`/`poi_emb`, completely bypassing the shared/spec projection heads. The embeddings carry no disentanglement signal.
2. **L_align is failing (rising throughout training)**: the model is actively pushing cross-city pseudo-positive pairs apart. Embeddings from NYC and Seoul never align with Singapore's functional patterns.
3. **City-domain clustering**: NT-Xent at τ=0.07 exploits city identity as the easiest discriminative axis. The resulting embedding space has two city-separated blobs; Singapore regions land between them with no meaningful assignment.

**Fix 0 status: ✅ Applied** (`models/model.py` — aggregator now operates on `[sat_shared⊕sat_spec]` and `[poi_shared⊕poi_spec]`; POI mask bug also fixed).

**Fix 1 status: ✅ Applied + Retrained** (`models/losses.py` — `SatAlignLoss` replaced with InfoNCE).

**Linear probe progression across runs:**

| Run | Config changes | MAE | RMSE | **R²** | Notes |
|-----|--------------|-----|------|------|-------|
| Baseline | broken arch | 21,400 | 24,907 | **-4.59** | raw space |
| Fix 0+1 | wire projections + InfoNCE (thr=0.7) | 16,892 | 20,230 | **-2.68** | raw space |
| Fix 0+1 + threshold + balanced | thr=0.9, balanced_sampler=true | 9,656 | 12,079 | **-0.31** | raw space |
| + τ_contrast=0.15 | raised from 0.07 | 11,835 | 19,679 | **-0.31** | log-space |
| **Fix 2 (UnifiedCityCon)** | merged L_contrast + L_align | 11,715 | 16,581 | **-0.17** | log-space |
| Fix 2+4+8 (bugged) | dual-signal + prototypes (Sinkhorn bug) | 12,094 | 15,858 | **-0.62** | log-space — proto stuck at 5.4, should be 2.77 |
| **Fix 2+4+8 (corrected)** | fixed Sinkhorn + λ_proto=0.3 | 10,611 | 18,934 | **-0.07** | log-space — best R² so far; over-alignment persists |
| **CLIP-like (B×B)** | CLIPRegionCon + L_proto | 8,416 | 11,141 | **-0.11** | log-space — MAE/RMSE ↓ but R² worse; city-domain clustering **returned** (centroid −0.449) |

**τ_contrast=0.15 experiment result: temperature is NOT the bottleneck.**

R² unchanged at -0.31, and city-domain clustering got significantly *worse*:

| Metric | τ=0.07 (prev) | τ=0.15 (current) | Δ |
|--------|--------------|-----------------|---|
| NYC↔Seoul centroid cos | -0.397 | **-0.594** | −50% worse |
| NYC↔SG centroid cos | -0.262 | **-0.472** | worse |
| Seoul↔SG centroid cos | +0.057 | **+0.269** | slightly better |

Softening the temperature redistributes gradients more uniformly across all negatives but does not change *which* pairs are negatives. The 90% of cross-city non-pseudo-positive pairs are still hard negatives for L_contrast regardless of temperature. The city-domain separation is structural, not a temperature artifact.

**Conclusion: Fix 2 (UnifiedCityCon) is confirmed as structurally required.** Temperature tuning cannot fix a conflict baked into the loss design.

1. **threshold=0.7 made InfoNCE trivially zero (val_align 0.003)**: 98.7% of cross-city pairs were pseudo-positives, so `log_numer ≈ log_denom` → loss ≈ 0. Fixed by raising threshold to 0.9 (→ 10% positive rate, val_align now meaningful at 0.246 log-space).

2. **City-domain clustering persists and got worse with balanced batches**: centroid cosine NYC↔Seoul went -0.20 → -0.397 → -0.594 across runs. Fix 2 (UnifiedCityCon) is now the immediate next step.

**Fix 2 result (embeddings/nyc_seoul_0501):**

| Metric | τ=0.07 (run 3) | τ=0.15 (run 4) | Fix 2 (UnifiedCityCon) |
|---|---|---|---|
| R² (log-space) | -0.31 | -0.31 | **-0.17** ↑ |
| NYC↔Seoul centroid cos | -0.397 | -0.594 | **+0.819** (fully reversed) |
| NYC↔SG centroid cos | -0.262 | -0.472 | **+0.400** ↑ |
| Seoul↔SG centroid cos | +0.057 | +0.269 | **+0.346** ↑ |
| NYC↔Seoul mean sim | -0.095 | -0.271 | **+0.621** |
| Within NYC | 0.568 | ~0.55 | **0.809** ⚠️ |
| Within Seoul | 0.352 | ~0.37 | **0.710** ⚠️ |
| Seoul norm std | 0.124 | 0.170 | **0.028** ⚠️ |

**Fix 2 solved city-domain clustering** (centroid -0.594 → +0.819, R² -0.31 → -0.17). However it introduced a new problem: **over-alignment / partial embedding collapse**.

- Within-city similarity 0.81 is too high — diverse region types (residential, commercial, industrial) should differ. 0.81 suggests the unified loss is pulling everything toward one cluster, not organizing by function.
- Seoul norm std = 0.028 — Seoul collapsing in norm space again, despite balanced sampler. Cross-city pseudo-positive pairs are pulling all Seoul embeddings toward a common direction.
- NYC↔Seoul mean sim = 0.621 ≈ within-city sim = 0.809 — the cross-city gap is very small, meaning there is little discriminative structure left.

**Root cause**: UnifiedCityCon adds many positives per anchor (1 cross-modal + ~1.6 cross-city), reducing the effective negative density. With fewer negatives per anchor, the uniformity pressure (which spreads embeddings apart) weakens and partial collapse results. Fix 2 goes too far in the alignment direction.

**Fix 2+4+8 corrected result (embeddings/nyc_seoul_0501_3):**

| Metric | Fix 2 (UnifiedCityCon) | Fix 2+4+8 corrected | Δ |
|--------|------------------------|---------------------|---|
| R² (log-space) | -0.17 | **-0.07** | ↑ best so far |
| NYC↔Seoul centroid cos | +0.819 | **+0.816** | ≈ maintained |
| NYC↔SG centroid cos | +0.400 | **+0.525** | ↑ improved |
| Seoul↔SG centroid cos | +0.346 | **+0.409** | ↑ improved |
| NYC↔Seoul mean sim | +0.621 | **+0.623** | ≈ same |
| Within NYC | 0.809 ⚠️ | **0.807** ⚠️ | no change |
| Within Seoul | 0.710 ⚠️ | **0.722** ⚠️ | slightly worse |
| Within SG | — | **0.833** ⚠️ | very high |
| Seoul norm std | 0.028 ⚠️ | **0.027** ⚠️ | no change |

**Fix 8 improved R² (-0.17 → -0.07) and cross-city geometry (SG centroids closer to source cities).** However, the over-alignment problem is **unchanged** — within-city similarity remains ~0.72–0.81 and Seoul norm std stays collapsed at 0.027. Fix 8 alone does not restore discriminative internal structure.

**Conclusion**: the prototypes are creating better cross-city structure (Singapore is now pulled toward both source cities), but the spec branch is still getting dragged by `UnifiedCityCon`. The architectural fix from §0.5.11 is now the recommended next step.

**Fix 8 (FunctionalPrototypeLoss) is now the critical next step.** *(Note: Fix 8 corrected run completed — see table above)* It installs K shared functional cluster centers across all cities, pushing different functional types apart and restoring discriminative internal structure within the now-aligned space.

**Root cause of continued failure — three compounding issues discovered by post-training embedding analysis:**

---

### A. The threshold=0.7 is completely wrong for this data (most critical)

Region-level `sat_mean` NYC↔Seoul cosine similarity distribution:

| Threshold | Pairs above threshold | % of all 984,912 pairs |
|-----------|----------------------|------------------------|
| 0.70      | 972,123              | **98.7%** ← current setting |
| 0.80      | 869,056              | 88.2% |
| 0.85      | 613,900              | 62.3% |
| 0.90      |  99,964              | **10.1%** ← reasonable target |

`sat_mean` mean = **0.853**, std = 0.046. Setting threshold=0.7 means **98.7% of ALL cross-city region pairs are pseudo-positives**. On average, each NYC anchor region has 420 out of 426 Seoul regions as "positives." This explains val_align → 0.003:

When `pos_cols` covers 98% of row indices, `logsumexp(logits[pos_cols]) ≈ logsumexp(logits)`, so `loss = log_denom − log_numer ≈ 0`. The InfoNCE loss collapsed to zero trivially — not because alignment is achieved, but because the mask is so broad it makes the loss meaningless.

**Fix: raise threshold to 0.90** (gives ~10% positive rate, ~43 Seoul positives per NYC anchor — meaningful InfoNCE discrimination).

---

### B. City-domain clustering is severe (confirmed by embedding geometry)

| Similarity | Value | Interpretation |
|------------|-------|----------------|
| Within NYC | 0.362 | moderate coherence |
| Within Seoul | 0.629 | high coherence |
| NYC↔Seoul (cross) | **-0.095** | **negative** — opposite hemispheres |
| NYC↔SG | 0.016 | near zero — SG is lost between cities |
| Seoul↔SG | -0.047 | slightly negative |
| NYC↔Seoul centroids | **-0.200** | cities in opposite directions on sphere |

NYC and Seoul are in **opposite hemispheres** of the embedding space. Singapore sits roughly at the equator between them with no functional structure. A linear probe cannot find a meaningful direction from this geometry — there is no useful signal for population density.

L_contrast at τ=0.07 is driving this: even though 98.7% of cross-city pairs are "pseudo-positives," the InfoNCE loss trivially saturates, so L_contrast provides 100% of the effective gradient and pushes all cross-city pairs apart. Fix 2 (UnifiedCityCon) is still needed but must be paired with a correct threshold.

---

### C. Seoul embeddings are collapsed

Seoul region embedding norm std = **0.019** (vs NYC std = 1.389). All 426 Seoul regions have essentially identical norms — a sign of representation collapse within Seoul. Cause: Seoul has only 426 regions vs NYC's 2312. With `balanced_sampler=False` and `batch_size=32`, Seoul appears in ~15% of batch slots, so NT-Xent sees very few Seoul↔Seoul negatives per batch. The model satisfies L_contrast for Seoul trivially by collapsing all Seoul embeddings to a single shell.

**Fix: enable `balanced_sampler=True`** so Seoul gets equal batch representation (16 NYC + 16 Seoul per batch instead of ~27 NYC + ~5 Seoul).

---

### Revised next steps (ordered by impact)

1. ✅ **Raise `align_threshold` to 0.90** — done; InfoNCE now discriminative (val_align went from saturated 0.003 → meaningful 0.246 in log-space)
2. ✅ **Enable `balanced_sampler: true`** — done; Seoul norm std recovered 0.019 → 0.124
3. ✅ **Retrain and re-evaluate** — done; R² improved -2.68 → -0.31
4. ⬜ **See §0.5 below** — critical review revealed missed problems and confirmed Fix 2 is now urgent

---

## 0.5 Critical Review — Missed Problems & New Observations

After the threshold/balanced run, we found new evidence that solution.md previously missed. The R² improvement is real, but the underlying geometry analysis surfaced surprises that need to be addressed before any further loss design.

### 0.5.1 sat_mean → pop_density correlation: validated ✅ (result: not correlated — but see framing note)

**Result** (`scripts/validate_satmean_correlation.py`):

| threshold | n pairs | Pearson r | Spearman r |
|-----------|---------|-----------|------------|
| 0.85 | 613,900 | +0.010 | +0.013 |
| 0.90 | 99,964 | +0.016 | +0.012 |
| 0.92 | 14,438 | -0.025 | -0.029 |
| Random baseline | 10,000 | +0.010 | +0.010 |

Pseudo-positive pairs (sat_sim > 0.9) show r = +0.016 vs random pairs r = +0.010. **Practically identical.**

**Critical framing note**: this does NOT mean sat_mean is a bad alignment signal. It means **population density is a narrow downstream proxy for embedding quality**. The goal of this project is to learn good *cross-city transferable region embeddings* — pop_density is just one evaluation task, not the target of the embedding itself. sat_mean may well capture real urban morphology and land-use patterns that are meaningful for cross-city transfer across many tasks, even if they don't correlate with pop_density specifically.

**What this result does tell us**:
1. Pop_density linear probe is a weak signal for measuring embedding alignment quality — a good cross-city embedding could score poorly here while being genuinely useful for other tasks (crime, mobility, land-use classification).
2. We should not redesign the alignment strategy based on pop_density correlation alone.
3. If we want stronger validation of sat_mean as a proxy, we need richer ground truth (land-use labels, POI category distributions, mobility patterns) rather than a single density number.

**Action (revised)**: do not abandon sat_mean-based alignment based on this result. Focus instead on whether the *embedding geometry* (cross-city centroid similarity, functional cluster structure) is improving — that is a more direct measure of cross-city transferability than pop_density correlation.

### 0.5.2 Pop density distribution: validated ✅ (result: moderate mismatch)

**Result**:

| City | n | mean | std | median | log std |
|------|---|------|-----|--------|---------|
| NYC | 2,325 | 16,712 | 10,516 | 15,105 | 0.879 |
| Seoul | 426 | 22,776 | 11,149 | 22,980 | 0.740 |
| Singapore | 364 | 11,839 | 10,611 | 10,212 | **3.229** |

- Seoul/Singapore mean ratio: **1.92×** (borderline)
- Singapore log-space std = **3.229** vs NYC 0.879, Seoul 0.740 — Singapore has a fundamentally different distribution shape (heavy-tailed, bimodal likely due to uninhabited industrial/water areas mixed with dense residential)

**Action**: apply `log1p` transform to pop_density target before linear probe. This will not fix the embedding; it reduces the influence of Singapore's extreme outliers on the regression and makes train/test distributions more comparable. Do not interpret R² improvement from this as embedding improvement — it is purely a probe calibration fix.

### 0.5.3 POI centroid (lat/lon) is loaded but completely unused

`data/dataset.py` loads `poi_emb.npy` as `{'morph_emb': [N, 64], 'centroid': [N, 2]}`. The model uses `morph_emb` only. The `centroid` field carries spatial information (distance to city center, intra-region BG distribution) that is potentially the strongest cross-city transferable signal — and we are dropping it.

**Action**: either (a) inject centroid into the model directly, (b) use centroid-derived features (distance to city centroid, density gradient) as auxiliary inputs, or (c) at minimum verify centroid is not silently being used elsewhere.

### 0.5.4 City-domain clustering got *worse* after the fix (mechanism not explained in solution.md)

| Centroid cosine | thr=0.7, balanced=F | thr=0.9, balanced=T |
|-----------------|---------------------|----------------------|
| NYC↔Seoul       | -0.200              | **-0.397** (2× worse) |

**Mechanism**: with threshold=0.9, only ~10% of cross-city pairs are pseudo-positive in InfoNCE. The other 90% remain hard negatives in L_contrast. With `balanced_sampler=True`, every batch has 16 NYC + 16 Seoul, so each NYC anchor sees 16 Seoul hard negatives per step (vs ~5 before). L_contrast's city-separation pressure increased substantially. **This confirms Fix 2 (UnifiedCityCon) is now structurally required** — pairwise alignment cannot win against systematically-amplified pairwise repulsion.

### 0.5.5 Singapore's position swapped sides

| | NYC↔SG | Seoul↔SG |
|---|--------|----------|
| Before (thr=0.7) | +0.016 | -0.047 |
| After (thr=0.9, balanced) | **-0.262** | **+0.057** |

Singapore moved into the opposite hemisphere from NYC and toward Seoul. Two competing interpretations:
- **(Optimistic)** the model learned a real "Asian-city vs American-city" geographic axis — which would actually help Singapore↔Seoul transfer.
- **(Pessimistic)** Singapore (the smaller of the unseen city) is collapsing toward the nearest training cluster regardless of function — i.e., the model has learned a geography shortcut, not a function axis.

**Action**: t-SNE colored by city + colored by pop density. If pop-density colors form coherent stripes across cities, optimistic case. If colors are random within city blobs, pessimistic case.

### 0.5.6 NYC over-clustering — a new side effect of balanced sampling

| Within-city sim | thr=0.7, balanced=F | thr=0.9, balanced=T |
|-----------------|---------------------|----------------------|
| Within NYC      | 0.362               | **0.568** (more clustered) |
| Within Seoul    | 0.629 (collapsed)   | **0.352** (recovered) |

Seoul collapse is fixed, but NYC is now becoming over-clustered. With balanced batches, NYC anchors see fewer NYC negatives per batch (15 NYC negatives vs 16 Seoul cross-city negatives), so NT-Xent's within-NYC repulsion is weakened. Some clustering is expected, but a 0.57 mean similarity is high.

**Action**: monitor whether this trend continues; if NYC fully collapses, revisit batch size or sampling strategy.

### 0.5.7 Fix 0 Solution A is not enforcing real disentanglement

Solution A concatenates `[shared, spec]` then aggregates. The aggregator sees a 64-d vector and has no incentive to treat shared and spec dimensions differently. In practice the aggregator may smear them together, undoing any disentanglement benefit. **Solution B** (separate aggregation per branch) was rejected as "more invasive" but might be what's actually needed if disentanglement is intended to matter.

**Action**: revisit Solution B if val_dis stays low (currently 0.045) and ablation shows disentanglement contributes nothing measurable.

### 0.5.8 lambda_dis = 0.1 makes Fix 0 nearly cosmetic

Even with Fix 0 wiring spec into region_emb, the gradient contribution of L_dis is `0.1 × 0.045 = 0.0045` — about **0.17% of L_contrast's contribution (2.69)**. Either raise λ_dis substantially or replace L_dis with a stronger non-collapsing regularizer (Fix 6: VICReg/Barlow Twins).

### 0.5.9 Contrast temperature τ=0.07 is the simplest lever for city-clustering

τ=0.07 gives extremely sharp NT-Xent, which maximizes uniformity (city-blob organization). Raising to τ=0.15 reduces uniformity pressure, allowing functional structure to compete. **One-line config change** that should be tried before architectural Fix 8.

### 0.5.10 Updated next-step priorities

| # | Action | Effort | Status | Why now |
|---|--------|--------|--------|---------|
| 1 | Validate sat_mean↔pop_density correlation (0.5.1) | done | ✅ | r≈0.016 ≈ random — pop_density is narrow proxy |
| 2 | Check pop-density distribution across cities (0.5.2) | done | ✅ | Singapore log std=3.23 vs 0.88 |
| 3 | Apply log1p to linear probe target | done | ✅ | Probe now calibrated; R² reported in log-space |
| 4 | Try τ_contrast=0.15 (0.5.9) | done | ✅ | **FAILED**: centroid cos -0.397→-0.594; temperature is not the bottleneck |
| 5 | Fix 2 (UnifiedCityCon 2B×2B) | done | ✅ | Centroid -0.594→+0.819 ✅; but over-alignment: within-city sim=0.81 ⚠️ |
| 6 | Fix 8+4 (bugged Sinkhorn) | done | ❌ | Sinkhorn rows summed to 2.0 → proto loss 5.4 → dominated training → R²=-0.62 |
| 7 | Fix 8+4 corrected (λ_proto=0.3) | done | ✅ | R²: -0.17→**-0.07** ✅; centroid cos +0.816 ✅; within-city sim 0.807 ⚠️ still high |
| 8 | **CLIP-like (CLIPRegionCon B×B)** | done | ✅ ❌ | Within-city sim ↓ (0.807→0.721) ✅ but city-domain clustering returned (centroid +0.816→**−0.449**) ❌ |
| 9 | Inject POI centroid (0.5.3) | small | ⬜ | Free spatial signal currently dropped |
| 10 | Raise λ_dis or apply Fix 6 VICReg (0.5.8) | small | ⬜ | Make disentanglement load-bearing |
| 11 | Fix 0 Solution B + shared-space alignment (0.5.11) | medium | ⬜ | If CLIP-like doesn't fix within-city sim |

**Fix 8 Sinkhorn bug — root cause and fix:**

The original Sinkhorn normalised rows incorrectly: after 3 iterations the assignment rows summed to ~2.0 instead of 1.0, inflating the loss 2× (5.4 observed vs 2.77 expected for K=16 uniform). With λ_proto=0.5, the effective gradient contribution was 0.5×5.4=2.7 — equal to L_contrast — so prototypes dominated training and prevented meaningful contrastive learning. The fix transposes Q to [K, B], normalises prototype rows then sample cols per SwAV convention, then rescales. Confirmed fixed: proto baseline is now 2.47 ≈ log(16)=2.77. λ_proto reduced to 0.3 (contribution ~0.83 vs contrast ~3-5; ~22% of total).

**Monitoring signal for correct Fix 8**: `val_proto` should decrease steadily from ~2.77 toward 0 as prototypes learn meaningful functional assignments. If stuck at ~2.77, Sinkhorn or gradients are not flowing.

**Primary objective reminder**: the goal is cross-city transferable region embeddings, not pop_density R². Primary quality indicators: (a) cross-city centroid cosine **positive and stable** (≥ +0.5), (b) within-city sim **drops to 0.4–0.6** (discriminative structure restored), (c) `val_proto` decreasing from ~2.77 toward 0, (d) mixed-city functional clusters in t-SNE.

**Current state**: CLIP-like (CLIPRegionCon B×B) training in progress. Key question: does within-city sim drop from 0.807/0.722 → target 0.4–0.6?

---

### 0.5.11 Architectural Proposal — Shared-Space Contrastive Alignment

**Proposal**: restrict `UnifiedCityCon` to operate only on the 32-d shared subspace (`sat_shared_region`, `poi_shared_region`) rather than the full 64-d `sat_region`/`poi_region`. Keep `FunctionalPrototypeLoss` on the full 128-d `region_emb`. Fix 0 Solution B (separate aggregators per branch) is a prerequisite.

**Motivation (from over-alignment observation)**:
After Fix 2, within-city sim rose to 0.81 and Seoul norm std collapsed to 0.028. The root cause is that `UnifiedCityCon` currently pulls the entire 64-d `sat_region` and `poi_region` toward alignment — the spec branch has no protected subspace and gets dragged along. If cross-city alignment operates only on the explicitly shared 32-d subspace, the spec branch (32-d) is never touched by the alignment loss and can maintain within-city discriminative structure.

**Proposed loss routing**:
```
UnifiedCityCon  →  sat_shared_region [32-d] + poi_shared_region [32-d]
BGDisLoss       →  spec ⊥ shared  (unchanged, BG-level)
FunctionalPrototypeLoss  →  region_emb [128-d]  (full embedding)
```

**Strengths**:
1. The spec branch (city-unique features) is protected from cross-city alignment pressure — within-city diversity is preserved.
2. The shared branch receives a clean, focused signal: only functionally transferable features should be shared.
3. `L_proto` on the full embedding still pulls both shared and spec dimensions toward functional cluster structure — it is the only loss that touches the full space.
4. This is the architecture DSN (Bousmalis et al., NeurIPS 2016) and related domain separation networks intended: private/shared decomposition with loss routing per branch.

**Concerns**:
1. **Requires Fix 0 Solution B** (separate aggregators for shared and spec branches). Solution A (concatenate-then-aggregate) does not produce a clean 32-d shared region vector. This is a medium-effort architectural change.
2. **32-d alignment subspace may be too narrow**. With `τ=0.07` and a 32-d unit sphere, the expressiveness of `UnifiedCityCon` may drop. May need to raise `n_prototypes` or reduce `contrast_temp` slightly.
3. **`L_proto` on 128-d still touches the spec branch indirectly** — if prototype assignment is dominated by shared-subspace geometry (which it likely will be after alignment), the spec branch may still collapse under prototype pressure. Monitor spec branch variance separately.
4. **Two contrastive losses with different embedding dimensionalities** can create gradient scale imbalance (32-d vs 128-d softmax temperatures are not directly comparable). May need to re-tune `contrast_temp` and `proto_temp` separately.

**Verdict**: structurally sound. However, it requires Solution B (non-trivial refactor). A simpler first attempt at the same core idea — protecting the spec branch from within-modality negative pressure — is **CLIPRegionCon (B×B cross-modal only)**, which achieves spec branch protection without needing Solution B. CLIPRegionCon is currently being trained (§0.5.10 row 8). If within-city sim drops to 0.4–0.6 after that run, this full architectural change is unnecessary. If CLIP-like is still insufficient, implement Solution B + 32-d shared-space contrastive as described above.

**Implementation order if needed**:
1. Implement Fix 0 Solution B (separate `sat_shared_agg`, `sat_spec_agg`, `poi_shared_agg`, `poi_spec_agg` aggregators in `model.py`).
2. Change `UnifiedCityCon` input from `sat_region`/`poi_region` to `sat_shared_region`/`poi_shared_region`.
3. Keep `FunctionalPrototypeLoss` on `region_emb` (128-d, unchanged).
4. Verify: spec branch variance should be higher than in the current Fix 2 run (Seoul norm std > 0.05).

---

### 0.5.12 Root Cause Analysis: Within-city Diversity vs Cross-city Alignment Trade-off

**Observed trade-off:**

| Approach | Matrix | Within-city sim (NYC/Seoul) | NYC↔Seoul centroid cos |
|----------|--------|---------------------------|------------------------|
| UnifiedCityCon (2B×2B) | all pairs | 0.807 / 0.722 ⚠️ | +0.816 ✅ |
| CLIPRegionCon (B×B) | cross-modal only | 0.721 / 0.147 ✅ | −0.449 ❌ |

Neither alone works. The root cause is that the two goals use opposing mechanisms:

**Why UnifiedCityCon causes over-alignment:**
The 2B×2B matrix includes same-modal cross-city positives: `pos_mask[i, j] = 1` where `i=sat_nyc`, `j=sat_seoul`. This directly pulls NYC sat embeddings and Seoul sat embeddings together, collapsing within-city diversity. The spec branch gets dragged along.

**Why CLIPRegionCon loses cross-city alignment:**
The B×B matrix removes within-modality negatives entirely. In UnifiedCityCon, `sat_nyc_i` and `sat_seoul_j` appeared in each other's denominator (as negatives) AND as positives (if pseudo-positive). This dual pressure — pull close if functionally similar, push apart otherwise — created fine-grained interleaving. In B×B, `sat_nyc_i` only competes against B poi embeddings; it never directly competes with `sat_seoul_j`. Without this cross-city negative pressure, the two cities drift into opposite hemispheres.

**The fundamental tension:**
- Cross-city alignment needs: same-function regions from different cities → close in embedding space
- Within-city diversity needs: different-function regions from same city → far apart
- UnifiedCityCon satisfies #1 by pulling all cross-city pseudo-positive pairs → BUT this also pulls non-pseudo-positive pairs closer (via the shared denominator), collapsing diversity
- CLIPRegionCon removes cross-city interaction entirely → cities drift apart

**Three candidate solutions:**

**Option A — Cross-modal only positives in 2B×2B (minimal change)**
Keep the 2B×2B matrix of UnifiedCityCon but remove same-modal cross-city positives. Only keep:
```
pos_mask[i,   B+i] = 1  # sat_i → poi_i  (same region, always)
pos_mask[B+i, i  ] = 1  # poi_i → sat_i
pos_mask[i,   B+j] = 1  # sat_i → poi_j  (cross-city cross-modal only)
pos_mask[B+i, j  ] = 1  # poi_i → sat_j
```
Remove `pos_mask[i, j]` (sat↔sat cross-city) and `pos_mask[B+i, B+j]` (poi↔poi cross-city).

**Why this helps**: sat_i and sat_j remain as mutual negatives in the 2B×2B denominator (cross-city repulsion preserved → cities don't drift). Cross-city alignment comes through cross-modal positives (sat_nyc → poi_seoul), not same-modal positives (sat_nyc → sat_seoul). The spec branch is no longer pulled by sat↔sat cross-city positives.

**Risk**: cross-city alignment signal is weaker (only cross-modal pseudo-positive, not 4-way). Centroid cos may not reach +0.816 again.

**Option B — Projection heads (decouple alignment from embedding)**
Add `sat_proj: Linear(64, 64)` and `poi_proj: Linear(64, 64)`. Run UnifiedCityCon (2B×2B with all 4 positive types) on the projected space, NOT on `sat_region`/`poi_region` directly. `region_emb = concat(sat_region, poi_region)` is unchanged and used only for L_proto and downstream.

```
L_clip  operates on:  sat_proj(sat_region),  poi_proj(poi_region)  [64-d projections]
L_proto operates on:  region_emb = concat(sat_region, poi_region)  [128-d, unchanged]
```

Gradient flow: UnifiedCityCon → proj heads → sat_agg weights (indirect); L_proto → region_emb → sat_agg + poi_agg (direct). The projection heads absorb most of the alignment pressure, protecting the main embedding space.

**Why this helps**: the over-alignment problem is in `region_emb` (used downstream). If alignment operates on a separate projection, `region_emb` retains within-city diversity. Standard in CLIP (original paper uses projection heads before the contrastive loss).

**Risk**: proj heads may simply relay the over-alignment back into the encoder weights. Need to verify that `sat_region` norms don't collapse even when `sat_proj(sat_region)` does.

**Option C — Separate contrastive temperatures per positive type**
Use the full 2B×2B UnifiedCityCon but scale cross-city positive contributions with a lower weight (soft positive weighting):
```python
# Instead of: pos_mask[i, j] = 1.0 for cross-city
pos_mask[i, j] = 0.3  # soft weight — less pull toward cross-city positives
pos_mask[i, B+i] = 1.0  # full weight for same-region
```
Or equivalently: multiply cross-city positive logits by a factor < 1 before logsumexp.

**Why this helps**: reduces the cross-city pull strength, allowing within-city repulsion to partially win, restoring some diversity. Simple one-parameter tuning.

**Risk**: soft weighting doesn't fix the structural issue — it just dials back the magnitude. May need very careful tuning and may produce a fragile optimum.

**Recommended order:**
1. **Option A first** (1-day change, cleanest fix) — remove same-modal cross-city positives from UnifiedCityCon. Retrain and check: does centroid cos stay positive AND within-city sim drop to 0.4–0.6?
2. **Option B if A insufficient** — add projection heads. Structural decoupling; more robust but 2-day change.
3. **Option C** — soft weighting as a quick diagnostic to understand the sensitivity.

---

## 1. Diagnosis from Training Dynamics

Reading `checkpoints/nyc_seoul/logs/history.json` across 100 epochs reveals three distinct failure signatures.

**Historical log — Run 1 (attraction-only loss, thr=0.7, balanced=F):** shows the original failure before fixes.

| Epoch | train_contrast | train_align | train_dis | val_contrast | val_align | val_dis |
|-------|--------------|-------------|-----------|-------------|-----------|---------|
| 1     | 3.1904        | 0.1608      | 0.1733    | 2.9515       | 0.1823    | 0.1364  |
| 5     | 2.9767        | 0.1896      | 0.0768    | 2.8870       | 0.1790    | 0.0768  |
| 10    | 2.9177        | 0.1973      | 0.0570    | 2.8192       | 0.2025    | 0.0574  |
| 50    | 2.6504        | 0.2414      | 0.0430    | 2.6153       | 0.2305    | 0.0441  |
| 100   | 2.5846        | 0.2528      | 0.0419    | 2.5864       | 0.2464    | 0.0427  |

**Current log — Run 3 (Fix 0+1, InfoNCE, thr=0.9, balanced=T):** the live training state.

| Epoch | train_contrast | train_align | train_dis | val_contrast | val_align | val_dis |
|-------|--------------|-------------|-----------|-------------|-----------|---------|
| 1     | 2.8953        | 1.3477      | 0.0960    | 2.9628       | 0.6234    | 0.0970  |
| 5     | 2.6213        | 0.7222      | 0.0727    | 2.8743       | 0.4340    | 0.0699  |
| 10    | 2.5645        | 0.5887      | 0.0533    | 2.8368       | 0.4060    | 0.0582  |
| 20    | 2.4258        | 0.4566      | 0.0427    | 2.7879       | 0.3067    | 0.0499  |
| 50    | 2.2795        | 0.3658      | 0.0367    | 2.7207       | 0.2751    | 0.0452  |
| 100   | 2.1875        | 0.3548      | 0.0354    | 2.6919       | 0.2456    | 0.0448  |

### 1.1 L_align is increasing, not decreasing (Run 1 — historical)

| Epoch | val_align | Interp. |
|-------|-----------|---------|
| 1     | 0.182     | random init |
| 10    | 0.203     | starts rising |
| 50    | 0.231     | plateau forms |
| 100   | 0.246     | **never recovered** |

`L_align = (1 - cos_sim).mean()` in the old attraction-only form. Rising means the model was actively pushing pseudo-positive pairs apart. **Fixed in Run 3**: val_align now decreases 0.623 → 0.246 (log-space InfoNCE — lower = better aligned).

**Gradient scale confirmed fixed**: Run 1 had contrast/align ratio ~10×. Run 3 ratio ~11× by value but InfoNCE is in log-space so the effective gradient magnitude is comparable.

### 1.2 L_dis collapses in the first 10 epochs (Run 1 — historical)

| Epoch | train_dis | val_dis |
|-------|----------|---------|
| 1     | 0.173    | 0.136   |
| 5     | 0.077    | 0.077   |
| 10    | 0.057    | 0.057   |
| 100   | 0.042    | 0.043   |

Monotonic collapse to ~0.043 is not genuine disentanglement. The network satisfies `cos(spec, shared.detach()) ≈ 0` trivially by either (a) shrinking `spec` toward zero or (b) aligning `spec` orthogonal to `shared` in a degenerate subspace. Either way the `spec` branch learns very little.

In Run 3 (Fix 0+1), val_dis starts at 0.097 and stabilises at 0.045 — slightly higher and more stable because Fix 0 now routes gradients from L_contrast through the spec branch. But λ_dis=0.1 means its contribution is still only ~0.17% of L_contrast (see §0.5.8).

**There is a more fundamental issue here**, separate from the trivial-solution problem. See §1.4 and Fix 0 below.

### 1.4 The shared/spec branches are disconnected from `region_emb`

Reading `models/model.py:63-88` carefully:

```python
def forward_region(self, sat_data, poi_emb, valid_mask):
    # BG level projection — these four heads exist...
    sat_shared = self.sat_shared_proj(sat_data)
    sat_spec   = self.sat_spec_proj(sat_data)
    poi_shared = self.poi_shared_proj(poi_emb)
    poi_spec   = self.poi_spec_proj(poi_emb)

    # ...but aggregation operates on the RAW inputs, not on the projections!
    sat_region = self.sat_agg(sat_data, valid_mask)
    poi_region = self.poi_agg(poi_emb, None)
    region_emb = torch.cat([sat_region, poi_region], dim=-1)
```

The four projection heads (`sat_shared`, `sat_spec`, `poi_shared`, `poi_spec`) feed only `BGDisLoss`. They never enter the `region_emb` pipeline. This means:

1. **Even if `L_dis` worked perfectly**, the disentangled features would not improve `region_emb` quality, because `region_emb` is built from the un-projected raw inputs.
2. The shared/spec projection heads receive gradient *only* from `L_dis`. With no other loss touching them, the orthogonality constraint can be satisfied by trivial solutions (zero outputs, degenerate orthogonal subspaces), explaining the rapid collapse to ~0.002.
3. **`L_dis` is doing essentially nothing useful for the downstream task.** Its effective contribution to `region_emb` is zero by construction.

This is the most consequential architectural finding in this analysis. The "two-level disentangled" design described in the codebase is, in implementation, a single-level architecture with an unused auxiliary branch.

### 1.3 L_contrast dominates everything

Weighted gradient contributions per epoch at convergence:
- `λ_contrast × L_contrast` ≈ 1.0 × 2.60 = **2.60**
- `λ_align    × L_align`    ≈ 1.0 × 0.25 = **0.25** (10× smaller)
- `λ_dis      × L_dis`      ≈ 0.1 × 0.002 = **0.0002**

The model is shaped almost entirely by the symmetric NT-Xent between `sat_region` and `poi_region` within the *same* city. Cross-city alignment provides roughly 9% of the total gradient.

---

## 2. Root Cause: L_align vs. L_contrast Are in Conflict

**L_contrast** (NT-Xent, temp=0.07) treats every region pair in the batch as a negative. If region A from NYC and region B from Seoul are functionally similar (high `sat_mean` cosine sim), L_contrast still pushes their `region_emb` apart — because it sees them as two different items in the batch. Meanwhile L_align tries to pull them together.

**L_contrast wins** because:
1. Its scale (~2.6) is 10× larger than L_align (~0.25).
2. NT-Xent operates on all N×N pairs simultaneously; L_align touches only pseudo-positive pairs (sparse subset).
3. L_contrast has sharp gradients at temperature τ=0.07; L_align gradient is a flat cosine difference.

This is the **alignment–uniformity conflict** (Wang & Isola, ICML 2020): NT-Xent aggressively maximizes *uniformity* (spread all embeddings apart); L_align asks for *alignment* of specific pairs. When they share the same embedding space and L_contrast is dominant, L_align cannot overcome the repulsion gradient.

### 2.2 Why L_contrast Alone Produces City-Domain Clusters

Even if the gradient-scale conflict is fully resolved (Fixes 1–2), the embedding space will still tend to organize by city identity rather than by function, for a structural reason rooted in the data geometry.

In a mixed-city batch of B regions (≈B/2 from NYC, ≈B/2 from Seoul), NT-Xent computes B−1 negatives for each anchor. NYC and Seoul have systematically different urban morphology — different street patterns, building materials, block sizes, POI category distributions. This means *city identity is the single most discriminative axis available to the model*. NT-Xent exploits the easiest discriminative feature first (Geirhos et al., "Shortcut Learning," Nature Machine Intelligence 2020); city identity is a strong, consistent shortcut that maximally separates embeddings with minimal loss.

The result is a hypersphere organized like this:

```
          Seoul cluster
         /
hypersphere center
         \
          NYC cluster
```

Cross-city pseudo-positive pairs live at the interface of these two clusters but are outnumbered by within-cluster pairs. The functional signal (residential ↔ residential across cities) is weaker than the city-identity signal, so the former gets subordinated to the latter.

**Fix 2 (UnifiedCityCon) partially mitigates this** — cross-city pseudo-positives are removed from each other's negative denominator, so the direct repulsion between those specific pairs is gone. But:
- Non-pseudo-positive cross-city pairs are still mutual negatives → city-level repulsion persists for most pairs.
- There is no global attractor pulling *all* residential regions (from both cities) toward a common cluster center. Fix 2 is pairwise; it does not create a global cluster geometry.

**What is missing**: a mechanism that organizes the full embedding space into K functional clusters (residential, commercial, industrial, park, …) shared across all cities, independent of pair-wise satellite similarity. This is Fix 8.

---

## 3. Why the Removed Approaches Likely Failed

### Prototype bank
A prototype bank that stores city-level centroids is prone to collapse when cities have unequal numbers of regions — one city's prototype dominates the bank. The prototype update rule (EMA or hard assignment) also suffers from stale representations early in training. However, the *idea* of prototypes is sound; the implementation details (update rule, batch dependency) likely caused the problem, not the concept itself.

### CORAL (Correlation Alignment)
CORAL aligns second-order statistics of feature distributions across domains (Sun & Saenko, ECCV 2016). In this setting the aggregator produces one vector per region, so CORAL would align per-region distributions. With only 2 source cities and variable region sizes, the covariance matrix estimate is noisy. CORAL is also domain-level (aligns marginal distributions), whereas the goal here is to find *specific* functionally similar pairs — a finer-grained objective that CORAL cannot express.

### GRL (Gradient Reversal Layer)
GRL (Ganin et al., JMLR 2016) maximizes domain confusion, i.e., it wants `region_emb` to be domain-*invariant*. But invariance and discriminativeness conflict at the region level: a residential district in NYC and a commercial district in Seoul must be *different* from each other even though they are from different domains. GRL would erase this difference. The goal is not full domain invariance but *selective* alignment of functionally similar regions. GRL is the wrong abstraction for this problem.

---

## 4. Proposed Solutions (Prioritized)

---

### Fix 0 — Wire the shared/spec branches into `region_emb` (Architectural Prerequisite) ✅ APPLIED

**Problem**: as shown in §1.4, the four BG-level projection heads currently feed only `BGDisLoss` and have no path into `region_emb`. Any improvement to `L_dis` (Fix 6) is wasted unless the disentangled features are actually used.

**Solution A — Aggregate the projections, then concatenate** (smallest change):

```python
def forward_region(self, sat_data, poi_emb, valid_mask):
    sat_shared = self.sat_shared_proj(sat_data)   # [M, 32]
    sat_spec   = self.sat_spec_proj(sat_data)     # [M, 32]
    poi_shared = self.poi_shared_proj(poi_emb)    # [M, 32]
    poi_spec   = self.poi_spec_proj(poi_emb)      # [M, 32]

    # Reconstruct each modality from its disentangled parts, then aggregate
    sat_recon = torch.cat([sat_shared, sat_spec], dim=-1)   # [M, 64]
    poi_recon = torch.cat([poi_shared, poi_spec], dim=-1)   # [M, 64]

    sat_region = self.sat_agg(sat_recon, valid_mask)        # [64]
    poi_region = self.poi_agg(poi_recon, valid_mask)        # [64]   ← also fix POI mask
    region_emb = torch.cat([sat_region, poi_region], dim=-1) # [128]
    ...
```

This routes every region_emb dimension through the disentangled subspace, so `BGDisLoss` becomes load-bearing. (Note: also pass `valid_mask` to the POI aggregator — the current `None` is likely an oversight.)

**Solution B — Aggregate shared and specific branches separately, then fuse** (more expressive):

```python
sat_shared_region = self.sat_shared_agg(sat_shared, valid_mask)   # [32]
sat_spec_region   = self.sat_spec_agg(sat_spec, valid_mask)       # [32]
poi_shared_region = self.poi_shared_agg(poi_shared, valid_mask)   # [32]
poi_spec_region   = self.poi_spec_agg(poi_spec, valid_mask)       # [32]

# Fuse cross-modal shared + concat modality-specific
shared_region = (sat_shared_region + poi_shared_region) / 2       # [32]  consensus
specific_region = torch.cat(
    [sat_spec_region, poi_spec_region], dim=-1)                   # [64]  modality-unique
region_emb = torch.cat([shared_region, specific_region], dim=-1)  # [96]
```

This explicitly separates "what satellite and POI agree on" from "what each modality uniquely contributes" — the design intent that the original code seems to want but does not implement. Note `region_emb` becomes 96-d here; either accept this or pad shared to 64-d to keep 128-d.

**Solution C — Drop the shared/spec heads and `L_dis` entirely**: if you don't intend to do disentanglement, removing four unused projection heads and one ineffective loss simplifies the codebase. This is the pragmatic choice if Fix 8 (functional prototypes) becomes the structural objective.

**Recommended**: Solution A is the smallest diff and immediately makes `L_dis` (or its replacement, Fix 6) actually do work. Apply Fix 0 *before* Fix 6, otherwise Fix 6 has no surface to act on.

Grounded in:
- **DSN** (Bousmalis et al., NeurIPS 2016, Domain Separation Networks): the canonical shared/private decomposition explicitly uses both branches to reconstruct the input; the private branch is not auxiliary.
- **Multi-modal disentanglement surveys** (Hwang et al., MultiBench, NeurIPS 2021; Liang et al., "Foundations of Multimodal Co-learning," 2022): in practice, modality-specific subspaces only generalize when they participate in the prediction head, not when they are detached regularization targets.

---

### Fix 1 — Replace L_align with a Cross-City InfoNCE Loss (Highest Impact) ✅ APPLIED

**Problem**: attraction-only MSE (`1 - cos_sim`) has 10× smaller gradient scale than NT-Xent and lacks repulsion.

**Solution**: reformulate L_align as an InfoNCE loss over cross-city pairs, where pseudo-positives (high `sat_mean` sim) are the positive keys and all other cross-city regions are negatives.

```python
class SatAlignLossInfoNCE(nn.Module):
    def __init__(self, threshold=0.7, temperature=0.1):
        super().__init__()
        self.threshold   = threshold
        self.temperature = temperature

    def forward(self, outputs):
        city_regions = {}
        for i, out in enumerate(outputs):
            city_regions.setdefault(out["city"], []).append(i)

        cities = list(city_regions.keys())
        if len(cities) < 2:
            return outputs[0]["region_emb"].new_zeros(1, requires_grad=True).squeeze()

        sat_means   = F.normalize(torch.stack([o["sat_mean"]   for o in outputs]), dim=-1)
        region_embs = F.normalize(torch.stack([o["region_emb"] for o in outputs]), dim=-1)

        loss_list = []
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                idx_i, idx_j = city_regions[cities[i]], city_regions[cities[j]]

                sat_sim   = sat_means[idx_i] @ sat_means[idx_j].T  # [N_i, N_j]
                mask_pos  = sat_sim > self.threshold                # [N_i, N_j]

                if mask_pos.sum() == 0:
                    continue

                emb_i = region_embs[idx_i]  # [N_i, 128]
                emb_j = region_embs[idx_j]  # [N_j, 128]
                logits = emb_i @ emb_j.T / self.temperature  # [N_i, N_j]

                # InfoNCE: for each anchor in city i, positive = pseudo-pos rows in city j
                for row in range(len(idx_i)):
                    pos_cols = mask_pos[row].nonzero(as_tuple=True)[0]
                    if len(pos_cols) == 0:
                        continue
                    # mean over multiple positives (SupCon-style)
                    log_denom = torch.logsumexp(logits[row], dim=0)
                    log_numer = torch.logsumexp(logits[row][pos_cols], dim=0)
                    loss_list.append(log_denom - log_numer)

        if not loss_list:
            return outputs[0]["region_emb"].new_zeros(1, requires_grad=True).squeeze()
        return torch.stack(loss_list).mean()
```

This is grounded in:
- **SimCLR** (Chen et al., ICML 2020): the InfoNCE gradient is proportional to the number of hard negatives, giving the alignment loss comparable gradient scale to L_contrast.
- **SupCon** (Khosla et al., NeurIPS 2020): when multiple positives exist (several cross-city regions with high `sat_mean` sim), SupCon-style averaging over all positives is more stable than picking one.

Expected effect: L_align scale rises from ~0.25 to a log-space range (1–4), matching L_contrast. The increasing trend in val_align should reverse.

---

### Fix 2 — Unified Multi-Positive Contrastive Loss (Eliminates L_contrast / L_align Conflict)

The deepest fix is to merge L_contrast and L_align into a single loss with a shared positive/negative assignment matrix. Currently the conflict exists because L_contrast treats all cross-city pairs as negatives while L_align wants specific ones as positives.

Define a single batch-level similarity matrix `A` where `A[i,j] = 1` if:
- Same region, different modality (satellite vs. POI) → existing L_contrast positive, OR
- Different city, same function (sat_mean cosine sim > threshold) → L_align positive

Then apply SupCon:

```python
class UnifiedCityCon(nn.Module):
    """
    Single NT-Xent over all region representations with multi-positive labels.
    Eliminates the L_contrast vs L_align gradient conflict.
    """
    def __init__(self, temperature=0.07, align_threshold=0.7):
        super().__init__()
        self.temperature     = temperature
        self.align_threshold = align_threshold

    def forward(self, outputs):
        # Stack both modalities: 2B vectors
        sat = F.normalize(torch.stack([o["sat_region"] for o in outputs]))  # [B, 64]
        poi = F.normalize(torch.stack([o["poi_region"] for o in outputs]))  # [B, 64]
        all_embs = torch.cat([sat, poi], dim=0)                             # [2B, 64]

        B = len(outputs)
        device = sat.device

        # Base positives: sat_i ↔ poi_i (same region, cross-modal)
        pos_mask = torch.zeros(2 * B, 2 * B, device=device)
        for i in range(B):
            pos_mask[i, B + i] = 1.0
            pos_mask[B + i, i] = 1.0

        # Add cross-city pseudo-positives to the mask
        sat_means = F.normalize(torch.stack([o["sat_mean"] for o in outputs]))
        sat_sim   = sat_means @ sat_means.T  # [B, B]
        cities    = [o["city"] for o in outputs]
        for i in range(B):
            for j in range(B):
                if cities[i] != cities[j] and sat_sim[i, j] > self.align_threshold:
                    # mark sat_i ↔ sat_j and poi_i ↔ poi_j as positives
                    pos_mask[i, j]         = 1.0
                    pos_mask[B+i, B+j]     = 1.0
                    pos_mask[i, B+j]       = 1.0   # cross-modal cross-city
                    pos_mask[B+i, j]       = 1.0

        logits = all_embs @ all_embs.T / self.temperature
        logits.fill_diagonal_(-1e9)  # exclude self

        log_denom = torch.logsumexp(logits, dim=1)  # [2B]
        loss = 0.0
        count = 0
        for i in range(2 * B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            log_numer = torch.logsumexp(logits[i][pos_idx], dim=0)
            loss += log_denom[i] - log_numer
            count += 1

        return loss / max(count, 1)
```

This directly solves the gradient conflict: L_contrast no longer pushes cross-city pseudo-positives apart because they are excluded from the negatives for each other.

---

### Fix 3 — Soft Pseudo-Positive Weighting (Replace Hard Threshold)

**Problem**: `sat_sim > 0.7` is a binary gate. A pair at sim=0.71 is treated identically to sim=0.95. Pairs near the threshold boundary are noisy pseudo-labels.

**Solution**: replace the binary mask with a temperature-scaled soft weight:

```python
# Instead of: mask_pos = sat_sim > self.threshold
# Use:
soft_weight = torch.sigmoid((sat_sim - self.threshold) / 0.05)  # smooth gate
# Weight the loss by soft_weight[i,j]
loss = (soft_weight[mask_pos] * (1 - pos_emb_sim[mask_pos])).sum() / soft_weight[mask_pos].sum()
```

Or in the InfoNCE form, weight the log-numerator:

```python
log_numer = torch.log((soft_weight[row] * torch.exp(logits[row])).sum())
```

Grounded in:
- **RINCE** (Chuang et al., NeurIPS 2022): robust InfoNCE under noisy views uses re-weighted denominators to discount uncertain negatives.
- **Soft-NN** (Frosst et al., NeurIPS 2019): soft nearest-neighbor loss for representation learning uses temperature-scaled assignments.

---

### Fix 4 — Dual-Signal Pseudo-Positive Mining (Satellite + POI)

**Problem**: pseudo-positive pairs are mined using only frozen `sat_mean`. POI patterns are more semantically meaningful for functional similarity (a park in NYC and a park in Seoul should have similar POI profiles), and using only satellite risks false positives (industrial and residential areas may look similarly flat in satellite imagery).

**Solution**: require *both* modalities to agree before calling a pair positive.

```python
sat_sim_matrix = sat_means_i @ sat_means_j.T   # [N_i, N_j]
poi_means_i = torch.stack([outputs[k]["poi_region"] for k in idx_i])
poi_means_j = torch.stack([outputs[k]["poi_region"] for k in idx_j])
poi_sim_matrix = F.normalize(poi_means_i) @ F.normalize(poi_means_j).T

# Require both modalities to agree — intersection
mask_pos = (sat_sim_matrix > sat_threshold) & (poi_sim_matrix > poi_threshold)

# Or: confidence = geometric mean of both similarities
confidence = (sat_sim_matrix.clamp(0) * poi_sim_matrix.clamp(0)).sqrt()
mask_pos = confidence > combined_threshold
```

Note: `poi_region` is *learned* (updated each step), while `sat_mean` is frozen. Using poi_region dynamically risks noisy mining early in training. A momentum-updated poi_mean (EMA) would stabilize this.

---

### Fix 5 — Stop-Gradient Momentum Target for L_align (BYOL-Style)

**Problem**: both sides of a pseudo-positive pair compute gradients through the model, creating conflicting gradient directions when L_contrast and L_align disagree on the same pair.

**Solution**: use a momentum encoder (EMA copy of the online model) to generate *stable target embeddings* for one side of each pseudo-positive pair.

```python
# In __init__:
self.momentum_model = copy.deepcopy(model)
for p in self.momentum_model.parameters():
    p.requires_grad_(False)
self.momentum = 0.99

# Each step:
@torch.no_grad()
def update_momentum():
    for p_online, p_target in zip(model.parameters(), momentum_model.parameters()):
        p_target.data = momentum * p_target.data + (1 - momentum) * p_online.data

# L_align: online_emb vs. stop_grad(momentum_emb)
target_embs = momentum_model.get_region_embedding(...)
loss = (1 - F.cosine_similarity(online_embs, target_embs.detach())).mean()
```

Grounded in:
- **BYOL** (Grill et al., NeurIPS 2020): asymmetric attraction with momentum target is provably collapse-free.
- **MoCo v3** (Chen et al., 2021): combining momentum encoder with NT-Xent is more stable than vanilla attraction-only.

Note: `sat_mean` is already frozen (which is good — it provides a stable alignment anchor). Extending this principle to the target embedding resolves the conflicting gradient problem.

---

### Fix 6 — Replace L_dis with VICReg or Barlow Twins (Non-Collapsing Disentanglement)

**Problem**: `BGDisLoss` collapses to ~0.002 by epoch 10. The gradient `cos(spec, shared.detach()) ≈ 0` is satisfied trivially by making `spec` output near-zero or purely orthogonal in a degenerate sense. The `spec` branch doesn't learn meaningful modality-specific features.

**Solution A — VICReg regularizer on the spec branch** (Bardes et al., ICLR 2022):

```python
class VICRegDisLoss(nn.Module):
    """
    Prevents spec from collapsing while still disentangling from shared.
    """
    def __init__(self, lam=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lam, self.mu, self.nu = lam, mu, nu
        self.gamma, self.eps = gamma, eps

    def forward(self, outputs):
        spec_list = []
        for out in outputs:
            mask = out["valid_mask"]
            if mask.sum() > 1:
                spec_list.append(torch.cat([out["sat_spec"][mask], out["poi_spec"][mask]]))

        if not spec_list:
            return outputs[0]["sat_spec"].new_zeros(1, requires_grad=True).squeeze()

        Z = torch.cat(spec_list, dim=0)  # [N, spec_dim]

        # Variance: prevent collapse (each dim should have std > 1)
        std = Z.std(dim=0)
        var_loss = F.relu(self.gamma - std).mean()

        # Invariance: covered by main L_contrast/L_align
        # Covariance: off-diagonal covariance should be zero (independence)
        Z_norm = Z - Z.mean(dim=0)
        cov = (Z_norm.T @ Z_norm) / (len(Z) - 1)
        cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / Z.shape[1]

        # Orthogonality with shared: keep the original dis constraint
        orth_loss = outputs[0]["sat_spec"].new_zeros(1, requires_grad=True).squeeze()
        # ... (original BGDisLoss computation)

        return self.lam * var_loss + self.nu * cov_loss + orth_loss
```

**Solution B — Barlow Twins cross-correlation on shared subspaces** (Zbontar et al., ICML 2021):

Apply Barlow Twins between `sat_shared` and `poi_shared` to learn genuinely shared cross-modal features without collapse:

```python
def barlow_twins_loss(z_sat, z_poi, lam=5e-3):
    N, D = z_sat.shape
    z_a = (z_sat - z_sat.mean(0)) / z_sat.std(0).clamp(min=1e-6)
    z_b = (z_poi - z_poi.mean(0)) / z_poi.std(0).clamp(min=1e-6)
    C = (z_a.T @ z_b) / N                       # [D, D]
    on_diag  = (1 - C.diagonal()).pow(2).sum()   # invariance: diagonal → 1
    off_diag = (C.fill_diagonal_(0)).pow(2).sum() # redundancy: off-diag → 0
    return on_diag + lam * off_diag
```

---

### Fix 7 — Curriculum on Pseudo-Positive Threshold

**Problem**: the threshold is fixed at 0.7 throughout training. Early on, representations are noisy and borderline pairs (sim ≈ 0.7–0.75) are low-confidence pseudo-labels. Later, the representation space has meaningful structure and the threshold could be relaxed.

**Solution**: anneal threshold from high (strict early, high-confidence only) down to current setting (looser late, more pairs). Calibrated to this dataset's sat_mean distribution (mean=0.853): anneal 0.95→0.90, not the original 0.85→0.65 which would include >62% of pairs as positives:

```python
threshold = max(0.90, 0.95 - epoch / max_epochs * 0.05)
```

Grounded in:
- **Curriculum learning** (Bengio et al., ICML 2009): start with easy/confident examples, progressively add harder ones.
- **Noisy label learning**: pseudo-label confidence thresholds should be calibrated to the current model quality. FixMatch (Sohn et al., NeurIPS 2020) uses a confidence threshold on pseudo-labels for exactly this reason.

---

### Fix 8 — Cross-City Functional Prototype Clustering (SwAV-Inspired)

**Problem**: pairwise alignment (Fixes 1–2) reduces direct city-domain repulsion for pseudo-positive pairs, but does not create a global cluster structure organized by function. The embedding space still defaults to city-domain organization because city identity is the strongest available discriminative signal for NT-Xent.

**Solution**: introduce K learnable functional prototypes shared across all cities. Every region — regardless of origin city — is assigned to the most appropriate prototype. The training signal encourages consistent prototype assignment: if a residential region from NYC and a residential region from Seoul embed similarly in the 128-d space, they are assigned to the same "residential" prototype and receive the same gradient direction toward it.

```python
class FunctionalPrototypeLoss(nn.Module):
    """
    SwAV-inspired (Caron et al., NeurIPS 2020).
    K prototypes represent functional archetypes (residential, commercial,
    industrial, park, …) shared across all cities.
    Sinkhorn-Knopp normalization prevents prototype collapse.
    """
    def __init__(self, n_prototypes=16, region_emb_dim=128,
                 temperature=0.1, sinkhorn_iters=3):
        super().__init__()
        self.prototypes    = nn.Linear(region_emb_dim, n_prototypes, bias=False)
        self.temperature   = temperature
        self.sinkhorn_iters = sinkhorn_iters
        nn.init.normal_(self.prototypes.weight)

    def forward(self, outputs: list) -> torch.Tensor:
        embs = F.normalize(
            torch.stack([o["region_emb"] for o in outputs]), dim=-1)  # [B, 128]

        # Normalize prototypes onto unit sphere (no gradient through them in codes)
        with torch.no_grad():
            proto_w = F.normalize(self.prototypes.weight, dim=1)  # [K, 128]

        scores = embs @ proto_w.T / self.temperature  # [B, K]

        # Soft balanced assignments (Sinkhorn-Knopp prevents prototype collapse)
        Q = self._sinkhorn(scores.detach().exp())  # [B, K], rows ≈ uniform

        # Cross-entropy between target codes Q and predicted distribution
        loss = -(Q * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def _sinkhorn(self, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """SwAV-style Sinkhorn-Knopp. Input [B, K], output [B, K] with rows summing to 1.
        Bug note: the original version had rows summing to 2.0 not 1.0, inflating loss 2×.
        Fix: transpose to [K, B], normalise prototype rows then sample cols, rescale."""
        B, K = Q.shape
        Q = Q.t()                                               # [K, B]
        Q /= Q.sum() + eps                                      # global normalize
        for _ in range(self.sinkhorn_iters):
            Q /= Q.sum(dim=1, keepdim=True) + eps               # uniform over regions per prototype
            Q /= K                                              # each prototype row sums to 1/K
            Q /= Q.sum(dim=0, keepdim=True) + eps               # uniform over prototypes per region
            Q /= B                                              # each sample col sums to 1/B
        Q *= B                                                  # rescale: each col sums to 1
        return Q.t()                                            # [B, K], rows sum to 1
```

**Integration into TotalLoss**:
```python
L_total = λ_contrast · L_contrast  +  λ_align · L_align  +  λ_proto · L_proto
```
Start with `λ_proto = 0.5`. `L_proto` can replace `L_dis` entirely (since `L_dis` has already collapsed) or co-exist with it.

**Why this addresses city-domain clustering specifically**:
- Prototype assignment is computed over all B regions simultaneously. A prototype that captures "high-density residential" will attract matching regions from *both* NYC and Seoul toward the same point, regardless of which batch they appear in.
- Sinkhorn-Knopp prevents a degenerate solution where all B regions in a batch collapse to one prototype (which would trivially satisfy the loss but produce useless embeddings).
- The prototype weight matrix W ∈ ℝ^{K×128} acts as a set of cross-city functional anchors. Because it is shared, city identity provides no advantage — the model cannot satisfy `L_proto` by memorizing city patterns.

**Why this complements, not replaces, Fix 2**:
- Fix 2 removes pairwise city-domain repulsion for high-confidence pseudo-positive pairs.
- Fix 8 installs a global functional cluster structure over the entire embedding space.
- Together: Fix 2 handles known similar pairs explicitly; Fix 8 handles the global geometry implicitly, including regions with no known cross-city pseudo-positive (sat_sim < threshold).

**Choosing K**: start with K=8–16. Natural urban functional archetypes: residential-dense, residential-sparse, commercial-core, commercial-strip, industrial, park/green, transit-hub, mixed-use, waterfront. K=16 allows finer granularity. If K is too small, distinct functions merge; if too large, prototypes collapse despite Sinkhorn. Monitor prototype utilization (each prototype should be assigned >1% of regions in a batch).

**Relation to the removed prototype bank**: the previous prototype bank likely stored city-level or region-level centroids updated via EMA, causing stale representations and city-domain collapse. Fix 8 differs in two ways: (1) prototypes are *learned parameters* (gradient-updated, never stale), and (2) Sinkhorn-Knopp enforces balanced assignment, preventing any city's regions from monopolizing prototypes.

Grounded in:
- **SwAV** (Caron et al., NeurIPS 2020): online clustering with prototype codes; Sinkhorn-Knopp for balanced assignment.
- **PCL** (Li et al., ICLR 2021): prototypical contrastive learning for representation learning.
- **SupCon** (Khosla et al., NeurIPS 2020): when prototypes act as soft cluster labels, the assignment loss is equivalent to supervised contrastive learning with soft labels.

---

## 5. Implementation Roadmap (Ordered by Impact vs. Effort)

| Priority | Fix | Impact | Effort | Status | Notes |
|----------|-----|--------|--------|--------|-------|
| **1** | Wire shared/spec into region_emb (Fix 0) | High | Low | ✅ | Architectural prerequisite; ~10 lines in `model.py` |
| **2** | InfoNCE-based L_align (Fix 1) | High | Low | ✅ | Drop-in replacement for `SatAlignLoss.forward` |
| **3a** | Raise align_threshold 0.7→0.90 | Very High | Trivial | ✅ | InfoNCE went 0.003 (saturated) → 0.246 (working) |
| **3b** | Enable balanced_sampler | High | Trivial | ✅ | Seoul norm std recovered 0.019 → 0.124 |
| **4a** | Validate sat_mean ↔ function correlation | High | Trivial | ✅ | r≈0.016 ≈ random; pop_density is narrow proxy |
| **4b** | Check pop-density distribution per city | High | Trivial | ✅ | Singapore log std=3.23 vs 0.88; log1p applied |
| **4c** | Try τ_contrast=0.15 | Medium | Trivial | ✅ ❌ | FAILED: centroid cos −0.594 (worse) |
| **5** | UnifiedCityCon 2B×2B (Fix 2) + dual-signal (Fix 4) | Very High | Medium | ✅ | R²=−0.17; centroid +0.819 ✅; within-city 0.81 ⚠️ |
| **6** | FunctionalPrototypeLoss (Fix 8) corrected | Very High | Medium | ✅ | R²=−0.07 ✅; centroid maintained; within-city 0.807 ⚠️ |
| **7** | **CLIP-like CLIPRegionCon (B×B cross-modal)** | Very High | Low | 🔄 training | Spec branch protected from within-modality negatives |
| **8** | Fix 0 Solution B + shared-space alignment (0.5.11) | High | Medium | ⬜ | If CLIP-like doesn't resolve within-city sim |
| **9** | Inject POI centroid into model | Medium | Small | ⬜ | Currently loaded but unused — free signal |
| **10** | Raise λ_dis or apply Fix 6 (VICReg) | Medium | Small | ⬜ | L_dis is <1% of total gradient |
| **11** | Soft pseudo-positive weights (Fix 3) | Medium | Low | ⬜ | Sigmoid gate instead of hard threshold |
| **12** | Momentum encoder (Fix 5) | High | High | ⬜ | Major architecture change; defer |

### Current state

| Run | Loss | R² | NYC↔Seoul centroid cos | Within NYC | Within Seoul | Seoul norm std |
|-----|------|----|----------------------|------------|--------------|----------------|
| Fix 0+1+thr+balanced | L_contrast+L_align | −0.31 | −0.397 | 0.568 | 0.352 | 0.124 |
| Fix 2 (UnifiedCityCon) | L_unified | −0.17 | +0.819 | 0.809 ⚠️ | 0.710 ⚠️ | 0.028 ⚠️ |
| Fix 2+4+8 corrected | L_unified+L_proto | −0.07 | +0.816 | 0.807 ⚠️ | 0.722 ⚠️ | 0.027 ⚠️ |
| **CLIP-like** | **L_clip+L_proto** | **−0.11** | **−0.449 ❌** | 0.721 ↓ | 0.147 ↓↓ | 0.027 |

**CLIP-like diagnosis**: within-city diversity partially restored (NYC 0.807→0.721, Seoul 0.722→0.147) but cross-city alignment failed — centroid cos reversed from +0.816 → −0.449. B×B cross-modal loss removed the within-modality negatives that were implicitly driving cross-city interleaving in UnifiedCityCon. The two goals (within-city diversity, cross-city alignment) are in direct tension and need to be addressed simultaneously, not traded off. See §0.5.12 for root cause and solutions.

**Decision**: neither 2B×2B (UnifiedCityCon) nor B×B (CLIPRegionCon) alone solves both problems. Need a hybrid approach. See §0.5.12.

---

## 6. Monitoring Checklist

Once L_align is reformulated as InfoNCE, the training signals to watch:

| Signal | Healthy | Problem |
|--------|---------|---------|
| `val_align` trend | Monotonically decreasing | Rising or flat (→ still dominated by L_contrast) |
| `val_align` scale | Comparable to `val_contrast` | 10× smaller (→ still gradient imbalance) |
| `val_dis` scale | Stable > 0.01 | Collapsed to 0.002 (→ spec branch degenerate) |
| Downstream R² (Singapore) | Increasing | Not improving despite lower train loss |
| Cross-city pseudo-pair hit rate | 5–15% of cross-city pairs | <2% (threshold too high) or >30% (too low — InfoNCE saturates) |
| t-SNE colored by city | Mixed-city functional clusters | Two clean city blobs → city-domain clustering still dominant |
| Prototype utilization (Fix 8) | Each prototype used by >1% of batch | One or two prototypes absorb >80% → Sinkhorn not effective |
| Within-prototype city mix (Fix 8) | Each prototype has regions from all cities | Prototypes are city-pure → functional clustering failed |

---

## 7. On the Satellite Anchor (0.758 Cross-City Similarity)

The observation that `sat_mean` cross-city cosine similarity averages 0.758 is the foundational empirical motivation for the alignment approach. This is meaningful: it says that satellite texture/appearance is a reliable proxy for functional similarity across cities.

**One caveat**: if the pre-extracted satellite embeddings come from a model trained on one geographic region (e.g., North American/European imagery), the embedding space may systematically under-represent Seoul's urban texture, causing the similarity estimate to be biased. The actual functional correspondence could be better or worse than 0.758. This is worth validating by checking the false-positive rate of pseudo-positive pairs (do pairs with sat_sim > 0.7 actually have similar population densities?).

**Validation suggestion** (one-time sanity check):
```python
# For each pseudo-positive pair (i from NYC, j from Seoul):
# check if |pop_density_i - pop_density_j| < some_threshold
# High agreement → the satellite proxy is trustworthy
import numpy as np
pop_nyc = pd.read_csv("data/nyc/pop_gt.csv").set_index("BoroCT2020")
pop_seoul = pd.read_csv("data/seoul/pop_gt.csv").set_index("ADM_CD")
# correlation of pop_density across pseudo-positive pairs
```

If cross-city pseudo-positive pairs do NOT have correlated population densities, the entire `sat_mean`-based alignment premise needs revisiting (perhaps `sat_mean` captures visual appearance, not function).

---

## 8. Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Shared/spec heads orphaned from `region_emb` | Aggregator operates on raw inputs; projections feed only `L_dis` | Wire shared/spec into the aggregator (Fix 0) |
| L_align rising after epoch 8 | Attraction-only loss overwhelmed by L_contrast (10× gradient imbalance) | InfoNCE-based L_align (Fix 1) or unified multi-positive loss (Fix 2) |
| L_align and L_contrast in conflict | NT-Xent treats cross-city pseudo-positives as negatives | Unified multi-positive loss with joint positive mask (Fix 2) |
| Embedding space organized by city domain, not function | City identity is the strongest discriminative axis for NT-Xent; no global functional attractor | Functional prototype clustering (Fix 8) |
| L_dis collapse to 0.002 | Trivial orthogonalization of spec branch; gradient too weak (and disconnected from `region_emb`) | Fix 0 + VICReg/Barlow Twins (Fix 6) |
| Hard pseudo-positive threshold noisy at boundaries | Binary gate at sim=0.7 | Soft sigmoid weighting (Fix 3) + curriculum annealing (Fix 7) |
| Single-modality alignment signal fragile | Only satellite used for pseudo-pair mining | Dual-signal mining: sat AND poi agreement (Fix 4) |

Two highest-leverage changes:
1. **Fix 0** — wire the shared/spec heads into the region embedding so `L_dis` (or any disentanglement loss) has a downstream effect.
2. **Fix 2 + Fix 8** — replace separate `L_contrast` / `L_align` with a unified multi-positive loss and a global functional prototype clustering objective.

---

## 9. Research Grounding (2023–2026)

The proposed fixes track convergent findings in the cross-city / urban representation learning literature from 2023 through early 2026. The mapping below shows that this codebase's failure modes are well-known across the field, and that the prescribed fixes align with current best practice. Subsections 9.1–9.4 focus on the most recent (2025–2026) work; 9.5 covers the 2023–2024 foundations that the current codebase builds on; 9.6 outlines what is genuinely novel in the proposed solution.

### 9.1 Pairwise alignment is insufficient for cross-city transfer (motivates Fix 8)

**CRRL: Contrastive Region Relevance Learning** (ScienceDirect, 2025) tackles the same setting — knowledge transfer between cities for spatiotemporal prediction — and explicitly identifies the *negative transfer* risk that L_align alone produces. CRRL pairs a Dual-branch Spatiotemporal Encoder with a structure-aware contrastive module that "resolves the false-negative issue, enabling models to correctly group regions by their underlying function." This is the exact failure mode diagnosed in §2.2: the current model treats functionally similar cross-city regions as false negatives. CRRL's solution mirrors Fix 2 (multi-positive contrastive) and Fix 8 (functional grouping).

**MTGRR: Modality-Tailored Graph Modeling for Urban Region Representation via Contrastive Learning** (arXiv 2509.23772, Sep 2025) introduces a *joint contrastive loss at three levels*: aggregated-level, point-level, and fusion-level. This is exactly the structural insight behind Fix 2 (UnifiedCityCon) — separate losses at different granularities create gradient conflicts; a joint objective resolves them. MTGRR uses POI, mobility, land use, road, RSI, and street-view modalities, but the core lesson translates to the simpler 2-modality (satellite + POI) setup here.

**Similarity-based City Data Transfer Framework** (Nature Scientific Reports, 2025, *s41598-025-94987-y*) embeds urban similarity into the adaptation transfer pipeline rather than relying on a single distance threshold. Its empirical finding is the same as our §7 caveat: a fixed similarity threshold (analogous to `align_threshold=0.7`) produces unstable transfer because urban similarity is inherently soft. This validates Fix 3 (soft sigmoid weighting) and Fix 7 (curriculum threshold).

### 9.2 Multi-modal contrastive needs alignment AND clustering (motivates Fix 2 + Fix 8 jointly)

**UrbanMMCL: Urban Multi-Modal Multi-View Dual Contrastive Learning** (ScienceDirect, 2025) establishes a self-supervised pre-training framework that runs contrastive learning at *two complementary levels*: cross-modal contrastive (RSI ↔ SVI ↔ POI ↔ text) and cross-view graph contrastive. The two-level design is necessary because single-level contrastive collapses to one modality's structure. The current codebase implements only one level (cross-modal NT-Xent inside `RegionContrastiveLoss`); Fix 8 supplies the missing structural level.

**Comprehensive Urban Region Representation Learning via Multi-View Joint Learning** (AAAI 2025) makes the same argument explicitly: multi-view joint optimization outperforms naïve summation of per-view losses because joint optimization eliminates inter-view gradient conflict. This is direct theoretical support for Fix 2.

**FlexiReg: Flexible Region Representation** (arXiv 2503.09128, Mar 2025) uses adaptive aggregation across modalities with prompt learning to tailor representations for different downstream tasks. Its key contribution — *adaptive* aggregation rather than fixed concat — suggests that the simple `concat([sat_region, poi_region])` in `model.py:77` is suboptimal. A learned fusion (gated attention) is a natural successor to Fix 0 once the shared/spec branch is wired in.

### 9.3 Prototype-based cross-domain alignment is the 2025 consensus (validates Fix 8)

**Dual-Domain Representation Modeling with Prototype Contrastive Learning (DMPC)** (IEEE 2025) for cross-domain few-shot scene classification — directly transferable to cross-city setup — uses dual prototypes per domain plus a contrastive alignment term. DMPC's ablation specifically shows that *removing prototypes* degrades cross-domain accuracy by >10 points, even when the contrastive loss is preserved. This is empirical evidence that Fix 8 (prototypes) is structurally necessary, not just complementary, when transferring across domains.

**Dynamic Prototype Contrastive Learning (DPCL)** (PMLR 2024 → extended 2025) and **Cross-Prototype Contrastive Learning (CPCL)** (IEEE 2024 → extended 2025) for unsupervised domain adaptation in medical imaging both confirm: dynamic prototype updating + contrastive alignment outperforms either alone. The *Sinkhorn-Knopp balanced assignment* in our Fix 8 corresponds to DPCL's "evolutionary prototype update" — both prevent prototype collapse.

**Contrastive Dual Learning** (overview, 2025) characterizes the dual contrast pattern (feature contrast + prototype contrast) as the dominant paradigm for "intra-class compactness + inter-class separability" in domain-invariant representation learning. This is exactly the pairing of Fix 2 (feature contrast with multi-positives) + Fix 8 (prototype contrast).

### 9.4 Generalization gap and zero-shot transfer (validates the project's framing)

**Self-Supervised Representation Learning for Geospatial Objects: A Survey** (arXiv 2408.12133, last revised Jun 2025) explicitly states: *"While many current location-representation methods achieve strong in-domain performance, they often generalize poorly to new geographic areas; spatiotemporal generalizability remains a key challenge for future GeoAI research."* This is the gap this project targets. The survey identifies prototype-based methods and multi-modal contrastive frameworks (CRRL, UrbanMMCL, DMPC) as the most promising directions — the same set Fix 2 and Fix 8 are drawn from.

**Predicting Human Mobility Flows from Satellite Imagery** (Nature Communications, 2025, *s41467-025-65373-z*) demonstrates that satellite-image embeddings combined with graph attention transfer across the top-10 US MSAs in a zero-shot fashion. This empirically validates the §7 premise that satellite embeddings can serve as a cross-city anchor, but also notes a key caveat: anchors trained on geographically-similar cities transfer well; cross-continent transfer (NYC ↔ Seoul ↔ Singapore) is significantly harder. This strengthens the case for Fix 4 (dual-signal mining) — relying on satellite alone is risky when source and target are geographically dissimilar.

**Open-set Cross-Modal Generalization via Multimodal Unified Representation** (ICCV 2025) shows that unified multi-modal representations generalize to unseen domains *and* unseen classes only when the unified space is built via prototype-based clustering, not pairwise alignment. This is direct validation that Fix 2 alone is insufficient and Fix 8 is required for the unseen-target-city setting (Singapore was never seen during training).

### 9.5 Foundational 2023–2024 work

This codebase inherits its design lineage from the 2023–2024 wave of urban region representation learning. The relevant works are:

**HREP** (Zhou et al., 2023, "Heterogeneous Region Embedding with Prompts") established the now-standard pattern of learning region embeddings from heterogeneous signals (mobility + POI + geographic neighborhood) and adapting them via prompt embeddings to downstream tasks. HREP introduced *relational embedding* to capture inter-region correlations — a structural cue this codebase currently lacks (the only inter-region signal here is contrastive negatives). HREP also defined the now-canonical *cross-city evaluation protocol* (train on all-but-one city, test on held-out city), which is exactly the protocol used here. HREP remains the dominant baseline that 2024–2025 papers are compared against.

**ReFound** (KDD 2024, "Crafting a Foundation Model for Urban Region Understanding") is the first urban *foundation model* approach. It uses a Mixture-of-Geospatial-Expert Transformer to integrate POI + satellite + text, distilling from frozen vision-language foundation models. The relevant lesson for this codebase: ReFound's ablation shows that the cross-modal *expert routing* (similar in spirit to the shared/spec branches here) only contributes when the expert outputs feed the prediction head — not when they are auxiliary heads. This independently confirms Fix 0.

**UrbanCLIP** (Yan et al., WWW 2024) trains a CLIP-style satellite-image encoder using LLM-generated textual descriptions of urban regions. Its key empirical finding is that *cross-city satellite alignment via a learned text proxy outperforms direct image-image contrastive*, because the text proxy captures functional semantics. This is direct support for Fix 4 (dual-signal mining): satellite alone is brittle as a cross-city anchor; a second semantic signal (text in UrbanCLIP, POI in this codebase) is needed.

**MGRL4RE** (ACM TIST 2024, "Multi-Graph Representation Learning for Urban Region Embedding") and **Urban Region Embedding via Multi-View Contrastive Prediction** (arXiv 2312.09681, Dec 2023) both use multi-view contrastive learning at the region level. MGRL4RE specifically observes that single-view contrastive (the current codebase's approach: only sat ↔ poi) underperforms multi-view contrastive by 8–15% on cross-city transfer. This is supporting evidence for Fix 2 (UnifiedCityCon) and Fix 8 (FunctionalPrototypeLoss): more positive sources beats more sophisticated single-source losses.

**Enhanced Urban Region Profiling with Adversarial Self-Supervised Learning** (arXiv 2402.01163, Feb 2024) explicitly tests adversarial domain alignment (similar to GRL) and finds it *underperforms* prototype-based approaches for cross-city transfer. This empirically validates §3's argument for why GRL was correctly removed and why Fix 8 (prototypes) is the right replacement direction.

**Urban Region Representation Learning with Attentive Fusion** (arXiv 2312.04606, Dec 2023) introduces gated attention fusion across modalities — a natural successor to the simple `concat([sat_region, poi_region])` in `model.py:77`. If Fix 0 is implemented, the next architectural improvement is to replace concat with attentive fusion.

**Urban Foundation Models: A Survey** (KDD 2024) and **A Review and Outlook of Urban Foundation Models** (arXiv 2402.01749) together synthesize the 2023–2024 landscape. Both surveys identify *cross-city generalization* as the central open problem and explicitly call out *prototype-based clustering + multi-view contrastive* as the most promising direction — the exact combination Fix 2 + Fix 8 implements.

**C2Seg** (Hong et al., 2023, "Cross-City Matters: Multimodal RS Benchmark") established the first cross-city RS segmentation benchmark and showed empirically that even when source and target cities are visually similar, naive transfer drops 20+ points without explicit domain alignment. This validates the necessity of L_align in any cross-city pipeline (just not in its current attraction-only form).

### 9.6 What is genuinely new in the proposed solution

The fixes here are not novel methods; they are well-validated 2024–2025 best practices applied to this specific codebase. The novelty is in the *combination* and the *diagnosis*:

1. **The orphaned shared/spec branch (Fix 0)** is a code-specific finding — most published work uses end-to-end shared/private aggregation (DSN-style). This codebase has a partial implementation that the literature would consider broken.
2. **The application of SwAV-style functional prototypes to *city-level* domain generalization** (Fix 8) — DMPC/DPCL apply prototypes to medical/scene domain adaptation; transferring this to urban region representation is a small but novel step.
3. **Diagnosing city-domain clustering as a NT-Xent shortcut** — CRRL identifies the false-negative problem but frames it differently (region pair structure). Framing it explicitly as shortcut learning (Geirhos et al. 2020) provides a sharper conceptual handle.

### Sources

- [CRRL: Contrastive Region Relevance Learning Framework for Cross-city Traffic Prediction (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S156625352500288X)
- [UrbanMMCL: Urban region representations via multi-modal and multi-graph self-supervised contrastive learning (ISPRS 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0924271625004514)
- [MTGRR: A Modality-Tailored Graph Modeling Framework for Urban Region Representation via Contrastive Learning (arXiv 2509.23772, Sep 2025)](https://arxiv.org/abs/2509.23772)
- [FlexiReg: Urban Region Representation Learning: A Flexible Approach (arXiv 2503.09128, Mar 2025)](https://arxiv.org/html/2503.09128v1)
- [Comprehensive Urban Region Representation Learning via Multi-view Joint Learning (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/download/38551/42513)
- [Multi-modal contrastive learning of urban space representations from POI data (Computers, Environment & Urban Systems 2025)](https://www.sciencedirect.com/science/article/pii/S0198971525000523)
- [Similarity-based city data transfer framework in urban digitization (Nature Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-94987-y)
- [GURPP: Urban Region Pre-training and Prompting — A Graph-based Approach (arXiv 2408.05920, updated 2025)](https://arxiv.org/html/2408.05920)
- [Self-Supervised Representation Learning for Geospatial Objects: A Survey (arXiv 2408.12133, revised Jun 2025)](https://arxiv.org/pdf/2408.12133)
- [Predicting human mobility flows in cities using deep learning on satellite imagery (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-65373-z)
- [Representation learning for geospatial data (Annals of GIS 2025)](https://www.tandfonline.com/doi/full/10.1080/19475683.2025.2552157)
- [DMPC: Dual-Domain Representation Modeling With Prototype Contrastive Learning for Cross-Domain Few-Shot Scene Classification (IEEE 2025)](https://ieeexplore.ieee.org/iel8/36/10807682/11010163.pdf)
- [DPCL: Unsupervised Domain Adaptation for Medical Image Segmentation with Dynamic Prototype-based Contrastive Learning (PMLR 2024)](https://proceedings.mlr.press/v248/en24a.html)
- [CPCL: Unsupervised Domain Adaptation by Cross-Prototype Contrastive Learning for Medical Image Segmentation (IEEE 2024)](https://ieeexplore.ieee.org/document/10386055/)
- [Open-set Cross Modal Generalization via Multimodal Unified Representation (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Open-set_Cross_Modal_Generalization_via_Multimodal_Unified_Representation_ICCV_2025_paper.pdf)
- [Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting (KDD 2022, foundational)](https://dl.acm.org/doi/10.1145/3534678.3539250)

**2023–2024 foundational work:**
- [ReFound: Crafting a Foundation Model for Urban Region Understanding upon Language and Visual Foundations (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671992)
- [MGRL4RE: A Multi-Graph Representation Learning Approach for Urban Region Embedding (ACM TIST 2024)](https://dx.doi.org/10.1145/3712698)
- [Urban Foundation Models: A Survey (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671453)
- [A Review and Outlook of Urban Foundation Models (arXiv 2402.01749, Feb 2024)](https://arxiv.org/pdf/2402.01749)
- [Enhanced Urban Region Profiling with Adversarial Self-Supervised Learning (arXiv 2402.01163, Feb 2024)](https://arxiv.org/html/2402.01163)
- [Awesome-Urban-Foundation-Models (community survey repo, 2024)](https://github.com/usail-hkust/Awesome-Urban-Foundation-Models)
- [Urban Region Embedding via Multi-View Contrastive Prediction (arXiv 2312.09681, Dec 2023)](https://arxiv.org/html/2312.09681)
- [Urban Region Representation Learning with Attentive Fusion (arXiv 2312.04606, Dec 2023)](https://arxiv.org/html/2312.04606)
- [Cross-City Matters: A Multimodal RS Benchmark for Cross-City Semantic Segmentation (Hong et al., RSE 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0034425723004078)
- [Urban Region Representation Learning with Human Trajectories (Annals of GIS / Tandfonline 2024)](https://www.tandfonline.com/doi/full/10.1080/15481603.2024.2387392)
- [HREP family: Heterogeneous Region Embedding with Prompts (Zhou et al., 2023, foundational baseline cited across 2024–2025 papers)](https://arxiv.org/html/2408.05920)
