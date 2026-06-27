# Diagnosed Problems — Cross-City Embedding Space

This document tracks confirmed problems with empirical evidence.

---

## Problem 1: City-Domain Clustering (Partially Fixed, Not Resolved)

**Status**: ⚠️ Partially addressed — UnifiedCityCon reversed global centroid direction but CLIPRegionCon regressed it.

**Evidence — centroid cosine similarity across runs:**

| Run | NYC↔Seoul | NYC↔SG |
|-----|-----------|--------|
| Baseline (L_contrast+L_align) | -0.397 | -0.262 |
| Fix 2 UnifiedCityCon | **+0.819** | **+0.400** |
| Fix 2+4+8 corrected | +0.816 | +0.525 |
| **CLIP-like (current)** | **−0.449** | — |

CLIPRegionCon removed within-modality negative pressure to protect the spec branch, but this simultaneously removed the cross-city repulsion brake. Cities drifted back to opposite hemispheres.

**Root cause**: City-domain clustering is not caused by the loss design alone — it originates from the raw feature space (see Problem 6). Any loss operating on top of city-biased features will need to fight uphill against the inherent city separation.

---

## Problem 2: Over-Alignment / Embedding Collapse After UnifiedCityCon

**Status**: ✅ Resolved for within-city sim by CLIPRegionCon, but city-domain clustering returned (trade-off, not a net fix).

**Evidence (Fix 2 UnifiedCityCon):**

| Metric | Before Fix 2 | After Fix 2 |
|--------|-------------|-------------|
| Within NYC sim | 0.568 | **0.809** ⚠️ |
| Within Seoul sim | 0.352 | **0.710** ⚠️ |
| Seoul norm std | 0.124 | **0.028** ⚠️ |
| NYC↔Seoul centroid cos | -0.397 | +0.819 |

**After CLIPRegionCon (current):**

| Metric | After Fix 2 | After CLIP-like |
|--------|-------------|----------------|
| Within NYC sim | 0.809 | **0.721** ↓ |
| Within Seoul sim | 0.710 | **0.147** ↓↓ |
| NYC↔Seoul centroid cos | +0.819 | **−0.449** ↓↓ |

CLIPRegionCon reduced within-city similarity but at the cost of city-domain clustering. Neither approach achieves both simultaneously.

**Root cause**: Too many positives per anchor (UnifiedCityCon) vs. too few cross-city signals (CLIPRegionCon). The fundamental tension cannot be resolved by changing the loss matrix shape alone.

---

## Problem 3: No Cross-City Functional Cluster Structure

**Status**: ⚠️ Not achieved by any run so far.

**Evidence — visual:**
t-SNE in all runs shows city-segregated sub-clusters. No run has produced mixed-city functional clusters (e.g., [NYC residential + Seoul residential] forming one cluster).

**Evidence — land use composition analysis (analyze_3.txt):**
Within-city clustering by raw satellite or POI features does separate meaningful land use types (e.g., Seoul low-density = Green & Open Space dominant; Seoul high-density = Residential + Res/Comm Mixed). But these within-city clusters are 100% city-pure (AvgCityPurity=0.999 at K=16).

**Root cause**: The raw feature space has no cross-city functional structure. Cross-city overlap is confined to high-density regions (see Problem 6). FunctionalPrototypeLoss cannot create cross-city functional clusters if the input features themselves are city-separated.

---

## Problem 4: Dual-Signal Mining is Non-Functional

**Status**: 🔴 Confirmed broken — poi_threshold=0.5 is never satisfied in raw/early-training space.

**Evidence (analyze_1.txt):**

| Pseudo-positive criterion | NYC↔Seoul pairs above threshold |
|--------------------------|--------------------------------|
| sat_mean > 0.9 | 99,964 / 984,912 (10.1%) |
| poi_region > 0.5 (raw POI) | **0 / 984,912** |
| Both conditions (current config) | ~0 early training → spurious after collapse |

Raw POI cross-city similarity distribution: mean=0.092, std=0.028, max=0.271. The threshold `poi_threshold=0.5` is above the maximum possible raw POI similarity.

**Consequence**: CLIPRegionCon effectively runs as satellite-only mining (`poi_threshold` ignored) until embedding collapse inflates POI similarity spuriously. After collapse, the 76% of pairs exceeding 0.5 are collapse artifacts, not genuine functional correspondences.

**Immediate fix**: Set `poi_threshold: 0.0` to use satellite-only pseudo-positive mining. This is consistent with the analyze_1.txt conclusion and removes the spurious poi_threshold gate.

---

## Problem 5: Singapore Distribution Mismatch (Probe Calibration)

**Status**: ✅ Partially addressed — log1p applied to probe target.

**Evidence:**

| City | pop density mean | std | log1p std |
|------|-----------------|-----|-----------|
| NYC | 16,712 | 10,516 | 0.879 |
| Seoul | 22,776 | 11,149 | 0.740 |
| Singapore | 11,839 | 10,611 | **3.229** |

Singapore log-space std is 3.7× larger. Land use composition analysis (analyze_3.txt) explains why: Singapore has Public/Transport/Inst at 32.7% of regions (MRT, airports, roads, civic buildings with near-zero residential population) mixed with dense residential areas. This creates a bimodal distribution alien to NYC and Seoul.

**Remaining gap**: log1p reduces but cannot eliminate the structural mismatch. R² ceiling for any embedding on Singapore pop prediction may be fundamentally low.

---

## Problem 6: Pre-Trained Feature City-Specific Bias (Root Cause)

**Status**: 🔴 Not addressed — fundamental blocker.

**Evidence (analyze_1.txt, analyze_2.txt):**

| Feature | K-means K=3 ARI | K=16 AvgCityPurity | Meaning |
|---------|-----------------|-------------------|---------|
| Raw satellite | 0.866 | 0.999 | 3 clusters ≈ 3 cities |
| Raw POI | 0.943 | 0.974 | Even stronger city separation |

PCA analysis: POI PC1 alone explains 65.3% of variance. This first principal component is almost certainly the city-identity axis (US vs Korean vs Singaporean POI taxonomy). The satellite PC1 explains 40.5% — also likely contains a city-style axis.

**Root cause mechanism**: The satellite embeddings and POI morphological embeddings were pre-trained separately, without cross-city alignment. The resulting feature spaces encode city-level visual/functional style rather than city-agnostic functional categories. Any downstream contrastive loss that operates on these features must fight against 87–94% accurate city-label information already present in the input.

**Implication for FunctionalPrototypeLoss**: K-means K=16 on raw features produces city-pure sub-clusters (AvgCityPurity=0.999). Applying Sinkhorn assignment to these features means prototypes learn "city sub-regions" (e.g., "NYC high-density", "Seoul residential"), not cross-city functional archetypes. This is confirmed by the lack of functional cluster structure in any run.

**Fix options (ranked by effort):**
1. City-mean subtraction on input features (low effort, validated by analyze_2.txt: POI PC1=65.3% is the city axis)
2. Gradient reversal layer (domain adversarial training on input features)
3. Replace pre-trained features with a city-agnostic encoder (e.g., SatCLIP, UrbanCLIP)

---

## Problem 7: POI Modality Domain Gap for Cross-City Mining

**Status**: 🔴 Not addressed — affects all cross-city alignment signals.

**Evidence (analyze_1.txt):**

Raw POI cross-city NYC↔Seoul cosine similarity:
- mean = 0.092, std = 0.028, **max = 0.271**
- Pairs exceeding 0.5: **0 / 984,912**

After embedding collapse, learned POI similarity:
- mean = 0.626, std = 0.227, max = 0.988
- Pairs exceeding 0.5: 744,272 / 984,912 (**76%**)

The jump from 0% to 76% is not genuine functional alignment — it is embedding collapse. The model cannot distinguish "Seoul is functionally similar to NYC" from "Seoul collapsed toward the mean."

**Root cause**: US and Korean POI schemas are completely different taxonomies. NYC uses PLUTO land use codes; Seoul uses Korean administrative land use codes; Singapore uses URA Master Plan categories. The `morph_emb` field encodes morphological patterns within each schema, but the schemas themselves are non-overlapping.

**Implication**: POI modality cannot provide meaningful cross-city alignment signals without cross-city POI schema alignment. This is a data-level problem, not a model-level problem.

---

## Problem 8: BG POI Aggregation Uses No Validity Mask

**Status**: 🟡 Minor — no mask applied to POI aggregation.

In `models/model.py:79`, the POI aggregator uses `valid_mask` (passed from `forward_region`). Looking at the code, `poi_recon` goes through `self.poi_agg(poi_recon, valid_mask)` — so the same validity mask (based on satellite coverage) is applied to both modalities. This means POI BGs without a corresponding satellite embedding are excluded, which may drop valid POI data unnecessarily.

---

## Summary Table

| # | Problem | Evidence | Status | Recommended Fix |
|---|---------|----------|--------|-----------------|
| 1 | City-domain clustering | Centroid cos; t-SNE | ⚠️ Partially fixed then regressed | Fix raw features (Problem 6) |
| 2 | Over-alignment after UnifiedCityCon | Within-city sim=0.81; Seoul norm std=0.028 | ✅ Reduced by CLIPRegionCon (but trade-off) | VICReg variance term |
| 3 | No cross-city functional cluster structure | t-SNE; K-means purity=0.999 | ⚠️ Not achieved | Fix raw features first, then L_proto |
| 4 | poi_threshold=0.5 never satisfied | Max raw POI sim=0.271 | 🔴 Mining broken | Set poi_threshold=0.0 |
| 5 | Singapore pop distribution mismatch | log1p std = 3.23 vs 0.88 | ✅ Partially (log1p) | Use land use eval too |
| 6 | Pre-trained feature city-specific bias | K=3 ARI=0.87–0.94; K=16 purity=0.999 | 🔴 Root cause, not addressed | City-mean subtraction; DANN; new encoder |
| 7 | POI domain gap (different taxonomies) | Max cross-city sim=0.271; 0% pairs >0.5 | 🔴 Not addressed | Sat-only mining; LLM schema mapping |
| 8 | BGDisLoss gradient <1% of total | λ_dis=0.1 × ~0.05 = 0.005 | 🟡 Cosmetic | VICReg or raise λ |
