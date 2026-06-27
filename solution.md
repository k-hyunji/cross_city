# Solution Analysis: What's Missing, What's Wrong, and How to Fix It

Based on empirical analysis (analyze_1–3.txt), experimental runs (solution_old.md), and recent literature.

---

## TL;DR — Three-Line Summary

1. **The root cause is not the loss design — it's the input features.** Pre-trained satellite/POI embeddings encode city-specific style (ARI=0.87–0.94 at K=3). Any contrastive loss operating on city-biased features will learn city clusters first.
2. **Two of the three loss terms are broken**: `poi_threshold=0.5` is never satisfied in raw space (max POI sim=0.271); `BGDisLoss` contributes <1% of gradients.
3. **The fix sequence is inverted.** Current effort goes into loss design (L_clip, L_proto) while the data-level problem (city bias in features) is untouched. The recommended order: **feature alignment first, then loss design.**

---

## 1. Root Cause: Pre-Trained Feature City Bias

### 1.1 What You've Missed

The current diagnosis focuses on loss design (city-domain clustering caused by NT-Xent, over-alignment caused by UnifiedCityCon, etc.). But the actual root cause is upstream: **the input features are already city-separated before any training begins.**

From K-means analysis on raw features (analyze_1.txt):

| Feature | K=3 ARI | K=16 AvgCityPurity | Interpretation |
|---------|---------|-------------------|----------------|
| Satellite | 0.866 | 0.999 | 3 clusters = 3 cities with 87% accuracy |
| POI | 0.943 | 0.974 | Even stronger city separation |

POI PC1 alone explains **65.3% of variance** (analyze_2.txt). This axis is almost certainly city-identity (US vs Korean POI taxonomy). The satellite space has 40.5% in PC1.

**Critical implication**: Any learning algorithm that processes these features without first removing the city-identity axis will use that axis as the primary discriminative signal. The 100 epochs of contrastive training are essentially fighting against the ARI=0.943 city-separation already present in the data.

**This is not a loss problem — it is a data preprocessing problem.**

### 1.2 Literature Support

Domain adaptation literature has addressed this class of problem since **DANN** (Ganin & Lempitsky, JMLR 2016) and **CORAL** (Sun & Saenko, ECCV 2016). The key insight in both is that feature alignment must happen at the representation level, not just at the prediction level.

More recently, **AdaBN** (Li et al., ECCV 2018) showed that simply normalizing feature statistics per domain (batch norm with domain-specific statistics) can dramatically reduce domain shift without adversarial training. In the urban context, **UrbanVLP** (Yan et al., KDD 2024) explicitly addresses the city-style bias in satellite imagery by using language descriptions as city-agnostic anchors.

For urban region representation specifically, **RegionDCL** (Liu et al., AAAI 2023) — which this project's architecture is based on — operates within a single city. Cross-city extension is not the design intent of RegionDCL, and the city-specific feature bias is exactly the gap.

### 1.3 Recommended Fixes (Ranked by Effort)

**Fix A: City-mean subtraction [Immediate, Zero-cost]**

```python
# Before feeding to DualModalNet:
city_sat_mean = sat_arr.mean(dim=0, keepdim=True)  # per city
sat_normalized = sat_arr - city_sat_mean

city_poi_mean = poi_arr.mean(dim=0, keepdim=True)
poi_normalized = poi_arr - city_poi_mean
```

Rationale from analyze_2.txt: POI PC1 (65.3% of variance) is the city-identity axis. Subtracting city means removes this axis, exposing the within-city functional structure (35% of POI variance) for the model to learn.

Validation: Run K-means K=16 on city-mean-subtracted features. If AvgCityPurity drops from 0.999 to <0.7, the fix is working.

**Precedent**: City-mean subtraction is standard in cross-city mobility prediction (e.g., Liu et al., "Cross-city transfer learning", KDD 2019; Yao et al., "Representing urban functions", WWW 2018) to remove city-level effects before modeling.

**Fix B: Gradient Reversal Layer on Input Features [Medium Effort]**

Add a city discriminator after the first projection layer, with gradient reversal (Ganin et al., JMLR 2016). The encoder learns to produce city-agnostic projections.

```
Input [64] → Projection [32] → GRL → City Classifier
                            → (continue to aggregator)
```

This was tried and removed (see train.py:108-110 residue). The concern is that a poorly tuned GRL can destroy functional information along with city information. The fix: apply GRL only to the `shared_proj` output, not the `spec_proj`. Domain adversarial pressure should be applied only to the branch intended to be shared.

**Reference**: **Domain Separation Networks** (Bousmalis et al., NeurIPS 2016) uses this exact architecture: a shared encoder subject to domain adversarial loss, plus city-specific encoders that are not adversarially trained.

**Fix C: Replace Pre-Trained Features with City-Agnostic Encoders [High Effort, Highest Impact]**

The fundamental issue is that the 64-d satellite embeddings and POI morphological embeddings were generated by models trained without cross-city alignment. Replacing these with:

- **SatCLIP** (Klemmer et al., 2023): CLIP-trained on satellite imagery aligned with geographic coordinates. Produces geographically-grounded, visually-comparable features across cities.
- **RemoteCLIP** (Liu et al., 2024): Remote sensing CLIP, trained on RS images with text descriptions. Better semantic alignment than pure visual features.
- **Multilingual text embeddings for POI** (e.g., LaBSE, mUSE): Map POI category names through a multilingual embedding model. Korean "주거지역" and American "Residential" map to similar vectors in semantic space.

**Reference**: **UrbanCLIP** (Yan et al., 2024) demonstrates that using CLIP's image encoder for satellite imagery provides significantly more cross-city transferable features than domain-specific pretrained encoders, because CLIP was trained to align visual and linguistic concepts at a semantic level.

---

## 2. Loss Design: What's Working, What's Not

### 2.1 CLIPRegionCon — Partially Valid, Incomplete

**What's valid**: The B×B cross-modal design correctly avoids within-modality negative pressure on the spec branch. This is the right architectural choice (cf. CLIP, Radford et al., 2021).

**What's broken**: The POI pseudo-positive condition (`poi_threshold=0.5`) is never satisfied in raw space. The loss effectively runs as satellite-only mining, which is weaker than the paper's design intent.

**What it introduces**: Without cross-city repulsion in the denominator, city-domain clustering returns (centroid cos: +0.816 → −0.449). CLIPRegionCon solves within-city collapse at the cost of city-domain clustering — a lateral trade-off, not a net improvement.

**Fix**: Set `poi_threshold: 0.0` (satellite-only mining, honest about data limitations). Then add a separate within-city uniformity term to prevent collapse without needing cross-city repulsion.

### 2.2 FunctionalPrototypeLoss — Correct Mechanism, Wrong Input

**What's valid**: SwAV-style prototype learning (Caron et al., NeurIPS 2020) is the right approach for city-agnostic functional structure. The Sinkhorn balanced assignment correctly prevents prototype collapse. The implementation is correct after the Sinkhorn bug fix.

**What's broken**: Prototypes cannot be city-agnostic if the input features are city-specific. K=16 K-means on raw features produces city-pure sub-clusters (purity=0.999). FunctionalPrototypeLoss with city-biased inputs will learn "NYC commercial", "Seoul residential", etc. — not cross-city functional archetypes.

**Partial win**: Despite the feature bias, L_proto improves Singapore geometry (NYC↔SG: +0.400→+0.525, Seoul↔SG: +0.346→+0.409). This is meaningful — Singapore is being pulled closer to the source city embedding space through prototype pressure, even if the prototypes themselves are city-biased.

**Fix**: City-mean subtraction first (Fix A above), then L_proto should converge to genuine functional archetypes. Additionally, K may need to be increased (K=32 or K=64) once features are city-agnostic, as the functional variety across cities is greater than within one city.

### 2.3 BGDisLoss — Structural Dead Weight

**What's broken**: Two compounding issues:
1. λ_dis = 0.1 gives gradient contribution ~0.5% of total.
2. The aggregator receives `cat([shared, spec], dim=-1)` = 64-d, identical to the original input. There is no structural mechanism that forces the aggregator to treat shared and spec dimensions differently. L_dis regularizes auxiliary projections that have no downstream effect.

**What the literature says**: **VICReg** (Bardes et al., ICLR 2022) directly addresses this — instead of enforcing orthogonality between branches, it penalizes variance collapse and feature correlation at the output level. The variance term:

```
L_var = (1/d) Σ_j max(0, γ - std(z_j))   # γ=1.0, z = region_emb
```

penalizes the embedding space from collapsing (all embeddings becoming similar), which is exactly what went wrong after UnifiedCityCon (within-city sim=0.81).

The covariance term:
```
L_cov = (1/(d-1)) Σ_{i≠j} [C_z]²_{ij}    # C_z = (1/B) Z^T Z
```

penalizes redundant dimensions, acting as a softer version of BGDisLoss without requiring the shared/spec architecture.

**Recommended replacement**: Add VICReg's variance term to prevent within-city collapse:

```python
class VICRegVarianceLoss(nn.Module):
    def forward(self, outputs, gamma=1.0):
        embs = torch.stack([F.normalize(o["region_emb"]) for o in outputs])
        # per-city variance penalty
        cities = {}
        for i, o in enumerate(outputs):
            cities.setdefault(o["city"], []).append(i)
        loss = 0.0
        for idx in cities.values():
            z = embs[idx]
            std = z.std(dim=0)
            loss += F.relu(gamma - std).mean()
        return loss / len(cities)
```

This directly addresses the within-city collapse without requiring the disentanglement architecture.

**Reference**: **Barlow Twins** (Zbontar et al., ICML 2021) achieves identical effects through cross-correlation matrix decorrelation. Both VICReg and Barlow Twins have been shown to be more training-stable than contrastive losses for asymmetric data distributions.

---

## 3. Architecture: What's Missing

### 3.1 POI Centroid is Loaded but Never Used

`data/dataset.py:195` loads `centroid: [N_bg, 2]` (lat/lon per BG) but the model ignores it. This is a free spatial signal.

**What centroid encodes**: within-region BG distribution pattern (dense downtown vs sparse suburb), distance to city center proxy, and intra-region spatial variance. These are cross-city transferable signals — a BG 5km from the city center in NYC and one 5km from Gangnam-gu in Seoul have comparable spatial context regardless of POI taxonomy.

**Recommended use**: Normalize BG centroids to within-region relative coordinates (offset from region centroid, scaled by region radius). This gives cross-city comparable spatial context without absolute coordinates.

```python
# In dataset.py __getitem__:
centroid_rel = centroid[bg_indices] - centroid[bg_indices].mean(0)  # [M, 2]
centroid_std = centroid_rel.std(0).clamp(min=1e-6)
centroid_norm = centroid_rel / centroid_std  # [M, 2]
# Concatenate with poi_emb: [M, 64+2]
```

**Literature**: **SpaBERT** (Lee et al., EMNLP 2022) shows that relative spatial encoding (offset from reference point) is cross-dataset transferable, while absolute coordinates are not. **GeoAware-SC** (2024) uses normalized spatial context as an auxiliary feature for satellite image understanding.

### 3.2 No Spatial Graph Between BGs

The AttentionAggregator treats BGs as an unordered set (permutation-invariant). Urban regions have inherent spatial structure — nearby BGs interact, form neighborhoods, and exhibit spatial autocorrelation.

**What's missing**: A spatial graph over BGs within a region, where edges are weighted by proximity. GNN message passing would allow BGs to pool information from their spatial neighbors before the final region-level pooling.

**Why this matters for cross-city transfer**: Spatial adjacency patterns (ring structures, grid patterns, radial patterns) are more cross-city comparable than feature distributions. A GNN that learns "adjacent BGs with high POI density form a commercial cluster" generalizes across cities, whereas a flat attention that learns "BGs at positions 3, 7, 12 tend to be commercial" does not.

**Reference**: **HGCL** (Huang et al., WWW 2022) uses hierarchical graph contrastive learning for region representation. **MGEO** (Yan et al., AAAI 2023) builds a multi-granularity graph (BG → region → city) with cross-granularity message passing.

**Effort**: Medium. Adding k-NN spatial graph within each region (k=4–8 nearest BGs by Euclidean distance) and one GNN layer before pooling is a modest change.

### 3.3 No Multi-Task Objective

The model is trained with purely self-supervised objectives. Singapore has `data/singapore/landuse_gt_list.csv` — land use labels that are unused during any phase of training or evaluation.

**Why this matters**: Land use labels provide explicit semantic anchors that are cross-city comparable. Residential areas in NYC, Seoul, and Singapore all have the label "Residential." If even a small fraction of Singapore regions have known land use, they could be used to:
1. Guide prototype initialization (assign each prototype to a land use category using Singapore labels)
2. Evaluate embedding quality on a richer task (land use classification R² >> pop density R² for this data)
3. Provide few-shot supervision during linear probe (use 10% of Singapore regions as calibration set)

**Strict zero-shot constraint**: If the project must not use any Singapore labels during training, use land use labels only as an evaluation metric. This gives a richer picture than pop density R² alone.

**Reference**: **CTLE** (Xu et al., KDD 2022) shows that multi-task training (crime prediction + land use classification + population prediction jointly) significantly improves cross-city transfer compared to single-task training. Joint supervision provides richer, less overfitted features.

---

## 4. Evaluation Protocol: What's Misleading

### 4.1 R² May Be Fundamentally Bounded for Singapore Pop Prediction

Singapore's land use distribution: Public/Transport/Inst = 32.7% (roads, MRT stations, airports, hospitals). These regions have near-zero residential population density. NYC and Seoul have comparable proportions of low-density public areas but without Singapore's extreme bimodal gap between "0 people in a transit corridor" and "50,000+ in a HDB block."

**Implication**: Even with perfect cross-city embeddings, a Ridge regressor trained on NYC + Seoul pop densities will fail on Singapore because the density-generating process (HDB high-rise blocks interspersed with large-footprint public infrastructure) is structurally different from US census tracts or Korean dong units.

**Action**: Add land use classification as a secondary evaluation task. A model that correctly clusters Singapore residential areas with NYC/Seoul residential areas has learned something useful even if pop density R² is near zero.

### 4.2 Ridge Alpha Selection by Source Train R² is Wrong

Current code selects alpha by train-set R² on source cities:
```python
r2 = r2_score(y_log_train, m.predict(X_train))  # no held-out validation
```

This selects the alpha that best fits the source training distribution — which may not be the alpha that best generalizes to Singapore. For cross-city transfer, alpha should be selected by held-out source validation.

**Simple fix**: Use `sklearn.model_selection.cross_val_score` with 5-fold CV on source data to select alpha.

### 4.3 Model Selection Criterion is Misaligned with Objective

The best checkpoint is selected by `val_total = L_clip + L_dis + L_proto` on source cities. This penalizes a model that generalizes to Singapore if it sacrifices source-city loss in the process.

**What's misaligned**: A model with low source val_total but poor cross-city geometry (centroid cos negative) could be selected over a model with slightly higher val_total but better cross-city structure.

**Recommended addition**: Log cross-city embedding statistics (centroid cosine, within-city sim) after each epoch. Use a composite score for model selection:

```
score = val_total + α × within_city_sim + β × max(0, -centroid_cos)
```

where α and β penalize collapse and city-domain clustering respectively.

---

## 5. Overlooked Opportunities

### 5.1 Use City-Mean Subtracted Features as Baseline (Zero-Code-Change)

Before any model training, compute:
```python
sat_city_mean = {city: sat_arr.mean(0) for city, sat_arr in ...}
poi_city_mean = {city: poi_arr.mean(0) for city, poi_arr in ...}
```

Then at dataset load time, subtract the city mean. This single-line change addresses the root cause (Problem 6) with zero architectural changes. Expected result: K-means K=16 on city-mean-subtracted features should show AvgCityPurity < 0.8, allowing FunctionalPrototypeLoss to actually learn cross-city functional clusters.

This should be **the first experiment before any further loss design iteration.**

### 5.2 Validation on Land Use Classification (Singapore has labels)

`data/singapore/landuse_gt_list.csv` contains land use labels. Instead of (or in addition to) pop density R², evaluate:
```python
# After extract_embeddings.py:
from sklearn.linear_model import LogisticRegression
# Train on Singapore land use with 80/20 split
# Report land use classification accuracy
```

This provides a much more interpretable signal: if Singapore Residential regions cluster with NYC/Seoul Residential regions, land use accuracy will be high even if pop density R² is low.

### 5.3 More Source Cities (Only 2 is Very Few)

The model is trained on 2 source cities (NYC + Seoul): n=2,312 + n=426 = 2,738 regions. This is a small training set for a 128-d embedding with multiple loss terms.

**What's likely happening**: the model is overfit to the specific visual/POI style of NYC and Seoul. Adding a third source city (e.g., Chicago, London, or Tokyo) would:
- Increase training diversity
- Force the shared representation to be less NYC+Seoul-specific
- Improve the frozen `sat_mean` pseudo-positive mining (more diverse reference prototypes)

**Reference**: **UrbanVLP** (2024) trains on 5+ cities and shows clear cross-city generalization improvement with each additional source city. **MetaGRL** (Pan et al., KDD 2022) explicitly uses meta-learning across multiple cities to learn city-agnostic initialization.

---

## 6. Is the Current Approach Valid?

### What's Correct

- **CLIPRegionCon (B×B cross-modal)**: Architecturally sound. CLIP-style cross-modal contrastive learning (Radford et al., 2021) is the right approach for multi-modal region representation. The issue is not the design but the input features.
- **Sinkhorn-Knopp balanced assignment**: Correctly prevents prototype collapse. SwAV (Caron et al., NeurIPS 2020) validates this for visual representation; the urban extension is conceptually justified.
- **Frozen satellite mean as cross-city anchor**: Reasonable proxy for visual functional similarity. Low correlation with pop density (r=0.016) does not invalidate it — pop density is a narrow proxy.
- **log1p probe target normalization**: Correct. Reduces outlier impact and makes source/target distributions more comparable in regression.
- **Balanced sampler**: Necessary for equal city representation. Without it, Seoul collapses (norm std=0.019).

### What's Incorrect

- **poi_threshold=0.5 with raw POI features**: Structurally broken. Max achievable in raw space = 0.271. This has been known since analyze_1.txt (May 2026) but not yet fixed in the config.
- **BGDisLoss as primary disentanglement mechanism**: The shared/spec architecture does not propagate to region_emb in a way the aggregator can exploit. The loss regularizes unused projections.
- **Loss iteration without feature alignment**: All Fix variants (1–8) operate at the loss level while the data-level root cause is untouched. The pattern of improvement (+R²) followed by new problem (collapse, clustering) suggests the approach is in a local optimum rather than progressing toward a genuine solution.

### Overall Assessment

The project correctly identifies the symptoms (city-domain clustering, over-alignment, functional cluster absence) and the loss-level causes. The experimental iterations are thorough and well-documented. However, the root cause (pre-trained feature city bias) has been identified in analyze_1.txt but not yet acted upon.

The current R² = -0.07 (best run) is negative R², meaning the model is still worse than predicting Singapore's mean population density. This suggests the problem is more fundamental than loss tuning.

---

## 7. Recommended Experiment Sequence

| Priority | Experiment | Expected Outcome | Effort |
|----------|-----------|-----------------|--------|
| 1 | Set `poi_threshold: 0.0` | Remove spurious gate; mining becomes honest sat-only | 1 line |
| 2 | City-mean subtraction on input features | K-means purity <0.8; L_proto learns functional clusters | 10 lines in dataset.py |
| 3 | Add VICReg variance term (λ_var=0.1) | Within-city sim stays 0.4–0.6 without collapse | 20 lines in losses.py |
| 4 | Add land use classification eval | Interpretable quality signal independent of pop density | 30 lines in linear_probe.py |
| 5 | Add normalized POI centroid to BG features | Free spatial signal; within-region structure | 5 lines in dataset.py |
| 6 | UnifiedCityCon + VICReg variance (revisit Fix 2 with collapse prevention) | Cross-city alignment without collapse | Retrain |
| 7 | GRL on sat_shared_proj (domain-adversarial on shared branch only) | City-agnostic shared representation | 1 new loss class |
| 8 | SatCLIP / RemoteCLIP satellite features | City-agnostic visual features from foundation model | Data pipeline |

---

## 8. Key Metrics to Track (Revised)

| Metric | Current Best | Target (realistic) | What it measures |
|--------|-------------|-------------------|-----------------|
| R² Singapore (log) | -0.07 | > 0.0 | Zero-shot transfer quality |
| Land use acc. Singapore | — | > 50% | Functional cluster quality |
| K=16 AvgCityPurity | ~0.999 (raw) | < 0.7 | City-agnostic feature quality |
| NYC↔Seoul centroid cos | -0.449 (CLIP) / +0.816 (Fix2) | > +0.5 stable | Cross-city alignment |
| Within-city sim | 0.14–0.72 | 0.3–0.5 | Discriminative structure |
| Prototype city mixing | ~0 (city-pure) | > 0.3 (each prototype has 30%+ second city) | Functional prototype quality |

---

## References

- Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments" (SwAV), NeurIPS 2020
- Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (CLIP), ICML 2021
- Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022
- Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", ICML 2021
- Khosla et al., "Supervised Contrastive Learning" (SupCon), NeurIPS 2020
- Ganin & Lempitsky, "Domain-Adversarial Training of Neural Networks" (DANN), JMLR 2016
- Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
- Bousmalis et al., "Domain Separation Networks" (DSN), NeurIPS 2016
- Li et al., "Adaptive Batch Normalization for practical domain adaptation" (AdaBN), ECCV 2018
- Liu et al., "Region Representation Learning via Large-Scale Point Clouds" (RegionDCL), AAAI 2023
- Yan et al., "UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining", WWW 2024
- Klemmer et al., "SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery", arXiv 2023
- Huang et al., "Hierarchical Graph Contrastive Learning for Urban Region Representation", WWW 2022
- Yan et al., "MGEO: Multi-Modal Geographic Language Model Pre-Training", ACL 2023
- Xu et al., "Cross-city Transfer Learning for Deep Spatio-Temporal Prediction", IJCAI 2022 (CTLE)
- Pan et al., "A Unified Approach to Urban Knowledge Representation and Transfer", KDD 2022 (MetaGRL)
- Lee et al., "SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation", EMNLP 2022
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo), CVPR 2020
