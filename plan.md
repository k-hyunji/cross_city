# Implementation Plan: Level-specific Architecture + Shared/Specific Loss Routing

## Saved Backups

| Original | Backup |
|----------|--------|
| `models/model.py` | `models/model_v1.py` |
| `models/losses.py` | `models/losses_v1.py` |
| `models/trainer.py` | `models/trainer_v1.py` |
| `configs/base.yaml` | `configs/base_v1.yaml` |

---

## Problem Being Solved

**Current flaw**: `DualModalNet` projects BGs into shared/spec branches at BG level, but then the aggregator receives `cat([shared, spec])` = 64-d and re-mixes everything. Result:
- `L_clip` operates on `sat_region[64]` which contains spec — so spec branch gets cross-city alignment gradient (over-alignment)
- `BGDisLoss` forces spec⊥shared at BG level, but the aggregator undoes this at region level
- No architectural protection for spec at region level

**Goal**: Separate region-level representations for shared (cross-city alignment) and specific (within-city diversity). Route each loss only to the branches it should influence.

---

## Design Concerns and Revisions

### 1. Input feature city bias — not directly addressed by architecture alone

City-mean subtraction (already in `dataset.py`) removes the per-city mean but the city-separating direction (POI PC1 = 65.3% of variance) remains intact. K-means on city-mean-subtracted features is still 87–94% city-accurate. The shared branch will learn city-identity first (easy gradient) unless explicitly prevented.

Raw satellite images and a city-agnostic encoder (SatCLIP, UrbanCLIP) would fix this at the source, but raw images are not currently available. POI encoder retraining is also blocked by cross-city taxonomy mismatch (PLUTO vs Korean codes vs URA — a data-level problem, not an encoder problem).

**Mitigation**: Adversarial city classifier (DANN) on `region_shared` — see DANN section under File 1. This directly penalizes `region_shared` for being city-discriminative without requiring raw data or a new encoder.

---

### 2. `region_shared` simple average → MLP fusion

`(sat_shared_region + poi_shared_region) / 2` assigns equal weight to both modalities at every training step. In early training POI shared embeddings are noisy (city-biased input), so averaging imports that noise at 50% weight into the signal used by `CrossCitySharedLoss`.

**Revised**: `cat([sat_shared_region, poi_shared_region]) → Linear(64→32) → ReLU`. The MLP learns to weight modalities based on their reliability. Adds 2,080 parameters.

---

### 3. `CrossCitySharedLoss` hard threshold → soft combined mining

Hard threshold `sat_mean_sim > 0.5` only mines visually similar cross-city pairs. Regions with the same function but different visual appearance (NYC brick low-rise vs Seoul high-rise apartment, both high-density residential) share functional meaning but may have sat_mean sim below 0.5.

**Revised**: soft positive weight combining frozen sat_mean similarity and learned `poi_shared_region` similarity:
```
combined_ij = 0.6 * sat_sim_ij + 0.4 * (poi_shared_sim_ij + 1) / 2
pos if: cities differ AND combined_ij > 0.2 (lower min threshold)
```
POI shared sim is normalized from [-1,1] → [0,1] before combining. The 0.6/0.4 split keeps sat dominant early in training when POI shared embeddings are still noisy, while opening a channel for functionally similar but visually different pairs as training progresses.

---

## Architecture After Change

```
BG Level (UNCHANGED):
  sat_emb[M,64] → sat_shared_proj → sat_shared[M,32]
               → sat_spec_proj   → sat_spec[M,32]
  poi_emb[M,64] → poi_shared_proj → poi_shared[M,32]
               → poi_spec_proj   → poi_spec[M,32]

Region Level (NEW — 4 aggregators instead of 2):
  sat_shared[M,32] → sat_shared_agg(dim=32) → sat_shared_region[32]
  sat_spec[M,32]   → sat_spec_agg(dim=32)   → sat_spec_region[32]
  poi_shared[M,32] → poi_shared_agg(dim=32) → poi_shared_region[32]
  poi_spec[M,32]   → poi_spec_agg(dim=32)   → poi_spec_region[32]

Fusion (shape unchanged — downstream eval unaffected):
  region_shared = shared_fusion(cat([sat_shared_region, poi_shared_region]))  # [32]  MLP fusion
  sat_region    = cat([sat_shared_region, sat_spec_region])                   # [64]  backward compat
  poi_region    = cat([poi_shared_region, poi_spec_region])                   # [64]  backward compat
  region_emb    = cat([sat_region, poi_region])                               # [128] UNCHANGED

DANN (optional, use_adv flag):
  region_shared → GRL(lambda) → Linear(32, n_cities) → city_pred
  L_adv = CrossEntropy(city_pred, city_label)   # encoder maximizes this, classifier minimizes
```

### Loss Routing

| Loss | Acts on | Gradient reaches |
|------|---------|-----------------|
| `CLIPSharedCon` (replaces L_clip) | `sat_shared_region[32]`, `poi_shared_region[32]` | shared proj + shared agg only |
| `CrossCitySharedLoss` (NEW) | `region_shared[32]` | shared proj + shared agg + shared_fusion |
| `FunctionalPrototypeLoss` | `region_emb[128]` | all 4 proj + all 4 agg |
| `BGDisLoss` | `sat_spec[M,32]` ⊥ `sat_shared[M,32].detach()` | spec proj only |
| `AdvCityLoss` (optional DANN) | `region_shared[32]` via GRL | negated → penalizes shared for city info |

**spec branch never receives gradient from alignment losses** — architectural guarantee.

---

## File 1: `models/model.py`

### What changes
- Add `use_sep_agg` config flag (default `False` for backward compat)
- When `use_sep_agg=True`: create 4 aggregators each with `dim=32` instead of 2 with `dim=64`
- Add `shared_fusion`: `Linear(64→32) + ReLU` replacing simple average for `region_shared`
- Add DANN components (optional `use_adv` flag): GRL + `Linear(32, n_cities)` city classifier
- New `forward_region()` branch for sep_agg path
- New output keys: `sat_shared_region`, `sat_spec_region`, `poi_shared_region`, `poi_spec_region`, `region_shared`
- All existing output keys (`sat_region[64]`, `poi_region[64]`, `region_emb[128]`) preserved

### Full new `__init__` block (aggregator section)

```python
# Aggregator
agg_mode    = m.get("aggregator",      "attention")
heads       = m.get("aggregator_heads", 4)
use_sep_agg = m.get("use_sep_agg",     False)
use_adv     = m.get("use_adv",         False)
n_cities    = m.get("n_cities",        2)
self.use_sep_agg = use_sep_agg
self.use_adv     = use_adv

if use_sep_agg:
    if agg_mode == "attention":
        self.sat_shared_agg = AttentionAggregator(shared_dim, heads)
        self.sat_spec_agg   = AttentionAggregator(spec_dim,   heads)
        self.poi_shared_agg = AttentionAggregator(shared_dim, heads)
        self.poi_spec_agg   = AttentionAggregator(spec_dim,   heads)
    else:
        self.sat_shared_agg = MeanAggregator()
        self.sat_spec_agg   = MeanAggregator()
        self.poi_shared_agg = MeanAggregator()
        self.poi_spec_agg   = MeanAggregator()

    # MLP fusion: cat([sat_shared_region, poi_shared_region])[64] → region_shared[32]
    self.shared_fusion = nn.Sequential(
        nn.Linear(shared_dim * 2, shared_dim),
        nn.ReLU(),
    )

    # DANN: city classifier with gradient reversal on region_shared
    if use_adv:
        self.city_classifier = nn.Linear(shared_dim, n_cities)
else:
    if agg_mode == "attention":
        self.sat_agg = AttentionAggregator(sat_dim, heads)
        self.poi_agg = AttentionAggregator(poi_dim, heads)
    else:
        self.sat_agg = MeanAggregator()
        self.poi_agg = MeanAggregator()

print(f"[Model] sat_dim={sat_dim}, poi_dim={poi_dim}")
print(f"[Model] shared_dim={shared_dim}, region_emb_dim={self.region_emb_dim}")
print(f"[Model] aggregator={agg_mode}, use_sep_agg={use_sep_agg}, use_adv={use_adv}")
```

GRL is implemented as a custom autograd Function — forward pass is identity, backward pass negates the gradient scaled by `lambda_adv`:

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None
```

### New `forward_region()` sep_agg branch

```python
def forward_region(self, sat_data, poi_emb, valid_mask) -> dict:
    sat_shared = self.sat_shared_proj(sat_data)   # [M, 32]
    sat_spec   = self.sat_spec_proj(sat_data)     # [M, 32]
    poi_shared = self.poi_shared_proj(poi_emb)    # [M, 32]
    poi_spec   = self.poi_spec_proj(poi_emb)      # [M, 32]

    if self.use_sep_agg:
        sat_shared_region = self.sat_shared_agg(sat_shared, valid_mask)  # [32]
        sat_spec_region   = self.sat_spec_agg(sat_spec,     valid_mask)  # [32]
        poi_shared_region = self.poi_shared_agg(poi_shared,  valid_mask)  # [32]
        poi_spec_region   = self.poi_spec_agg(poi_spec,      valid_mask)  # [32]

        # MLP fusion instead of simple average — learns modality weights
        region_shared = self.shared_fusion(
            torch.cat([sat_shared_region, poi_shared_region], dim=-1)
        )                                                                  # [32]

        sat_region = torch.cat([sat_shared_region, sat_spec_region], dim=-1)   # [64]
        poi_region = torch.cat([poi_shared_region, poi_spec_region], dim=-1)   # [64]
        region_emb = torch.cat([sat_region, poi_region], dim=-1)               # [128]

        return {
            # BG-level (for BGDisLoss)
            "sat_shared": sat_shared, "sat_spec": sat_spec,
            "poi_shared": poi_shared, "poi_spec": poi_spec, "valid_mask": valid_mask,
            # Region-level new keys (for CLIPSharedCon, CrossCitySharedLoss, DANN)
            "sat_shared_region": sat_shared_region,
            "sat_spec_region":   sat_spec_region,
            "poi_shared_region": poi_shared_region,
            "poi_spec_region":   poi_spec_region,
            "region_shared":     region_shared,
            # Backward-compat keys (for FunctionalPrototypeLoss, extract_embeddings)
            "sat_region": sat_region,
            "poi_region": poi_region,
            "region_emb": region_emb,
        }
    else:
        # Legacy 2-agg path (unchanged)
        sat_recon  = torch.cat([sat_shared, sat_spec], dim=-1)   # [M, 64]
        poi_recon  = torch.cat([poi_shared, poi_spec], dim=-1)   # [M, 64]
        sat_region = self.sat_agg(sat_recon, valid_mask)
        poi_region = self.poi_agg(poi_recon, valid_mask)
        region_emb = torch.cat([sat_region, poi_region], dim=-1)
        return {
            "sat_shared": sat_shared, "sat_spec": sat_spec,
            "poi_shared": poi_shared, "poi_spec": poi_spec, "valid_mask": valid_mask,
            "sat_region": sat_region, "poi_region": poi_region, "region_emb": region_emb,
        }
```

---

## File 2: `models/losses.py`

### Why `CLIPRegionCon` is removed

`CLIPRegionCon` (current active loss, `use_clip: true`) operates at region level on `sat_region[64]` × `poi_region[64]`.

The problem is what `sat_region[64]` contains. In the current 2-aggregator architecture:
```
sat_recon  = cat([sat_shared[M,32], sat_spec[M,32]])   # [M, 64] — mixed back together
sat_region = sat_agg(sat_recon)                         # [64]    — aggregation of the mix
```

`sat_region` is the aggregation of the re-concatenated shared+spec features. So when `CLIPRegionCon` computes:
```
logits[i, j] = sat_region_i · poi_region_j / τ
```
and backpropagates, the gradient flows back through `sat_agg → sat_recon → sat_shared_proj AND sat_spec_proj`. **Both branches receive cross-city alignment gradient.** The spec branch, which is supposed to capture city/local-specific diversity, is pulled toward cross-city alignment — the same over-alignment problem that `UnifiedCityCon` caused.

`BGDisLoss` tries to counteract this by pushing `sat_spec ⊥ sat_shared` at BG level, but it operates on `[M, 32]` tensors before the aggregator. Once the aggregator mixes them back into `sat_region[64]`, the BG-level orthogonality constraint has no effect on the region-level gradient from `CLIPRegionCon`. The two losses fight each other and neither wins cleanly.

**The fix is not to change the loss formula — it is to change what the loss sees.** With 4 separate aggregators, `sat_shared_region[32]` and `sat_spec_region[32]` never mix before the loss is computed. A loss operating only on `sat_shared_region` cannot send gradient to `sat_spec_agg` or `sat_spec_proj` by construction.

`CLIPRegionCon` is therefore disabled (`use_clip: false`) and replaced by two losses that operate on the separated shared subspace.

---

### What changes — overview

Two new loss classes replace `CLIPRegionCon`. Both act at **region level** (after aggregation), but only on the **shared subspace** outputs (`sat_shared_region[32]`, `poi_shared_region[32]`, `region_shared[32]`). The spec branch (`sat_spec_region`, `poi_spec_region`) is never an input to either loss.

| Loss | Replaces | Level | Input | What it enforces |
|------|----------|-------|-------|-----------------|
| `CLIPRegionCon` | — | Region | `sat_region[64]`, `poi_region[64]` (mixed) | Modality alignment, but leaks into spec branch |
| `CLIPSharedCon` | `CLIPRegionCon` | Region | `sat_shared_region[32]`, `poi_shared_region[32]` | Modality alignment in shared subspace only |
| `CrossCitySharedLoss` | (new) | Region | `region_shared[32]` | Cross-city functional alignment in shared subspace |

`BGDisLoss` and `FunctionalPrototypeLoss` are unchanged. The spec branch only receives gradient from `BGDisLoss` (BG level) and `FunctionalPrototypeLoss` (region level via `region_emb[128]`).

---

### `CLIPSharedCon` — what it does

**Level**: region  
**Input**: `sat_shared_region[32]` and `poi_shared_region[32]` per region in the batch

**Purpose**: align satellite and POI **modalities** in the shared subspace. This is exactly what `CLIPRegionCon` was doing, but restricted to the 32-d shared vectors — so gradient cannot reach the spec branch.

**How it works**:

Step 1 — build a B×B cross-modal similarity matrix:
```
sat = normalize(sat_shared_region)   # [B, 32]
poi = normalize(poi_shared_region)   # [B, 32]
logits[i, j] = sat_i · poi_j / τ    # [B, B],  τ = 0.07
```

Step 2 — build positive mask:
```
pos_mask[i, i] = 1                  # diagonal: same region, cross-modal (always)
pos_mask[i, j] = 1 (cross-city)     # if sat_mean_i · sat_mean_j > 0.5 AND cities differ
```
Pseudo-positives use frozen `sat_mean` similarity only — poi_threshold is dropped (it was never reachable at max 0.271).

Step 3 — SupCon loss in both directions (sat→poi and poi→sat):
```
for each anchor i (sat→poi):
    loss_i = logsumexp(logits[i, :]) - logsumexp(logits[i, pos_idx])

for each anchor j (poi→sat):
    loss_j = logsumexp(logits[:, j]) - logsumexp(logits[pos_idx, j])

L_clip_shared = mean over all anchors with ≥1 positive
```

**What it achieves**: `sat_shared_region` and `poi_shared_region` are pulled together for same-region pairs and for cross-city pairs with similar satellite context. The spec aggregators (`sat_spec_agg`, `poi_spec_agg`) are completely outside this computation.

---

### `CrossCitySharedLoss` — what it does

**Level**: region  
**Input**: `region_shared[32]`, `poi_shared_region[32]`, `sat_mean[64]` per region in the batch

**Purpose**: pull cross-city regions with similar functional context together in the shared space, while keeping them separated from within-city regions. Uses a **soft combined mining** signal — not just visual (sat_mean) similarity alone.

**How it works**:

Step 1 — build similarity matrix over all B regions:
```
shared = normalize(region_shared)       # [B, 32]
logits[i, j] = shared_i · shared_j / τ # [B, B],  τ = 0.1
logits.fill_diagonal_(-inf)             # exclude self
```

Step 2 — compute soft positive weight for each cross-city pair:
```
sat_sim_ij     = normalize(sat_mean_i) · normalize(sat_mean_j)      # frozen, [0,1]
poi_sim_ij     = normalize(poi_shared_i) · normalize(poi_shared_j)  # learned, [-1,1]
poi_sim_norm   = (poi_sim_ij + 1) / 2                               # → [0,1]

combined_ij    = 0.6 * sat_sim_ij + 0.4 * poi_sim_norm

soft_pos[i,j]  = combined_ij   if cities[i]≠cities[j] AND combined_ij > 0.2
               = 0              otherwise
```

Step 3 — soft-weighted InfoNCE with **full in-batch denominator**:
```
log_numer_i = log( Σ_j soft_pos[i,j] * exp(logits[i,j]) )
log_denom_i = logsumexp(logits[i, :])      # all B regions, within + cross-city

loss_i      = log_denom_i - log_numer_i    # only for anchors with Σ soft_pos > 0

L_cross_city = mean over valid anchors
```

**Why soft mining over hard threshold**:  
Hard `sat_mean_sim > 0.5` misses functionally similar but visually different pairs (NYC brick low-rise vs Seoul high-rise, both high-density residential). The `poi_shared_region` similarity provides a complementary functional signal that grows more meaningful as training progresses. The 0.6/0.4 split keeps sat dominant early (when POI shared embeddings are noisy), while gradually opening the functional channel.

**Why full denominator (including within-city)**:  
Without within-city regions in the denominator, there is no force separating cross-city functional clusters from within-city regions. Full denominator enforces: "NYC commercial must be closer to Seoul commercial than to NYC residential" — functional topology, not just city alignment.

---

### TotalLoss changes

```python
# New flags in __init__:
self.use_clip_shared = t.get("use_clip_shared", False)
self.use_cross_city  = t.get("use_cross_city",  False)
self.use_adv         = m.get("use_adv",         False)
self.lw_cross_city   = t.get("lambda_cross_city", 0.5)
self.lw_adv          = t.get("lambda_adv",        0.1)

if self.use_clip_shared:
    self.clip_shared = CLIPSharedCon(
        temperature=t.get("contrast_temp",    0.07),
        sat_threshold=t.get("align_threshold", 0.5),
    )

if self.use_cross_city:
    self.cross_city = CrossCitySharedLoss(
        temperature=t.get("cross_city_temp",          0.1),
        min_threshold=t.get("cross_city_min_threshold", 0.2),
        sat_weight=t.get("cross_city_sat_weight",       0.6),
    )

# In forward():
l_clip_shared = self.clip_shared(outputs) if self.use_clip_shared else zero
l_cross_city  = self.cross_city(outputs)  if self.use_cross_city  else zero
l_adv = (torch.stack([o["city_adv_loss"] for o in outputs]).mean()
         if self.use_adv and "city_adv_loss" in outputs[0] else zero)

total = (self.lw_contrast   * l_clip_shared
       + self.lw_dis        * l_dis
       + self.lw_proto      * l_proto
       + self.lw_cross_city * l_cross_city
       + self.lw_adv        * l_adv)

return {
    "total":      total,
    "contrast":   l_clip_shared,
    "dis":        l_dis,
    "align":      zero,
    "proto":      l_proto,
    "cross_city": l_cross_city,
    "adv":        l_adv,
}
```

DANN city_adv_loss is computed inside `model.forward()` when `use_adv=True`:
```python
# In model.forward(), after forward_region():
if self.use_adv:
    reversed_shared = GradReverse.apply(out["region_shared"], lambda_adv)
    city_pred       = self.city_classifier(reversed_shared)
    city_label      = torch.tensor(sample["city_idx"], device=device)
    out["city_adv_loss"] = F.cross_entropy(city_pred.unsqueeze(0),
                                            city_label.unsqueeze(0))
```
`lambda_adv` follows the standard DANN ramp: `2 / (1 + exp(-10 * p)) - 1` where `p = epoch / total_epochs`.

### `CLIPSharedCon` class

```python
class CLIPSharedCon(nn.Module):
    """CLIPRegionCon on 32-d shared subspace. Gradient never reaches spec branch."""
    def __init__(self, temperature=0.07, sat_threshold=0.5):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold

    def forward(self, outputs):
        B = len(outputs)
        if B < 2:
            return outputs[0]["sat_shared_region"].new_zeros(1, requires_grad=True).squeeze()

        sat = F.normalize(torch.stack([o["sat_shared_region"] for o in outputs]), dim=-1)  # [B, 32]
        poi = F.normalize(torch.stack([o["poi_shared_region"] for o in outputs]), dim=-1)  # [B, 32]
        logits = sat @ poi.T / self.temperature   # [B, B]

        pos_mask = torch.eye(B, device=sat.device)

        sat_means = F.normalize(torch.stack([o["sat_mean"] for o in outputs]), dim=-1)  # [B, 64]
        sat_sim   = sat_means @ sat_means.T   # [B, B]
        cities    = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if sat_sim[i, j] > self.sat_threshold:
                    pos_mask[i, j] = 1.0
                    pos_mask[j, i] = 1.0

        log_denom_row = torch.logsumexp(logits, dim=1)
        log_denom_col = torch.logsumexp(logits, dim=0)

        loss, count = 0.0, 0
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            loss += log_denom_row[i] - torch.logsumexp(logits[i, pos_idx], dim=0)
            count += 1
        for j in range(B):
            pos_idx = pos_mask[:, j].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            loss += log_denom_col[j] - torch.logsumexp(logits[pos_idx, j], dim=0)
            count += 1

        return loss / max(count, 1)
```

### `CrossCitySharedLoss` class

```python
class CrossCitySharedLoss(nn.Module):
    """InfoNCE on region_shared[32], full in-batch denominator (within+cross-city)."""
    def __init__(self, temperature=0.1, sat_threshold=0.5):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold

    def forward(self, outputs):
        cities = [o["city"] for o in outputs]
        if len(set(cities)) < 2:
            return outputs[0]["region_shared"].new_zeros(1, requires_grad=True).squeeze()

        shared    = F.normalize(torch.stack([o["region_shared"] for o in outputs]), dim=-1)  # [B, 32]
        sat_means = F.normalize(torch.stack([o["sat_mean"]      for o in outputs]), dim=-1)  # [B, 64]
        sat_sim   = sat_means @ sat_means.T   # [B, B]

        B      = len(outputs)
        logits = shared @ shared.T / self.temperature   # [B, B]
        logits.fill_diagonal_(-1e9)

        loss, count = 0.0, 0
        for i in range(B):
            pos_idx = []
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if sat_sim[i, j] > self.sat_threshold:
                    pos_idx.append(j)
            if not pos_idx:
                continue
            pos_idx = torch.tensor(pos_idx, device=shared.device)
            log_denom  = torch.logsumexp(logits[i], dim=0)
            log_numer  = torch.logsumexp(logits[i][pos_idx], dim=0)
            loss      += log_denom - log_numer
            count     += 1

        return loss / max(count, 1) if count > 0 else shared.new_zeros(1, requires_grad=True).squeeze()
```

### `TotalLoss` changes

```python
# New flags in __init__:
self.use_clip_shared = t.get("use_clip_shared", False)
self.use_cross_city  = t.get("use_cross_city",  False)
self.lw_cross_city   = t.get("lambda_cross_city", 0.5)

if self.use_clip_shared:
    self.clip_shared = CLIPSharedCon(
        temperature=t.get("contrast_temp",    0.07),
        sat_threshold=t.get("align_threshold", 0.5),
    )

if self.use_cross_city:
    self.cross_city = CrossCitySharedLoss(
        temperature=t.get("cross_city_temp",   0.1),
        sat_threshold=t.get("align_threshold", 0.5),
    )

# In forward():
l_clip_shared = self.clip_shared(outputs)  if self.use_clip_shared else zero
l_cross_city  = self.cross_city(outputs)   if self.use_cross_city  else zero

total = (self.lw_contrast   * l_contrast
       + self.lw_dis        * l_dis
       + self.lw_align      * l_align
       + self.lw_proto      * l_proto
       + self.lw_contrast   * l_clip_shared   # same λ_contrast weight
       + self.lw_cross_city * l_cross_city)

return {
    "total":      total,
    "contrast":   l_contrast if not self.use_clip_shared else l_clip_shared,
    "dis":        l_dis,
    "align":      l_align,
    "proto":      l_proto,
    "cross_city": l_cross_city,
}
```

---

## File 3: `models/trainer.py`

### What changes
- `keys` list: add `"cross_city"`
- `pbar.set_postfix`: add `cc=` field
- epoch print line: add `cc=` field
- `run()` loss_desc string: add `use_clip_shared` branch

### Diff summary

```python
# Before:
keys = ["total", "contrast", "dis", "align", "proto"]

# After:
keys = ["total", "contrast", "dis", "align", "proto", "cross_city"]

# Before (pbar):
pbar.set_postfix(
    loss=f"{losses['total'].item():.4f}",
    cont=f"{losses['contrast'].item():.4f}",
    proto=f"{losses['proto'].item():.4f}",
)

# After (pbar):
pbar.set_postfix(
    loss=f"{losses['total'].item():.4f}",
    cont=f"{losses['contrast'].item():.4f}",
    cc=f"{losses['cross_city'].item():.4f}",
    proto=f"{losses['proto'].item():.4f}",
)

# Before (epoch print):
f"(cont={tl['contrast']:.4f} dis={tl['dis']:.4f} proto={tl['proto']:.4f})"

# After (epoch print):
f"(cont={tl['contrast']:.4f} cc={tl['cross_city']:.4f} dis={tl['dis']:.4f} proto={tl['proto']:.4f})"

# Before (loss_desc):
if t.get("use_clip", False):
    loss_desc = "L_clip (CLIP B×B) + L_dis + L_proto"

# After (loss_desc):
if t.get("use_clip_shared", False):
    loss_desc = "CLIPSharedCon (32-d shared) + CrossCitySharedLoss + L_dis + L_proto"
elif t.get("use_clip", False):
    loss_desc = "L_clip (CLIP B×B) + L_dis + L_proto"
```

---

## File 4: `configs/base.yaml`

### What changes

```yaml
model:
  aggregator: "attention"
  aggregator_heads: 4
  use_sep_agg: true          # NEW: 4 branch-level aggregators (was 2)

training:
  use_clip:         false    # OLD CLIPRegionCon — disabled
  use_clip_shared:  true     # NEW CLIPSharedCon on 32-d shared subspace
  use_cross_city:   true     # NEW CrossCitySharedLoss on region_shared[32]
  use_unified_con:  false    # still off
  use_proto:        true     # still on

  lambda_contrast:  1.0      # unchanged (used by CLIPSharedCon)
  lambda_dis:       0.3      # was 0.1 — raised because BGDisLoss now reaches region level
  lambda_proto:     0.3      # unchanged
  lambda_cross_city: 0.5     # NEW
  cross_city_temp:   0.1     # NEW

  align_threshold:  0.5      # unchanged (calibrated for city-mean-sub space)
  poi_threshold:    0.0      # unchanged (fixed)
  contrast_temp:    0.07     # unchanged
```

---

## Verification Steps

```bash
# 1. Shape verification (no training)
python - <<'EOF'
import yaml, torch
from data.dataset import UrbanRegionDataset, collate_regions
from torch.utils.data import DataLoader
from models.model import DualModalNet
from models.losses import TotalLoss

with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["model"]["n_cities"] = 2

ds     = UrbanRegionDataset("nyc", "data")
loader = DataLoader(ds, batch_size=4, collate_fn=collate_regions)
batch  = next(iter(loader))

model   = DualModalNet(cfg)
outputs = model(batch)
o       = outputs[0]
print("sat_shared_region:", o["sat_shared_region"].shape)   # [32]
print("poi_shared_region:", o["poi_shared_region"].shape)   # [32]
print("region_shared:",     o["region_shared"].shape)       # [32]
print("sat_region:",        o["sat_region"].shape)          # [64]
print("region_emb:",        o["region_emb"].shape)          # [128]

losses = TotalLoss(cfg)
out    = losses(outputs)
print("loss keys:", list(out.keys()))
print("total:", out["total"].item())
EOF

# 2. Full training
python train.py \
    --config configs/base.yaml \
    --source_cities nyc seoul \
    --target_city singapore \
    --checkpoint_dir checkpoints/nyc_seoul_sep_agg

# 3. Extract + probe
python scripts/extract_embeddings.py \
    --config configs/base.yaml \
    --checkpoint checkpoints/nyc_seoul_sep_agg/epoch_XXX_best.pt \
    --cities nyc seoul singapore \
    --output_dir embeddings/nyc_seoul_sep_agg

python scripts/linear_probe.py \
    --source_city nyc seoul \
    --target_city singapore \
    --emb_dir embeddings/nyc_seoul_sep_agg
```

---

## Health Metrics to Monitor

| Signal | Healthy | Problem |
|--------|---------|---------|
| `val_contrast` (= CLIPSharedCon) | Decreasing from ~3.4 | Rising after ep 20 |
| `val_cross_city` | Decreasing from ~3+ toward 1 | Stuck → cross-city alignment not working |
| `val_proto` | Decreasing from ~2.77 | Stuck at 2.77 → prototypes not learning |
| NYC↔Seoul centroid cos | Positive (≥ +0.3) | Negative → city-domain clustering in shared space |
| Within-city sim | 0.3–0.6 | >0.7 → over-aligned; <0.2 → collapsed |
| Seoul norm std | >0.05 | <0.03 → Seoul collapsed |

---

## What Does NOT Change

- `models/aggregator.py`: `AttentionAggregator(dim=32, n_heads=4)` is already valid (8-d/head)
- `data/dataset.py`: city_mean_subtraction fix already applied
- `scripts/extract_embeddings.py`: uses `out["region_emb"]` only — unaffected
- `scripts/linear_probe.py`: uses region_emb — unaffected
- `train.py`: unchanged
- `region_emb[128]` shape: unchanged — all downstream code works as-is
