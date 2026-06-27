# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-city urban region representation learning. Trains on source cities (e.g., NYC + Seoul) and transfers to a target city (e.g., Singapore) for downstream tasks like population density prediction.

## Commands

### Training
```bash
python train.py \
    --config configs/base.yaml \
    --source_cities nyc seoul \
    --target_city singapore \
    --checkpoint_dir experiments/nyc_seoul
```

### Extract Region Embeddings (after training)
```bash
python scripts/extract_embeddings.py \
    --config configs/base.yaml \
    --checkpoint experiments/nyc_seoul/checkpoints/epoch_best.pt \
    --cities nyc seoul singapore \
    --output_dir embeddings/nyc_seoul
```

### Downstream Evaluation (linear probe)
```bash
python scripts/linear_probe.py \
    --source_city nyc seoul \
    --target_city singapore \
    --emb_dir embeddings/nyc_seoul
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Full Pipeline
**Train** → **Extract embeddings** → **Linear probe evaluation**

### Data Layer (`data/dataset.py`)
- `UrbanRegionDataset`: loads `satellite_emb.npy` and `poi_emb.npy` per city, then spatially joins block group (BG) centroids to region polygons via `region.shp`. Each `__getitem__` returns all BGs belonging to one region as `[M, dim]` tensors.
- `MultiCityDataset`: concatenates multiple `UrbanRegionDataset`s with per-city indexing.
- `collate_regions`: intentionally returns a plain `list[dict]` (no batching) — the aggregator processes each region independently.
- `LinearProbeDataset`: wraps pre-extracted `{city}_region_emb.npy` for evaluation.

Expected data layout per city:
```
data/{city}/
    satellite_emb.npy   # [N_bg, 64] pre-computed satellite embeddings
    poi_emb.npy         # dict with 'morph_emb' [N_bg, 64] and 'centroid' [N_bg, 2]
    region.shp          # region polygon boundaries
    pop_gt.csv          # population density ground truth
```
Supported cities with their configs: `nyc` (BoroCT2020), `seoul` (ADM_CD), `singapore` (row_index).

### Model (`models/model.py`)
`DualModalNet` operates at two levels:

1. **BG level** — each block group's satellite and POI embedding is projected to shared (32-d) and specific (32-d) subspaces via linear projections.
2. **Region level** — BG embeddings are aggregated to a single region vector using `AttentionAggregator` (transformer-style cross-attention pooling) or `MeanAggregator`. Final `region_emb = concat(sat_region, poi_region)` → 128-d.

The model also stores frozen per-region satellite mean (`sat_mean`) used for cross-city alignment.

### Losses (`models/losses.py`)
```
L_total = λ_contrast · L_contrast  +  λ_dis · L_dis  +  λ_align · L_align
```
- **`RegionContrastiveLoss`** — symmetric NT-Xent between `sat_region` and `poi_region` embeddings (within-region positive pair, between-region negatives).
- **`BGDisLoss`** — cosine similarity between specific and shared projections should be zero (disentanglement).
- **`SatAlignLoss`** — uses frozen `sat_mean` vectors to find cross-city positive pairs (cosine sim > threshold); pulls their `region_emb`s closer. Requires ≥2 cities in the batch.

### Aggregator (`models/aggregator.py`)
- `AttentionAggregator`: self-attention over BGs → learnable pooling query → FFN → single vector.
- `MeanAggregator`: simple mean of valid BG features.

### Trainer (`models/trainer.py`)
Adam + CosineAnnealingLR. Saves `epoch_NNN_best.pt` when validation loss improves, and periodic checkpoints every `save_every` epochs. History logged to `{log_dir}/history.json`.

## Configuration (`configs/base.yaml`)

Key hyperparameters:
- `lambda_contrast / lambda_dis / lambda_align` — loss weights
- `align_threshold` — satellite cosine similarity threshold for cross-city positive pairs (default 0.7)
- `balanced_sampler` — equalizes city sample counts when training on multiple cities
- `aggregator` — `"attention"` or `"mean"`
- `satellite_mode` — `"npy"` (pre-extracted embeddings) or `"image"` (raw PNG files)
