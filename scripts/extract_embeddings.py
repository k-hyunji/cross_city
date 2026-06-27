# scripts/extract_embeddings.py
"""
학습된 모델로 region embedding 추출.

Usage:
    python scripts/extract_embeddings.py \
        --config configs/base.yaml \
        --checkpoint experiments/nyc_seoul/checkpoints/epoch_best.pt \
        --cities nyc seoul singapore \
        --output_dir embeddings/nyc_seoul
"""

import argparse
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import UrbanRegionDataset, collate_regions
from models.model import DualModalNet


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    all_embs, all_rids = [], []

    for batch in loader:
        for sample in batch:
            sat  = sample["sat_data"].to(device)
            poi  = sample["poi_emb"].to(device)
            mask = sample["valid_mask"].to(device)
            emb  = model.get_region_embedding(sat, poi, mask)
            all_embs.append(emb.cpu().numpy())
            all_rids.append(sample["region_id"])

    return np.stack(all_embs), all_rids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/base.yaml")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--cities",      nargs="+", required=True)
    parser.add_argument("--output_dir",  default="embeddings")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device   = args.device if torch.cuda.is_available() else "cpu"
    sat_mode = cfg.get("satellite_mode", "npy")

    model = DualModalNet(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"[extract] Loaded: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    for city in args.cities:
        ds     = UrbanRegionDataset(city, cfg["data"]["data_root"],
                                    satellite_mode=sat_mode)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_regions)

        embs, rids = extract(model, loader, device)
        out_path   = os.path.join(args.output_dir, f"{city}_region_emb.npy")
        np.save(out_path, {"embeddings": embs, "region_ids": rids})
        print(f"[extract] {city}: {len(rids)} regions, "
              f"emb_dim={embs.shape[1]} → {out_path}")


if __name__ == "__main__":
    main()
