# train.py
"""
Cross-city urban region representation learning.

Usage:
    python train.py \
        --config configs/base.yaml \
        --source_cities nyc seoul \
        --target_city singapore \
        --checkpoint_dir experiments/nyc_seoul
"""

import argparse
import json
import os
import sys
import yaml

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from data.dataset import UrbanRegionDataset, MultiCityDataset, collate_regions
from models.model import DualModalNet
from models.trainer import Trainer


def save_config(cfg, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/base.yaml")
    parser.add_argument("--source_cities",  nargs="+", required=True)
    parser.add_argument("--target_city",    required=True)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--num_workers",    type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint_dir:
        cfg["logging"]["checkpoint_dir"] = args.checkpoint_dir + "/checkpoints"
        cfg["logging"]["log_dir"]        = args.checkpoint_dir + "/logs"

    # n_cities 자동 설정
    cfg["model"]["n_cities"] = len(args.source_cities)

    save_config(cfg, cfg["logging"]["log_dir"])

    device   = args.device if torch.cuda.is_available() else "cpu"
    sat_mode = cfg.get("satellite_mode", "npy")
    bs       = cfg["training"]["batch_size"]
    nw       = args.num_workers

    # Dataset
    source_ds = {
        city: UrbanRegionDataset(city, cfg["data"]["data_root"],
                                 satellite_mode=sat_mode)
        for city in args.source_cities
    }
    train_ds = MultiCityDataset(source_ds)

    # Train / val split (8:2)
    n_train = int(len(train_ds) * 0.8)
    n_val   = len(train_ds) - n_train
    train_split, val_split = random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Balanced sampler: 도시별 균등 샘플링
    use_balanced = cfg["training"].get("balanced_sampler", True)
    if use_balanced and len(source_ds) > 1:
        train_indices = train_split.indices
        subset_weights = []
        n_cities = len(source_ds)
        city_sizes = {c: len(ds) for c, ds in source_ds.items()}
        for idx in train_indices:
            city, _ = train_ds.index[idx]
            w = 1.0 / (city_sizes[city] * n_cities)
            subset_weights.append(w)
        sampler = WeightedRandomSampler(
            subset_weights, num_samples=len(train_indices), replacement=True)
        print(f"[Sampler] Balanced: {city_sizes}")
        train_loader = DataLoader(
            train_split, batch_size=bs, sampler=sampler,
            num_workers=nw, pin_memory=True, collate_fn=collate_regions,
        )
    else:
        train_loader = DataLoader(
            train_split, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=True, collate_fn=collate_regions,
        )

    val_loader = DataLoader(
        val_split, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, collate_fn=collate_regions,
    )

    # Model
    model = DualModalNet(cfg)

    # city → index 매핑 (GRL용)
    # city_to_idx = {city: i for i, city in enumerate(sorted(source_ds.keys()))}
    # print(f"[Main] city_to_idx: {city_to_idx}")

    # Trainer
    trainer = Trainer(cfg, model, train_loader, val_loader, device=device)

    # Train
    history = trainer.run()

    # Save history
    log_dir = cfg["logging"]["log_dir"]
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Main] History saved → {log_dir}/history.json")


if __name__ == "__main__":
    main()
