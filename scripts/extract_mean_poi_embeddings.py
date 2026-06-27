# scripts/extract_mean_poi_embeddings.py
"""
Extract region embeddings by mean-aggregating RegionContext/POI BG embeddings.

This does not load a trained checkpoint. It directly groups BG-level
`morph_emb` vectors into regions by spatial join and applies
models.aggregator.MeanAggregator.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.dataset import build_region_groups, get_city_config, load_poi_data
from models.aggregator import MeanAggregator


def resolve_poi_path(city, data_root, poi_npy):
    if poi_npy:
        return poi_npy
    return os.path.join(data_root, city, "poi_emb.npy")


@torch.no_grad()
def extract_city(city, data_root, poi_npy):
    city = city.lower()
    cfg = get_city_config(city)
    city_dir = os.path.join(data_root, city)
    region_shp = os.path.join(city_dir, "region.shp")
    if not os.path.exists(region_shp):
        raise FileNotFoundError(f"Missing region shapefile: {region_shp}")

    poi_path = resolve_poi_path(city, data_root, poi_npy)
    poi_arr, centroid = load_poi_data(poi_path)
    poi_tensor = torch.tensor(poi_arr.astype(np.float32))

    region_groups = build_region_groups(
        centroid=centroid,
        region_shp_path=region_shp,
        region_id_col=cfg["region_id_col"],
        region_id_str=cfg.get("region_id_str", False),
    )

    agg = MeanAggregator()
    region_ids = list(region_groups.keys())
    embs = []
    for rid in region_ids:
        bg_idx = region_groups[rid]
        emb = agg(poi_tensor[bg_idx], valid_mask=None)
        embs.append(emb.numpy().astype(np.float32))

    return np.stack(embs, axis=0), region_ids, poi_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--singapore_poi_npy",
        default=None,
        help="Optional override for Singapore RegionContext/POI npy.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for city in args.cities:
        poi_override = args.singapore_poi_npy if city.lower() == "singapore" else None
        embs, region_ids, poi_path = extract_city(
            city=city,
            data_root=args.data_root,
            poi_npy=poi_override,
        )
        out_path = os.path.join(args.output_dir, f"{city.lower()}_region_emb.npy")
        np.save(out_path, {"embeddings": embs, "region_ids": region_ids})
        print(
            f"[mean-poi] {city.lower()}: regions={len(region_ids)} "
            f"dim={embs.shape[1]} source={poi_path} -> {out_path}"
        )


if __name__ == "__main__":
    main()
