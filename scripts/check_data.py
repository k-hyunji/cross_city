# scripts/check_data.py
"""
데이터 검증 스크립트

Usage:
    python scripts/check_data.py --cities nyc seoul singapore
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.dataset import UrbanRegionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()

    for city in args.cities:
        print(f"\n{'='*50}")
        print(f"  {city.upper()}")
        print(f"{'='*50}")
        try:
            ds = UrbanRegionDataset(city, args.data_root, satellite_mode="npy")
            print(f"  Regions: {len(ds)}")

            # Sample check
            sample = ds[0]
            print(f"  sat_data shape: {sample['sat_data'].shape}")
            print(f"  poi_emb shape:  {sample['poi_emb'].shape}")
            print(f"  valid_mask sum: {sample['valid_mask'].sum()}/{len(sample['valid_mask'])}")
            print(f"  region_id: {sample['region_id']}")
            print(f"  city: {sample['city']}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
