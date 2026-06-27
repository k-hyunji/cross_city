# scripts/convert_satellite_geojson.py
"""
Google Earth alpha embedding geojson → satellite_emb.npy 변환

Usage:
    python scripts/convert_satellite_geojson.py \
        --geojson alpha_2024_nyc_mean.geojson \
        --output data/nyc/satellite_emb.npy
"""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()

    with open(args.geojson) as f:
        gj = json.load(f)

    features = gj["features"]
    print(f"[convert] {len(features)} features")

    # A00 ~ A63 컬럼에서 embedding 추출
    emb_cols = [f"A{i:02d}" for i in range(64)]
    embs = []
    for feat in features:
        props = feat["properties"]
        vec = [props.get(c, None) for c in emb_cols]
        if any(v is None for v in vec):
            embs.append(np.zeros(64, dtype=np.float32))
        else:
            embs.append(np.array(vec, dtype=np.float32))

    embs = np.stack(embs)
    np.save(args.output, embs)
    print(f"[convert] Saved: {args.output} | shape={embs.shape}")
    print(f"[convert] None ratio: {(embs.sum(axis=1) == 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
