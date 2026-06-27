# scripts/linear_probe.py
"""
Downstream evaluation: region_emb → Ridge → population density

Usage:
    python scripts/linear_probe.py \
        --source_city nyc seoul \
        --target_city singapore \
        --emb_dir embeddings/nyc_seoul
"""

import argparse
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.dataset import LinearProbeDataset


def run_probe(X_train, y_log_train, X_test, y_log_test):
    """y_log_* are log1p-transformed population densities (not z-scored)."""
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    best = {"r2": -float("inf"), "alpha": None, "model": None}
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m  = Ridge(alpha=alpha).fit(X_train, y_log_train)
        r2 = r2_score(y_log_train, m.predict(X_train))
        if r2 > best["r2"]:
            best = {"r2": r2, "alpha": alpha, "model": m}

    pred_log = best["model"].predict(X_test)
    pred_orig = np.expm1(pred_log)
    true_orig = np.expm1(y_log_test)
    return {
        "MAE":   float(mean_absolute_error(true_orig, pred_orig)),
        "RMSE":  float(np.sqrt(mean_squared_error(true_orig, pred_orig))),
        "R2":    float(r2_score(y_log_test, pred_log)),   # log-space R²
        "alpha": best["alpha"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/base.yaml")
    parser.add_argument("--source_city",  nargs="+", required=True)
    parser.add_argument("--target_city",  required=True)
    parser.add_argument("--emb_dir",      default="embeddings")
    parser.add_argument("--data_root",    default="data")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Source: 여러 도시 embedding 합치기
    X_train_list, y_train_list = [], []
    for city in args.source_city:
        src_emb = os.path.join(args.emb_dir, f"{city}_region_emb.npy")
        src_ds  = LinearProbeDataset(
            city, args.data_root,
            emb_npy=src_emb, normalize_pop=False,
        )
        X_train_list.append(src_ds.embs.numpy())
        y_train_list.append(np.log1p(src_ds.pops.numpy()))

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Target
    tgt_emb = os.path.join(args.emb_dir, f"{args.target_city}_region_emb.npy")
    tgt_ds  = LinearProbeDataset(
        args.target_city, args.data_root,
        emb_npy=tgt_emb, normalize_pop=False,
    )
    X_test = tgt_ds.embs.numpy()
    y_test = np.log1p(tgt_ds.pops.numpy())

    results = run_probe(X_train, y_train, X_test, y_test)

    tgt_pops_raw = tgt_ds.pops.numpy()
    src_cities = " + ".join(args.source_city)
    print(f"\n{'='*50}")
    print(f"Linear probe: {src_cities} → {args.target_city}")
    print(f"{'='*50}")
    print(f"  Train regions : {len(y_train)}")
    print(f"  Test  regions : {len(y_test)}")
    print(f"  Target pop  mean={tgt_pops_raw.mean():.1f}  std={tgt_pops_raw.std():.1f}  (raw)")
    print(f"  [log1p space]  mean={y_test.mean():.3f}  std={y_test.std():.3f}")
    print(f"{'─'*50}")
    print(f"  MAE   : {results['MAE']:.2f}  (명/km²)")
    print(f"  RMSE  : {results['RMSE']:.2f}  (명/km²)")
    print(f"  R²    : {results['R2']:.4f}  (log-space)")
    print(f"  Ridge alpha: {results['alpha']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
