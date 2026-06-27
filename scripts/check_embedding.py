# scripts/check_embedding.py
"""
학습된 모델의 embedding 분석.

Usage:
    python scripts/check_embedding.py \
        --emb_dir embeddings/nyc_seoul \
        --cities nyc seoul singapore
"""

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--cities", nargs="+", required=True)
    args = parser.parse_args()

    embs = {}
    for city in args.cities:
        data = np.load(f"{args.emb_dir}/{city}_region_emb.npy",
                       allow_pickle=True).item()
        embs[city] = data["embeddings"]
        print(f"{city}: {embs[city].shape}")

    # 도시 간 cosine similarity
    print(f"\n{'='*40}")
    print("City-pair mean cosine similarity:")
    cities = list(embs.keys())
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            c_i, c_j = cities[i], cities[j]
            mean_i = embs[c_i].mean(0, keepdims=True)
            mean_j = embs[c_j].mean(0, keepdims=True)
            sim = cosine_similarity(mean_i, mean_j)[0][0]
            print(f"  {c_i} vs {c_j}: {sim:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
