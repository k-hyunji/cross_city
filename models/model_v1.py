# models/model.py
"""
DualModalNet + Satellite-based Cross-city Alignment

BG level:
    projection → sat_shared/poi_shared, sat_spec/poi_spec

Aggregator:
    sat_agg → sat_region [64]
    poi_agg → poi_region [64]
    region_emb = concat [128]

Alignment:
    원본 sat_emb (frozen) 기반 cross-city alignment
    → satellite 공간에서 같은 기능 region pair 찾기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregator import AttentionAggregator, MeanAggregator


class DualModalNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["data"]
        m = cfg["model"]

        sat_dim    = d["sat_dim"]                    # 64
        poi_dim    = d["poi_dim"]                    # 64
        shared_dim = min(sat_dim, poi_dim) // 2      # 32
        spec_dim   = min(sat_dim, poi_dim) // 2      # 32

        self.shared_dim     = shared_dim
        self.spec_dim       = spec_dim
        self.region_emb_dim = sat_dim + poi_dim      # 128

        # BG level projection
        self.sat_shared_proj = nn.Linear(sat_dim, shared_dim)
        self.sat_spec_proj   = nn.Linear(sat_dim, spec_dim)
        self.poi_shared_proj = nn.Linear(poi_dim, shared_dim)
        self.poi_spec_proj   = nn.Linear(poi_dim, spec_dim)

        # Aggregator
        agg_mode = m.get("aggregator", "attention")
        heads    = m.get("aggregator_heads", 4)
        if agg_mode == "attention":
            self.sat_agg = AttentionAggregator(sat_dim, heads)
            self.poi_agg = AttentionAggregator(poi_dim, heads)
        else:
            self.sat_agg = MeanAggregator()
            self.poi_agg = MeanAggregator()



        print(f"[Model] sat_dim={sat_dim}, poi_dim={poi_dim}")
        print(f"[Model] shared_dim={shared_dim}, region_emb_dim={self.region_emb_dim}")
        print(f"[Model] aggregator={agg_mode}")


    def forward_region(self,
                       sat_data: torch.Tensor,
                       poi_emb:  torch.Tensor,
                       valid_mask: torch.Tensor,
                       ) -> dict:
        # BG level projection
        sat_shared = self.sat_shared_proj(sat_data)
        sat_spec   = self.sat_spec_proj(sat_data)
        poi_shared = self.poi_shared_proj(poi_emb)
        poi_spec   = self.poi_spec_proj(poi_emb)

        # Reconstruct each modality from disentangled parts, then aggregate
        sat_recon = torch.cat([sat_shared, sat_spec], dim=-1)   # [M, 64]
        poi_recon = torch.cat([poi_shared, poi_spec], dim=-1)   # [M, 64]

        sat_region = self.sat_agg(sat_recon, valid_mask)
        poi_region = self.poi_agg(poi_recon, valid_mask)
        region_emb = torch.cat([sat_region, poi_region], dim=-1)

        return {
            "sat_shared":  sat_shared,
            "sat_spec":    sat_spec,
            "poi_shared":  poi_shared,
            "poi_spec":    poi_spec,
            "valid_mask":  valid_mask,
            "sat_region":  sat_region,
            "poi_region":  poi_region,
            "region_emb":  region_emb,
        }

    def forward(self, batch: list) -> list:
        outputs = []
        device  = next(self.parameters()).device
        region_embs = []

        for sample in batch:
            sat  = sample["sat_data"].to(device)
            poi  = sample["poi_emb"].to(device)
            mask = sample["valid_mask"].to(device)
            out  = self.forward_region(sat, poi, mask)
            out["region_id"] = sample["region_id"]
            out["city"]      = sample["city"]
            out["sat_mean"]  = sample["sat_mean"].to(device)  # [64] frozen
            region_embs.append(out["region_emb"])
            outputs.append(out)

        return outputs

    @torch.no_grad()
    def get_region_embedding(self, sat_data, poi_emb, valid_mask):
        self.eval()
        out = self.forward_region(sat_data, poi_emb, valid_mask)
        return out["region_emb"]
