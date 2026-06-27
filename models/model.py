# models/model.py
"""
DualModalNet — level-specific architecture with shared/specific loss routing.

BG level:
    projection → sat_shared[M,32] / sat_spec[M,32]
               → poi_shared[M,32] / poi_spec[M,32]

Region level (use_sep_agg=True):
    4 separate aggregators → sat_shared_region[32], sat_spec_region[32],
                              poi_shared_region[32], poi_spec_region[32]
    shared_fusion MLP → region_shared[32]   (used by CLIPSharedCon, CrossCitySharedLoss)
    sat_region[64] = cat([sat_shared_region, sat_spec_region])
    poi_region[64] = cat([poi_shared_region, poi_spec_region])
    region_emb[128] = cat([sat_region, poi_region])  — shape unchanged

DANN (use_adv=True):
    GradReverse(region_shared) → city_classifier → city_adv_loss
    Penalizes region_shared for being city-discriminative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregator import AttentionAggregator, MeanAggregator


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (DANN)
# ---------------------------------------------------------------------------

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


# ---------------------------------------------------------------------------
# DualModalNet
# ---------------------------------------------------------------------------

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

    def forward_region(self,
                       sat_data:   torch.Tensor,
                       poi_emb:    torch.Tensor,
                       valid_mask: torch.Tensor,
                       city_idx:   int = 0,
                       lam_adv:    float = 0.0,
                       ) -> dict:
        # BG level projection
        sat_shared = self.sat_shared_proj(sat_data)   # [M, 32]
        sat_spec   = self.sat_spec_proj(sat_data)     # [M, 32]
        poi_shared = self.poi_shared_proj(poi_emb)    # [M, 32]
        poi_spec   = self.poi_spec_proj(poi_emb)      # [M, 32]

        if self.use_sep_agg:
            sat_shared_region = self.sat_shared_agg(sat_shared, valid_mask)  # [32]
            sat_spec_region   = self.sat_spec_agg(sat_spec,     valid_mask)  # [32]
            poi_shared_region = self.poi_shared_agg(poi_shared,  valid_mask) # [32]
            poi_spec_region   = self.poi_spec_agg(poi_spec,      valid_mask) # [32]

            # MLP fusion — learns modality weighting instead of fixed average
            region_shared = self.shared_fusion(
                torch.cat([sat_shared_region, poi_shared_region], dim=-1)
            )                                                                 # [32]

            sat_region = torch.cat([sat_shared_region, sat_spec_region], dim=-1)  # [64]
            poi_region = torch.cat([poi_shared_region, poi_spec_region], dim=-1)  # [64]
            region_emb = torch.cat([sat_region, poi_region], dim=-1)              # [128]

            out = {
                "sat_shared":        sat_shared,
                "sat_spec":          sat_spec,
                "poi_shared":        poi_shared,
                "poi_spec":          poi_spec,
                "valid_mask":        valid_mask,
                "sat_shared_region": sat_shared_region,
                "sat_spec_region":   sat_spec_region,
                "poi_shared_region": poi_shared_region,
                "poi_spec_region":   poi_spec_region,
                "region_shared":     region_shared,
                "sat_region":        sat_region,
                "poi_region":        poi_region,
                "region_emb":        region_emb,
            }

            # DANN: adversarial city classifier on gradient-reversed region_shared
            if self.use_adv and self.training:
                reversed_shared  = GradReverse.apply(region_shared, lam_adv)
                city_pred        = self.city_classifier(reversed_shared)   # [n_cities]
                label            = torch.tensor([city_idx], device=sat_data.device)
                out["city_adv_loss"] = F.cross_entropy(
                    city_pred.unsqueeze(0), label
                )

            return out

        else:
            sat_recon  = torch.cat([sat_shared, sat_spec], dim=-1)   # [M, 64]
            poi_recon  = torch.cat([poi_shared, poi_spec], dim=-1)   # [M, 64]
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

    def forward(self, batch: list, lam_adv: float = 0.0) -> list:
        outputs = []
        device  = next(self.parameters()).device

        for sample in batch:
            sat      = sample["sat_data"].to(device)
            poi      = sample["poi_emb"].to(device)
            mask     = sample["valid_mask"].to(device)
            city_idx = sample.get("city_idx", 0)

            out = self.forward_region(sat, poi, mask,
                                      city_idx=city_idx, lam_adv=lam_adv)
            out["region_id"] = sample["region_id"]
            out["city"]      = sample["city"]
            out["city_idx"]  = city_idx
            out["sat_mean"]  = sample["sat_mean"].to(device)   # [64] frozen
            outputs.append(out)

        return outputs

    @torch.no_grad()
    def get_region_embedding(self, sat_data, poi_emb, valid_mask):
        self.eval()
        out = self.forward_region(sat_data, poi_emb, valid_mask)
        return out["region_emb"]
