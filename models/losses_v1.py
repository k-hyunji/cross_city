# models/losses.py
"""
Training losses

Default (use_unified_con: false):
  L_total = λ_contrast · L_contrast  +  λ_dis · L_dis  +  λ_align · L_align

Fix 2 (use_unified_con: true):
  L_total = λ_contrast · L_unified   +  λ_dis · L_dis
  UnifiedCityCon merges L_contrast and L_align into a single SupCon loss.
  Cross-city pseudo-positives are positives — never negatives.

CLIP-like (use_clip: true):
  L_total = λ_contrast · L_clip  +  λ_dis · L_dis  +  λ_proto · L_proto
  CLIPRegionCon uses a B×B cross-modal similarity matrix (sat rows × poi cols).
  Within-modality pairs are never explicit negatives — only cross-modal pairs.
  Cross-city pseudo-positives: sat_i ↔ poi_j when dual-signal thresholds met.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Region level NT-Xent (symmetric) ────────────────────────

class RegionContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs: list) -> torch.Tensor:
        sat_list = [F.normalize(out["sat_region"], dim=-1) for out in outputs]
        poi_list = [F.normalize(out["poi_region"], dim=-1) for out in outputs]
        sat = torch.stack(sat_list, dim=0)
        poi = torch.stack(poi_list, dim=0)
        B = sat.size(0)
        if B < 2:
            return sat.new_zeros(1, requires_grad=True).squeeze()
        labels = torch.arange(B, device=sat.device)
        sim_sp = sat @ poi.T / self.temperature
        sim_ps = poi @ sat.T / self.temperature
        loss = (F.cross_entropy(sim_sp, labels) + F.cross_entropy(sim_ps, labels)) / 2
        return loss


# ── BG level L_dis ───────────────────────────────────────────

class BGDisLoss(nn.Module):
    def forward(self, outputs: list) -> torch.Tensor:
        sat_cos_list, poi_cos_list = [], []
        for out in outputs:
            mask = out["valid_mask"]
            if mask.sum() == 0:
                continue
            sat_cos = F.cosine_similarity(
                out["sat_spec"][mask],
                out["sat_shared"][mask].detach(), dim=-1).abs()
            poi_cos = F.cosine_similarity(
                out["poi_spec"][mask],
                out["poi_shared"][mask].detach(), dim=-1).abs()
            sat_cos_list.append(sat_cos)
            poi_cos_list.append(poi_cos)
        if not sat_cos_list:
            return outputs[0]["sat_shared"].new_zeros(1, requires_grad=True).squeeze()
        return (torch.cat(sat_cos_list).mean() + torch.cat(poi_cos_list).mean()) * 0.5


# ── Satellite-based Cross-city Alignment Loss (InfoNCE) ──────
# Fix 1: replaces attraction-only cosine loss with InfoNCE.
# Gradient scale rises from ~0.25 to log-scale (1–4), matching
# L_contrast. SupCon-style numerator handles multiple positives.

class SatAlignLoss(nn.Module):
    def __init__(self, threshold: float = 0.7, temperature: float = 0.1):
        super().__init__()
        self.threshold   = threshold
        self.temperature = temperature

    def forward(self, outputs: list) -> torch.Tensor:
        city_regions: dict = {}
        for i, out in enumerate(outputs):
            city_regions.setdefault(out["city"], []).append(i)

        cities = list(city_regions.keys())
        if len(cities) < 2:
            return outputs[0]["region_emb"].new_zeros(1, requires_grad=True).squeeze()

        sat_means   = F.normalize(torch.stack([o["sat_mean"]   for o in outputs]), dim=-1)
        region_embs = F.normalize(torch.stack([o["region_emb"] for o in outputs]), dim=-1)

        loss_list = []
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                idx_i = city_regions[cities[i]]
                idx_j = city_regions[cities[j]]

                sat_sim  = sat_means[idx_i] @ sat_means[idx_j].T   # [N_i, N_j]
                mask_pos = sat_sim > self.threshold

                if mask_pos.sum() == 0:
                    continue

                emb_i  = region_embs[idx_i]                        # [N_i, 128]
                emb_j  = region_embs[idx_j]                        # [N_j, 128]
                logits = emb_i @ emb_j.T / self.temperature        # [N_i, N_j]

                for row in range(len(idx_i)):
                    pos_cols = mask_pos[row].nonzero(as_tuple=True)[0]
                    if len(pos_cols) == 0:
                        continue
                    log_denom  = torch.logsumexp(logits[row], dim=0)
                    log_numer  = torch.logsumexp(logits[row][pos_cols], dim=0)
                    loss_list.append(log_denom - log_numer)

        if not loss_list:
            return outputs[0]["region_emb"].new_zeros(1, requires_grad=True).squeeze()
        return torch.stack(loss_list).mean()


# ── Fix 2 + Fix 4: Unified Multi-Positive Contrastive Loss ───
# Fix 2: merges L_contrast and L_align — cross-city pseudo-positives
#         are never in each other's negative denominator.
# Fix 4: dual-signal mining — requires sat_mean sim > sat_threshold
#         AND learned poi_region sim > poi_threshold to be positive.
#         POI distributions are a stronger functional discriminator
#         than satellite appearance alone (r=+0.016 vs random for sat-only).

class UnifiedCityCon(nn.Module):
    def __init__(self, temperature: float = 0.07,
                 sat_threshold: float = 0.9,
                 poi_threshold: float = 0.5):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold
        self.poi_threshold = poi_threshold

    def forward(self, outputs: list) -> torch.Tensor:
        B = len(outputs)
        if B < 2:
            return outputs[0]["sat_region"].new_zeros(1, requires_grad=True).squeeze()

        sat = F.normalize(torch.stack([o["sat_region"] for o in outputs]), dim=-1)  # [B, 64]
        poi = F.normalize(torch.stack([o["poi_region"] for o in outputs]), dim=-1)  # [B, 64]
        all_embs = torch.cat([sat, poi], dim=0)  # [2B, 64]
        device   = all_embs.device

        pos_mask = torch.zeros(2 * B, 2 * B, device=device)

        # (a) cross-modal same-region positives
        for i in range(B):
            pos_mask[i,     B + i] = 1.0
            pos_mask[B + i, i    ] = 1.0

        # (b) Fix 4 dual-signal cross-city pseudo-positives:
        #     sat_mean sim (frozen) AND poi_region sim (learned) must both exceed threshold
        sat_means = F.normalize(
            torch.stack([o["sat_mean"] for o in outputs]), dim=-1)   # [B, 64]
        sat_sim = sat_means @ sat_means.T                             # [B, B]
        poi_sim = poi @ poi.T                                         # [B, B]  (detached from pos_mask construction)
        cities  = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if (sat_sim[i, j] > self.sat_threshold and
                        poi_sim[i, j].item() > self.poi_threshold):
                    pos_mask[i,     j    ] = 1.0  # sat_i ↔ sat_j
                    pos_mask[B + i, B + j] = 1.0  # poi_i ↔ poi_j
                    pos_mask[i,     B + j] = 1.0  # sat_i ↔ poi_j
                    pos_mask[B + i, j    ] = 1.0  # poi_i ↔ sat_j

        logits = all_embs @ all_embs.T / self.temperature  # [2B, 2B]
        logits.fill_diagonal_(-1e9)

        log_denom = torch.logsumexp(logits, dim=1)  # [2B]

        loss, count = 0.0, 0
        for i in range(2 * B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            log_numer = torch.logsumexp(logits[i][pos_idx], dim=0)
            loss += log_denom[i] - log_numer
            count += 1

        return loss / max(count, 1)


# ── Fix 8: Functional Prototype Loss (SwAV-inspired) ─────────
# K learnable prototypes shared across all cities represent functional
# archetypes (residential, commercial, industrial, park, …).
# Sinkhorn-Knopp balanced assignment prevents prototype collapse —
# every prototype must be used by a fair share of regions.
# Because prototypes are city-agnostic, city identity gives no advantage;
# the model must map regions by function, not origin.

class FunctionalPrototypeLoss(nn.Module):
    def __init__(self, n_prototypes: int = 16,
                 region_emb_dim: int = 128,
                 temperature: float = 0.1,
                 sinkhorn_iters: int = 3):
        super().__init__()
        self.prototypes     = nn.Linear(region_emb_dim, n_prototypes, bias=False)
        self.temperature    = temperature
        self.sinkhorn_iters = sinkhorn_iters
        nn.init.normal_(self.prototypes.weight)

    def forward(self, outputs: list) -> torch.Tensor:
        embs    = F.normalize(
            torch.stack([o["region_emb"] for o in outputs]), dim=-1)  # [B, 128]
        proto_w = F.normalize(self.prototypes.weight, dim=1)          # [K, 128]  grad flows here
        scores  = embs @ proto_w.T / self.temperature                 # [B, K]
        Q       = self._sinkhorn(scores.detach().exp())               # [B, K]  no grad (target)
        loss    = -(Q * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def _sinkhorn(self, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """SwAV-style Sinkhorn-Knopp. Input Q: [B, K]. Returns [B, K] with rows summing to 1."""
        B, K = Q.shape
        Q = Q.t()                                               # [K, B]
        Q /= Q.sum() + eps                                      # global normalize
        for _ in range(self.sinkhorn_iters):
            Q /= Q.sum(dim=1, keepdim=True) + eps               # uniform over regions per prototype
            Q /= K                                              # each prototype row sums to 1/K
            Q /= Q.sum(dim=0, keepdim=True) + eps               # uniform over prototypes per region
            Q /= B                                              # each sample col sums to 1/B
        Q *= B                                                  # rescale: each col sums to 1
        return Q.t()                                            # [B, K], rows sum to 1


# ── CLIP-style Cross-Modal Contrastive Loss ──────────────────
# B×B similarity matrix: sat (rows) × poi (cols).
# Positive mask:
#   (a) diagonal: sat_i ↔ poi_i  (same region, cross-modal)
#   (b) cross-city pseudo-positives: sat_i ↔ poi_j when
#       sat_mean_i · sat_mean_j > sat_threshold  AND
#       poi_region_i · poi_region_j > poi_threshold
# Both directions (sat→poi and poi→sat) are computed symmetrically.
# Within-modality pairs (sat_i vs sat_j) are never explicit negatives,
# so the spec branch is shielded from cross-city alignment pressure.

class CLIPRegionCon(nn.Module):
    def __init__(self, temperature: float = 0.07,
                 sat_threshold: float = 0.9,
                 poi_threshold: float = 0.5):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold
        self.poi_threshold = poi_threshold

    def forward(self, outputs: list) -> torch.Tensor:
        B = len(outputs)
        if B < 2:
            return outputs[0]["sat_region"].new_zeros(1, requires_grad=True).squeeze()

        sat = F.normalize(torch.stack([o["sat_region"] for o in outputs]), dim=-1)  # [B, 64]
        poi = F.normalize(torch.stack([o["poi_region"] for o in outputs]), dim=-1)  # [B, 64]

        logits = sat @ poi.T / self.temperature  # [B, B]

        # (a) same-region diagonal positives
        pos_mask = torch.eye(B, device=sat.device)

        # (b) dual-signal cross-city pseudo-positives
        sat_means = F.normalize(
            torch.stack([o["sat_mean"] for o in outputs]), dim=-1)  # [B, 64]
        sat_sim = sat_means @ sat_means.T   # [B, B]  frozen proxy
        poi_sim = poi @ poi.T               # [B, B]  learned similarity
        cities  = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if (sat_sim[i, j] > self.sat_threshold and
                        poi_sim[i, j].item() > self.poi_threshold):
                    pos_mask[i, j] = 1.0  # sat_i → poi_j
                    pos_mask[j, i] = 1.0  # sat_j → poi_i

        # SupCon: sat_i → poi direction
        log_denom_row = torch.logsumexp(logits, dim=1)  # [B]
        # SupCon: poi_j → sat direction
        log_denom_col = torch.logsumexp(logits, dim=0)  # [B]

        loss, count = 0.0, 0
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            loss += log_denom_row[i] - torch.logsumexp(logits[i, pos_idx], dim=0)
            count += 1

        for j in range(B):
            pos_idx = pos_mask[:, j].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            loss += log_denom_col[j] - torch.logsumexp(logits[pos_idx, j], dim=0)
            count += 1

        return loss / max(count, 1)


# ── Total Loss ───────────────────────────────────────────────

class TotalLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        t = cfg["training"]
        self.lw_contrast = t.get("lambda_contrast", 1.0)
        self.lw_dis      = t.get("lambda_dis",       0.1)
        self.lw_align    = t.get("lambda_align",     1.0)
        self.lw_proto    = t.get("lambda_proto",     0.0)
        self.use_clip    = t.get("use_clip",         False)
        self.use_unified = t.get("use_unified_con",  False)
        self.use_proto   = t.get("use_proto",        False)

        self.dis = BGDisLoss()

        if self.use_clip:
            self.clip_con = CLIPRegionCon(
                temperature=t.get("contrast_temp",    0.07),
                sat_threshold=t.get("align_threshold", 0.9),
                poi_threshold=t.get("poi_threshold",   0.5),
            )
        elif self.use_unified:
            self.unified = UnifiedCityCon(
                temperature=t.get("contrast_temp",    0.07),
                sat_threshold=t.get("align_threshold", 0.9),
                poi_threshold=t.get("poi_threshold",   0.5),
            )
        else:
            self.contrast = RegionContrastiveLoss(t.get("contrast_temp", 0.07))
            self.align    = SatAlignLoss(
                threshold=t.get("align_threshold", 0.7),
                temperature=t.get("align_temp",    0.1),
            )

        if self.use_proto:
            m = cfg.get("model", {})
            emb_dim = m.get("sat_dim", 64) + m.get("poi_dim", 64)  # 128
            self.proto_loss = FunctionalPrototypeLoss(
                n_prototypes=t.get("n_prototypes",  16),
                region_emb_dim=emb_dim,
                temperature=t.get("proto_temp",     0.1),
                sinkhorn_iters=t.get("sinkhorn_iters", 3),
            )

    def forward(self, outputs: list) -> dict:
        l_dis = self.dis(outputs)
        zero  = l_dis.new_zeros(1).squeeze()

        if self.use_clip:
            l_contrast = self.clip_con(outputs)
            l_align    = zero
        elif self.use_unified:
            l_contrast = self.unified(outputs)
            l_align    = zero
        else:
            l_contrast = self.contrast(outputs)
            l_align    = self.align(outputs)

        l_proto = self.proto_loss(outputs) if self.use_proto else zero

        total = (self.lw_contrast * l_contrast
               + self.lw_dis      * l_dis
               + self.lw_align    * l_align
               + self.lw_proto    * l_proto)

        return {
            "total":    total,
            "contrast": l_contrast,
            "dis":      l_dis,
            "align":    l_align,
            "proto":    l_proto,
        }
