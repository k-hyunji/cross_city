# models/losses.py
"""
Training losses

Legacy (use_clip: true):
  L_total = λ_contrast · L_clip  +  λ_dis · L_dis  +  λ_proto · L_proto
  CLIPRegionCon — B×B cross-modal on sat_region[64] × poi_region[64].
  Gradient leaks into spec branch (architectural mismatch with BGDisLoss).

New (use_clip_shared: true, use_sep_agg: true):
  L_total = λ_contrast · L_clip_shared
          + λ_dis      · L_dis
          + λ_proto    · L_proto
          + λ_cross    · L_cross_city
          + λ_adv      · L_adv          (optional DANN)

  CLIPSharedCon    — B×B cross-modal on sat_shared_region[32] × poi_shared_region[32].
                     Gradient cannot reach spec branch.
  CrossCitySharedLoss — InfoNCE on region_shared[32], soft combined mining
                        (sat_mean sim + learned poi_shared sim), full in-batch denominator.
  L_adv            — DANN adversarial city loss, computed in model.forward().
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

                sat_sim  = sat_means[idx_i] @ sat_means[idx_j].T
                mask_pos = sat_sim > self.threshold

                if mask_pos.sum() == 0:
                    continue

                emb_i  = region_embs[idx_i]
                emb_j  = region_embs[idx_j]
                logits = emb_i @ emb_j.T / self.temperature

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

        sat = F.normalize(torch.stack([o["sat_region"] for o in outputs]), dim=-1)
        poi = F.normalize(torch.stack([o["poi_region"] for o in outputs]), dim=-1)
        all_embs = torch.cat([sat, poi], dim=0)
        device   = all_embs.device

        pos_mask = torch.zeros(2 * B, 2 * B, device=device)
        for i in range(B):
            pos_mask[i,     B + i] = 1.0
            pos_mask[B + i, i    ] = 1.0

        sat_means = F.normalize(torch.stack([o["sat_mean"] for o in outputs]), dim=-1)
        sat_sim   = sat_means @ sat_means.T
        poi_sim   = poi @ poi.T
        cities    = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if (sat_sim[i, j] > self.sat_threshold and
                        poi_sim[i, j].item() > self.poi_threshold):
                    pos_mask[i,     j    ] = 1.0
                    pos_mask[B + i, B + j] = 1.0
                    pos_mask[i,     B + j] = 1.0
                    pos_mask[B + i, j    ] = 1.0

        logits = all_embs @ all_embs.T / self.temperature
        logits.fill_diagonal_(-1e9)
        log_denom = torch.logsumexp(logits, dim=1)

        loss, count = 0.0, 0
        for i in range(2 * B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0:
                continue
            log_numer = torch.logsumexp(logits[i][pos_idx], dim=0)
            loss += log_denom[i] - log_numer
            count += 1

        return loss / max(count, 1)


# ── Functional Prototype Loss (SwAV-inspired) ────────────────

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
            torch.stack([o["region_emb"] for o in outputs]), dim=-1)
        proto_w = F.normalize(self.prototypes.weight, dim=1)
        scores  = embs @ proto_w.T / self.temperature
        Q       = self._sinkhorn(scores.detach().exp())
        loss    = -(Q * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def _sinkhorn(self, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        B, K = Q.shape
        Q = Q.t()
        Q /= Q.sum() + eps
        for _ in range(self.sinkhorn_iters):
            Q /= Q.sum(dim=1, keepdim=True) + eps
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True) + eps
            Q /= B
        Q *= B
        return Q.t()


# ── CLIP-style Cross-Modal Contrastive Loss (legacy) ─────────
# Operates on sat_region[64] × poi_region[64] — gradient reaches spec branch.
# Superseded by CLIPSharedCon when use_sep_agg=True.

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

        sat = F.normalize(torch.stack([o["sat_region"] for o in outputs]), dim=-1)
        poi = F.normalize(torch.stack([o["poi_region"] for o in outputs]), dim=-1)
        logits = sat @ poi.T / self.temperature

        pos_mask  = torch.eye(B, device=sat.device)
        sat_means = F.normalize(torch.stack([o["sat_mean"] for o in outputs]), dim=-1)
        sat_sim   = sat_means @ sat_means.T
        poi_sim   = poi @ poi.T
        cities    = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if (sat_sim[i, j] > self.sat_threshold and
                        poi_sim[i, j].item() > self.poi_threshold):
                    pos_mask[i, j] = 1.0
                    pos_mask[j, i] = 1.0

        log_denom_row = torch.logsumexp(logits, dim=1)
        log_denom_col = torch.logsumexp(logits, dim=0)

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


# ── CLIPSharedCon: CLIPRegionCon on 32-d shared subspace ─────
# Replaces CLIPRegionCon when use_sep_agg=True.
# Acts on sat_shared_region[32] × poi_shared_region[32] — gradient
# cannot reach sat_spec_agg or sat_spec_proj by construction.
# Pseudo-positives: sat_mean sim only (poi_threshold dropped — was non-functional).

class CLIPSharedCon(nn.Module):
    def __init__(self, temperature: float = 0.07, sat_threshold: float = 0.5):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold

    def forward(self, outputs: list) -> torch.Tensor:
        B = len(outputs)
        if B < 2:
            return outputs[0]["sat_shared_region"].new_zeros(1, requires_grad=True).squeeze()

        sat = F.normalize(torch.stack([o["sat_shared_region"] for o in outputs]), dim=-1)  # [B,32]
        poi = F.normalize(torch.stack([o["poi_shared_region"] for o in outputs]), dim=-1)  # [B,32]
        logits = sat @ poi.T / self.temperature   # [B, B]

        pos_mask  = torch.eye(B, device=sat.device)
        sat_means = F.normalize(torch.stack([o["sat_mean"] for o in outputs]), dim=-1)
        sat_sim   = sat_means @ sat_means.T
        cities    = [o["city"] for o in outputs]

        for i in range(B):
            for j in range(B):
                if i == j or cities[i] == cities[j]:
                    continue
                if sat_sim[i, j] > self.sat_threshold:
                    pos_mask[i, j] = 1.0
                    pos_mask[j, i] = 1.0

        log_denom_row = torch.logsumexp(logits, dim=1)
        log_denom_col = torch.logsumexp(logits, dim=0)

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


# ── CrossCitySharedLoss: InfoNCE on region_shared[32] ────────
# Full in-batch denominator (within + cross-city) forces functional topology:
# "NYC commercial must be closer to Seoul commercial than to NYC residential."
# Positive pairs: cross-city only, sat_mean sim > sat_threshold.
# sat_threshold=0.4 → ~15% positive rate (~38 pairs/batch of 256).
# poi_sim removed: learned poi_shared is near-constant (mean≈-0.02) during
# early training and adds only a constant offset, killing gradient signal.

class CrossCitySharedLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, sat_threshold: float = 0.4,
                 denom_mode: str = "full"):
        super().__init__()
        self.temperature   = temperature
        self.sat_threshold = sat_threshold
        self.denom_mode    = denom_mode  # "full": within+cross, "cross_only": cross-city만

    def forward(self, outputs: list) -> torch.Tensor:
        cities = [o["city"] for o in outputs]
        if len(set(cities)) < 2:
            return outputs[0]["region_shared"].new_zeros(1, requires_grad=True).squeeze()

        B         = len(outputs)
        shared    = F.normalize(torch.stack([o["region_shared"] for o in outputs]), dim=-1)  # [B,32]
        sat_means = F.normalize(torch.stack([o["sat_mean"]      for o in outputs]), dim=-1)  # [B,64]

        sat_sim = sat_means @ sat_means.T   # [B,B] frozen

        logits = shared @ shared.T / self.temperature   # [B,B]
        logits.fill_diagonal_(-1e9)

        loss, count = 0.0, 0

        for i in range(B):
            pos_indices = []
            for j in range(B):
                if j == i or cities[j] == cities[i]:
                    continue
                if sat_sim[i, j].item() > self.sat_threshold:
                    pos_indices.append(j)

            if not pos_indices:
                continue

            pos_t = torch.tensor(pos_indices, device=shared.device)

            if self.denom_mode == "cross_only":
                cross_indices = [j for j in range(B) if j != i and cities[j] != cities[i]]
                denom_t   = torch.tensor(cross_indices, device=shared.device)
                log_denom = torch.logsumexp(logits[i][denom_t], dim=0)
            else:
                log_denom = torch.logsumexp(logits[i], dim=0)

            log_numer  = torch.logsumexp(logits[i][pos_t], dim=0)
            loss      += log_denom - log_numer
            count     += 1

        if count == 0:
            return shared.new_zeros(1, requires_grad=True).squeeze()
        return loss / count


# ── Within-City VICReg Variance Term ─────────────────────────
# 각 도시의 평균을 먼저 제거한 뒤 전체 배치로 분산을 계산.
# → cross-city 공통 방향(over-alignment)은 분산 계산에서 배제됨.
# → Seoul 내부 + NYC 내부 분산을 합쳐서 유지 (32개 샘플 기반 → 안정적).
# std 계산: sqrt(var + ε) 로 붕괴 시 기울기 폭발 방지.

class WithinCityVarLoss(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs: list) -> torch.Tensor:
        city_embs: dict = {}
        for o in outputs:
            city_embs.setdefault(o["city"], []).append(o["region_emb"])

        if len(city_embs) < 1:
            return outputs[0]["region_emb"].new_zeros(1, requires_grad=True).squeeze()

        centered = []
        for embs in city_embs.values():
            e = torch.stack(embs)                    # [n_c, 128]
            centered.append(e - e.mean(0, keepdim=True))  # 도시 평균 제거

        z   = torch.cat(centered, dim=0)             # [B, 128]
        std = (z.var(dim=0) + 1e-6).sqrt()           # [128], ε로 안정화
        loss = F.relu(self.gamma - std).pow(2).mean()
        return loss


# ── Total Loss ───────────────────────────────────────────────

class TotalLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        t = cfg["training"]
        m = cfg.get("model", {})

        self.lw_contrast    = t.get("lambda_contrast",    1.0)
        self.lw_dis         = t.get("lambda_dis",          0.1)
        self.lw_align       = t.get("lambda_align",        1.0)
        self.lw_proto       = t.get("lambda_proto",        0.0)
        self.lw_cross_city  = t.get("lambda_cross_city",  0.5)
        self.lw_adv         = t.get("lambda_adv",          0.1)

        self.use_clip        = t.get("use_clip",         False)
        self.use_unified     = t.get("use_unified_con",  False)
        self.use_proto       = t.get("use_proto",        False)
        self.use_clip_shared = t.get("use_clip_shared",  False)
        self.use_cross_city  = t.get("use_cross_city",   False)
        self.use_adv         = m.get("use_adv",          False)
        self.use_vicreg      = t.get("use_vicreg",       False)
        self.lw_vicreg       = t.get("lambda_vicreg",    1.0)

        self.dis = BGDisLoss()

        if self.use_clip_shared:
            self.clip_shared = CLIPSharedCon(
                temperature=t.get("contrast_temp",    0.07),
                sat_threshold=t.get("align_threshold", 0.5),
            )
        elif self.use_clip:
            self.clip_con = CLIPRegionCon(
                temperature=t.get("contrast_temp",    0.07),
                sat_threshold=t.get("align_threshold", 0.9),
                poi_threshold=t.get("poi_threshold",   0.0),
            )
        elif self.use_unified:
            self.unified = UnifiedCityCon(
                temperature=t.get("contrast_temp",    0.07),
                sat_threshold=t.get("align_threshold", 0.9),
                poi_threshold=t.get("poi_threshold",   0.0),
            )
        else:
            self.contrast = RegionContrastiveLoss(t.get("contrast_temp", 0.07))
            self.align    = SatAlignLoss(
                threshold=t.get("align_threshold", 0.7),
                temperature=t.get("align_temp",    0.1),
            )

        if self.use_cross_city:
            self.cross_city_loss = CrossCitySharedLoss(
                temperature=t.get("cross_city_temp",         0.1),
                sat_threshold=t.get("cross_city_sat_threshold", 0.4),
                denom_mode=t.get("cross_city_denom_mode",    "full"),
            )

        if self.use_vicreg:
            self.vicreg_loss = WithinCityVarLoss(
                gamma=t.get("vicreg_gamma", 1.0),
            )

        if self.use_proto:
            emb_dim = m.get("sat_dim", 64) + m.get("poi_dim", 64)   # 128
            self.proto_loss = FunctionalPrototypeLoss(
                n_prototypes=t.get("n_prototypes",   16),
                region_emb_dim=emb_dim,
                temperature=t.get("proto_temp",      0.1),
                sinkhorn_iters=t.get("sinkhorn_iters", 3),
            )

    def forward(self, outputs: list) -> dict:
        l_dis = self.dis(outputs)
        zero  = l_dis.new_zeros(1).squeeze()

        if self.use_clip_shared:
            l_contrast = self.clip_shared(outputs)
            l_align    = zero
        elif self.use_clip:
            l_contrast = self.clip_con(outputs)
            l_align    = zero
        elif self.use_unified:
            l_contrast = self.unified(outputs)
            l_align    = zero
        else:
            l_contrast = self.contrast(outputs)
            l_align    = self.align(outputs)

        l_proto      = self.proto_loss(outputs)       if self.use_proto      else zero
        l_cross_city = self.cross_city_loss(outputs)  if self.use_cross_city else zero
        l_vicreg     = self.vicreg_loss(outputs)      if self.use_vicreg     else zero

        # DANN loss is pre-computed per region inside model.forward()
        if self.use_adv and "city_adv_loss" in outputs[0]:
            l_adv = torch.stack([o["city_adv_loss"] for o in outputs]).mean()
        else:
            l_adv = zero

        total = (self.lw_contrast   * l_contrast
               + self.lw_dis        * l_dis
               + self.lw_align      * l_align
               + self.lw_proto      * l_proto
               + self.lw_cross_city * l_cross_city
               + self.lw_adv        * l_adv
               + self.lw_vicreg     * l_vicreg)

        return {
            "total":      total,
            "contrast":   l_contrast,
            "dis":        l_dis,
            "align":      l_align,
            "proto":      l_proto,
            "cross_city": l_cross_city,
            "adv":        l_adv,
            "vicreg":     l_vicreg,
        }
