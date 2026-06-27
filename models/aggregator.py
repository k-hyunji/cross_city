# models/aggregator.py
"""
Building group → Region aggregation.
region 하나 [M, dim] → [dim].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        self.pool_query = nn.Parameter(torch.randn(1, 1, dim))
        self.pool_attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self,
                bg_feats: torch.Tensor,
                valid_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        if valid_mask is not None:
            bg_feats = bg_feats[valid_mask]

        M = len(bg_feats)
        if M == 0:
            return bg_feats.new_zeros(self.dim)
        if M == 1:
            return bg_feats.squeeze(0)

        x = bg_feats.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        q = self.pool_query
        pooled, _ = self.pool_attn(q, x, x)
        pooled = self.norm2(pooled)
        out    = self.norm3(pooled + self.ffn(pooled))
        return out.squeeze(0).squeeze(0)


class MeanAggregator(nn.Module):
    def forward(self,
                bg_feats: torch.Tensor,
                valid_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        if valid_mask is not None:
            bg_feats = bg_feats[valid_mask]
        if len(bg_feats) == 0:
            return bg_feats.new_zeros(bg_feats.shape[-1])
        return bg_feats.mean(dim=0)
