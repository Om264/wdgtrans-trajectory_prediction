from dataclasses import dataclass

import torch
import torch.nn as nn

from .graph import DynamicWeightedGraph, velocities
from .layers import MultiModalDecoder, SpatialLayer, TemporalLayer


@dataclass
class WDGTransConfig:
    obs_len: int = 8
    pred_len: int = 12
    dim: int = 64
    heads: int = 4
    depth: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    num_modes: int = 20
    edge_hidden: int = 32
    init_sigma: float = 2.0


class WDGTrans(nn.Module):
    """Weighted Dynamic Graph Transformer.

    obs:  (B, T_obs, N, 2) absolute positions of N (padded) agents
    mask: (B, N) bool, True for real agents

    Returns dict with
        pred:   (B, K, T_pred, N, 2) absolute future positions per mode
        logits: (B, N, K)            mode scores
        adj:    (B, T_obs, N, N)     dynamic distance-prior graph (for inspection)
    """

    def __init__(self, cfg: WDGTransConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.embed = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.time_pos = nn.Parameter(torch.randn(1, cfg.obs_len, 1, d) * 0.02)
        self.graph = DynamicWeightedGraph(cfg.heads, cfg.edge_hidden, cfg.init_sigma)
        self.spatial = nn.ModuleList(
            [SpatialLayer(d, cfg.heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.temporal = nn.ModuleList(
            [TemporalLayer(d, cfg.heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.decoder = MultiModalDecoder(d, cfg.pred_len, cfg.num_modes)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        B, T, N, _ = obs.shape
        last = obs[:, -1]  # (B,N,2)
        feats = torch.cat([obs - last.unsqueeze(1), velocities(obs)], dim=-1)  # translation-invariant
        x = self.embed(feats) + self.time_pos

        bias, adj = self.graph(obs)
        eye = torch.eye(N, dtype=torch.bool, device=obs.device)
        blocked = ~(mask.unsqueeze(1) | eye.unsqueeze(0))  # padded keys hidden; self always visible

        for sp, tp in zip(self.spatial, self.temporal):
            x = sp(x, bias, blocked)
            x = tp(x)

        h = self.norm(x[:, -1])  # last-step token has attended over the whole window
        offsets, logits = self.decoder(h)  # (B,N,K,Tp,2), (B,N,K)
        pred = last[:, :, None, None] + offsets.cumsum(3)
        return {"pred": pred.permute(0, 2, 3, 1, 4), "logits": logits, "adj": adj}
