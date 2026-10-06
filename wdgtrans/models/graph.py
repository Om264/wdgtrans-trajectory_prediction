"""Dynamic weighted interaction graph.

For every observed time step t a fully-connected graph over the agents is
built. Each directed edge (i <- j) receives a *weight* in log-space, per
attention head:

    bias_ij^t = -d_ij^2 / (2 sigma^2)  +  MLP([p_i - p_j, v_i - v_j, d_ij])

The first term is a learnable Gaussian distance prior (nearby agents matter
more); the second is a learned correction that can encode approach speed,
relative bearing, etc. The bias is added to the attention logits of the
spatial transformer, so the graph changes at every time step as agents move.
"""
import math

import torch
import torch.nn as nn


def velocities(pos: torch.Tensor) -> torch.Tensor:
    """Finite-difference velocities. pos: (B, T, N, 2) -> same shape (v_0 = 0)."""
    vel = torch.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    return vel


class DynamicWeightedGraph(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int = 32, init_sigma: float = 2.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_heads)
        )
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))

    def forward(self, pos: torch.Tensor):
        """pos: (B, T, N, 2) absolute positions.

        Returns
            bias: (B, T, N, N, H) additive attention bias (log edge weights)
            adj:  (B, T, N, N)    interpretable distance-prior weights in (0, 1]
        """
        diff = pos.unsqueeze(3) - pos.unsqueeze(2)  # (B,T,N,N,2): p_i - p_j
        vel = velocities(pos)
        dvel = vel.unsqueeze(3) - vel.unsqueeze(2)
        dist2 = diff.pow(2).sum(-1)
        dist = (dist2 + 1e-6).sqrt()  # safe sqrt (diagonal has zero distance)
        feat = torch.cat([diff, dvel, dist.unsqueeze(-1)], dim=-1)
        prior = -dist2 / (2.0 * self.log_sigma.exp() ** 2)
        bias = prior.unsqueeze(-1) + self.edge_mlp(feat)
        return bias, prior.exp()
