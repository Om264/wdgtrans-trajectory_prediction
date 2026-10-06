import torch
import torch.nn as nn


class GraphAttention(nn.Module):
    """Multi-head self-attention over agents, biased by the dynamic graph."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, blocked):
        # x: (B,T,N,D)  bias: (B,T,N,N,H)  blocked: (B,N,N) bool, True = cannot attend
        B, T, N, D = x.shape
        q, k, v = self.qkv(x).view(B, T, N, 3, self.h, self.dh).unbind(3)
        q, k, v = (t.permute(0, 1, 3, 2, 4) for t in (q, k, v))  # (B,T,H,N,dh)
        logits = q @ k.transpose(-1, -2) / self.dh ** 0.5  # (B,T,H,N,N)
        logits = logits + bias.permute(0, 1, 4, 2, 3)
        logits = logits.masked_fill(blocked[:, None, None], float("-inf"))
        attn = self.drop(logits.softmax(-1))
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, T, N, D)
        return self.proj(out)


def _ffn(dim, mult, dropout):
    return nn.Sequential(
        nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(dim * mult, dim), nn.Dropout(dropout),
    )


class SpatialLayer(nn.Module):
    """Pre-norm transformer block: attention across agents at each time step."""

    def __init__(self, dim, heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GraphAttention(dim, heads, dropout)
        self.ff = _ffn(dim, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, blocked):
        x = x + self.drop(self.attn(self.n1(x), bias, blocked))
        return x + self.ff(self.n2(x))


class TemporalLayer(nn.Module):
    """Pre-norm transformer block: attention across time for each agent."""

    def __init__(self, dim, heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = _ffn(dim, ff_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        y = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h = self.n1(y)
        y = y + self.drop(self.attn(h, h, h, need_weights=False)[0])
        y = y + self.ff(self.n2(y))
        return y.view(B, N, T, D).permute(0, 2, 1, 3)


class MultiModalDecoder(nn.Module):
    """Predicts K future displacement sequences and a probability logit per mode."""

    def __init__(self, dim, pred_len, num_modes):
        super().__init__()
        self.k, self.t = num_modes, pred_len
        self.traj = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, num_modes * pred_len * 2)
        )
        self.cls = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_modes))

    def forward(self, h):
        B, N, _ = h.shape
        offsets = self.traj(h).view(B, N, self.k, self.t, 2)
        return offsets, self.cls(h)
