import math

import torch


def random_rotate_flip(obs, fut):
    """Per-scene random rotation + optional y-flip. obs/fut: (B,T,N,2)."""
    B = obs.shape[0]
    theta = torch.rand(B, device=obs.device) * 2 * math.pi
    c, s = theta.cos(), theta.sin()
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)
    flip = (torch.rand(B, device=obs.device) < 0.5).float() * 2 - 1
    F_ = torch.diag_embed(torch.stack([torch.ones_like(flip), flip], -1))  # (B,2,2)
    M = R @ F_
    return torch.einsum("bij,btnj->btni", M, obs), torch.einsum("bij,btnj->btni", M, fut)
