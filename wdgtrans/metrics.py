import torch


@torch.no_grad()
def min_ade_fde(pred, fut, mask, k=None, logits=None):
    """Best-of-K ADE/FDE summed over valid agents.

    If `k` is given the K most probable modes (by `logits`) are used.
    Returns (sum_ade, sum_fde, n_agents).
    """
    err = (pred - fut.unsqueeze(1)).pow(2).sum(-1).sqrt()  # (B,K,T,N)
    ade = err.mean(2).permute(0, 2, 1)  # (B,N,K)
    fde = err[:, :, -1].permute(0, 2, 1)
    if k is not None and logits is not None and k < ade.shape[-1]:
        idx = logits.topk(k, dim=-1).indices
        ade, fde = ade.gather(-1, idx), fde.gather(-1, idx)
    return ade.min(-1).values[mask].sum().item(), fde.min(-1).values[mask].sum().item(), int(mask.sum())


class MetricMeter:
    def __init__(self):
        self.ade = self.fde = 0.0
        self.n = 0

    def update(self, a, f, n):
        self.ade += a; self.fde += f; self.n += n

    def compute(self):
        n = max(self.n, 1)
        return {"minADE": self.ade / n, "minFDE": self.fde / n}
