import torch
import torch.nn.functional as F


def wta_loss(pred, logits, fut, mask, relax=0.05, fde_weight=0.5, cls_weight=0.1):
    """Relaxed winner-takes-all regression + mode classification.

    pred (B,K,T,N,2), logits (B,N,K), fut (B,T,N,2), mask (B,N) bool.
    """
    B, K, T, N, _ = pred.shape
    err = ((pred - fut.unsqueeze(1)).pow(2).sum(-1) + 1e-6).sqrt()  # (B,K,T,N)
    cost = err.mean(2) + fde_weight * err[:, :, -1]  # (B,K,N)
    best, best_k = cost.min(1)  # (B,N)
    reg = (1 - relax) * best + relax * cost.mean(1)
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    reg = (reg * m).sum() / denom
    ce = F.cross_entropy(logits.reshape(-1, K), best_k.reshape(-1), reduction="none").view(B, N)
    cls = (ce * m).sum() / denom
    return reg + cls_weight * cls, {"reg": reg.item(), "cls": cls.item()}
