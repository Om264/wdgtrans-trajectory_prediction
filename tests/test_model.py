import torch

from wdgtrans import WDGTrans, WDGTransConfig
from wdgtrans.data import SyntheticTrajectoryDataset, collate_scenes
from wdgtrans.losses import wta_loss
from wdgtrans.metrics import min_ade_fde


def make(modes=4):
    cfg = WDGTransConfig(obs_len=8, pred_len=12, dim=32, heads=4, depth=2, num_modes=modes, dropout=0.0)
    return WDGTrans(cfg).eval()


def batch(n_scenes=4):
    ds = SyntheticTrajectoryDataset(num_scenes=n_scenes, max_agents=6)
    return collate_scenes([ds[i] for i in range(n_scenes)])


def test_shapes():
    b, m = batch(), make()
    out = m(b["obs"], b["mask"])
    B, N = b["mask"].shape
    assert out["pred"].shape == (B, 4, 12, N, 2)
    assert out["logits"].shape == (B, N, 4)
    assert out["adj"].shape == (B, 8, N, N)
    assert torch.isfinite(out["pred"]).all()


def test_padding_invariance():
    b, m = batch(), make()
    ref = m(b["obs"], b["mask"])["pred"]
    obs2 = torch.cat([b["obs"], torch.randn(b["obs"].shape[0], 8, 3, 2) * 50], 2)
    mask2 = torch.cat([b["mask"], torch.zeros(b["mask"].shape[0], 3, dtype=torch.bool)], 1)
    out2 = m(obs2, mask2)["pred"][:, :, :, : b["mask"].shape[1]]
    assert torch.allclose(ref[:, :, :, :][b["mask"][:, None, None, :, None].expand_as(ref)],
                          out2[b["mask"][:, None, None, :, None].expand_as(out2)], atol=1e-4)


def test_permutation_equivariance():
    b, m = batch(), make()
    N = b["mask"].shape[1]
    perm = torch.randperm(N)
    a = m(b["obs"], b["mask"])["pred"]
    p = m(b["obs"][:, :, perm], b["mask"][:, perm])["pred"]
    assert torch.allclose(a[:, :, :, perm], p, atol=1e-4)


def test_translation_invariance():
    b, m = batch(), make()
    a = m(b["obs"], b["mask"])["pred"]
    shift = torch.tensor([10.0, -7.0])
    p = m(b["obs"] + shift, b["mask"])["pred"]
    assert torch.allclose(a + shift, p, atol=1e-3)


def test_loss_and_backward():
    b, m = batch(), make()
    m.train()
    out = m(b["obs"], b["mask"])
    loss, _ = wta_loss(out["pred"], out["logits"], b["fut"], b["mask"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_metrics_perfect_prediction():
    b = batch()
    pred = b["fut"].unsqueeze(1).repeat(1, 3, 1, 1, 1)
    a, f, n = min_ade_fde(pred, b["fut"], b["mask"])
    assert a < 1e-5 and f < 1e-5 and n == int(b["mask"].sum())
