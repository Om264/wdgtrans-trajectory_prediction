"""Plot predictions for one test scene.

    python scripts/visualize.py --ckpt runs/synthetic/best.pt --index 0 --out scene.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import build_model  # noqa: E402
from wdgtrans.data import build_datasets, collate_scenes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="scene.png")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ckpt["cfg"]); model.load_state_dict(ckpt["model"]); model.eval()
    _, _, test_ds = build_datasets(ckpt["cfg"]["data"])
    b = collate_scenes([test_ds[args.index]])
    with torch.no_grad():
        out = model(b["obs"], b["mask"])

    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(int(b["mask"].sum())):
        c = f"C{i % 10}"
        o, f = b["obs"][0, :, i], b["fut"][0, :, i]
        ax.plot(*o.T, "-o", c=c, ms=3, label="observed" if i == 0 else None)
        ax.plot(*torch.cat([o[-1:], f]).T, "--", c=c, alpha=0.8, label="ground truth" if i == 0 else None)
        for k in out["logits"][0, i].topk(min(args.topk, out["pred"].shape[1])).indices:
            p = out["pred"][0, k, :, i]
            ax.plot(*torch.cat([o[-1:], p]).T, ":", c=c, lw=2, label="prediction" if i == 0 and k == out["logits"][0, i].argmax() else None)
    ax.set_aspect("equal"); ax.legend(); ax.set_title("W-DGTrans predictions")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
