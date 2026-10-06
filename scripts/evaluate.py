"""Evaluate a checkpoint on the test split.

    python scripts/evaluate.py --ckpt runs/eth_ucy_eth/best.pt --k 20
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import build_model  # noqa: E402
from wdgtrans.data import build_datasets, collate_scenes  # noqa: E402
from wdgtrans.metrics import MetricMeter, min_ade_fde  # noqa: E402
from wdgtrans.utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=None, help="use the top-k most probable modes")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _, _, test_ds = build_datasets(cfg["data"])
    dl = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_scenes)

    meter = MetricMeter()
    with torch.no_grad():
        for b in dl:
            obs, fut, mask = b["obs"].to(device), b["fut"].to(device), b["mask"].to(device)
            out = model(obs, mask)
            meter.update(*min_ade_fde(out["pred"], fut, mask, k=args.k, logits=out["logits"]))
    r = meter.compute()
    print(f"minADE@{args.k or model.cfg.num_modes}: {r['minADE']:.3f}   "
          f"minFDE@{args.k or model.cfg.num_modes}: {r['minFDE']:.3f}   ({meter.n} agents)")


if __name__ == "__main__":
    main()
