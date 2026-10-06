"""Train W-DGTrans.

    python scripts/train.py --config configs/synthetic.yaml
    python scripts/train.py --config configs/eth_ucy.yaml --set data.test_set=hotel train.epochs=50
"""
import argparse
import csv
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wdgtrans import WDGTrans, WDGTransConfig
from wdgtrans.data import build_datasets, collate_scenes, random_rotate_flip
from wdgtrans.losses import wta_loss
from wdgtrans.metrics import MetricMeter, min_ade_fde
from wdgtrans.utils import count_params, get_device, load_config, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = MetricMeter()
    for b in loader:
        obs, fut, mask = b["obs"].to(device), b["fut"].to(device), b["mask"].to(device)
        out = model(obs, mask)
        meter.update(*min_ade_fde(out["pred"], fut, mask))
    return meter.compute()


def build_model(cfg):
    d = cfg["data"]
    return WDGTrans(WDGTransConfig(obs_len=d["obs_len"], pred_len=d["pred_len"], **cfg["model"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(cfg["data"])
    tc = cfg["train"]
    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=shuffle, num_workers=tc["num_workers"],
        collate_fn=collate_scenes, drop_last=False)
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)
    print(f"scenes  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    model = build_model(cfg).to(device)
    print(f"parameters: {count_params(model):,}  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm, total = tc["warmup_epochs"], tc["epochs"]
    lr_fn = lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(total - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    log_path = os.path.join(cfg["out_dir"], "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_minADE", "val_minFDE", "lr", "sec"])

    best = float("inf")
    for epoch in range(total):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for b in train_dl:
            obs, fut, mask = b["obs"].to(device), b["fut"].to(device), b["mask"].to(device)
            if tc["augment"]:
                obs, fut = random_rotate_flip(obs, fut)
            out = model(obs, mask)
            loss, _ = wta_loss(out["pred"], out["logits"], fut, mask, **cfg["loss"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            run += loss.item(); nb += 1
        sched.step()
        val = evaluate(model, val_dl, device)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch + 1:3d}/{total}  loss {run / nb:.4f}  "
              f"val minADE {val['minADE']:.3f}  minFDE {val['minFDE']:.3f}  ({time.time() - t0:.1f}s)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, run / nb, val["minADE"], val["minFDE"], lr, time.time() - t0])
        state = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1, "val": val}
        torch.save(state, os.path.join(cfg["out_dir"], "last.pt"))
        if val["minADE"] < best:
            best = val["minADE"]
            torch.save(state, os.path.join(cfg["out_dir"], "best.pt"))

    ckpt = torch.load(os.path.join(cfg["out_dir"], "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_dl, device)
    print(f"TEST (best val ckpt)  minADE {test['minADE']:.3f}  minFDE {test['minFDE']:.3f}")


if __name__ == "__main__":
    main()
