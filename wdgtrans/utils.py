import copy
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_dotted(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(path, overrides=None):
    """Load YAML and apply overrides such as ['train.epochs=5', 'model.dim=128']."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, val = item.split("=", 1)
        _set_dotted(cfg, key, yaml.safe_load(val))
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
