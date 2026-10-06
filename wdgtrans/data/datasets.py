"""Datasets. Each item is one *scene window*: all agents seen for obs_len+pred_len frames.

Items are dicts {"obs": (T_obs, n, 2), "fut": (T_pred, n, 2)}; `collate_scenes`
pads to the largest n in the batch and returns a boolean agent mask.
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


def collate_scenes(batch):
    B = len(batch)
    To, Tp = batch[0]["obs"].shape[0], batch[0]["fut"].shape[0]
    N = max(b["obs"].shape[1] for b in batch)
    obs = torch.zeros(B, To, N, 2)
    fut = torch.zeros(B, Tp, N, 2)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["obs"].shape[1]
        obs[i, :, :n], fut[i, :, :n], mask[i, :n] = b["obs"], b["fut"], True
    return {"obs": obs, "fut": fut, "mask": mask}


class SyntheticTrajectoryDataset(Dataset):
    """Pedestrians walking toward goals with pairwise repulsion. Good for smoke tests."""

    def __init__(self, num_scenes=2000, obs_len=8, pred_len=12, max_agents=8, seed=0, dt=0.4):
        rng = np.random.RandomState(seed)
        T = obs_len + pred_len
        self.obs_len, self.pred_len, self.scenes = obs_len, pred_len, []
        for _ in range(num_scenes):
            n = rng.randint(3, max_agents + 1)
            pos = rng.uniform(-5, 5, (n, 2))
            ang = rng.uniform(0, 2 * np.pi, n)
            speed = rng.uniform(0.8, 1.6, n)
            goal_dir = np.stack([np.cos(ang), np.sin(ang)], 1)
            vel = goal_dir * speed[:, None]
            traj = [pos.copy()]
            for _ in range(T - 1):
                d = pos[:, None] - pos[None]
                dist = np.maximum(np.linalg.norm(d, axis=-1), 0.5) + np.eye(n) * 1e9  # clamp: bounded forces
                rep = (d / dist[..., None] ** 3 * (dist < 2.5)[..., None]).sum(1)
                vel = 0.9 * vel + 0.1 * goal_dir * speed[:, None] + 0.4 * rep * dt
                vel *= np.minimum(1.0, 2.5 / np.maximum(np.linalg.norm(vel, axis=-1, keepdims=True), 1e-6))
                pos = pos + vel * dt
                traj.append(pos.copy())
            self.scenes.append(np.stack(traj).astype(np.float32))  # (T, n, 2)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        s = torch.from_numpy(self.scenes[i])
        return {"obs": s[: self.obs_len], "fut": s[self.obs_len:]}


class ETHUCYDataset(Dataset):
    """Social-GAN style ETH/UCY text files: columns = frame, ped_id, x, y.

    `data_dir` contains *.txt files (e.g. data/eth_ucy/eth/train/*.txt).
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_agents=1):
        seq_len = obs_len + pred_len
        self.obs_len, self.pred_len, self.scenes = obs_len, pred_len, []
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {data_dir}")
        for path in files:
            data = np.loadtxt(path)  # whitespace/tab separated
            frames = np.unique(data[:, 0])
            by_frame = [data[data[:, 0] == f] for f in frames]
            for start in range(0, len(frames) - seq_len + 1, skip):
                window = np.concatenate(by_frame[start:start + seq_len], 0)
                ids, counts = np.unique(window[:, 1], return_counts=True)
                full = ids[counts == seq_len]  # agents present in every frame
                if len(full) < min_agents:
                    continue
                seqs = [window[window[:, 1] == pid][:, 2:4] for pid in full]
                self.scenes.append(np.stack(seqs, 1).astype(np.float32))  # (T, n, 2)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        s = torch.from_numpy(self.scenes[i])
        return {"obs": s[: self.obs_len], "fut": s[self.obs_len:]}


def build_datasets(dcfg):
    """Returns (train, val, test) datasets from the `data` section of a config."""
    To, Tp = dcfg["obs_len"], dcfg["pred_len"]
    if dcfg["name"] == "synthetic":
        kw = dict(obs_len=To, pred_len=Tp, max_agents=dcfg.get("max_agents", 8))
        n = dcfg.get("num_scenes", 2000)
        return (SyntheticTrajectoryDataset(n, seed=0, **kw),
                SyntheticTrajectoryDataset(max(n // 5, 50), seed=1, **kw),
                SyntheticTrajectoryDataset(max(n // 5, 50), seed=2, **kw))
    if dcfg["name"] == "eth_ucy":
        root, test_set = dcfg["root"], dcfg["test_set"]
        train_sets = [s for s in dcfg["all_sets"] if s != test_set]  # leave-one-out
        mk = lambda s, split: ETHUCYDataset(
            os.path.join(root, s, split), To, Tp, dcfg.get("skip", 1))
        train = ConcatDataset([mk(s, "train") for s in train_sets])
        val = ConcatDataset([mk(s, "val") for s in train_sets])
        test = mk(test_set, "test")
        return train, val, test
    raise ValueError(f"Unknown dataset {dcfg['name']}")
