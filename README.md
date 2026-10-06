# W-DGTrans — Weighted Dynamic Graph Transformer for Trajectory Prediction

PyTorch implementation of a multi-agent trajectory predictor that combines a
**dynamic, weighted interaction graph** with **spatial + temporal Transformer blocks**
and a **multi-modal (best-of-K) decoder**.

> **Note on the name.** This repo implements "W-DGTrans" as: *W*eighted *D*ynamic *G*raph
> *Trans*former. If your reference paper defines W differently (e.g. wavelet, window),
> the places to change are marked under [Customising](#customising).

## Architecture

```
obs (B,T,N,2) ──► features [pos − last_pos, velocity] ──► MLP embed + temporal pos-emb
      │                                                          │
      └─► DynamicWeightedGraph ──► per-step, per-head edge bias ─┤
                                                                 ▼
                              ┌─ SpatialLayer  (graph-biased attention across agents, each time step)
                  × depth ────┤
                              └─ TemporalLayer (attention across time, each agent)
                                                                 ▼
                                   last-step token ──► MultiModalDecoder
                                                         ├─ K trajectories (cumulative displacements)
                                                         └─ K mode logits
```

**Dynamic weighted graph** (`wdgtrans/models/graph.py`). At every observed step a fully
connected graph is built; the weight of edge *i←j* (per attention head, in log-space) is

```
w_ij(t) = −d_ij(t)² / (2σ²)  +  MLP([p_i − p_j, v_i − v_j, d_ij])
```

a learnable Gaussian distance prior plus a learned correction from relative position and
velocity. It is added to the spatial attention logits, so who-attends-to-whom changes as
agents move. Padded agents are masked out.

**Training** (`wdgtrans/losses.py`): relaxed winner-takes-all regression (ADE + 0.5·FDE of the
best mode, small weight on the others to avoid mode collapse) + cross-entropy on the mode logits.

**Properties verified by tests:** invariance to padding agents, equivariance to agent
permutation, invariance to global translation.

## Install

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Quick start (no data needed)

```bash
python scripts/train.py --config configs/synthetic.yaml
python scripts/evaluate.py --ckpt runs/synthetic/best.pt --k 3
python scripts/visualize.py --ckpt runs/synthetic/best.pt --out scene.png
```

The synthetic set (goal-directed pedestrians with pairwise repulsion) is only a sanity
check that the pipeline learns; it says nothing about real-world performance.

## ETH/UCY

Place the Social-GAN formatted data as

```
data/eth_ucy/<eth|hotel|univ|zara1|zara2>/{train,val,test}/*.txt   # columns: frame, ped_id, x, y
```

(the Social-GAN repo, `agrimgupta92/sgan`, ships a download script for this layout), then run
the standard leave-one-out protocol — 8 observed frames (3.2 s) → 12 predicted frames (4.8 s), best-of-20:

```bash
for s in eth hotel univ zara1 zara2; do
  python scripts/train.py --config configs/eth_ucy.yaml \
      --set data.test_set=$s out_dir=runs/eth_ucy_$s
done
```

Overrides use dotted keys, e.g. `--set model.dim=128 train.epochs=50`.
Evaluate with `--k 20` (or any smaller K to use the K most probable modes).

## Layout

```
configs/            synthetic.yaml, eth_ucy.yaml
wdgtrans/
  models/graph.py       dynamic weighted graph
  models/layers.py      graph attention, spatial/temporal blocks, decoder
  models/wdgtrans.py    full model + config dataclass
  data/datasets.py      synthetic + ETH/UCY datasets, collate with agent masks
  data/transforms.py    random rotation / flip augmentation
  losses.py  metrics.py  utils.py
scripts/            train.py  evaluate.py  visualize.py
tests/              test_model.py
```

## Customising

- **Graph definition** → `DynamicWeightedGraph.forward` (add edge features, sparsify with a radius, etc.).
- **Different "W"** (e.g. wavelet temporal encoder) → replace `TemporalLayer` in `models/layers.py`.
- **Other datasets** (SDD, nuScenes, Argoverse) → add a `Dataset` that returns `{"obs": (T_o,n,2), "fut": (T_p,n,2)}` and register it in `build_datasets`.
- **Scene context / maps** → concatenate a context embedding to the agent tokens before the decoder.

## Related work

Spatial/temporal separated Transformers such as STAR (Yu et al., ECCV 2020) and
best-of-K training from Social-GAN-style benchmarks are the main inspiration for this design.

## License

MIT
