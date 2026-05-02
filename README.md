# banditdl

Hydra-first decentralized learning experiments under Byzantine attacks.

## Setup

```bash
uv sync
```

## Run (Hydra)

```bash
uv run -m banditdl
```

Pick a profile:

```bash
uv run -m banditdl profile=cifar_dynamic
uv run -m banditdl profile=mnist_dynamic
uv run -m banditdl profile=cifar_fixed
uv run -m banditdl profile=mnist_fixed
```

Override parameters (no code edits):

```bash
uv run -m banditdl profile=cifar_dynamic sweep.supercharge=4 sweep.seeds=[0,1,2]
uv run -m banditdl profile=mnist_dynamic profile.nb_neighbors_list=[3,5,7] profile.attacks=[ALIE,SF]
```

Hydra multirun:

```bash
uv run -m banditdl -m profile=mnist_dynamic profile.nb_neighbors_list='[3,4,5]'
```

## Config Layout

- `conf/profile/*.yaml`: experiment presets
- `conf/train/*.yaml`: dynamic vs fixed training entry
- `conf/sweep/*.yaml`: runtime knobs (seeds, supercharge)

## Bandit Extension Surface

- Add samplers in `banditdl/sampling/`
- Wire selection in `banditdl/experiments/train_p2p.py`
- Sampling call site is `banditdl/core/training/dynamic/worker.py`
