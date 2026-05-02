# banditdl

Hydra-driven experiments for Byzantine-resilient decentralized learning.

## Quick Start

```bash
uv sync
uv run -m banditdl --cfg job
uv run -m banditdl profile=cifar_dynamic
```

## How To Run

Default run (uses `conf/config.yaml` defaults):

```bash
uv run -m banditdl
```

Available built-in profiles:

```bash
uv run -m banditdl profile=cifar_dynamic
uv run -m banditdl profile=mnist_dynamic
uv run -m banditdl profile=cifar_fixed
uv run -m banditdl profile=mnist_fixed
```

Hydra multirun example:

```bash
uv run -m banditdl -m profile=mnist_dynamic profile.nb_neighbors_list='[3,4,5]'
```

## Existing Setups In This Repo

- `cifar_dynamic`: dynamic peer-to-peer CIFAR-10
- `mnist_dynamic`: dynamic peer-to-peer MNIST
- `cifar_fixed`: fixed-graph CIFAR-10 baseline
- `mnist_fixed`: fixed-graph MNIST baseline

Defined in `conf/profile/*.yaml`.

## Hydra Configs and Parameters

Top-level config groups:

- `profile`: experiment definition (`conf/profile/*.yaml`)
- `train`: trainer wiring and plotting selectors (`conf/train/*.yaml`)
- `sweep`: runtime sweep settings (`conf/sweep/*.yaml`)

Commonly used keys:

- `profile.mode`: `dynamic` or `fixed`
- `profile.dataset`: `mnist`, `cifar10`, etc.
- `profile.model`: model name used by training scripts
- `profile.nb_workers`: number of workers
- `profile.alpha`: Dirichlet alpha
- `profile.result_directory`, `profile.plot_directory`
- `profile.byzcounts`: byzantine counts to sweep
- `profile.b_hat_list`: optional per-`byzcounts` override
- `profile.nb_neighbors_list`: neighbor counts to sweep
- `profile.nb_local_steps`: local steps sweep
- `profile.attacks`: attack names
- `profile.method_values`: fixed-graph method sweep values (`null` for none)
- `profile.params_common`: CLI params forwarded to train program
- `train.train_program`: module path of training entry script
- `train.neighbor_sampler`: current sampler selector (`uniform`)
- `train.plot_location`, `train.plot_column`, `train.plot_reduction`
- `sweep.seeds`: repeated seeds
- `sweep.supercharge`: parallel jobs per device
- `devices` (optional top-level override): `auto` or list/string of devices

Inspect effective config:

```bash
uv run -m banditdl --cfg job
```

## How To Create A New Experiment

1. Copy a profile file in `conf/profile/`, e.g. `mnist_dynamic.yaml` -> `mnist_bandit.yaml`.
2. Edit the profile sweep space (`attacks`, `nb_neighbors_list`, `nb_local_steps`, `byzcounts`, `params_common`, etc.).
3. Run it:

```bash
uv run -m banditdl profile=mnist_bandit
```

Optional: add a dedicated train config in `conf/train/` if you need a different training entrypoint.

## Where To Modify Sampling / Bandits

- Sampling implementation: `banditdl/core/sampling.py`
- Training-time sampler wiring: `banditdl/experiments/train_p2p.py`
- Dynamic worker sampler usage: `banditdl/core/training/dynamic/worker.py`

## Notes

- Hydra run outputs: `.hydra_runs/...`
- Hydra multirun outputs: `.hydra_multirun/...`
