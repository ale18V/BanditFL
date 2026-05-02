# banditdl

Hydra-multirun experiments for Byzantine-resilient decentralized learning.

## Setup

```bash
uv sync
```

If `uv` cache is not writable in your environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Run One Experiment

```bash
uv run -m banditdl
```

Example overrides:

```bash
uv run -m banditdl profile=mnist_dynamic profile.nb_neighbors=5 profile.byzcount=1 seed=0
```

## Run Sweeps (Hydra Multirun)

Hydra does the orchestration. No custom in-repo scheduler is used by the main entrypoint.

### Option A: Preset matrix from profile

Each profile contains its own `hydra.sweeper.params` matrix.

```bash
uv run -m banditdl -m profile=cifar_dynamic
uv run -m banditdl -m profile=mnist_dynamic
uv run -m banditdl -m profile=cifar_fixed
uv run -m banditdl -m profile=mnist_fixed
```

### Option B: Ad-hoc sweep from CLI

```bash
uv run -m banditdl -m \
  profile=mnist_dynamic \
  seed=0,1 \
  profile.nb_neighbors=3,5 \
  profile.attack=ALIE,SF \
  profile.nb_local_steps=1,3
```

## Existing Profiles

- `cifar_dynamic`
- `mnist_dynamic`
- `cifar_fixed`
- `mnist_fixed`

## Hydra Parameters Handled

Top-level:
- `profile` (config group)
- `train` (config group)
- `seed`
- `device`

`profile` fields:
- `mode`: `dynamic` or `fixed`
- `dataset`, `model`, `nb_workers`, `alpha`
- `result_directory`, `plot_directory`
- `byzcount`, `b_hat`
- `nb_neighbors`, `nb_local_steps`
- `attack`, `method`
- `params_common` (forwarded as CLI args to training runner)
- `hydra.sweeper.params` (preset sweep matrix)

`train` fields:
- `train_program`: module path (`train_p2p` or `fx_train_p2p`)
- `neighbor_sampler`: currently `uniform`

Inspect resolved config:

```bash
uv run -m banditdl --cfg job
```

## How To Create A New Experiment

1. Copy a profile in `conf/profile/`.
2. Set scalar defaults for single-run behavior.
3. Add/update `hydra.sweeper.params` in the same profile for preset matrix sweeps.

Example:

```yaml
# conf/profile/my_new_profile.yaml
mode: dynamic
...
hydra:
  sweeper:
    params:
      seed: 0,1
      profile.nb_neighbors: 3,5,7
      profile.attack: ALIE,SF
```

Run it:

```bash
uv run -m banditdl -m profile=my_new_profile
```

## Runtime Logic (Short)

1. `uv run -m banditdl` -> `banditdl.__main__` -> `banditdl.experiments.hydra_run`.
2. Hydra composes config from `conf/`.
3. `hydra_run` builds one training command for that run.
4. In multirun mode, Hydra launches many runs (one per parameter combination).
5. Each run executes `train_p2p` or `fx_train_p2p` and writes result files.

## Sampling / Bandit Hook Points

- `banditdl/core/sampling.py`
- `banditdl/experiments/train_p2p.py`
- `banditdl/core/training/dynamic/worker.py`
