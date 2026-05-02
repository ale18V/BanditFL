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


## Runtime Architecture

This section describes execution logic, not folder layout.

### End-to-end Flow

1. You launch `uv run -m banditdl ...`.
2. `banditdl.__main__` dispatches to `banditdl.experiments.hydra_run`.
3. Hydra loads config (`conf/config.yaml` + selected `profile/train/sweep`).
4. `hydra_run` translates config into sweep jobs and calls `run_sweep(...)`.
5. `run_sweep` creates a `Jobs` scheduler (parallel devices/seeds), then submits one training process per combination of:
   - byzantine count
   - neighbors
   - attack
   - local steps
   - method (fixed-graph mode)
   - seed
6. Each job runs a training entrypoint module (`train_p2p` or `fx_train_p2p`) as a separate Python process.
7. Training process builds workers, runs train/eval loop, writes results in its own result directory.
8. After all jobs finish, `run_sweep` loads result files and generates plots.

### Responsibilities By Runtime Module

- `banditdl.experiments.hydra_run`
  - Control-plane adapter between Hydra config and sweep engine.
  - Defines how config fields map to training CLI parameters.

- `banditdl.experiments.common`
  - Sweep execution engine.
  - Builds commands, schedules jobs, waits for completion, triggers plotting.

- `banditdl.core.tools.jobs`
  - Concurrent job runner.
  - Handles seed replication, device assignment, subprocess execution, stdout/stderr capture.

- `banditdl.experiments.train_p2p` (dynamic) and `banditdl.experiments.fx_train_p2p` (fixed)
  - Per-job runtime script.
  - Parses job CLI args, creates datasets/workers, performs iterative training/evaluation, persists metrics.

- `banditdl.core.training.*`
  - Worker logic and update rules.
  - Dynamic/fixed communication behavior, aggregation calls, attack handling integration.

- `banditdl.core.robustness.*`
  - Attack implementations and robust aggregation/summation algorithms.

- `banditdl.data.*`
  - Dataset creation/partitioning and model instantiation.

- `banditdl.core.sampling`
  - Neighbor sampler strategy used by dynamic workers.
  - Current baseline: uniform sampling.

- `banditdl.core.analysis.study` + `banditdl.core.common`
  - Result loading, reduction, plotting helpers used after job completion.

### Interaction Contracts

- **Hydra -> Sweep Engine**: structured config objects become explicit sweep loops and CLI parameter dictionaries.
- **Sweep Engine -> Training Processes**: subprocess contract is pure CLI arguments; each run is isolated by result directory.
- **Training -> Core Algorithms**: training entrypoints orchestrate, core modules compute.
- **Core -> Data**: workers request models/data loaders, then operate on tensor updates/aggregation.
- **Post-processing**: analysis reads generated result files only (no in-memory coupling with training processes).

### Why This Split

- Experiment iteration speed lives in config + orchestration.
- Algorithm correctness and simulation behavior live in core/training/robustness.
- Sampling research can evolve independently via `core.sampling` and training wiring.

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
