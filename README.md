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
uv run -m banditdl profile=mnist_dynamic profile.nodes=100 profile.sampling=0.05 train.neighbor_sampler=uniform seed=0
```

Runs print lightweight progress to stdout: start metadata, result directory, periodic decentralized-learning rounds, evaluation accuracy when available, and completion.

## Run Sweeps (Hydra Multirun)

Hydra does orchestration. The custom in-repo scheduler is no longer the main path.

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
  profile.nodes=50,100 \
  profile.sampling=0.03,0.05 \
  train.neighbor_sampler=uniform \
  profile.nb_local_steps=1,3
```

## Existing Profiles

- `cifar_dynamic`
- `mnist_dynamic`
- `cifar_fixed`
- `mnist_fixed`

## Config Reference

Hydra config lives in `conf/`. The main entry point is `conf/config.yaml`.

Inspect the resolved config before launching a large run:

```bash
uv run -m banditdl --cfg job
```

### Top-Level Config

- `profile`: experiment profile config group. Existing values: `mnist_dynamic`, `cifar_dynamic`, `mnist_fixed`, `cifar_fixed`.
- `train`: training/sampler config group. Existing values: `dynamic`, `fixed`.
- `seed`: random seed. Use comma-separated values under `-m` for sweeps.
- `device`: `auto`, `cpu`, or a torch device string such as `cuda`.

### Profile Config

Profiles are in `conf/profile/`. They define dataset, topology, training defaults, result paths, and preset sweep matrices.

- `mode`: topology mode. `dynamic` resamples neighbors every round. `fixed` builds one graph.
- `dataset`: dataset name passed to the loader. Common values: `mnist`, `cifar10`.
- `model`: model constructor from `banditdl/data/models.py`, for example `cnn_mnist` or `cnn_cifar_old`.
- `nodes`: total simulated participants, including Byzantine participants.
- `sampling`: dynamic-mode sampling ratio. The dynamic worker samples about `round((nodes - 1) * sampling)` neighbors.
- `degree`: fixed-mode graph degree target. Used only by fixed profiles.
- `alpha`: Dirichlet data heterogeneity parameter passed as `dirichlet-alpha`.
- `result_directory`: root directory for per-run saved artifacts.
- `plot_directory`: legacy profile field; current plotting uses `scripts/plot_results.py`.
- `byzcount`: number of declared and real Byzantine workers currently instantiated by the Hydra adapter.
- `byzantine_budget`: robustness budget `b_hat`. If unset/null, defaults to `byzcount`.
- `nb_local_steps`: local SGD steps per communication round.
- `attack`: Byzantine attack name or `null`. Available attacks include `SF`, `LF`, `FOE`, `ALIE`, `mimic`, `auto_ALIE`, `auto_FOE`, `inf`.
- `method`: fixed-graph summation method. Current values used by fixed profiles: `cs+`, `cs_he`, `gts`.
- `params_common`: training hyperparameters passed to the engine.
- `hydra.sweeper.params`: preset multirun matrix for `uv run -m banditdl -m profile=<name>`.

### Train Config

Train configs are in `conf/train/`.

- `neighbor_sampler`: neighbor selection strategy. Values: `uniform`, `bandit`, `epsilon_greedy`.
- `bandit_epsilon`: epsilon-greedy exploration rate. Used by `bandit`/`epsilon_greedy`.
- `bandit_initial_value`: initial reward estimate for unseen arms.
- `bandit_reward`: reward strategy. Current value: `parameter_distance`.

`fixed.yaml` currently only defines `neighbor_sampler: uniform`; fixed mode does not use dynamic sampling.

### Common Training Params

These live under `profile.params_common`.

- `batch-size`: training batch size.
- `batch-size-test`: test batch size. Defaults to `100` if omitted.
- `loss`: torch loss class name, for example `NLLLoss`.
- `learning-rate`: SGD learning rate. Defaults to `0.5` if omitted.
- `learning-rate-decay`: decay scale used by the worker learning-rate schedule.
- `learning-rate-decay-delta`: step interval for learning-rate decay checks.
- `weight-decay`: SGD weight decay.
- `momentum-worker`: worker momentum.
- `nb-steps`: number of communication/training rounds.
- `evaluation-delta`: evaluate every N rounds.
- `numb-labels`: number of dataset labels.
- `pre-aggregator`: optional first-stage robust aggregation rule, commonly `nnm`.
- `aggregator`: robust aggregator, commonly `trmean`.
- `rag`: dynamic-mode robust aggregation flag. Dynamic Hydra runs force this to `true`.
- `mimic-learning-phase`: optional learning phase length for mimic attacks.
- `bucket-size`: robust aggregation bucket size. Defaults to `1`.
- `gradient-clip`: optional gradient clipping threshold.
- `server-clip`: optional server clipping flag.
- `hetero`: dataset heterogeneity flag. Defaults to `false`.
- `distinct-data`: give workers distinct local datasets. Defaults to `false`.
- `nb-datapoints`: local datapoint count for distinct-data setups.

Available robust aggregators include `average`, `trmean`, `median`, `geometric_median`, `krum`, `multi_krum`, `nnm`, `bucketing`, `pmk`, `cc`, `mda`, `mva`, `monna`, `meamed`.

### Sweep Syntax

Preset profile sweep:

```bash
uv run -m banditdl -m profile=mnist_dynamic
```

Ad-hoc sweep:

```bash
uv run -m banditdl -m \
  profile=mnist_dynamic \
  profile.nodes=50,100 \
  profile.sampling=0.03,0.05 \
  train.neighbor_sampler=uniform,bandit \
  seed=0,1,2
```

Hydra takes the Cartesian product of comma-separated override values.

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
      profile.nodes: 50,100
      profile.sampling: 0.03,0.05
      profile.nb_local_steps: 1,3
```

Run it:

```bash
uv run -m banditdl -m profile=my_new_profile
```

## Plot Saved Results

Experiments write results first. Plotting is a standalone offline step, kept out of the Python package tree.

Plot one run:

```bash
uv run python scripts/plot_results.py \
  results_mnist/results-data-mnist-iclr/<run-dir> \
  -o plots/example.png
```

Compare multiple runs:

```bash
uv run python scripts/plot_results.py \
  results_mnist/results-data-mnist-iclr/<run-a> \
  results_mnist/results-data-mnist-iclr/<run-b> \
  --label uniform \
  --label bandit \
  -o plots/comparison.png
```

Aggregate seed runs:

```bash
uv run python scripts/plot_results.py \
  results_mnist/results-data-mnist-iclr/mnist-*-seed_* \
  --aggregate \
  --label "uniform mean" \
  -o plots/uniform_seed_mean.png
```

Useful options:
- `--metric accuracies`: plot from `accuracies.npy` (default).
- `--metric eval`: plot average accuracy from `eval`.
- `--metric eval_worst`: plot worst-worker accuracy from `eval_worst`.
- `--metric regret`: plot regret against the best fixed neighbor subset in hindsight.
- `--metric normalized_regret`: plot regret divided by oracle reward.
- `--metric reward_algorithm|reward_oracle`: plot cumulative reward curves.
- `--stat mean|worst`: choose mean worker or worst worker; for regret, worst means highest regret.
- `--legend outside|best|none`: choose legend placement; default keeps it below the plot.
- `--max-label-length 48`: cap auto-generated labels.

## Runtime Architecture

This section describes runtime execution logic and module interactions.

### Runtime Interaction Diagram

```mermaid
flowchart TD
    A[User: uv run -m banditdl ...] --> B[banditdl.__main__]
    B --> C[experiments.hydra_run]
    C --> D[Hydra config composition
conf/config.yaml + profile/train]

    D --> E{Hydra mode}
    E -->|single run| F[One composed config]
    E -->|multirun -m| G[Cartesian expansion from
hydra.sweeper.params + CLI overrides]

    F --> H[hydra_run dispatches training engine]
    G --> H

    H --> I1[Training engine
experiments.engine::run_dynamic]
    H --> I2[Training engine
    experiments.engine::run_fixed]

    I1 --> J1[data.*
    models + dataset loaders]
    I1 --> K1[core.worker.dynamic
    local updates + neighbor sampling]
    K1 --> L1[core.robustness.*
    attacks + aggregators]

    I2 --> J2[data.*
    models + dataset loaders]
    I2 --> K2[core.worker.fixed
    fixed-graph updates]
    K2 --> L2[core.robustness.*
    attacks + summations]

    I1 --> M[Per-run result directory
    eval, eval_worst, logs]
    I2 --> M
```

### End-to-end Flow

1. You run `uv run -m banditdl ...`.
2. `banditdl.__main__` dispatches to `banditdl.experiments.hydra_run`.
3. Hydra composes config from `conf/`.
4. In multirun mode, Hydra generates one run per parameter combination.
5. For each run, `hydra_run` dispatches the corresponding training engine function.
6. Training engine (`experiments.engine`) executes and writes results.

### Responsibilities By Module

- `banditdl.experiments.hydra_run`
  - Hydra-to-engine adapter.
  - Converts composed config into one concrete training call.

- `banditdl.experiments.engine`
  - Per-run execution logic for dynamic/fixed settings.
  - Drives training/evaluation loops and persistence.

- `banditdl.core.worker.*`
  - Worker logic for local updates and communication.

- `banditdl.core.robustness.*`
  - Byzantine attacks and robust aggregation/summation rules.

- `banditdl.data.*`
  - Dataset loading/partitioning and model construction.

- `banditdl.core.sampling`
  - Neighbor sampling strategy used in dynamic worker mode.


### Terminology: Worker = Node

In this repository, a **worker** is one decentralized learning participant (node/client):
- it owns local train/test data loaders,
- performs local optimization steps,
- communicates with neighbors,
- applies robust aggregation logic under Byzantine settings.

Honest participants are modeled as `DynamicWorker`/`FixedGraphWorker`; Byzantine participants are modeled as explicit attack-only nodes.

### Decentralized Structure Diagram

```mermaid
flowchart LR
    subgraph Topology["Decentralized Topology (N workers)"]
        W0["Worker 0 (possibly Byzantine)"]
        W1["Worker 1"]
        W2["Worker 2"]
        W3["Worker 3"]
        W0 --- W1
        W1 --- W2
        W2 --- W3
        W3 --- W0
        W0 --- W2
    end

    W0 --> S["core.sampling: choose neighbors (dynamic mode)"]
    W1 --> S
    W2 --> S
    W3 --> S

    S --> U["core.worker.*: local SGD/update + send/receive"]
    U --> A["core.robustness: attack model + robust aggregation"]
    A --> M["Updated model state per worker"]

    D["data.*: local shards + model ctor"] --> U
```

Interpretation:
- Each worker is a simulated node with its own local data and model copy.
- Communication is peer-to-peer, not centralized; each node exchanges updates with selected neighbors.
- In dynamic mode, neighbor sets are re-sampled each round (`core.sampling`).
- Received updates pass through Byzantine attack/aggregation logic before updating local state.

## Sampling / Bandit Hook Points

- `banditdl/core/sampling.py`
- `banditdl/experiments/engine.py`
- `banditdl/core/worker/`

Use the multi-armed bandit sampler in dynamic mode:

```bash
uv run -m banditdl \
  profile=mnist_dynamic \
  train.neighbor_sampler=bandit \
  train.bandit_epsilon=0.1 \
  profile.sampling=0.05 \
  seed=0
```

Current bandit feedback:
- each neighbor is one arm,
- MABWiser provides the epsilon-greedy bandit implementation,
- dynamic workers update selected arms after receiving neighbor weights,
- reward is selected through a strategy object; the default is `parameter_distance`,
- `parameter_distance` uses `1 / (1 + parameter_distance)` against the local model before aggregation.

Dynamic runs also save hindsight diagnostics for every sampler, including uniform:
- `reward_algorithm.npy`: cumulative reward achieved by sampled neighbors.
- `reward_oracle.npy`: cumulative reward of the best fixed neighbor subset in hindsight.
- `regret.npy`: `reward_oracle - reward_algorithm`.
- `normalized_regret.npy`: regret divided by oracle reward.
- `selected_neighbors.npy`: sampled neighbors per round and worker.
- `oracle_neighbors.npy`: best fixed hindsight neighbors per round and worker.

This is intentionally small: sampler choice and bandit parameters are Hydra-controlled, while reward design remains isolated behind the reward strategy API in `banditdl/core/sampling.py`.
