# BanditFL
Project for Optimization for ML class at EPFL

This is a framework for Byzantine-resilient decentralized learning, implementing the algorithms from the paper:
**"Byzantine Epidemic Learning: Breaking the Communication Barrier in Robust Collaborative Learning"**

## Installation 

This project uses `uv` for environment and dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

## Reproducing the Experiments

All experiment scripts are located in the `scripts/` directory.

### CIFAR-10 Experiments
To reproduce the CIFAR-10 experiments:
```bash
uv run scripts/run_cifar.py --supercharge 2
```
*(The `--supercharge` parameter launches multiple experiments in parallel; use with caution to avoid memory overflow.)*

### MNIST Experiments
To reproduce the MNIST experiments:
```bash
uv run scripts/run_mnist.py --supercharge 2
```

### Fixed-Graph Baselines
To reproduce the fixed-graph baselines:
- MNIST: `uv run scripts/baselines/fx_run_mnist.py --supercharge 2`
- CIFAR-10: `uv run scripts/baselines/fx_run_cifar.py --supercharge 2`

### Simulations
To reproduce the theoretical simulations (Figure 3 in the paper):
```bash
uv run scripts/analysis/simulations.py
```

## Customization
- To test with different parameters, edit the dictionaries inside `scripts/run_cifar.py` or `scripts/run_mnist.py`.
- For detailed argument descriptions, see `scripts/train_p2p.py` or `scripts/baselines/fx_train_p2p.py`.

## Repository Structure
- `BanditFL/`: Core package containing the decentralized learning logic.
    - `core/`: Worker and model definitions.
    - `data/`: Dataset loading and partitioning.
    - `robustness/`: Byzantine attacks and robust aggregators.
    - `network/`: Topology and graph utilities.
    - `utils/`: Internal helpers.
- `scripts/`: Main entry points for experiments.
- `tools/`: Job management and CLI utilities.
