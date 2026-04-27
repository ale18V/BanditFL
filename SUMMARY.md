# BanditFL Project Summary

This repository implements a framework for **Byzantine-resilient decentralized learning**, specifically focusing on "Byzantine Epidemic Learning". It serves as a base for comparing uniform random neighbor sampling with more advanced strategies, such as bandit-based sampling.

## Repository Structure

### Entry Points & Scripts (`scripts/`)
- `train_p2p.py`: The primary training script. Manages the training loop and communication steps.
- `run_cifar.py` / `run_mnist.py`: Scripts to launch batch experiments.
- `baselines/`: Scripts for fixed-graph baseline experiments.
- `analysis/`: Plotting and theoretical simulations.

### Core Implementation (`BanditFL/`)
- `core/worker_p2p.py`: Defines the `P2PWorker` class. Logic for local SGD and neighbor sampling.
- `core/models.py`: Neural network architectures.
- `data/dataset.py`: Data loading and partitioning (IID/Non-IID).
- `robustness/aggregators.py`: Byzantine-resilient aggregation rules (Trimmed Mean, etc.).
- `robustness/attacks.py` & `robustness/byz_worker.py`: Byzantine behavior and attacks.
- `network/topo.py`: Graph topology utilities.
- `utils/`: Internal utilities for metrics and logging.
- `tools/`: Framework utilities for job management.

## Focus for Bandit Sampling Experiments

To replace uniform sampling with bandit algorithms, you should focus on:
1. **`BanditFL/core/worker_p2p.py`**:
   - Locate `random.sample(indices_list, self.nb_neighbors)`.
   - Replace this with your Bandit selection logic.
2. **`scripts/train_p2p.py`**:
   - Coordinator script where you can track rewards for the bandit algorithm.

## How to Run
```bash
uv run scripts/run_cifar.py --supercharge 2
```
