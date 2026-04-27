# BanditFL Project Summary

This repository implements a framework for **Byzantine-resilient decentralized learning**, specifically focusing on "Byzantine Epidemic Learning". It serves as a base for comparing uniform random neighbor sampling with more advanced strategies, such as bandit-based sampling.

## Repository Structure

### Entry Points & Scripts
- `run.py`: Main script to launch CIFAR-10 experiments.
- `mnist_run.py`: Main script to launch MNIST experiments.
- `fx_cifar10_run.py` / `fx_mnist_run.py`: Entry points for fixed-graph baseline experiments.
- `simulations.py`: Script to reproduce theoretical simulations (e.g., Figure 3 in the paper).
- `train_p2p.py`: The primary training script for decentralized learning. It manages the training loop, worker initialization, and communication steps.
- `fx_train_p2p.py`: Similar to `train_p2p.py` but tailored for fixed communication topologies.

### Core Implementation (`src/`)
- `worker_p2p.py`: Defines the `P2PWorker` class. This is the **most relevant file** for your experiments. It contains the logic for:
    - Local SGD updates (`perform_local_step`).
    - **Neighbor Sampling**: Implemented in `aggregate_and_update_parameters_cgplus` and `aggregate_and_update_parameters_dec_with_rag` using `random.sample`.
- `dataset.py`: Handles data loading, preprocessing, and distribution among workers. It supports IID and non-IID (Dirichlet-based) partitioning.
- `robust_aggregators.py`: Implements various Byzantine-resilient aggregation rules (e.g., Trimmed Mean, Median, Krum).
- `byzWorker.py` & `byz_attacks.py`: Define Byzantine behavior and specific attacks (e.g., ALIE, FOE, Label Flipping).
- `models.py`: Contains PyTorch model definitions (CNNs for MNIST/CIFAR-10).
- `topo.py`: Utility for generating different graph topologies (fully connected, Erdos-Renyi, etc.).

### Utilities
- `tools/`: Internal framework utilities for job management, command-line processing, and logging.
- `utils/`: Miscellaneous helpers for tensor conversions and gossip protocols.

## Focus for Bandit Sampling Experiments

To replace uniform sampling with bandit algorithms, you should focus on:
1. **`src/worker_p2p.py`**:
   - Locate `random.sample(indices_list, self.nb_neighbors)`.
   - This is where the worker chooses which peers to communicate with.
   - You can replace this logic with a Bandit agent that selects neighbors based on their past contributions or reliability.
2. **`train_p2p.py`**:
   - This script coordinates the "rounds". You might need to track rewards (e.g., loss reduction or validation accuracy improvement) here to feed back into your bandit algorithm.

## How to Run
To start a basic experiment:
```bash
python3 run.py --supercharge 2
```
This will trigger multiple training runs as defined in the `run.py` parameter grid.
