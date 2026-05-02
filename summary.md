# Repository Summary

## Core Idea

Hydra is the single experiment control plane.

## Package Structure

- `banditdl/core/`: stable simulator/training stack
- `banditdl/data/`: model + dataset loading
- `banditdl/experiments/`: orchestration and entrypoints
- `banditdl/sampling/`: pluggable neighbor samplers
- `conf/`: Hydra configs (profile/train/sweep)

## Main Entrypoint

- `uv run -m banditdl`

## What You Edit Most

- Experiment presets: `conf/profile/*.yaml`
- Sweep/runtime knobs: `conf/sweep/*.yaml`
- Bandit samplers: `banditdl/sampling/*`
