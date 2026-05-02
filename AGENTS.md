# AGENTS.md

This file records repository-specific engineering preferences and workflow rules.

## Core Principles

- Optimize for conciseness and simplicity.
- Prefer "good enough" practical solutions over complex designs.
- Avoid overengineering and unnecessary abstractions.
- Do not introduce "enterprise" layering for small/medium tasks.

## Package/Layout Rules

- The Python codebase lives under `banditdl/`.
- Avoid ambiguous technical names in import paths when a clearer name is available.
- Keep config files outside the Python package tree.
  - Use top-level `conf/` for Hydra configs.
- Keep module boundaries straightforward and discoverable.
- Follow the principles of software engineering of modularity, separation of responsabilities while not overcomplicating the codebase.

## Naming and Structure Preferences

- Prefer simple top-level modules over deep hierarchies.
- Use `banditdl/utils/` for reusable helpers.
- Do not bury reusable helpers under `core/` unless they are truly core-internal.
- Keep "things we actively change" separated from stable simulator/model code.

## Hydra / Experiment Workflow

- Hydra is the primary experiment control plane.
- Main run path should be:
  - `uv run -m banditdl`
- Use Hydra overrides rather than editing code for parameter sweeps.
- Keep profiles and sweep knobs in `conf/`.

## Git Workflow

- Use Git for operations.
- Stage and commit changes regularly.
- Do not leave large, uncommitted refactors hanging.
- Use clear commit messages describing intent.

## Communication Style

- Be concise.
- Prefer direct answers.
- Minimize long explanations unless explicitly requested.
- If a design choice is questionable, state it plainly and fix it.

## Current Decision: Sampling Module

- Sampling is currently in `banditdl/sampling/` to isolate frequently changing experiment logic.
- If preferred, it can be collapsed to `banditdl/core/sampling.py` (or `banditdl/sampling.py`) in a follow-up simplification pass.
- Default preference for future changes: choose the flatter option when both are acceptable.
