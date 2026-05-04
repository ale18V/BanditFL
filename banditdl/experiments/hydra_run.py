from __future__ import annotations

import pathlib
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.engine import run_dynamic, run_fixed


def _run_name(cfg: DictConfig, byzantine_budget: int, nb_neighbors: int) -> str:
    topology_token = (
        f"-sampling_{cfg.topology.sampling}"
        if cfg.topology.mode == "dynamic"
        else f"-degree_{nb_neighbors}"
    )
    base = (
        f"{cfg.local.dataset}-n_{cfg.topology.nodes}"
        f"-model_{cfg.local.model}"
        f"-attack_{cfg.adversary.attack}"
        f"-agg_{cfg.local.params_common.aggregator}"
        f"{topology_token}"
        f"-sampler_{cfg.topology.neighbor_sampler}"
        f"-f_{cfg.adversary.byzcount}"
        f"-alpha_{cfg.local.alpha}"
        f"-byz_budget_{byzantine_budget}"
        f"-nb-local_{cfg.local.nb_local_steps}"
    )
    method = cfg.topology.get("method")
    if method is not None:
        base += f"-{method}"
    if cfg.topology.neighbor_sampler in {"bandit", "epsilon_greedy"}:
        base += (
            f"-eps_{cfg.topology.get('bandit_epsilon', 0.1)}"
            f"-init_{cfg.topology.get('bandit_initial_value', 0.0)}"
        )
    return base


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    params_common = OmegaConf.to_container(cfg.local.params_common, resolve=True)
    assert isinstance(params_common, dict)

    # Build one concrete training run from config.
    params: dict[str, Any] = dict(params_common)
    nodes = int(cfg.topology.nodes)
    if cfg.topology.mode == "dynamic":
        sampling = float(cfg.topology.sampling)
        nb_neighbors = max(1, min(nodes - 1, int(round((nodes - 1) * sampling))))
    else:
        nb_neighbors = int(cfg.topology.degree)

    params["dataset"] = cfg.local.dataset
    params["model"] = cfg.local.model
    params["nb-workers"] = nodes
    params["dirichlet-alpha"] = float(cfg.local.alpha)
    params["nb-decl-byz"] = int(cfg.adversary.byzcount)
    params["nb-real-byz"] = int(cfg.adversary.byzcount)
    params["nb-neighbors"] = nb_neighbors
    if cfg.adversary.attack is not None:
        params["attack"] = cfg.adversary.attack
    params["nb-local-steps"] = int(cfg.local.nb_local_steps)
    params["neighbor-sampler"] = cfg.topology.neighbor_sampler
    params["bandit-epsilon"] = float(cfg.topology.get("bandit_epsilon", 0.1))
    params["bandit-initial-value"] = float(
        cfg.topology.get("bandit_initial_value", 0.0)
    )
    params["bandit-reward"] = cfg.topology.get("bandit_reward", "parameter_distance")

    byz_budget_raw = cfg.adversary.get("byzantine_budget")
    byzantine_budget = int(
        cfg.adversary.byzcount if byz_budget_raw is None else byz_budget_raw
    )
    params["b-hat"] = byzantine_budget

    if cfg.topology.mode == "dynamic":
        params["rag"] = True
        params["sampling-ratio"] = float(cfg.topology.sampling)

    if not cfg.device or cfg.device == "auto":
        import torch

        cfg.device = torch.cuda.is_available() and "cuda" or "cpu"

    method = cfg.topology.get("method")
    if method is not None:
        params["method"] = method

    results_root = pathlib.Path(str(cfg.result_directory))
    run_name = _run_name(cfg, byzantine_budget, nb_neighbors)
    result_dir = results_root / f"{run_name}-seed_{cfg.seed}"
    if cfg.topology.mode == "dynamic":
        run_dynamic(
            params=params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=str(cfg.device),
        )
    else:
        run_fixed(
            params=params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=str(cfg.device),
        )


if __name__ == "__main__":
    main()
