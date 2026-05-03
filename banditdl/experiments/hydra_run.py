from __future__ import annotations

import pathlib
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.engine import run_dynamic, run_fixed
def _run_name(cfg: DictConfig, byzantine_budget: int, nb_neighbors: int) -> str:
    topology_token = (
        f"-sampling_{cfg.profile.sampling}"
        if cfg.profile.mode == "dynamic"
        else f"-degree_{nb_neighbors}"
    )
    base = (
        f"{cfg.profile.dataset}-n_{cfg.profile.nodes}"
        f"-model_{cfg.profile.model}"
        f"-attack_{cfg.profile.attack}"
        f"-agg_{cfg.profile.params_common.aggregator}"
        f"{topology_token}"
        f"-sampler_{cfg.train.neighbor_sampler}"
        f"-f_{cfg.profile.byzcount}"
        f"-alpha_{cfg.profile.alpha}"
        f"-byz_budget_{byzantine_budget}"
        f"-nb-local_{cfg.profile.nb_local_steps}"
    )
    method = cfg.profile.get("method")
    if method is not None:
        base += f"-{method}"
    return base


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    params_common = OmegaConf.to_container(cfg.profile.params_common, resolve=True)
    assert isinstance(params_common, dict)

    # Build one concrete training run from config.
    params: dict[str, Any] = dict(params_common)
    nodes = int(cfg.profile.nodes)
    if cfg.profile.mode == "dynamic":
        sampling = float(cfg.profile.sampling)
        nb_neighbors = max(1, min(nodes - 1, int(round((nodes - 1) * sampling))))
    else:
        nb_neighbors = int(cfg.profile.degree)

    params["dataset"] = cfg.profile.dataset
    params["model"] = cfg.profile.model
    params["nb-workers"] = nodes
    params["dirichlet-alpha"] = float(cfg.profile.alpha)
    params["nb-decl-byz"] = int(cfg.profile.byzcount)
    params["nb-real-byz"] = int(cfg.profile.byzcount)
    params["nb-neighbors"] = nb_neighbors
    if cfg.profile.attack is not None:
        params["attack"] = cfg.profile.attack
    params["nb-local-steps"] = int(cfg.profile.nb_local_steps)
    params["neighbor-sampler"] = cfg.train.neighbor_sampler

    byz_budget_raw = cfg.profile.get("byzantine_budget")
    byzantine_budget = int(cfg.profile.byzcount if byz_budget_raw is None else byz_budget_raw)
    params["b-hat"] = byzantine_budget

    if cfg.profile.mode == "dynamic":
        params["rag"] = True
        params["sampling-ratio"] = float(cfg.profile.sampling)

    method = cfg.profile.get("method")
    if method is not None:
        params["method"] = method

    results_root = pathlib.Path(cfg.profile.result_directory)
    run_name = _run_name(cfg, byzantine_budget, nb_neighbors)
    result_dir = results_root / f"{run_name}-seed_{cfg.seed}"
    if cfg.profile.mode == "dynamic":
        run_dynamic(params=params, result_dir=result_dir, seed=int(cfg.seed), device=str(cfg.device))
    else:
        run_fixed(params=params, result_dir=result_dir, seed=int(cfg.seed), device=str(cfg.device))


if __name__ == "__main__":
    main()
