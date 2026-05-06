from __future__ import annotations

import pathlib
from typing import Any
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.plotting import plot_all


def _is_dynamic_sampler(sampler: str) -> bool:
    return sampler in {"uniform", "bandit", "epsilon_greedy"}


def _run_name(cfg: DictConfig, byzantine_budget: int, nb_neighbors: int) -> str:
    sampler = str(cfg.topology.neighbor_sampler)
    is_dynamic = _is_dynamic_sampler(sampler)
    topology_token = (
        f"-sampling_{cfg.topology.sampling}"
        if is_dynamic
        else f"-degree_{nb_neighbors}"
    )
    base = (
        f"{cfg.dataset.dataset}-n_{cfg.nodes}"
        f"-model_{cfg.dataset.model}"
        f"-attack_{cfg.adversary.attack}"
        f"-agg_{cfg.aggregator.aggregator}"
        f"{topology_token}"
        f"-sampler_{cfg.topology.neighbor_sampler}"
        f"-f_{cfg.adversary.byzcount}"
        f"-alpha_{cfg.heterogeneity.alpha}"
        f"-byz_budget_{byzantine_budget}"
        f"-nb-local_{cfg.optimization.nb_local_steps}"
    )
    if cfg.topology.neighbor_sampler in {"bandit", "epsilon_greedy"}:
        base += (
            f"-eps_{cfg.topology.get('bandit_epsilon', 0.1)}"
            f"-init_{cfg.topology.get('bandit_initial_value', 0.0)}"
        )
    return base


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Build one concrete training run from config.
    params: dict[str, Any] = {}
    print("\n"+ OmegaConf.to_yaml(cfg, resolve=True) + "\n")
    nodes = int(cfg.nodes)
    sampler = str(cfg.topology.neighbor_sampler)
    is_dynamic = _is_dynamic_sampler(sampler)
    if is_dynamic:
        sampling = float(cfg.topology.sampling)
        nb_neighbors = max(1, min(nodes - 1, int(round((nodes - 1) * sampling))))
    else:
        nb_neighbors = int(cfg.topology.degree)

    # Build params from config groups
    params["dataset"] = cfg.dataset.dataset
    params["model"] = cfg.dataset.model
    params["nb-workers"] = nodes
    params["dirichlet-alpha"] = float(cfg.heterogeneity.alpha)
    params["nb-decl-byz"] = int(cfg.adversary.byzcount)
    params["nb-real-byz"] = int(cfg.adversary.byzcount)
    params["nb-neighbors"] = nb_neighbors
    if cfg.adversary.attack is not None:
        params["attack"] = cfg.adversary.attack
    params["nb-local-steps"] = int(cfg.optimization.nb_local_steps)
    params["neighbor-sampler"] = sampler
    params["bandit-epsilon"] = float(cfg.topology.get("bandit_epsilon", 0.1))
    params["bandit-initial-value"] = float(
        cfg.topology.get("bandit_initial_value", 0.0)
    )
    params["bandit-reward"] = cfg.topology.get("bandit_reward", "parameter_distance")
    
    # Add optimization parameters
    params["batch-size"] = int(cfg.optimization.get("batch_size"))
    params["loss"] = cfg.optimization.get("loss")
    params["weight-decay"] = float(cfg.optimization.get("weight_decay"))
    params["momentum-worker"] = float(cfg.optimization.get("momentum_worker"))
    params["nb-steps"] = int(cfg.optimization.get("nb_steps"))
    
    # Add aggregator parameters
    params["aggregator"] = cfg.aggregator.get("aggregator")
    params["pre-aggregator"] = cfg.aggregator.get("pre_aggregator")
    params["rag"] = bool(cfg.aggregator.get("rag"))
    
    # Add heterogeneity parameters
    params["numb-labels"] = int(cfg.heterogeneity.get("numb_labels"))
    
    # Add evaluation parameters
    params["evaluation-delta"] = int(cfg.evaluation.get("evaluation_delta"))

    byz_budget_raw = cfg.adversary.get("byzantine_budget")
    byzantine_budget = int(
        cfg.adversary.byzcount if byz_budget_raw is None else byz_budget_raw
    )
    params["b-hat"] = byzantine_budget

    if is_dynamic:
        params["rag"] = True
        params["sampling-ratio"] = float(cfg.topology.sampling)

    device = str(cfg.device)
    if not device or device == "auto":
        device = torch.cuda.is_available() and "cuda" or "cpu"

    if not is_dynamic:
        params["method"] = cfg.topology.get("method", sampler)

    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    result_dir = output_dir / "results"
    run_name = _run_name(cfg, byzantine_budget, nb_neighbors)
    if is_dynamic:
        run_dynamic(
            params=params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=device,
        )
    else:
        run_fixed(
            params=params,
            result_dir=result_dir,
            seed=int(cfg.seed),
            device=device,
        )
    plots_dir = output_dir / "plots"
    plot_all(run_dir=result_dir, plots_dir=plots_dir, run_label=run_name)


if __name__ == "__main__":
    main()
