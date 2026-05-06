from __future__ import annotations

from pathlib import Path

import hydra
import optuna
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from banditdl.experiments.engine import run_dynamic, run_fixed
from banditdl.utils.plot_sweep_base import (
    build_axis_metadata,
    enumerate_valid_param_dicts,
    normalize_directions,
    normalize_plot_modes,
    plot_sweep,
    trial_folder_name,
)


def _build_engine_params(cfg):
    """
    Convert Hydra config into engine argument dictionary.

    Args:
      cfg: DictConfig
        Composed run config.

    return: tuple
      (engine_params, run_mode)
    """
    params = {}
    nodes = int(cfg.nodes)
    sampler = str(cfg.topology.neighbor_sampler)
    is_dynamic = sampler in {"uniform", "bandit", "epsilon_greedy"}
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
    params["bandit-epsilon"] = float(cfg.topology.get("bandit_epsilon"))
    params["bandit-initial-value"] = float(cfg.topology.get("bandit_initial_value"))
    params["bandit-reward"] = cfg.topology.get("bandit_reward", "parameter_distance")

    params["batch-size"] = int(cfg.optimization.get("batch_size"))
    params["loss"] = cfg.optimization.get("loss")
    params["weight-decay"] = float(cfg.optimization.get("weight_decay"))
    params["momentum-worker"] = float(cfg.optimization.get("momentum_worker"))
    params["nb-steps"] = int(cfg.optimization.get("nb_steps"))

    params["aggregator"] = cfg.aggregator.get("aggregator")
    params["pre-aggregator"] = cfg.aggregator.get("pre_aggregator")
    params["rag"] = bool(cfg.aggregator.get("rag"))

    params["numb-labels"] = int(cfg.heterogeneity.get("numb_labels"))
    params["evaluation-delta"] = int(cfg.evaluation.get("evaluation_delta"))
    byz_budget_raw = cfg.adversary.get("byzantine_budget")
    byz_budget = int(cfg.adversary.byzcount if byz_budget_raw is None else byz_budget_raw)
    params["b-hat"] = byz_budget

    if is_dynamic:
        params["rag"] = True
        params["sampling-ratio"] = float(cfg.topology.sampling)

    method = cfg.topology.get("method")
    if not is_dynamic:
        params["method"] = method or sampler

    return params, "dynamic" if is_dynamic else "fixed"


def _pick_device(cfg):
    """
    Resolve device string from config.

    Args:
      cfg: DictConfig
        Composed run config.

    return: str
      Torch device string.
    """
    configured = str(cfg.device)
    if configured and configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _read_metric_file_max(metric_file):
    """
    Extract best metric value from a result file.

    Args:
      metric_file: Path
        Path to metric text file.

    return: float
      Maximum metric observed in the file.
    """
    if not metric_file.exists():
        raise FileNotFoundError(f"Missing metric file: {metric_file}")

    metric_values = []
    lines = metric_file.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("\t")
        if len(fields) < 2:
            continue
        metric_values.append(float(fields[1]))
    if not metric_values:
        raise ValueError(f"No metric values found in: {metric_file}")
    return max(metric_values)


def _resolved_trial_params(trial):
    """
    Return effective trial parameters for grid-driven sweeps.

    Args:
      trial: Trial | FrozenTrial
        Optuna trial object.

    return: dict
      Parameter mapping used for this trial.
    """
    if trial.params:
        return dict(trial.params)
    resolved = trial.user_attrs.get("resolved_params")
    if isinstance(resolved, dict):
        return dict(resolved)
    return {}


def _objective(trial, base_cfg, trials_root, axis_lookup, combos):
    """
    Execute one trial using fixed parameter assignments.

    Args:
      trial: Trial
        Optuna trial instance.
      base_cfg: DictConfig
        Base Hydra configuration.
      trials_root: Path
        Root folder storing per-trial artifacts.
      axis_lookup: dict
        Metadata mapping parameter paths to display attributes.
      combos: list
        Ordered list of parameter dictionaries for each trial index.

    return: float
      Validation accuracy objective.
    """
    trial_index = int(trial.number)
    if trial_index >= len(combos):
        raise IndexError(f"Trial index {trial_index} out of bounds for {len(combos)} combinations")
    trial_params = dict(combos[trial_index])

    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for path, value in trial_params.items():
        OmegaConf.update(trial_cfg, path, value, merge=False)

    folder_name = trial_folder_name(trial_params, axis_lookup)
    trial_result_dir = trials_root / folder_name / "results"
    trial_result_dir.mkdir(parents=True, exist_ok=True)
    params, run_mode = _build_engine_params(trial_cfg)
    seed_value = int(trial_cfg.seed) + int(trial.number)
    device = _pick_device(trial_cfg)

    if run_mode == "dynamic":
        run_dynamic(params=params, result_dir=trial_result_dir, seed=seed_value, device=device)
    else:
        run_fixed(params=params, result_dir=trial_result_dir, seed=seed_value, device=device)

    validation_metric = _read_metric_file_max(trial_result_dir / "validation")
    trial.set_user_attr("validation_accuracy", validation_metric)
    trial.set_user_attr("result_dir", str(trial_result_dir))
    trial.set_user_attr("seed", seed_value)
    trial.set_user_attr("resolved_params", trial_params)
    return validation_metric


def _run_best_trial_test_evaluation(best_trial, base_cfg, output_root):
    """
    Re-run the best trial with test evaluation enabled.

    Args:
      best_trial: FrozenTrial
        Optuna best trial struct.
      base_cfg: DictConfig
        Base Hydra configuration.
      output_root: Path
        Hydra run output directory.

    return: float
      Test accuracy from the re-run.
    """
    best_params = _resolved_trial_params(best_trial)
    best_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for param_path, sampled_value in best_params.items():
        OmegaConf.update(best_cfg, param_path, sampled_value, merge=False)

    best_result_dir = output_root / "best_trial_test_eval" / "results"
    best_result_dir.mkdir(parents=True, exist_ok=True)
    params, run_mode = _build_engine_params(best_cfg)
    params["evaluate-test"] = True
    seed_value = int(best_cfg.seed) + int(best_trial.number)
    device = _pick_device(best_cfg)

    if run_mode == "dynamic":
        run_dynamic(params=params, result_dir=best_result_dir, seed=seed_value, device=device)
    else:
        run_fixed(params=params, result_dir=best_result_dir, seed=seed_value, device=device)

    test_accuracy = _read_metric_file_max(best_result_dir / "test")
    return test_accuracy


def _metrics_list_from_cfg(cfg):
    """
    Normalize plot metric selections from Hydra config.

    Args:
      cfg: DictConfig
        User configuration object.

    return: list
      Metric stems for sweep plots.
    """
    raw = cfg.get("plot_metrics")
    if raw is None:
        return []
    return list(OmegaConf.to_container(raw, resolve=True))


@hydra.main(version_base=None, config_path="../../conf", config_name="sweep")
def main(cfg):
    """
    Launch categorical grid sweep with Optuna bookkeeping and sweep plots.
    """
    output_root = Path(HydraConfig.get().runtime.output_dir)
    trials_root = output_root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    if "optuna" not in cfg:
        raise ValueError("Missing 'optuna' section in Hydra config")
    optuna_cfg = cfg.optuna
    if "search_space" not in optuna_cfg:
        raise ValueError("Missing 'optuna.search_space' in Hydra config")

    search_space = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")

    combos = enumerate_valid_param_dicts(cfg, search_space)
    if not combos:
        raise ValueError(
            "No categorical grid combinations found. Use categorical sweeps or add list-style search_space entries."
        )

    _, axis_meta = build_axis_metadata(search_space)
    axis_lookup = {path: axis_meta.get(path, {}) for path in search_space.keys()}

    direction = str(optuna_cfg.direction)
    study = optuna.create_study(direction=direction)
    total_trials = len(combos)
    print(f"[optuna] grid trials={total_trials} | metric=validation_accuracy | trials_dir={trials_root}")
    study.optimize(lambda trial: _objective(trial, cfg, trials_root, axis_lookup, combos), n_trials=total_trials)

    best = study.best_trial
    best_dir = best.user_attrs.get("result_dir")
    print(f"[optuna] best trial: {best.number}")
    print(f"[optuna] best validation_accuracy: {best.value:.6f}")
    if best_dir:
        print(f"[optuna] best result directory: {best_dir}")
    print("[optuna] best parameters:")
    for name, value in _resolved_trial_params(best).items():
        print(f"  - {name}: {value}")

    final_test_accuracy = _run_best_trial_test_evaluation(best, cfg, output_root)
    print(f"[optuna] best trial final test directory: {output_root / 'best_trial_test_eval' / 'results'}")
    print(f"[optuna] best trial final test_accuracy: {final_test_accuracy:.6f}")

    metrics_list = _metrics_list_from_cfg(cfg)
    plot_modes = normalize_plot_modes(cfg.plot_mode)
    plot_directions = normalize_directions(cfg.get("direction"))
    sweep_plot_root = output_root / "sweep_artifacts"
    plot_sweep(
        plot_modes,
        plot_directions,
        trials_root,
        study,
        search_space,
        metrics_list,
        sweep_plot_root,
    )
    print(
        f"[optuna] sweep plots written to: {sweep_plot_root} | modes={plot_modes} directions={plot_directions}"
    )


if __name__ == "__main__":
    main()
