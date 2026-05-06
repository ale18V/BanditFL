from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import optuna
import torch
from omegaconf import OmegaConf

from banditdl.experiments.engine import run_dynamic, run_fixed


def _repo_root():
    """
    Resolve repository root from this module path.

    return: Path
      Repository root.
    """
    return Path(__file__).resolve().parents[2]


def _load_config(conf_dir, sweep_relative_path):
    """
    Load base Hydra config and merge sweep overrides.

    Args:
      conf_dir: Path
        Hydra config directory.
      sweep_relative_path: str
        Sweep config path relative to conf directory.
    return: DictConfig
      Merged config with base defaults and sweep metadata.
    """
    sweep_path = conf_dir / sweep_relative_path
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")
    with hydra.initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        base_cfg = hydra.compose(config_name="config")
    sweep_cfg = OmegaConf.load(sweep_path)
    merged = OmegaConf.merge(base_cfg, sweep_cfg)
    return merged


def _resolve_output_dir(base_dir, output_dir_cfg):
    """
    Resolve output directory from config.

    Args:
      base_dir: Path
        Repository root directory.
      output_dir_cfg: str
        Output directory from config.
    return: Path
      Absolute output path.
    """
    output_dir = Path(str(output_dir_cfg))
    if output_dir.is_absolute():
        return output_dir
    return (base_dir / output_dir).resolve()


def _sample_param(trial, param_name, spec):
    """
    Sample one parameter from Optuna search-space spec.

    Args:
      trial: Trial
        Active Optuna trial.
      param_name: str
        Name/path of sampled parameter.
      spec: Any
        Parameter specification.
    return: Any
      Sampled value for the parameter.
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(param_name, spec)
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid search space entry for '{param_name}': {spec}")

    param_type = str(spec.get("type", "")).lower()
    if param_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Categorical '{param_name}' must define non-empty choices")
        return trial.suggest_categorical(param_name, choices)

    if param_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        log_flag = bool(spec.get("log", False))
        step = spec.get("step")
        if step is None:
            return trial.suggest_float(param_name, low, high, log=log_flag)
        return trial.suggest_float(param_name, low, high, step=float(step), log=log_flag)

    if param_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        log_flag = bool(spec.get("log", False))
        step = int(spec.get("step", 1))
        return trial.suggest_int(param_name, low, high, step=step, log=log_flag)

    raise ValueError(f"Unsupported parameter type '{param_type}' for '{param_name}'")


def _conditions_met(cfg, conditions):
    if conditions is None:
        return True
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("optuna.search_space 'when' must be a non-empty mapping")
    for path, expected in conditions.items():
        actual = OmegaConf.select(cfg, path)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


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
    params["dataset"] = cfg.dataset_nn.dataset
    params["model"] = cfg.dataset_nn.model
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
      Device string.
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
        Path to metric file.
    return: float
      Maximum validation accuracy observed in the trial.
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


def _objective(trial, base_cfg, output_root, search_space):
    """
    Run one Optuna trial and return validation objective.

    Args:
      trial: Trial
        Active Optuna trial.
      base_cfg: DictConfig
        Base config merged with sweep metadata.
      output_root: Path
        Trial output root directory.
      search_space: dict
        Parameter path -> search space specification.
    return: float
      Trial objective value (best validation accuracy).
    """
    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for param_path, spec in search_space.items():
        if isinstance(spec, dict) and "when" in spec:
            if not _conditions_met(trial_cfg, spec.get("when")):
                continue
            spec = {k: v for k, v in spec.items() if k != "when"}
        sampled_value = _sample_param(trial, param_path, spec)
        OmegaConf.update(trial_cfg, param_path, sampled_value, merge=False)

    trial_result_dir = output_root / f"trial_{trial.number:04d}" / "results"
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
    return validation_metric


def _run_best_trial_test_evaluation(best_trial, base_cfg, output_root):
    """
    Re-run the best trial and store final test accuracy.

    Args:
      best_trial: FrozenTrial
        Best trial selected by validation accuracy.
      base_cfg: DictConfig
        Base config merged with sweep metadata.
      output_root: Path
        Optuna output root directory.
    return: float
      Final test accuracy of the re-run best trial.
    """
    best_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    for param_path, sampled_value in best_trial.params.items():
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


def _parse_args():
    """
    Parse CLI arguments.

    return: Namespace
      Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Optuna sweep runner for BanditDL.")
    parser.add_argument("--sweep-config", default="optuna/default.yaml", help="Path under conf/ to sweep yaml file.")
    return parser.parse_args()


def main():
    """
    Launch Optuna sweep from Hydra defaults and sweep override config.
    """
    args = _parse_args()
    repo_root = _repo_root()
    conf_dir = repo_root / "conf"
    cfg = _load_config(conf_dir, args.sweep_config)

    if "optuna" not in cfg:
        raise ValueError("Missing 'optuna' section in sweep config")
    optuna_cfg = cfg.optuna
    if "search_space" not in optuna_cfg:
        raise ValueError("Missing 'optuna.search_space' in sweep config")

    search_space = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping")

    output_root = _resolve_output_dir(repo_root, optuna_cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction=str(optuna_cfg.direction))
    trials_count = int(optuna_cfg.n_trials)
    print(f"[optuna] running {trials_count} trials | metric=validation_accuracy | output={output_root}")
    study.optimize(lambda trial: _objective(trial, cfg, output_root, search_space), n_trials=trials_count)

    best = study.best_trial
    best_dir = best.user_attrs.get("result_dir")
    print(f"[optuna] best trial: {best.number}")
    print(f"[optuna] best validation_accuracy: {best.value:.6f}")
    if best_dir:
        print(f"[optuna] best result directory: {best_dir}")
    print("[optuna] best parameters:")
    for name, value in best.params.items():
        print(f"  - {name}: {value}")

    final_test_accuracy = _run_best_trial_test_evaluation(best, cfg, output_root)
    print(f"[optuna] best trial final test directory: {output_root / 'best_trial_test_eval' / 'results'}")
    print(f"[optuna] best trial final test_accuracy: {final_test_accuracy:.6f}")


if __name__ == "__main__":
    main()
