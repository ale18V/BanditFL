from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf



def dict_to_cmdlist(dp):
    cmd = []
    for name, value in dp.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{name}")
        else:
            if isinstance(value, (list, tuple)):
                cmd.append(f"--{name}")
                for subval in value:
                    cmd.append(str(subval))
            elif value is not None:
                cmd.append(f"--{name}")
                cmd.append(str(value))
    return cmd


def _module_from_program_path(program_path: str) -> str:
    return pathlib.Path(program_path).with_suffix("").as_posix().replace("/", ".")


def _run_name(cfg: DictConfig, b_hat: int) -> str:
    base = (
        f"{cfg.profile.dataset}-n_{cfg.profile.nb_workers}"
        f"-model_{cfg.profile.model}"
        f"-attack_{cfg.profile.attack}"
        f"-agg_{cfg.profile.params_common.aggregator}"
        f"-neighbors_{cfg.profile.nb_neighbors}"
        f"-f_{cfg.profile.byzcount}"
        f"-alpha_{cfg.profile.alpha}"
        f"-b_hat_{b_hat}"
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
    params["dataset"] = cfg.profile.dataset
    params["model"] = cfg.profile.model
    params["nb-workers"] = int(cfg.profile.nb_workers)
    params["dirichlet-alpha"] = float(cfg.profile.alpha)
    params["nb-decl-byz"] = int(cfg.profile.byzcount)
    params["nb-real-byz"] = int(cfg.profile.byzcount)
    params["nb-neighbors"] = int(cfg.profile.nb_neighbors)
    params["attack"] = cfg.profile.attack
    params["nb-local-steps"] = int(cfg.profile.nb_local_steps)
    params["neighbor-sampler"] = cfg.train.neighbor_sampler

    b_hat_raw = cfg.profile.get("b_hat")
    b_hat = int(cfg.profile.byzcount if b_hat_raw is None else b_hat_raw)
    params["b-hat"] = b_hat

    if cfg.profile.mode == "dynamic":
        params["rag"] = True

    method = cfg.profile.get("method")
    if method is not None:
        params["method"] = method

    # Per-run output directory managed by Hydra.
    hydra_out = pathlib.Path(HydraConfig.get().runtime.output_dir)
    results_root = pathlib.Path(cfg.profile.result_directory)
    run_name = _run_name(cfg, b_hat)
    result_dir = results_root / f"{run_name}-seed_{cfg.seed}"

    module_name = _module_from_program_path(cfg.train.train_program)
    cmd = ["python3", "-OO", "-m", module_name]
    cmd += dict_to_cmdlist(params)
    cmd += ["--seed", str(cfg.seed), "--device", str(cfg.device), "--result-directory", str(result_dir)]

    hydra_out.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, capture_output=True, text=True)

    (hydra_out / "runner.stdout.log").write_text(completed.stdout)
    (hydra_out / "runner.stderr.log").write_text(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Training command failed with code {completed.returncode}. "
            f"See {hydra_out / 'runner.stderr.log'}"
        )


if __name__ == "__main__":
    main()
