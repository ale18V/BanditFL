from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from banditdl.experiments.common import run_sweep


def _configure_params(cfg: DictConfig):
    b_hat_list = list(cfg.profile.b_hat_list)

    def configure(params: dict[str, Any], f: int, _nb_neighbors: int, _attack: str, nb_local: int, byz_index: int, method_value: Any):
        b_hat = f if len(b_hat_list) == 0 else b_hat_list[byz_index]
        params["b-hat"] = b_hat
        params["nb-local-steps"] = nb_local
        params["neighbor-sampler"] = cfg.train.neighbor_sampler
        if cfg.profile.mode == "dynamic":
            params["rag"] = True
        if method_value is not None:
            params["method"] = method_value

    return configure


def _job_name_builder(cfg: DictConfig):
    dataset = cfg.profile.dataset
    nb_workers = cfg.profile.nb_workers
    model = cfg.profile.model
    alpha = cfg.profile.alpha

    def build(params: dict[str, Any], f: int, nb_neighbors: int, attack: str, nb_local: int, method_value: Any):
        base = (
            f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}"
            f"-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}"
        )
        if "b-hat" in params:
            base += f"-b_hat_{params['b-hat']}"
        if "learning-rate" in params:
            base += f"_lr_{params['learning-rate']}"
        if method_value is not None:
            base += f"-{method_value}"
        return base

    return build


def _result_name_builder(cfg: DictConfig):
    dataset = cfg.profile.dataset
    nb_workers = cfg.profile.nb_workers
    model = cfg.profile.model
    alpha = cfg.profile.alpha

    def build(params: dict[str, Any], f: int, nb_neighbors: int, attack: str, nb_local: int, method_value: Any):
        base = (
            f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}"
            f"-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}"
        )
        b_hat = f
        if len(cfg.profile.b_hat_list) > 0:
            # Keep historical naming if b_hat differs from f
            b_hat = cfg.profile.b_hat_list[list(cfg.profile.byzcounts).index(f)]
        base += f"-b_hat_{b_hat}"
        if "learning-rate" in params:
            base += f"_lr_{params['learning-rate']}"
        if method_value is not None:
            base += f"-{method_value}"
        return base

    return build


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    params_common = OmegaConf.to_container(cfg.profile.params_common, resolve=True)
    assert isinstance(params_common, dict)

    methods = list(cfg.profile.method_values)

    run_sweep(
        dataset=cfg.profile.dataset,
        result_directory=cfg.profile.result_directory,
        plot_directory=cfg.profile.plot_directory,
        params_common=params_common,
        nb_workers=int(cfg.profile.nb_workers),
        model=cfg.profile.model,
        alpha=float(cfg.profile.alpha),
        byzcounts=list(cfg.profile.byzcounts),
        nb_neighbors_list=list(cfg.profile.nb_neighbors_list),
        attacks=list(cfg.profile.attacks),
        nb_local_steps=list(cfg.profile.nb_local_steps),
        train_program=cfg.train.train_program,
        configure_params=_configure_params(cfg),
        job_name_builder=_job_name_builder(cfg),
        result_name_builder=_result_name_builder(cfg),
        plot_filename_builder=lambda f, nb_neighbors, _nb_local: f"{cfg.profile.dataset}_f={f}_model={cfg.profile.model}_neighbors={nb_neighbors}.pdf",
        x_max=int(params_common["nb-steps"]),
        method_values=tuple(methods),
        seeds=tuple(cfg.sweep.seeds),
        plot_location=cfg.train.plot_location,
        plot_column=cfg.train.plot_column,
        plot_reduction=cfg.train.plot_reduction,
        legend_builder=(
            (lambda attack_list, method_list: [f"({method}, attack = {attack})" for method in method_list for attack in attack_list])
            if any(method is not None for method in methods)
            else None
        ),
        devices=cfg.get("devices", "auto"),
        supercharge=int(cfg.sweep.supercharge),
    )


if __name__ == "__main__":
    main()
