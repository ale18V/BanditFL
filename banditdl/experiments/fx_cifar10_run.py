
from banditdl.experiments.common import run_sweep


def main():
  params_cifar = {
    "batch-size": 50,
    "loss": "NLLLoss",
    "learning-rate": 0.5,
    "learning-rate-decay": 100,
    "learning-rate-decay-delta": 500,
    "weight-decay": 1e-2,
    "evaluation-delta": 20,
    "nb-steps": 2000,
    "momentum-worker": 0.99,
    "numb-labels": 10,
    "mimic-learning-phase": 400,
    "pre-aggregator": "nnm",
    "aggregator": "trmean",
    "method": "cs+",
  }

  dataset = "cifar10"
  nb_workers = 20
  model = "cnn_cifar_old"
  alpha = 1
  byzcounts = [3, 0]
  nb_neighbors_list = [6]
  attacks = ["dissensus", "ALIE", "SF", "FOE"]
  nb_local_steps = [3, 1]
  methods = ["cs+", "gts", "cs_he"]

  run_sweep(
    dataset=dataset,
    result_directory=f"results_cifar10/results-data-{dataset}-fx-dissensus",
    plot_directory=f"results_cifar10/results-plot-{dataset}-fx-dissensus",
    params_common=params_cifar,
    nb_workers=nb_workers,
    model=model,
    alpha=alpha,
    byzcounts=byzcounts,
    nb_neighbors_list=nb_neighbors_list,
    attacks=attacks,
    nb_local_steps=nb_local_steps,
    train_program="banditdl/experiments/fx_train_p2p.py",
    method_values=methods,
    configure_params=lambda params, f, nb_neighbors, attack, nb_local, byz_index, method_value: params.update({"b-hat": f, "nb-local-steps": nb_local, "method": method_value}),
    job_name_builder=lambda params, f, nb_neighbors, attack, nb_local, method_value: f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-b_hat_{params['b-hat']}_nb_local{nb_local}_lr_{params['learning-rate']}-{params['method']}",
    result_name_builder=lambda params, f, nb_neighbors, attack, nb_local, method_value: f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-b_hat_{f}_nb_local{nb_local}_lr_{params['learning-rate']}-{method_value}",
    plot_filename_builder=lambda f, nb_neighbors, nb_local: f"{dataset}_f={f}_model={model}_neighbors={nb_neighbors}_alpha={alpha}.pdf",
    x_max=params_cifar["nb-steps"],
    legend_builder=lambda attack_list, method_list: [f"({method}, attack = {attack})" for method in method_list for attack in attack_list],
    title_builder=lambda f, nb_neighbors, nb_local: None,
  )


if __name__ == "__main__":
  main()
