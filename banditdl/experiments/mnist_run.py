
from banditdl.experiments.common import run_sweep


def main():
  params_mnist = {
    "batch-size": 25,
    "loss": "NLLLoss",
    "weight-decay": 1e-4,
    "nb-steps": 200,
    "momentum-worker": 0.9,
    "numb-labels": 10,
    "rag": True,
    "pre-aggregator": "nnm",
    "aggregator": "trmean",
    "evaluation-delta": 20,
  }

  dataset = "mnist"
  nb_workers = 100
  model = "cnn_mnist"
  alpha = 1
  byzcounts = [1, 0]
  b_hat_list = [1, 0]
  nb_neighbors_list = [3, 4, 5]
  nb_local_steps = [1, 3]
  attacks = ["ALIE"]

  run_sweep(
    dataset=dataset,
    result_directory=f"results_mnist/results-data-{dataset}-iclr",
    plot_directory=f"results_mnist/results-plot-{dataset}-iclr",
    params_common=params_mnist,
    nb_workers=nb_workers,
    model=model,
    alpha=alpha,
    byzcounts=byzcounts,
    nb_neighbors_list=nb_neighbors_list,
    attacks=attacks,
    nb_local_steps=nb_local_steps,
    train_program="banditdl/experiments/train_p2p.py",
    configure_params=lambda params, f, nb_neighbors, attack, nb_local, byz_index, method_value: params.update({"b-hat": b_hat_list[byz_index], "rag": True}),
    job_name_builder=lambda params, f, nb_neighbors, attack, nb_local, method_value: f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}",
    result_name_builder=lambda params, f, nb_neighbors, attack, nb_local, method_value: f"{dataset}-n_{nb_workers}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}",
    plot_filename_builder=lambda f, nb_neighbors, nb_local: f"{dataset}_f={f}_model={model}_neighbors={nb_neighbors}.pdf",
    x_max=params_mnist["nb-steps"],
  )


if __name__ == "__main__":
  main()
