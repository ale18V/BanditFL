
#import banditdl.core.tools
from banditdl.core.analysis import study
from banditdl.core import common as misc
#tools.success("Module loading...")
import signal, torch


import matplotlib.pyplot as plt

# Set the font family to 'DejaVu Sans'
plt.rcParams['font.family'] = 'DejaVu Sans'


# Helper code for plotting the comparison results



params_mnist = {
  "batch-size": 25,
  "loss": "NLLLoss",
  "weight-decay": 1e-4,
  "nb-steps": 200,
  "momentum-worker": 0.9,
  "numb-labels": 10,
  "pre-aggregator": "nnm",
  "aggregator" : "trmean", #"multi_krum", #
  "evaluation-delta":20,
  "nb-workers": 30,
}

params = params_mnist.copy()
n = params['nb-workers']
dataset = "mnist"
model = "cnn_mnist"
attack = "ALIE"
alpha = 1
nb_local = 1
seeds = [0,1]
seeds_fx = [0,1] 
methods = ["cs+", "cs_he", "gts"]

result_directory_el = "results_mnist/results-data-" + dataset + "-final"
result_directory_fx = "results_mnist/results-data-" + dataset + "-fx" 
plot_directory = "plots_dissensus"





for f in [1]:#[0,3,6]:# [0,1]:#
  for nb_neighbors in [3,4,5]:#[15]:#
    for worst in [True, False]:
      name = f"{dataset}-n_{params['nb-workers']}-model_{model}-attack_{attack}-agg_{params['aggregator']}-neighbors_{nb_neighbors}-f_{f}-alpha_{alpha}-nb-local_{nb_local}"
      ev = "eval_worst" if worst else "eval"

      plot = study.LinePlot() 
      for method in methods:
        name_fx = name + f"-{method}"
        print(name_fx)
        brdl = misc.compute_avg_err_op(name_fx, seeds_fx, result_directory_fx, ev, ("Accuracy", "mean"))
        plot.include(brdl[0], "Accuracy", errs="-err", lalp=0.8)

      brdl_el = misc.compute_avg_err_op(name, seeds, result_directory_el, ev, ("Accuracy", "mean")) 
      plot.include(brdl_el[0], "Accuracy", errs="-err", lalp=0.8)


      #legend = [f"(attack = {attack})" for attack in attacks]
      legend = ["cs+", "ClippedGossip", "gts", "RPEL"]

      plot.finalize(rf"$n={n}$, $b={f}$, $s={nb_neighbors}$ ", "Step number", "Test accuracy", xmin=0, xmax=params['nb-steps'], ymin=0.1, ymax=1.0, legend=legend)

      pref = "/comp-worst-" if worst else "/comp-avg-"

      plot.save(plot_directory + pref + dataset + "_n=" + str(n) + "_f=" + str(f) + "_model=" + str(model) + "_neighbors=" + str(nb_neighbors) + ".pdf", xsize=3, ysize=1.5)

      print("Saved to " + plot_directory + pref + dataset + "_n=" + str(n) + "_f=" + str(f) + "_model=" + str(model) + "_neighbors=" + str(nb_neighbors) + ".pdf")



