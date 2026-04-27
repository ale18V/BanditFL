import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from BanditFL import tools

from BanditFL.utils import study, misc
tools.success("Module loading...")
import signal, torch
import matplotlib.pyplot as plt

# Set the font family to 'DejaVu Sans'
plt.rcParams['font.family'] = 'DejaVu Sans'


# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #


#Pick the dataset on which to run experiments
dataset = "cifar10"
result_directory = "results_cifar10/results-data-" + dataset + "-bandit-comp"
plot_directory = "results_cifar10/results-plot-" + dataset + "-bandit-comp" 


with tools.Context("cmdline", "info"):
  args = misc.process_commandline()
  # Make the result directories
  args.result_directory = misc.check_make_dir(result_directory)
  args.plot_directory = misc.check_make_dir(plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")



params_cifar = {
  "batch-size": 50,
  "loss": "NLLLoss",
  "learning-rate": 0.5,
  "learning-rate-decay": 100 ,
  "learning-rate-decay-delta" :500 ,
  "weight-decay": 1e-2,
  "evaluation-delta": 20,
  "nb-steps": 1000,
  "momentum-worker": 0.99,
  "numb-labels": 10,
  "pre-aggregator": "nnm",
  "aggregator": "trmean",
}



nb_workers = 20
model = "cnn_cifar_old"
alpha = 1
nb_neighbors_list = [5]
nb_local_steps = [1]
params_common = params_cifar 
sampling_methods = ["Uniform", "Bandit"]

# Command maker helper
def make_command(params):
  cmd = ["uv", "run", "scripts/train_p2p.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=[0])
seeds = jobs.get_seeds()

for nb_local in nb_local_steps:
  for nb_neighbors in nb_neighbors_list:
    for method in sampling_methods:
      params = params_common.copy()
      params["dataset"] = dataset
      params["model"] = model
      params["nb-workers"] = nb_workers
      params["dirichlet-alpha"] = alpha
      params["nb-neighbors"] = nb_neighbors
      params["nb-local-steps"] = nb_local
      if method == "Bandit":
        params["use-bandit"] = True
      
      jobs.submit(f"{dataset}-n_{nb_workers}-method_{method}-neighbors_{nb_neighbors}-alpha_{alpha}-nb_local{nb_local}", make_command(params))

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")

for nb_local in nb_local_steps:
  for nb_neighbors in nb_neighbors_list:
    plot = study.LinePlot()
    for method in sampling_methods:
      name = f"{dataset}-n_{nb_workers}-method_{method}-neighbors_{nb_neighbors}-alpha_{alpha}-nb_local{nb_local}"
      brdl = misc.compute_avg_err_op(name, seeds, result_directory, "eval", ("Accuracy", "max"))
      plot.include(brdl[0], "Accuracy", errs="-err", lalp=0.8)
    
    plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params_common['nb-steps'], ymin=0.1, ymax=1.0, legend=sampling_methods)
    plot.save(plot_directory + "/" + dataset + "_neighbors=" + str(nb_neighbors) +"_alpha="+str(alpha)+ "_bandit_comp.pdf", xsize=3, ysize=1.5)
