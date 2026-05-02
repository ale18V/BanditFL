import os
import signal
import pathlib

from banditdl.core import tools
from banditdl.core import common as misc
from banditdl.core.analysis import study


def bootstrap_run(result_directory, plot_directory):
  torch = __import__("torch")
  with tools.Context("cmdline", "info"):
    args = misc.process_commandline()
    args.result_directory = misc.check_make_dir(result_directory)
    args.plot_directory = misc.check_make_dir(plot_directory)
    if args.devices == "auto":
      if torch.cuda.is_available():
        args.devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
      else:
        args.devices = ["cpu"]
    else:
      args.devices = [name.strip() for name in args.devices.split(",")]

  exit_is_requested, exit_set_requested = tools.onetime("exit")
  signal.signal(signal.SIGINT, exit_set_requested)
  signal.signal(signal.SIGTERM, exit_set_requested)
  return args, exit_is_requested


def build_command(train_program, params):
  module_name = pathlib.Path(train_program).with_suffix("").as_posix().replace("/", ".")
  cmd = ["python3", "-OO", "-m", module_name]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)


def run_sweep(
  *,
  dataset,
  result_directory,
  plot_directory,
  params_common,
  nb_workers,
  model,
  alpha,
  byzcounts,
  nb_neighbors_list,
  attacks,
  nb_local_steps,
  train_program,
  configure_params,
  job_name_builder,
  result_name_builder,
  plot_filename_builder,
  x_max,
  method_values=(None,),
  seeds=(0, 1),
  plot_location="eval",
  plot_column="Accuracy",
  plot_reduction="max",
  legend_builder=None,
  title_builder=None,
  x_label="Step number",
  y_label="Test accuracy",
  y_min=0.1,
  y_max=1.0,
  devices="auto",
  supercharge=1,
):
  args, exit_is_requested = bootstrap_run(result_directory, plot_directory)
  if devices != "auto":
    if isinstance(devices, str):
      args.devices = [name.strip() for name in devices.split(",")]
    else:
      args.devices = list(devices)
  args.supercharge = int(supercharge)

  tools.success("Running experiments...")
  jobs = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge, seeds=seeds)
  seed_values = jobs.get_seeds()

  for nb_local in nb_local_steps:
    for byz_index, f in enumerate(byzcounts):
      for nb_neighbors in nb_neighbors_list:
        for method_value in method_values:
          for attack in attacks:
            params = params_common.copy()
            params["dataset"] = dataset
            params["model"] = model
            params["nb-workers"] = nb_workers
            params["dirichlet-alpha"] = alpha
            params["nb-decl-byz"] = params["nb-real-byz"] = f
            params["nb-neighbors"] = nb_neighbors
            params["attack"] = attack
            params["nb-local-steps"] = nb_local
            if method_value is not None:
              params["method"] = method_value
            configure_params(params, f, nb_neighbors, attack, nb_local, byz_index, method_value)
            jobs.submit(job_name_builder(params, f, nb_neighbors, attack, nb_local, method_value), build_command(train_program, params))

  jobs.wait(exit_is_requested)
  jobs.close()

  if exit_is_requested():
    return

  tools.success("Plotting results...")
  for nb_local in nb_local_steps:
    for f in byzcounts:
      for nb_neighbors in nb_neighbors_list:
        plot = study.LinePlot()
        plotted_any = False
        for method_value in method_values:
          for attack in attacks:
            name = result_name_builder(params_common, f, nb_neighbors, attack, nb_local, method_value)
            try:
              brdl = misc.compute_avg_err_op(name, seed_values, args.result_directory, plot_location, (plot_column, plot_reduction))
              plot.include(brdl[0], plot_column, errs="-err", lalp=0.8)
              plotted_any = True
            except Exception as err:
              tools.warning(f"Skipping plot data for {name!r} ({err})")

        if not plotted_any:
          tools.warning(f"Skipping plot {plot_filename_builder(f, nb_neighbors, nb_local)!r} because no data was available")
          continue

        legend = legend_builder(attacks, method_values) if legend_builder is not None else [f"(attack = {attack})" for attack in attacks for _ in method_values]
        title = title_builder(f, nb_neighbors, nb_local) if title_builder is not None else None
        plot.finalize(title, x_label, y_label, xmin=0, xmax=x_max, ymin=y_min, ymax=y_max, legend=legend)
        plot.save(os.path.join(args.plot_directory, plot_filename_builder(f, nb_neighbors, nb_local)), xsize=3, ysize=1.5)
