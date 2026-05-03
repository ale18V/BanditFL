from __future__ import annotations

import os
import pathlib
import random
from types import SimpleNamespace

import numpy as np
import torch

from banditdl.core import common as misc
from banditdl.core import tools
from banditdl.core.sampling import UniformNeighborSampler
from banditdl.core.topology.fxgraph import generate_connected_graph
from banditdl.core.topology.graph import CommunicationNetwork
from banditdl.core.worker.byzantine import ByzantineWorker, DecByzantineWorker
from banditdl.core.worker.dynamic import DynamicWorker
from banditdl.core.worker.fixed import FixedGraphWorker
from banditdl.data import dataset


def _setup_seed(seed: int) -> None:
    reproducible = seed >= 0
    if reproducible:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible


def _make_args(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> SimpleNamespace:
    args = dict(params)
    args.setdefault("hetero", False)
    args.setdefault("distinct-data", False)
    args.setdefault("dirichlet-alpha", None)
    args.setdefault("nb-datapoints", None)
    args.setdefault("numb-labels", None)
    args.setdefault("batch-size-test", 100)
    args.setdefault("loss", "NLLLoss")
    args.setdefault("weight-decay", 0)
    args.setdefault("momentum-worker", 0.99)
    args.setdefault("bucket-size", 1)
    args.setdefault("pre-aggregator", None)
    args.setdefault("aggregator", "average")
    args.setdefault("learning-rate", 0.5)
    args.setdefault("learning-rate-decay", 5000)
    args.setdefault("learning-rate-decay-delta", 1)
    args.setdefault("mimic-learning-phase", None)
    args.setdefault("gradient-clip", None)
    args.setdefault("server-clip", False)
    args.setdefault("rag", False)
    args.setdefault("method", "cs+")
    args.setdefault("attack", None)
    args["result-directory"] = str(result_dir)
    args["seed"] = seed
    args["device"] = device
    # normalize dashed keys for existing code style
    normalized = {k.replace("-", "_"): v for k, v in args.items()}
    normalized["nb_honests"] = normalized["nb_workers"] - normalized["nb_real_byz"]
    return SimpleNamespace(**normalized)


def _init_workers_dynamic(args, train_loader_dict, test_loader):
    neighbor_sampler = UniformNeighborSampler()
    workers = []
    for worker_id in range(args.nb_honests):
        w = DynamicWorker(
            worker_id,
            train_loader_dict[worker_id],
            test_loader,
            args.nb_workers,
            args.nb_decl_byz,
            args.nb_real_byz,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            args.model,
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_delta,
            args.weight_decay,
            args.loss,
            args.momentum_worker,
            args.device,
            args.attack == "LF",
            args.gradient_clip,
            args.numb_labels,
            args.nb_neighbors,
            getattr(args, "sampling_ratio", None),
            args.rag,
            args.b_hat,
            args.nb_local_steps,
            neighbor_sampler=neighbor_sampler,
        )
        if worker_id > 0:
            w.model.load_state_dict(workers[0].model.state_dict())
        workers.append(w)
    return workers


def run_dynamic(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    args = _make_args(params, result_dir, seed, device)
    _setup_seed(args.seed)

    train_loader_dict, test_loader = dataset.make_train_test_datasets(
        args.dataset,
        heterogeneity=args.hetero,
        numb_labels=args.numb_labels,
        alpha_dirichlet=args.dirichlet_alpha,
        distinct_datasets=args.distinct_data,
        nb_datapoints=args.nb_datapoints,
        honest_workers=args.nb_honests,
        train_batch=args.batch_size,
        test_batch=args.batch_size_test,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    workers = _init_workers_dynamic(args, train_loader_dict, test_loader)

    byz_workers = [
        ByzantineWorker(
            worker_id=i,
            nb_workers=args.nb_workers,
            nb_decl_byz=args.nb_decl_byz,
            nb_real_byz=args.nb_real_byz,
            attack=args.attack,
            aggregator=args.aggregator,
            second_aggregator=args.pre_aggregator,
            server_clip=args.server_clip,
            bucket_size=args.bucket_size,
            model_size=workers[0].model_size,
            mimic_learning_phase=args.mimic_learning_phase,
            gradient_clip=args.gradient_clip,
            device=args.device,
        )
        for i in range(args.nb_honests, args.nb_workers)
    ]

    fd_eval = (result_dir / "eval").open("w")
    fd_eval_worst = (result_dir / "eval_worst").open("w")
    misc.make_result_file(fd_eval, ["Step number", "Cross-accuracy"])
    misc.make_result_file(fd_eval_worst, ["Step number", "Cross-accuracy"])

    accuracies = []
    for current_step in range(args.nb_steps + 1):
        if args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0:
            accs = [w.compute_accuracy() for w in workers]
            accuracies.append(accs)
            misc.store_result(fd_eval, current_step, sum(accs) / len(accs))

        for w in workers:
            w.train()
        honest_weights = [w.pull(None) for w in workers]

        for w in workers:
            neighbor_indices = w._sample_neighbors()
            honest_neighbor_weights = [honest_weights[i] for i in neighbor_indices if i < args.nb_honests]
            byz_neighbor_ids = [i for i in neighbor_indices if i >= args.nb_honests]
            w.num_selected_byz.append(len(byz_neighbor_ids))
            byz_weights = [
                byz_workers[0].pull({"honest_weights": honest_weights, "step": current_step})
                for _ in byz_neighbor_ids
            ] if byz_neighbor_ids and byz_workers else []
            w.aggregate(honest_neighbor_weights + byz_weights)

    if accuracies:
        worst_idx = min(range(len(workers)), key=lambda i: accuracies[-1][i])
        for i, accs in enumerate(accuracies):
            misc.store_result(fd_eval_worst, i * args.evaluation_delta, accs[worst_idx])

    np.save(os.path.join(result_dir, "accuracies.npy"), np.array(accuracies))


def run_fixed(params: dict, result_dir: pathlib.Path, seed: int, device: str) -> None:
    args = _make_args(params, result_dir, seed, device)
    _setup_seed(args.seed)

    train_loader_dict, test_loader = dataset.make_train_test_datasets(
        args.dataset,
        heterogeneity=args.hetero,
        numb_labels=args.numb_labels,
        alpha_dirichlet=args.dirichlet_alpha,
        distinct_datasets=args.distinct_data,
        nb_datapoints=args.nb_datapoints,
        honest_workers=args.nb_honests,
        train_batch=args.batch_size,
        test_batch=args.batch_size_test,
    )

    nb_edges = args.nb_workers * args.nb_neighbors // 2
    g = generate_connected_graph(args.nb_workers, nb_edges, seed=args.seed)
    comm_graph = CommunicationNetwork(g, weights_method="metropolis", device=args.device if args.device != "auto" else "cpu")
    dissensus = args.attack == "dissensus"

    result_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for worker_id in range(args.nb_honests):
        w = FixedGraphWorker(
            worker_id,
            train_loader_dict[worker_id],
            test_loader,
            args.nb_workers,
            args.nb_decl_byz,
            args.nb_real_byz,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            args.model,
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_delta,
            args.weight_decay,
            args.loss,
            args.momentum_worker,
            args.device,
            args.attack == "LF",
            args.gradient_clip,
            args.numb_labels,
            args.nb_neighbors,
            args.rag,
            args.b_hat,
            args.nb_local_steps,
            args.method,
            comm_graph,
            dissensus,
        )
        if worker_id > 0:
            w.model.load_state_dict(workers[0].model.state_dict())
        workers.append(w)

    byz_workers = {
        i: ByzantineWorker(i, args.nb_workers, args.nb_decl_byz, args.nb_real_byz, args.attack, args.aggregator,
                           args.pre_aggregator, args.server_clip, args.bucket_size, workers[0].model_size,
                           args.mimic_learning_phase, args.gradient_clip, args.device)
        for i in range(args.nb_honests, args.nb_workers)
    }
    dec_byz_workers = {i: DecByzantineWorker(i, args.nb_honests, comm_graph, args.device) for i in range(args.nb_honests, args.nb_workers)}

    fd_eval = (result_dir / "eval").open("w")
    fd_eval_worst = (result_dir / "eval_worst").open("w")
    misc.make_result_file(fd_eval, ["Step number", "Cross-accuracy"])
    misc.make_result_file(fd_eval_worst, ["Step number", "Cross-accuracy"])

    accuracies = []
    for current_step in range(args.nb_steps + 1):
        if args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0:
            accs = [w.compute_accuracy() for w in workers]
            accuracies.append(accs)
            misc.store_result(fd_eval, current_step, sum(accs) / len(accs))

        for w in workers:
            w.train()
        honest_weights = [w.pull(None) for w in workers]

        for w in workers:
            neighbors = list(w.comm_graph.neighbors(w.worker_id)) + [w.worker_id]
            honest_neighbors = [i for i in neighbors if i < args.nb_honests]
            byz_neighbors = [i for i in neighbors if i >= args.nb_honests]
            w.num_selected_byz.append(len(byz_neighbors))
            honest_neighbor_weights = [honest_weights[i] for i in honest_neighbors]
            if dissensus:
                byz_weights = [
                    dec_byz_workers[i].pull({
                        "target": w.worker_id,
                        "honest_neighbors": honest_neighbors,
                        "pivot_params": w.pull(None),
                        "honest_local_params": honest_neighbor_weights,
                    })
                    for i in byz_neighbors
                ]
            else:
                byz_weights = [
                    byz_workers[byz_neighbors[0]].pull({"honest_weights": honest_weights, "step": current_step})
                    for _ in byz_neighbors
                ] if byz_neighbors else []
            w.aggregate(honest_neighbor_weights + byz_weights)

    if accuracies:
        worst_idx = min(range(len(workers)), key=lambda i: accuracies[-1][i])
        for i, accs in enumerate(accuracies):
            misc.store_result(fd_eval_worst, i * args.evaluation_delta, accs[worst_idx])

    np.save(os.path.join(result_dir, "accuracies.npy"), np.array(accuracies))
