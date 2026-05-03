import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.robustness.summations import cs_plus, gts, cs_he
from banditdl.core.topology.gossip import LaplacianGossipMatrix
from banditdl.core.worker.base import BaseWorker
from banditdl.core.worker.byzantine import ByzantineWorker, DecByzantineWorker


_METHODS = {"cs+": cs_plus, "cs_he": cs_he, "gts": gts}


class FixedGraphWorker(BaseWorker):
    def __init__(
        self,
        worker_id,
        data_loader,
        data_loader_test,
        nb_workers,
        nb_byz,
        nb_real_byz,
        aggregator,
        pre_aggregator,
        server_clip,
        bucket_size,
        model,
        learning_rate,
        learning_rate_decay,
        learning_rate_decay_delta,
        weight_decay,
        loss,
        momentum,
        device,
        labelflipping,
        gradient_clip,
        numb_labels,
        nb_neighbors,
        rag,
        b_hat,
        nb_local_steps,
        method,
        comm_graph,
        dissensus,
    ):
        super().__init__(
            worker_id,
            data_loader,
            data_loader_test,
            nb_workers,
            nb_byz,
            nb_real_byz,
            model,
            learning_rate,
            learning_rate_decay,
            learning_rate_decay_delta,
            weight_decay,
            loss,
            momentum,
            device,
            labelflipping,
            gradient_clip,
            numb_labels,
            nb_local_steps,
            rag,
            b_hat,
        )
        self.comm_graph = comm_graph
        self.method = method
        self.dissensus = dissensus
        self.rho = 1.0
        self.W = torch.tensor(LaplacianGossipMatrix(self.comm_graph)).to(device)

        self.robust_aggregator = RobustAggregator(
            aggregator,
            pre_aggregator,
            server_clip,
            nb_neighbors + 1 - b_hat,
            b_hat,
            bucket_size,
            self.model_size,
            self.device,
        )
        self.nb_neighbors = len(list(self.comm_graph.neighbors(self.worker_id)))

        metropolis = True
        if metropolis:
            self.byz_weights = 0
            neighbors = list(self.comm_graph.neighbors(self.worker_id))
            neighbors_degrees = sorted([comm_graph.degree(i) for i in neighbors])
            pivot_degree = comm_graph.degree(self.worker_id)
            for i in range(b_hat):
                try:
                    self.byz_weights += 1 / (neighbors_degrees[i] + pivot_degree + 1)
                except Exception:
                    print(
                        f"Warning: b_hat = {b_hat} is too large compared to the number of neighbors of worker {self.worker_id}, "
                        f"which is {len(neighbors)}"
                    )
        else:
            self.byz_weights = b_hat

        self.num_clipped = []

    def aggregate_and_update_parameters(self, honest_params, args, current_step):
        if self.dissensus:
            self._aggregate_fixed_graph_dissensus(honest_params)
        else:
            self._aggregate_fixed_graph(honest_params, args, current_step)

    def _aggregate_fixed_graph(self, honest_params, args, current_step):
        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))
        neighbors.append(self.worker_id)

        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)
        self.num_selected_byz.append(nb_selected_byz)

        byz_worker = ByzantineWorker(
            len(neighbors),
            nb_selected_byz,
            nb_selected_byz,
            args.attack,
            args.aggregator,
            args.pre_aggregator,
            args.server_clip,
            args.bucket_size,
            self.model_size,
            args.mimic_learning_phase,
            args.gradient_clip,
            args.device,
        )

        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]
        byzantine_params = byz_worker.compute_byzantine_vectors(honest_local_params, None, current_step)
        worker_params = honest_local_params + [byz_param for byz_param in byzantine_params]

        with torch.no_grad():
            worker_params = torch.stack(worker_params)
            differences = worker_params - pivot_params
            robust_aggregate, num_clipped = _METHODS[self.method](
                weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
                gradients=differences,
                byz_weights=self.byz_weights,
            )
            self.num_clipped.append(num_clipped)
            aggregate_params = pivot_params + self.rho * robust_aggregate
            self.set_model_parameters(aggregate_params)

    @torch.no_grad()
    def _aggregate_fixed_graph_dissensus(self, honest_params):
        pivot_params = honest_params[self.worker_id]
        neighbors = list(self.comm_graph.neighbors(self.worker_id))
        neighbors.append(self.worker_id)

        honest_neighbors = [i for i in neighbors if i < self.nb_honest]
        byz_neighbors = [i for i in neighbors if i >= self.nb_honest]
        nb_selected_byz = len(byz_neighbors)
        self.num_selected_byz.append(nb_selected_byz)

        byz_worker = DecByzantineWorker(
            target=self.worker_id,
            honest_neighbors=honest_neighbors,
            nb_honest=self.nb_honest,
            nb_byz_neighbors=nb_selected_byz,
            pivot_params=pivot_params,
            network=self.comm_graph,
            device=self.device,
        )

        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]
        byzantine_params = byz_worker.compute_byzantine_vectors(honest_local_params)
        worker_params = honest_local_params + [byz_param for byz_param in byzantine_params]

        worker_params = torch.stack(worker_params)
        differences = worker_params - pivot_params

        robust_aggregate, num_clipped = _METHODS[self.method](
            weights=self.comm_graph.weights(self.worker_id)[neighbors].clone().detach().requires_grad_(False),
            gradients=differences,
            byz_weights=self.byz_weights,
        )
        self.num_clipped.append(num_clipped)
        aggregate_params = pivot_params + self.rho * robust_aggregate
        self.set_model_parameters(aggregate_params)


FixedGraphP2PWorker = FixedGraphWorker
