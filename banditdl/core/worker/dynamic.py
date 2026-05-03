import torch

from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.sampling import UniformNeighborSampler
from banditdl.core.worker.base import BaseWorker
from banditdl.core.worker.byzantine import ByzantineWorker


class DynamicWorker(BaseWorker):
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
        neighbor_sampler=None,
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
        self.nb_neighbors = nb_neighbors
        self.neighbor_sampler = neighbor_sampler or UniformNeighborSampler()
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

    def aggregate_and_update_parameters(self, honest_params, args, current_step):
        if self.rag:
            self._aggregate_with_rag(honest_params, args, current_step)
        else:
            self._aggregate_cgplus(honest_params, args, current_step)

    def _sample_neighbors(self):
        indices_list = list(range(self.nb_honest + self.nb_byz))
        indices_list.remove(self.worker_id)
        return self.neighbor_sampler.sample(indices_list, self.nb_neighbors)

    def _aggregate_cgplus(self, honest_params, args, current_step):
        random_indices = self._sample_neighbors()
        selected_byzantine_indices = [i - self.nb_honest for i in random_indices if i >= self.nb_honest]
        nb_selected_byz = len(selected_byzantine_indices)
        byz_worker = ByzantineWorker(
            len(random_indices),
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
        self.num_selected_byz.append(nb_selected_byz)

        honest_local_params = [honest_params[i] for i in random_indices if i < self.nb_honest]
        byzantine_params = byz_worker.compute_byzantine_vectors(honest_local_params, None, current_step)

        worker_params = [honest_params[i] for i in random_indices if i < self.nb_honest] + [
            byz_param for byz_param in byzantine_params
        ]

        worker_params = torch.stack(worker_params)
        differences = worker_params - honest_params[self.worker_id]
        distances = differences.norm(dim=1)
        clipping_threshold = torch.topk(distances, 2 * self.b_hat).values[-1] if self.b_hat > 0 else torch.inf
        mask = distances[:, None].broadcast_to(differences.shape) > clipping_threshold
        clipped_differences = torch.where(mask, differences * (clipping_threshold / distances[:, None]), differences)

        communication_lr = 1 / (current_step // 250 + 1)
        aggregate_params = (
            honest_params[self.worker_id]
            + communication_lr * clipped_differences.sum(dim=0) * (1 / self.nb_neighbors)
        )
        self.set_model_parameters(aggregate_params)

    def _aggregate_with_rag(self, honest_params, args, current_step):
        random_indices = self._sample_neighbors()
        selected_byzantine_indices = [i - self.nb_honest for i in random_indices if i >= self.nb_honest]
        nb_selected_byz = len(selected_byzantine_indices)
        self.num_selected_byz.append(nb_selected_byz)

        byz_worker = ByzantineWorker(
            len(random_indices),
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
        honest_local_params = [honest_params[i] for i in random_indices if i < self.nb_honest]
        byzantine_params = byz_worker.compute_byzantine_vectors(honest_local_params, None, current_step)
        worker_params = [honest_params[i] for i in random_indices if i < self.nb_honest] + [
            byz_param for byz_param in byzantine_params
        ]
        worker_params.append(honest_params[self.worker_id])
        aggregate_params = self.robust_aggregator.aggregate(worker_params)
        self.set_model_parameters(aggregate_params)


P2PWorker = DynamicWorker
