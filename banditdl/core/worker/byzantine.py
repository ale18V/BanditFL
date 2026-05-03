"""Byzantine worker implementations for distributed training."""

import torch
from banditdl.core.robustness.attacks import ByzantineAttack
from banditdl.core.robustness.aggregators import RobustAggregator
from banditdl.core.worker.base import BaseWorker


class ByzantineWorker(BaseWorker):
    """Byzantine participant implementing the worker API."""

    def __init__(
        self,
        worker_id,
        nb_workers,
        nb_decl_byz,
        nb_real_byz,
        attack,
        aggregator,
        second_aggregator,
        server_clip,
        bucket_size,
        model_size,
        mimic_learning_phase,
        gradient_clip,
        device,
    ):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        robust_aggregator = RobustAggregator(
            aggregator,
            second_aggregator,
            server_clip,
            nb_workers,
            nb_decl_byz,
            bucket_size,
            model_size,
            device,
        )
        self.byzantine_attack = ByzantineAttack(
            attack,
            nb_real_byz,
            model_size,
            device,
            mimic_learning_phase,
            gradient_clip,
            robust_aggregator,
        )

    def emit_messages(self, honest_vectors, count, current_step):
        return self.byzantine_attack.generate_byzantine_vectors(honest_vectors, None, current_step)[:count]

    def perform_local_step(self, current_step):
        return None

    def aggregate_and_update_parameters(self, *args, **kwargs):
        return None

    def compute_accuracy(self):
        return None


class DecByzantineWorker(BaseWorker):
    """Decentralized Byzantine participant for fixed-graph dissensus."""

    def __init__(self, worker_id, nb_honest, network, device, epsilon=1):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        self.nb_honest = nb_honest
        self.network = network
        self.device = device
        self.epsilon = epsilon

    def emit_message(self, target, honest_neighbors, pivot_params, honest_local_params):
        W_i = self.network.weights(target)
        byz_neighbors = [k for k in self.network.neighbors(target) if k >= self.nb_honest]
        total_byz_weights = W_i[byz_neighbors].sum()
        honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - pivot_params
        byzantine_vector = pivot_params - self.epsilon / total_byz_weights * torch.matmul(
            (W_i[honest_neighbors]).unsqueeze(0), differences
        )
        return byzantine_vector.squeeze(0)

    def perform_local_step(self, current_step):
        return None

    def aggregate_and_update_parameters(self, *args, **kwargs):
        return None

    def compute_accuracy(self):
        return None
