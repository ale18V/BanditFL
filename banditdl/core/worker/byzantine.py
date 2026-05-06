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

    def train(self) -> None:
        return None

    def aggregate(self, weights) -> None:
        return None

    def pull(self, context):
        if context is None:
            return None
        honest_vectors = context.get("honest_weights", [])
        current_step = context.get("step", 0)
        vectors = self.byzantine_attack.generate_byzantine_vectors(honest_vectors, None, current_step)
        return vectors[0] if len(vectors) > 0 else None

    def compute_validation_accuracy(self):
        return None

    def compute_validation_loss(self):
        return None

    def compute_train_loss(self):
        return None


class DecByzantineWorker(BaseWorker):
    """Decentralized Byzantine participant for fixed-graph dissensus."""

    def __init__(self, worker_id, nb_honest, network, device, epsilon=1):
        super().__init__(worker_id=worker_id, is_byzantine=True)
        self.nb_honest = nb_honest
        self.network = network
        self.device = device
        self.epsilon = epsilon

    def train(self) -> None:
        return None

    def aggregate(self, weights) -> None:
        return None

    def pull(self, context):
        target = context["target"]
        honest_neighbors = context["honest_neighbors"]
        pivot_params = context["pivot_params"]
        honest_local_params = context["honest_local_params"]
        W_i = self.network.weights(target)
        byz_neighbors = [k for k in self.network.neighbors(target) if k >= self.nb_honest]
        total_byz_weights = W_i[byz_neighbors].sum()
        honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - pivot_params
        byzantine_vector = pivot_params - self.epsilon / total_byz_weights * torch.matmul(
            (W_i[honest_neighbors]).unsqueeze(0), differences
        )
        return byzantine_vector.squeeze(0)

    def compute_validation_accuracy(self):
        return None

    def compute_validation_loss(self):
        return None

    def compute_train_loss(self):
        return None
