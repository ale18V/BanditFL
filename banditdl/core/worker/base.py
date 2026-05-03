from abc import ABC, abstractmethod

import torch

from banditdl.data import models
from banditdl.core import common as misc


class BaseWorker(ABC):
    """Worker API for all participants (honest or Byzantine)."""

    def __init__(self, worker_id, is_byzantine=False):
        self.worker_id = worker_id
        self.is_byzantine = is_byzantine

    @abstractmethod
    def perform_local_step(self, current_step):
        """Execute local step and return the message payload for this round."""

    @abstractmethod
    def aggregate_and_update_parameters(self, *args, **kwargs):
        """Consume received messages and update local state."""

    @abstractmethod
    def compute_accuracy(self):
        """Return worker accuracy when meaningful, else None."""


class HonestWorker(BaseWorker):
    """Shared training logic for honest decentralized workers."""

    def __init__(
        self,
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
    ):
        super().__init__(worker_id=worker_id, is_byzantine=False)
        self.nb_byz = nb_byz
        self.nb_real_byz = nb_real_byz
        self.nb_honest = nb_workers - nb_byz
        self.rag = rag
        self.b_hat = b_hat

        self.loaders = {"train": data_loader, "test": data_loader_test}
        self.iterators = {"train": iter(data_loader), "test": iter(data_loader_test)}

        self.initial_learning_rate = self.current_learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delta = learning_rate_decay_delta

        self.device = device
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)()
        self.model.to(self.device)
        self.model_shapes = [param.shape for param in self.model.parameters()]
        self.model_size = len(misc.flatten(self.model.parameters()))

        if self.device == "cuda":
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay
        )
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = momentum
        self.gradient_clip = gradient_clip

        self.labelflipping = labelflipping
        self.numb_labels = numb_labels
        self.nb_local_steps = nb_local_steps
        self.num_selected_byz = []

    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except Exception:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])

    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()])

    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            self.gradient_labelflipping = self.backward_pass(inputs, targets_flipped)
            self.model.train()

        return self.backward_pass(inputs, targets)

    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1 - self.momentum)

        if self.gradient_clip is not None:
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient

    def local_model_update(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            return self.current_learning_rate

        new_learning_rate = update_learning_rate(current_step)
        if self.current_learning_rate != new_learning_rate:
            self.current_learning_rate = new_learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_learning_rate

        self.optimizer.step()

    def perform_local_step(self, current_step):
        for _ in range(self.nb_local_steps):
            self.set_gradient(self.compute_momentum())
            self.local_model_update(current_step)
        return misc.flatten(self.model.parameters())

    @torch.no_grad()
    def compute_accuracy(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs, targets in self.loaders["test"]:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct / total

    def set_gradient(self, gradient):
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()

    def set_model_parameters(self, params):
        params = misc.unflatten(params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = params[j].data.detach().clone()
