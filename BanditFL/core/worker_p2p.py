import torch, random, math
import numpy as np
from bayesianbandits import Agent, Arm, NormalRegressor, EXP3A
from . import models; from BanditFL.utils import misc

class P2PWorker(object):
    """A worker for decentralized learning in peer to peer model."""

    def __init__(self, worker_id, data_loader, data_loader_test, nb_workers, nb_byz, nb_real_byz, aggregator, pre_aggregator,server_clip, bucket_size, model,
                 learning_rate, learning_rate_decay, learning_rate_decay_delta, weight_decay, loss, momentum, device, labelflipping,
                 gradient_clip, numb_labels, nb_neighbors, rag, b_hat, nb_local_steps, use_bandit=False):
        self.worker_id = worker_id
        self.nb_workers = nb_workers
        self.nb_neighbors = nb_neighbors
        self.nb_local_steps = nb_local_steps
        self.use_bandit = use_bandit

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
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_learning_rate, weight_decay=weight_decay)
        
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        self.momentum = momentum
        self.gradient_clip = gradient_clip
        
        # Bandit setup
        if self.use_bandit:
            # Each worker views all other workers as potential arms
            arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) 
                    for i in range(nb_workers) if i != worker_id]
            # Standard EXP3: gamma > 0, ix_gamma = 0
            policy = EXP3A(gamma=0.1, eta=1.0, ix_gamma=0)
            self.bandit = Agent(arms, policy)
            self.last_sampled_indices = None
            self.last_loss = None

    def sample_batch(self, mode):
        try:
            return next(self.iterators[mode])
        except StopIteration:
            self.iterators[mode] = iter(self.loaders[mode])
            return next(self.iterators[mode])

    def backward_pass(self, inputs, targets):
        self.model.zero_grad()
        loss = self.loss(self.model(inputs), targets)
        loss.backward()
        return misc.flatten([param.grad for param in self.model.parameters()]), loss.item()

    def compute_gradients(self):
        self.model.train()
        inputs, targets = self.sample_batch("train")
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        grad, loss = self.backward_pass(inputs, targets)
        self.last_loss = loss # Store for bandit reward calculation
        return grad

    def compute_momentum(self):
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(self.compute_gradients(), alpha=1-self.momentum)

        if self.gradient_clip is not None:
            return misc.clip_vector(self.momentum_gradient, self.gradient_clip)

        return self.momentum_gradient

    def local_model_update(self, current_step):
        def update_learning_rate(step):
            if self.learning_rate_decay > 0 and step % self.learning_rate_decay_delta == 0:
                return self.initial_learning_rate / (step / self.learning_rate_decay + 1)
            else:
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

    def aggregate_and_update_parameters(self, all_params, current_step):
        """Neighbor selection using either Uniform or Bandit sampling."""
        indices_list = list(range(self.nb_workers))
        indices_list.remove(self.worker_id)
        
        if self.use_bandit:
            # 1. Update bandit if we have a previous round's outcome
            if self.last_sampled_indices is not None and self.last_loss is not None:
                # Reward is reduction in loss (normalized)
                # This is a placeholder reward logic
                current_loss = self.last_loss
                # We need a way to attribute loss reduction to specific peers.
                # For simplicity in this baseline, we give a shared reward to all sampled peers.
                # In a more advanced version, we'd evaluate the models individually.
                reward = 1.0 if current_loss < getattr(self, 'prev_loss', current_loss) else 0.0
                for idx in self.last_sampled_indices:
                    self.bandit.select_for_update(idx)
                    self.bandit.update(np.array([reward]))
                self.prev_loss = current_loss

            # 2. Sample new neighbors
            # pull(top_k=n) returns [[token1, token2, ...]]
            sampled_indices = self.bandit.pull(top_k=self.nb_neighbors)[0]
            self.last_sampled_indices = sampled_indices
        else:
            # Uniform baseline
            sampled_indices = random.sample(indices_list, self.nb_neighbors)
        
        neighbor_params = [all_params[i] for i in sampled_indices]
        neighbor_params.append(all_params[self.worker_id])
        aggregate_params = torch.stack(neighbor_params).mean(dim=0)

        self.set_model_parameters(aggregate_params)

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
        return correct/total

    def set_gradient(self, gradient):
        gradient = misc.unflatten(gradient, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.grad = gradient[j].detach().clone()

    def set_model_parameters(self, params):
        params = misc.unflatten(params, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = params[j].data.detach().clone()