from abc import ABC, abstractmethod
import random

import torch
from mabwiser.mab import MAB, LearningPolicy


class RewardStrategy(ABC):
    @abstractmethod
    def score(self, local_weights, neighbor_weights) -> list[float]:
        """Compute one reward per selected neighbor."""


class ParameterDistanceReward(RewardStrategy):
    def score(self, local_weights, neighbor_weights) -> list[float]:
        return [
            1 / (1 + torch.norm(weight - local_weights).item())
            for weight in neighbor_weights
        ]


def make_reward_strategy(name):
    if name == "parameter_distance":
        return ParameterDistanceReward()
    raise ValueError(f"Unknown bandit reward strategy: {name}")


class UniformNeighborSampler:
    """Uniformly sample neighbors without replacement."""

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if rng is None:
            return random.sample(population, k)
        return rng.sample(population, k)

    def update(self, population, rewards) -> None:
        return None


class MultiArmedBanditSampler:
    """MABWiser-backed epsilon-greedy neighbor sampler."""

    def __init__(self, epsilon=0.1, initial_value=0.0, seed=123456):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.seed = seed
        self._mab = None
        self._arms = set()

    def _ensure_mab(self, population):
        arms = set(population)
        if self._mab is not None and arms == self._arms:
            return
        self._arms = arms
        self._mab = MAB(
            arms=list(population),
            learning_policy=LearningPolicy.EpsilonGreedy(epsilon=self.epsilon),
            seed=self.seed,
        )
        self._mab.fit(
            decisions=list(population),
            rewards=[self.initial_value] * len(population),
        )

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if k == 0:
            return []

        rng = rng or random
        population = list(population)
        self._ensure_mab(population)

        if k == 1:
            return [self._mab.predict()]

        if rng.random() < self.epsilon:
            return rng.sample(population, k)

        rng.shuffle(population)
        expectations = self._mab.predict_expectations()
        return sorted(
            population,
            key=lambda arm: expectations.get(arm, self.initial_value),
            reverse=True,
        )[:k]

    def update(self, population, rewards) -> None:
        population = list(population)
        rewards = list(rewards)
        if not population:
            return None
        if self._mab is None or any(arm not in self._arms for arm in population):
            self._ensure_mab(population)
        self._mab.partial_fit(decisions=population, rewards=rewards)
        return None


def make_neighbor_sampler(name, **kwargs):
    if name == "uniform":
        return UniformNeighborSampler()
    if name in {"bandit", "epsilon_greedy"}:
        return MultiArmedBanditSampler(**kwargs)
    raise ValueError(f"Unknown neighbor sampler: {name}")
