import random


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
    """Epsilon-greedy neighbor sampler.

    Each neighbor is an arm. Rewards are supplied by the dynamic worker after
    observing selected neighbor weights.
    """

    def __init__(self, epsilon=0.1, initial_value=0.0):
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.counts = {}
        self.values = {}

    def sample(self, population, k, rng=None):
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > len(population):
            raise ValueError("k cannot exceed population size")
        if k == 0:
            return []

        rng = rng or random
        population = list(population)
        if rng.random() < self.epsilon:
            return rng.sample(population, k)

        rng.shuffle(population)
        return sorted(
            population,
            key=lambda arm: self.values.get(arm, self.initial_value),
            reverse=True,
        )[:k]

    def update(self, population, rewards) -> None:
        for arm, reward in zip(population, rewards, strict=True):
            previous_count = self.counts.get(arm, 0)
            previous_value = self.values.get(arm, self.initial_value)
            count = previous_count + 1
            self.counts[arm] = count
            self.values[arm] = previous_value + (reward - previous_value) / count


def make_neighbor_sampler(name, **kwargs):
    if name == "uniform":
        return UniformNeighborSampler()
    if name in {"bandit", "epsilon_greedy"}:
        return MultiArmedBanditSampler(**kwargs)
    raise ValueError(f"Unknown neighbor sampler: {name}")
