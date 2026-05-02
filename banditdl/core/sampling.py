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
