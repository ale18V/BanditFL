import pytest
import torch

from banditdl.core.sampling import (
    MultiArmedBanditSampler,
    ParameterDistanceReward,
    make_neighbor_sampler,
    make_reward_strategy,
)
from banditdl.experiments.engine import _best_fixed_subset


def test_bandit_sampler_prefers_high_reward_arm():
    sampler = MultiArmedBanditSampler(epsilon=0.0)
    sampler.update([1, 2, 3], [0.1, 0.9, 0.2])

    assert sampler.sample([1, 2, 3], 1) == [2]


def test_neighbor_sampler_factory():
    assert make_neighbor_sampler("uniform").sample([1, 2, 3], 2)
    assert isinstance(make_neighbor_sampler("bandit"), MultiArmedBanditSampler)


def test_parameter_distance_reward():
    reward = ParameterDistanceReward()

    assert reward.score(torch.tensor([1.0]), [torch.tensor([1.0])]) == [1.0]
    assert reward.score(torch.tensor([1.0]), [torch.tensor([3.0])]) == pytest.approx(
        [1 / 3]
    )


def test_reward_strategy_factory():
    assert isinstance(
        make_reward_strategy("parameter_distance"), ParameterDistanceReward
    )


def test_best_fixed_subset_excludes_self():
    selected, reward = _best_fixed_subset(torch.tensor([0.9, 0.5, 0.8, 0.1]), 0, 2)

    assert selected.tolist() == [2, 1]
    assert reward == pytest.approx(1.3)
