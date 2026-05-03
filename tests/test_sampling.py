from banditdl.core.sampling import MultiArmedBanditSampler, make_neighbor_sampler


def test_bandit_sampler_prefers_high_reward_arm():
    sampler = MultiArmedBanditSampler(epsilon=0.0)
    sampler.update([1, 2, 3], [0.1, 0.9, 0.2])

    assert sampler.sample([1, 2, 3], 1) == [2]


def test_neighbor_sampler_factory():
    assert make_neighbor_sampler("uniform").sample([1, 2, 3], 2)
    assert isinstance(make_neighbor_sampler("bandit"), MultiArmedBanditSampler)
