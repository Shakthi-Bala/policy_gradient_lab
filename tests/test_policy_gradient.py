"""Numerical invariants that catch return, boundary, and gradient mistakes."""

import numpy as np
import pytest
import torch
from torch import nn

from agents.pg_agent import PGAgent
from infrastructure import utils
from networks.policies import MLPPolicyPG


def make_agent(**kwargs):
    config = {
        "ob_dim": 2,
        "ac_dim": 2,
        "discrete": True,
        "n_layers": 1,
        "layer_size": 8,
        "gamma": 0.5,
        "learning_rate": 0.01,
        "use_baseline": False,
        "use_reward_to_go": True,
        "baseline_learning_rate": 0.01,
        "baseline_gradient_steps": 5,
        "gae_lambda": None,
        "normalize_advantages": False,
    }
    config.update(kwargs)
    return PGAgent(**config)


def test_returns_and_episode_independence():
    agent = make_agent()
    np.testing.assert_allclose(agent._discounted_reward_to_go([1, 2, 4]), [3, 4, 4])
    np.testing.assert_allclose(agent._discounted_return([1, 2, 4]), [3, 3, 3])
    q = agent._calculate_q_vals([np.array([1, 2]), np.array([100])])
    np.testing.assert_allclose(np.concatenate(q), [2, 2, 100])


class FixedCritic(nn.Module):
    def forward(self, observations):
        return observations[:, 0]


@pytest.mark.parametrize("lam", [0, 0.95, 1])
def test_gae_matches_explicit_residual_sum_and_stops_at_boundaries(lam):
    agent = make_agent(use_baseline=True, gae_lambda=lam)
    agent.critic = FixedCritic()
    obs = np.array([[3, 0], [4, 0], [50, 0]], dtype=np.float32)
    rewards = np.array([1, 2, 100], dtype=np.float32)
    q = np.array([2, 2, 100], dtype=np.float32)
    result = agent._estimate_advantage(obs, rewards, q, np.array([0, 1, 1]))
    # delta = [0, -2, 50]; no leakage of the second episode's value 50.
    np.testing.assert_allclose(result, [-lam, -2, 50], atol=1e-6)
    if lam == 1:
        np.testing.assert_allclose(result, q - obs[:, 0])


def test_advantage_normalization_handles_constant_batch():
    agent = make_agent(normalize_advantages=True)
    a = agent._estimate_advantage(np.zeros((3, 2)), np.ones(3), np.ones(3), np.ones(3))
    np.testing.assert_array_equal(a, np.zeros(3))


@pytest.mark.parametrize("discrete,action_dim", [(True, 2), (False, 3)])
def test_policy_shapes_and_positive_advantage_increases_likelihood(
    discrete, action_dim
):
    torch.manual_seed(2)
    policy = MLPPolicyPG(action_dim, 2, discrete, 1, 8, 0.01)
    obs = np.zeros((4, 2), dtype=np.float32)
    actions = (
        np.zeros(4, dtype=np.float32)
        if discrete
        else np.zeros((4, action_dim), dtype=np.float32)
    )
    action_tensor = torch.tensor(actions).long() if discrete else torch.tensor(actions)
    before = policy(torch.tensor(obs)).log_prob(action_tensor).detach()
    assert before.shape == (4,)
    policy.update(obs, actions, np.ones(4, dtype=np.float32))
    after = policy(torch.tensor(obs)).log_prob(action_tensor).detach()
    assert torch.all(after > before)
    sample = policy.get_action(obs[0])
    assert sample.shape == (() if discrete else (action_dim,))


def test_baseline_regression_reduces_error():
    torch.manual_seed(3)
    critic = make_agent(use_baseline=True).critic
    obs = np.zeros((8, 2), dtype=np.float32)
    targets = np.full(8, 2, dtype=np.float32)
    initial = critic.update(obs, targets)["Baseline Loss"]
    for _ in range(50):
        final = critic.update(obs, targets)["Baseline Loss"]
    assert final < initial / 10


def test_time_limit_and_sampling_budget():
    import gym

    env = gym.make("CartPole-v0", disable_env_checker=True)
    env.reset(seed=1)
    agent = make_agent(ob_dim=4)
    paths, steps = utils.sample_trajectories(env, agent.actor, 7, max_length=3)
    assert steps == 9
    assert len(paths) == 3
    for path in paths:
        np.testing.assert_array_equal(path["terminal"], [0, 0, 1])
    env.close()


def test_gae_requires_baseline():
    with pytest.raises(ValueError, match="GAE requires"):
        make_agent(gae_lambda=0.95)


@pytest.mark.parametrize("discrete", [True, False])
def test_numpy_rollout_matches_torch_parameters_and_invalidates_cache(discrete):
    torch.manual_seed(17)
    policy = MLPPolicyPG(3, 2, discrete, 2, 16, 0.01)
    obs = np.array([0.4, -0.7], dtype=np.float32)
    output, std = policy._numpy_distribution_parameters(obs)
    with torch.no_grad():
        net = policy.logits_net if discrete else policy.mean_net
        expected = net(torch.tensor(obs)).numpy()
    np.testing.assert_allclose(output, expected, atol=2e-7)
    if not discrete:
        np.testing.assert_allclose(std, policy.logstd.detach().exp().numpy())
    policy.rollout_backend = "numpy"
    assert policy.get_action(obs).shape == (() if discrete else (3,))
    batch_obs = np.repeat(obs[None], 4, axis=0)
    actions = (
        np.zeros(4, dtype=np.float32)
        if discrete
        else np.zeros((4, 3), dtype=np.float32)
    )
    policy.update(batch_obs, actions, np.ones(4, dtype=np.float32))
    updated, _ = policy._numpy_distribution_parameters(obs)
    with torch.no_grad():
        expected_updated = net(torch.tensor(obs)).numpy()
    assert not np.allclose(output, updated)
    np.testing.assert_allclose(updated, expected_updated, atol=2e-7)


def test_numpy_categorical_sampling_matches_distribution():
    torch.manual_seed(17)
    np.random.seed(17)
    policy = MLPPolicyPG(3, 2, True, 1, 8, 0.01)
    policy.rollout_backend = "numpy"
    obs = np.zeros(2, dtype=np.float32)
    expected = policy(torch.tensor(obs)).probs.detach().numpy()
    samples = np.array([policy.get_action(obs) for _ in range(4000)])
    observed = np.bincount(samples, minlength=3) / len(samples)
    np.testing.assert_allclose(observed, expected, atol=0.03)


@pytest.mark.parametrize("seed", [1, 17, 42])
def test_lunar_particle_cleanup_preserves_exact_dynamics(seed):
    from infrastructure.envs import make_env

    reference = make_env("LunarLander-v2", disable_env_checker=True)
    optimized = make_env(
        "LunarLander-v2", prune_lunar_particles=True, disable_env_checker=True
    )
    np.testing.assert_array_equal(
        reference.reset(seed=seed), optimized.reset(seed=seed)
    )
    rng = np.random.default_rng(seed)
    try:
        for step in range(1000):
            action = int(rng.integers(4))
            expected, actual = reference.step(action), optimized.step(action)
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:3] == actual[1:3]
            assert len(optimized.unwrapped.particles) == 0
            if expected[2]:
                np.testing.assert_array_equal(
                    reference.reset(seed=seed + step + 1),
                    optimized.reset(seed=seed + step + 1),
                )
    finally:
        reference.close()
        optimized.close()
