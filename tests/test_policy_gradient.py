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
