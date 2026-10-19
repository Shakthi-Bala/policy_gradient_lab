"""Monte Carlo policy gradients with reward-to-go, a value baseline, and GAE."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from infrastructure import pytorch_util as ptu
from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: float | None,
        baseline_gradient_steps: int | None,
        gae_lambda: float | None,
        normalize_advantages: bool,
    ):
        super().__init__()
        if not 0 <= gamma <= 1:
            raise ValueError("discount must be in [0, 1]")
        if gae_lambda is not None and (not use_baseline or not 0 <= gae_lambda <= 1):
            raise ValueError("GAE requires a baseline and lambda in [0, 1]")
        if use_baseline and (
            not baseline_gradient_steps or baseline_gradient_steps < 1
        ):
            raise ValueError("baseline_gradient_steps must be positive")
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )
        self.critic = (
            ValueCritic(ob_dim, n_layers, layer_size, baseline_learning_rate)
            if use_baseline
            else None
        )
        self.baseline_gradient_steps = baseline_gradient_steps
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        q_values = np.concatenate(self._calculate_q_vals(rewards))
        obs, actions, rewards, terminals = map(
            np.concatenate, (obs, actions, rewards, terminals)
        )
        # Compute advantages with the old value function, before either network updates.
        advantages = self._estimate_advantage(obs, rewards, q_values, terminals)
        info = self.actor.update(obs, actions, advantages)
        if self.critic is not None:
            losses = []
            for _ in range(self.baseline_gradient_steps):
                losses.append(self.critic.update(obs, q_values)["Baseline Loss"])
            info["Baseline Loss"] = float(np.mean(losses))
        return info

    def _discounted_return(self, rewards: Sequence[float]) -> np.ndarray:
        total = self._discounted_reward_to_go(rewards)[0]
        return np.full(len(rewards), total, dtype=np.float32)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> np.ndarray:
        returns = np.empty(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = float(rewards[t]) + self.gamma * running
            returns[t] = running
        return returns

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        estimator = (
            self._discounted_reward_to_go
            if self.use_reward_to_go
            else self._discounted_return
        )
        return [estimator(episode) for episode in rewards]

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        if self.critic is None:
            advantages = q_values.copy()
        else:
            with torch.no_grad():
                values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape
            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                values = np.append(values, 0.0)
                advantages = np.zeros(len(rewards) + 1, dtype=np.float32)
                for t in reversed(range(len(rewards))):
                    # A terminal marks every rollout boundary, including time limits.
                    # This follows the assignment's finite-horizon, zero-tail convention.
                    continuation = 1.0 - terminals[t]
                    delta = (
                        rewards[t]
                        + self.gamma * continuation * values[t + 1]
                        - values[t]
                    )
                    advantages[t] = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * continuation
                        * advantages[t + 1]
                    )
                advantages = advantages[:-1]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.astype(np.float32)
