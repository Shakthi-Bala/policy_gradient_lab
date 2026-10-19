"""Categorical and diagonal Gaussian policies with score-function updates."""

import itertools

import numpy as np
import torch
from torch import distributions as D
from torch import nn

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()
        self.discrete = discrete
        network = ptu.build_mlp(ob_dim, ac_dim, n_layers, layer_size)
        if discrete:
            self.logits_net = network
            parameters = network.parameters()
        else:
            self.mean_net = network
            self.logstd = nn.Parameter(torch.zeros(ac_dim, device=ptu.device))
            parameters = itertools.chain([self.logstd], network.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return ptu.to_numpy(
            self(ptu.from_numpy(np.asarray(obs, dtype=np.float32))).sample()
        )

    def forward(self, obs: torch.Tensor) -> D.Distribution:
        if self.discrete:
            return D.Categorical(logits=self.logits_net(obs))
        # Independent sums log probabilities across action coordinates exactly once.
        return D.Independent(D.Normal(self.mean_net(obs), self.logstd.exp()), 1)

    def update(self, obs, actions, *args, **kwargs):
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    def update(
        self, obs: np.ndarray, actions: np.ndarray, advantages: np.ndarray
    ) -> dict:
        distribution = self(ptu.from_numpy(obs))
        action_tensor = ptu.from_numpy(actions)
        if self.discrete:
            action_tensor = action_tensor.long()
        loss = -(
            distribution.log_prob(action_tensor) * ptu.from_numpy(advantages).detach()
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite actor loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"Actor Loss": loss.item()}
