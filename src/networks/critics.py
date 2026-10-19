"""State-value regression against unnormalized Monte Carlo returns."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    def __init__(
        self, ob_dim: int, n_layers: int, layer_size: int, learning_rate: float
    ):
        super().__init__()
        self.network = ptu.build_mlp(ob_dim, 1, n_layers, layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        loss = F.mse_loss(self(ptu.from_numpy(obs)), ptu.from_numpy(q_values))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite baseline loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"Baseline Loss": loss.item()}
