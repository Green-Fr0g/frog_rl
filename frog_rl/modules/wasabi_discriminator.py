"""Configurable feed-forward discriminator used by WASABI."""

from __future__ import annotations

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

from frog_rl.networks import EmpiricalNormalization, MLP


class WasabiDiscriminator(nn.Module):
    """Classify policy states against reference motion states."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        normalize_input: bool = True,
        normalization_until: int | None = int(1e8),
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("WASABI discriminator requires at least one hidden layer.")
        self.normalizer = EmpiricalNormalization(state_dim, until=normalization_until) if normalize_input else nn.Identity()
        self.trunk = MLP(state_dim, hidden_dims[-1], hidden_dims, activation=activation)
        self.logit = nn.Linear(hidden_dims[-1], 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return an unbounded discriminator logit for each state."""
        return self.logit(self.trunk(self.normalizer(states)))

    def compute_grad_pen(
        self,
        current_states: torch.Tensor,
        reference_states: torch.Tensor,
        lambda_: float = 10.0,
        gradient_tolerance: float = 0.0,
    ) -> torch.Tensor:
        """Compute the gradient penalty on current and reference states."""
        states = torch.cat([current_states, reference_states], dim=0)
        states.requires_grad = True
        logits = self(states)
        gradients = autograd.grad(
            outputs=logits,
            inputs=states,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        excess_norm = torch.clamp(gradients.norm(2, dim=1) - gradient_tolerance, min=0.0)
        return lambda_ * excess_norm.square().mean()

    @torch.no_grad()
    def reward(
        self,
        states: torch.Tensor,
        reward_type: str,
        reward_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a WASABI/AMP imitation reward without changing module mode."""
        was_training = self.training
        self.eval()
        logits = self(states)
        if reward_type == "log":
            reward = F.softplus(logits)
        elif reward_type == "quad":
            reward = torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
        elif reward_type == "wasserstein":
            reward = logits
        else:
            raise ValueError(f"Unsupported WASABI reward type: {reward_type}")
        self.train(was_training)
        return reward_coef * reward.squeeze(-1), logits.squeeze(-1)
