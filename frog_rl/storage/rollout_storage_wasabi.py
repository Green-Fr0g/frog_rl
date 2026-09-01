"""Rollout storage for WASABI-style state-pair discriminator updates."""

from __future__ import annotations

from typing import Generator, NamedTuple

import torch


class WasabiMiniBatch(NamedTuple):
    """A batch of current/reference states and episode termination flags."""

    current_states: torch.Tensor
    reference_states: torch.Tensor
    dones: torch.Tensor


class WasabiStorage:
    """Store one PPO rollout for discriminator training."""

    def __init__(self, num_steps: int, num_envs: int, state_dim: int, device: str) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.current_states = torch.zeros(num_steps, num_envs, state_dim, device=device)
        self.reference_states = torch.zeros_like(self.current_states)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        self.step = 0

    @property
    def num_samples(self) -> int:
        """Return the number of valid stored state pairs."""
        return self.step * self.num_envs

    def add(self, current_states: torch.Tensor, reference_states: torch.Tensor, dones: torch.Tensor) -> None:
        """Append one vectorized environment step."""
        if self.step >= self.num_steps:
            raise RuntimeError("WASABI storage overflow. Call clear() after each discriminator update.")
        self.current_states[self.step].copy_(current_states.detach())
        self.reference_states[self.step].copy_(reference_states.detach())
        self.dones[self.step].copy_(dones.reshape(-1).to(dtype=torch.bool))
        self.step += 1

    def states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all valid current and reference states as flattened tensors."""
        return (
            self.current_states[: self.step].flatten(0, 1),
            self.reference_states[: self.step].flatten(0, 1),
        )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int) -> Generator[WasabiMiniBatch, None, None]:
        """Yield shuffled feed-forward mini-batches for each discriminator epoch."""
        if self.step == 0:
            return

        current_states, reference_states = self.states()
        dones = self.dones[: self.step].flatten()
        batch_size = current_states.shape[0]
        for _ in range(num_epochs):
            for indices in torch.tensor_split(torch.randperm(batch_size, device=current_states.device), num_mini_batches):
                if indices.numel() != 0:
                    yield WasabiMiniBatch(current_states[indices], reference_states[indices], dones[indices])

    def clear(self) -> None:
        """Mark all stored transitions as reusable."""
        self.step = 0
