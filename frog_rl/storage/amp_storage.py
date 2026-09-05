"""Replay buffer storing AMP policy transitions."""

from __future__ import annotations

from typing import Generator

import numpy as np
import torch


class AMPStorage:
    """Fixed-size circular buffer to store AMP ``(state, next_state)`` transitions.

    Args:
        obs_dim: Dimension of a single state.
        buffer_size: Maximum number of transitions in the buffer.
        device: Device to store the tensors on.
    """

    def __init__(self, obs_dim: int, buffer_size: int, device: str) -> None:
        """Initialize an AMPStorage object."""
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def add(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Add new states to memory.

        Args:
            states: Current states of shape ``(num_states, obs_dim)``.
            next_states: Next states of shape ``(num_states, obs_dim)``.
        """
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[: self.buffer_size - self.step]
            self.next_states[self.step : self.buffer_size] = next_states[: self.buffer_size - self.step]
            self.states[: end_idx - self.buffer_size] = states[self.buffer_size - self.step :]
            self.next_states[: end_idx - self.buffer_size] = next_states[self.buffer_size - self.step :]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def mini_batch_generator(self, num_mini_batch: int, mini_batch_size: int) -> Generator:
        """Yield random mini-batches of ``(states, next_states)``.

        Args:
            num_mini_batch: Number of mini-batches to yield.
            mini_batch_size: Number of transitions per mini-batch.

        Yields:
            Tuple of ``(states, next_states)`` mini-batches.
        """
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (
                self.states[sample_idxs].to(self.device),
                self.next_states[sample_idxs].to(self.device),
            )