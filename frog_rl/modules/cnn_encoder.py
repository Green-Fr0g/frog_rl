# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from frog_rl.networks import CNN


class CNNEncoder(nn.Module):
    """CNN encoder for image observations stored in a TensorDict.

    This module owns one CNN per 4D observation group and concatenates the encoded
    image features into a single latent tensor. It intentionally does not own an
    MLP head, action distribution, actor, or critic.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the CNN encoder.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to encode, for example "policy" or "critic".
            cnn_cfg: Configuration of the CNN encoder(s).
        """
        super().__init__()

        self._resolve_obs_groups(obs, obs_groups, obs_set)

        if cnn_cfg is None:
            raise ValueError("CNN configurations must be provided.")
        # Create a cnn config for each 2D observation group in case only one is provided
        if not all(isinstance(v, dict) for v in cnn_cfg.values()):
            cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
        # Check that the number of configs matches the number of observation groups
        if len(cnn_cfg) != len(self.obs_groups_2d):
            raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
        # Create CNNs for each 2D observation
        cnns = {}
        for idx, obs_group in enumerate(self.obs_groups_2d):
            cnns[obs_group] = CNN(
                input_dim=self.obs_dims_2d[idx],
                input_channels=self.obs_channels_2d[idx],
                **cnn_cfg[obs_group],
            )

        # Compute latent dimension of the CNNs
        self._output_dim = 0
        for cnn in cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before using it as an encoder latent.")
            self._output_dim += int(cnn.output_dim)  # type: ignore[arg-type]

        # Register CNN encoders
        self.cnns = nn.ModuleDict(cnns)

    @property
    def output_dim(self) -> int:
        """Get the concatenated latent dimension."""
        return self._output_dim

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Encode configured image observations and concatenate their latents."""
        return torch.cat([self.cnns[obs_group](obs[obs_group]) for obs_group in self.obs_groups_2d], dim=-1)

    def _resolve_obs_groups(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> None:
        """Select active 4D observation groups and compute their CNN input shapes."""
        active_obs_groups = obs_groups[obs_set]
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        # Iterate through active observation groups and keep only image observations.
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) != 2:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided for the CNN encoder.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
