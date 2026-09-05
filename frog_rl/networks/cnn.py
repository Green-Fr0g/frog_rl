# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
from torch import nn as nn
from typing import TypeVar

from frog_rl.utils import resolve_nn_activation

T = TypeVar("T")


class CNN(nn.Sequential):
    """Convolutional Neural Network.

    The CNN network is a sequence of convolutional layers, optional normalization layers, optional activation functions,
    and optional pooling. The final output can be flattened.
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, ...] | list[int],
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_dim: Height and width of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each convolutional layer or a single kernel size for all layers.
            stride: List of strides for each convolutional layer or a single stride for all layers.
            dilation: List of dilations for each convolutional layer or a single dilation for all layers.
            padding: Padding type to use. Either 'none', 'zeros', 'reflect', 'replicate', or 'circular'.
            norm: List of normalization types for each convolutional layer or a single type for all layers. Either
                'none', 'batch', or 'layer'.
            activation: Activation function to use.
            max_pool: List of booleans indicating whether to apply max pooling after each convolutional layer or a
                single boolean for all layers.
            global_pool: Global pooling type to apply at the end. Either 'none', 'max', or 'avg'.
            flatten: Whether to flatten the output tensor.
        """
        super().__init__()

        # Resolve activation function
        activation_function = resolve_nn_activation(activation)

        # Create layers sequentially
        layers = []
        last_channels = input_channels
        last_dim = input_dim
        for idx in range(len(output_channels)):
            # Get parameters for the current layer
            k = _get_param(kernel_size, idx)
            s = _get_param(stride, idx)
            d = _get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else (0, 0)
            )

            # Append convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            )

            # Append normalization layer if specified
            n = _get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            # Append activation function
            layers.append(activation_function)

            # Apply max pooling if specified
            use_max_pool = _get_param(max_pool, idx)
            if use_max_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            # Update last channels and dimensions
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=use_max_pool)

        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        # Apply flattening if specified
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # Store final output dimension
        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1]

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """Get the output height and width or total output dimension if output is flattened."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize the weights of the CNN with Kaiming initialization."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN."""
        for layer in self:
            x = layer(x)
        return x


def _get_param(param: T | tuple[T, ...] | list[T], idx: int) -> T:
    """Return the indexed value for per-layer parameters, or the scalar value for shared parameters."""
    if isinstance(param, (tuple, list)):
        return param[idx]
    return param


def _to_pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Convert an integer or two-element sequence to a height/width pair."""
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected an int or a two-element sequence, got: {value}")
    return (int(value[0]), int(value[1]))


def _compute_padding(
    input_hw: tuple[int, int],
    kernel: int | tuple[int, int] | list[int],
    stride: int | tuple[int, int] | list[int],
    dilation: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Compute the optimal padding for the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    kernel_h, kernel_w = _to_pair(kernel)
    stride_h, stride_w = _to_pair(stride)
    dilation_h, dilation_w = _to_pair(dilation)
    h = math.ceil(
        (
            stride_h * math.floor(input_hw[0] / stride_h)
            - input_hw[0]
            - stride_h
            + dilation_h * (kernel_h - 1)
            + 1
        )
        / 2
    )
    w = math.ceil(
        (
            stride_w * math.floor(input_hw[1] / stride_w)
            - input_hw[1]
            - stride_w
            + dilation_w * (kernel_w - 1)
            + 1
        )
        / 2
    )
    return (h, w)


def _compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int | tuple[int, int] | list[int],
    stride: int | tuple[int, int] | list[int],
    dilation: int | tuple[int, int] | list[int],
    padding: tuple[int, int],
    is_max_pool: bool = False,
) -> tuple[int, int]:
    """Compute the output height and width of the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    kernel_h, kernel_w = _to_pair(kernel)
    stride_h, stride_w = _to_pair(stride)
    dilation_h, dilation_w = _to_pair(dilation)
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)

    return (h, w)
