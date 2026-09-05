# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .amp_discriminator import AMPDiscriminator
from .cnn_model import CNNModel
from .mlp_model import MLPModel
from .moe_model import MoEModel
from .rnn_model import RNNModel
from .transformer import Transformer
from .vae import VAE
from .wasabi_discriminator import WasabiDiscriminator

__all__ = [
    "AMPDiscriminator",
    "CNNModel",
    "MLPModel",
    "MoEModel",
    "RNNModel",
    "Transformer",
    "VAE",
    "WasabiDiscriminator",
]
