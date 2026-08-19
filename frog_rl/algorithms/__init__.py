# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .ppo import PPO
from .amp_discriminator import AMPDiscriminator
from .amp_ppo import AMPPPO
from .distillation import Distillation
from .wasabi import WasabiPPO
from .wasabi_discriminator import WasabiDiscriminator

__all__ = ["AMPDiscriminator", "AMPPPO", "Distillation", "PPO", "WasabiDiscriminator", "WasabiPPO"]
