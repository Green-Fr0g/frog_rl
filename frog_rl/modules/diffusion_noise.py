# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiffusionNoise(nn.Module):
    """Forward / reverse diffusion schedule for action sequences.

    Implements the DDPM forward process used by Diffusion Policy style trainers:

    .. math::

        x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon

    and the matching reverse helpers (``p_sample``, ``ddim_step``) plus epsilon /
    sample / v target conversions.

    Default action layout is ``(batch, horizon, action_dim)``. Coefficient extraction
    broadcasts over trailing dims, so ``(batch, action_dim)`` also works.
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: str = "cpu",
    ) -> None:
        """Initialize beta / alpha-bar buffers for q-sampling and reverse steps.

        Args:
            num_timesteps: Number of diffusion steps ``T``.
            schedule: Noise schedule type, ``"linear"`` or ``"cosine"``.
            beta_start: Linear schedule start beta.
            beta_end: Linear schedule end beta.
            cosine_s: Offset used by the cosine schedule.
            device: Device for registered buffers.
        """
        super().__init__()

        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"Unknown schedule '{schedule}'. Expected 'linear' or 'cosine'.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1 for the linear schedule.")
        if cosine_s < 0.0:
            raise ValueError("cosine_s must be non-negative.")

        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.device = device

        if schedule == "linear":
            betas = self._betas_linear(num_timesteps, beta_start, beta_end)
        else:
            betas = self._betas_cosine(num_timesteps, cosine_s)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])

        # q(x_t | x_0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # q(x_{t-1} | x_t, x_0) posterior
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.to(device)

    @staticmethod
    def _betas_linear(num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

    @staticmethod
    def _betas_cosine(num_timesteps: int, cosine_s: float) -> torch.Tensor:
        """Improved DDPM cosine schedule (Nichol & Dhariwal)."""
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((t / num_timesteps) + cosine_s) / (1.0 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=1e-5, max=0.999).to(dtype=torch.float32)

    @staticmethod
    def _extract(buffer: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gather schedule values for batch timesteps and reshape for broadcasting."""
        out = buffer.gather(0, t.long()).to(dtype=buffer.dtype)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def _validate_xt_t(self, xt: torch.Tensor, t: torch.Tensor) -> None:
        if xt.ndim < 2:
            raise ValueError(f"xt must have at least 2 dims (B, ...), got shape {tuple(xt.shape)}")
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D with shape (B,), got shape {tuple(t.shape)}")
        if t.shape[0] != xt.shape[0]:
            raise ValueError(f"Batch size mismatch: xt has {xt.shape[0]}, t has {t.shape[0]}")
        if t.numel() == 0:
            raise ValueError("t must be non-empty")
        if torch.any(t < 0) or torch.any(t >= self.num_timesteps):
            raise ValueError(f"t values must lie in [0, {self.num_timesteps})")

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Sample training timesteps uniformly from ``[0, num_timesteps)``.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device for the returned tensor. Defaults to this module's device.

        Returns:
            Long tensor of shape ``(batch_size,)``.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if device is None:
            device = self.betas.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``x_t`` from ``q(x_t | x_0)``.

        Args:
            x0: Clean action tensor, typically ``(B, T, action_dim)``.
            t: Integer timesteps of shape ``(B,)`` in ``[0, num_timesteps)``.
            noise: Optional epsilon with the same shape as ``x0``. Drawn from
                ``N(0, I)`` when omitted.

        Returns:
            Tuple ``(xt, noise)`` with the same shape as ``x0``.
        """
        self._validate_xt_t(x0, t)

        if noise is None:
            noise = torch.randn_like(x0)
        elif noise.shape != x0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match x0 shape {tuple(x0.shape)}")

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape).to(dtype=x0.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape).to(dtype=x0.dtype)
        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover ``x0`` from ``xt`` and predicted / true epsilon."""
        self._validate_xt_t(xt, t)
        if noise.shape != xt.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match xt shape {tuple(xt.shape)}")
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        return sqrt_recip * xt - sqrt_recipm1 * noise

    def predict_noise_from_x0(self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Recover epsilon from ``xt`` and predicted / true ``x0``."""
        self._validate_xt_t(xt, t)
        if x0.shape != xt.shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} must match xt shape {tuple(xt.shape)}")
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        return (xt - sqrt_alpha * x0) / sqrt_one_minus_alpha.clamp_min(1e-8)

    def predict_v_from_x0_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute v-parameterization target: ``v = sqrt(abar) * eps - sqrt(1-abar) * x0``."""
        self._validate_xt_t(x0, t)
        if noise.shape != x0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match x0 shape {tuple(x0.shape)}")
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape).to(dtype=x0.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape).to(dtype=x0.dtype)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x0

    def predict_x0_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover ``x0`` from ``xt`` and v-prediction."""
        self._validate_xt_t(xt, t)
        if v.shape != xt.shape:
            raise ValueError(f"v shape {tuple(v.shape)} must match xt shape {tuple(xt.shape)}")
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        return sqrt_alpha * xt - sqrt_one_minus_alpha * v

    def predict_noise_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover epsilon from ``xt`` and v-prediction."""
        self._validate_xt_t(xt, t)
        if v.shape != xt.shape:
            raise ValueError(f"v shape {tuple(v.shape)} must match xt shape {tuple(xt.shape)}")
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        return sqrt_one_minus_alpha * xt + sqrt_alpha * v

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean / variance of ``q(x_{t-1} | x_t, x_0)``."""
        self._validate_xt_t(xt, t)
        if x0.shape != xt.shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} must match xt shape {tuple(xt.shape)}")

        mean = (
            self._extract(self.posterior_mean_coef1, t, xt.shape).to(dtype=xt.dtype) * x0
            + self._extract(self.posterior_mean_coef2, t, xt.shape).to(dtype=xt.dtype) * xt
        )
        variance = self._extract(self.posterior_variance, t, xt.shape).to(dtype=xt.dtype)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, xt.shape).to(dtype=xt.dtype)
        return mean, variance, log_variance

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        noise: torch.Tensor | None = None,
        clip_denoised: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """Ancestral DDPM reverse step from ``x_t`` to ``x_{t-1}``.

        Args:
            xt: Noisy sample at timestep ``t``.
            t: Integer timesteps of shape ``(B,)``.
            pred_noise: Network-predicted (or true) epsilon.
            noise: Optional stochastic noise for ``t > 0``. Drawn if omitted.
            clip_denoised: Whether to clip the predicted ``x0`` before the step.
            clip_range: Inclusive clip bounds used when ``clip_denoised`` is True.

        Returns:
            Sample ``x_{t-1}`` with the same shape as ``xt``.
        """
        self._validate_xt_t(xt, t)
        if pred_noise.shape != xt.shape:
            raise ValueError(
                f"pred_noise shape {tuple(pred_noise.shape)} must match xt shape {tuple(xt.shape)}"
            )

        x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)
        if clip_denoised:
            x0_pred = x0_pred.clamp(clip_range[0], clip_range[1])

        mean, _, log_variance = self.q_posterior_mean_variance(x0_pred, xt, t)
        if noise is None:
            noise = torch.randn_like(xt)
        elif noise.shape != xt.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match xt shape {tuple(xt.shape)}")

        # No noise when t == 0.
        nonzero_mask = (t > 0).to(dtype=xt.dtype).reshape(xt.shape[0], *([1] * (xt.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    def ddim_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        eta: float = 0.0,
        noise: torch.Tensor | None = None,
        clip_denoised: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """DDIM reverse step from ``x_t`` to ``x_{t-1}``.

        Args:
            xt: Noisy sample at timestep ``t``.
            t: Integer timesteps of shape ``(B,)``.
            pred_noise: Network-predicted (or true) epsilon.
            eta: Stochasticity. ``0`` is deterministic DDIM; ``1`` matches DDPM variance.
            noise: Optional noise used when ``eta > 0``. Drawn if omitted.
            clip_denoised: Whether to clip the predicted ``x0`` before the step.
            clip_range: Inclusive clip bounds used when ``clip_denoised`` is True.

        Returns:
            Sample ``x_{t-1}`` with the same shape as ``xt``.
        """
        if eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        self._validate_xt_t(xt, t)
        if pred_noise.shape != xt.shape:
            raise ValueError(
                f"pred_noise shape {tuple(pred_noise.shape)} must match xt shape {tuple(xt.shape)}"
            )

        x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)
        if clip_denoised:
            x0_pred = x0_pred.clamp(clip_range[0], clip_range[1])
            pred_noise = self.predict_noise_from_x0(xt, t, x0_pred)

        alpha_t = self._extract(self.alphas_cumprod, t, xt.shape).to(dtype=xt.dtype)
        alpha_prev = self._extract(self.alphas_cumprod_prev, t, xt.shape).to(dtype=xt.dtype)

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t).clamp_min(1e-8))
            * torch.sqrt((1.0 - alpha_t / alpha_prev.clamp_min(1e-8)).clamp_min(0.0))
        )
        dir_xt = torch.sqrt((1.0 - alpha_prev - sigma**2).clamp_min(0.0)) * pred_noise
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        if eta > 0.0:
            if noise is None:
                noise = torch.randn_like(xt)
            elif noise.shape != xt.shape:
                raise ValueError(f"noise shape {tuple(noise.shape)} must match xt shape {tuple(xt.shape)}")
            nonzero_mask = (t > 0).to(dtype=xt.dtype).reshape(xt.shape[0], *([1] * (xt.ndim - 1)))
            x_prev = x_prev + nonzero_mask * sigma * noise
        return x_prev
