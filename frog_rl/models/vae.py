"""Variational autoencoder for vector observations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class VAE(nn.Module):
    """A compact beta-VAE with a diagonal Gaussian latent posterior."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] | None = None,
        activation: str = "elu",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        decoder_hidden_dims = encoder_hidden_dims if decoder_hidden_dims is None else decoder_hidden_dims
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, 2 * latent_dim, encoder_hidden_dims, activation).to(device)
        self.decoder = MLP(latent_dim, input_dim, decoder_hidden_dims, activation).to(device)

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_var = torch.chunk(self.encoder(observations), 2, dim=-1)
        var = torch.exp(log_var)
        return mean, var

    @staticmethod
    def reparameterize(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Sample from a diagonal Gaussian while preserving gradients."""
        if mean.shape != var.shape:
            raise ValueError("mean and var must have the same shape")
        standard_deviation = torch.sqrt(var)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(
        self, observations: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, posterior mean, and posterior variance."""
        _, reconstruction, mean, var = self.vae_forward(observations, sample)
        return reconstruction, mean, var

    def vae_forward(
        self, observations: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, reparameterize, and decode observations for training."""
        mean, var = self.encode(observations)
        latent = self.reparameterize(mean, var) if sample else mean
        reconstruction = self.decode(latent)
        return latent, reconstruction, mean, var

    def as_jit(self) -> nn.Module:
        """Return a deterministic TorchScript-friendly reconstruction model."""
        return _ExportableVAE(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a deterministic ONNX export wrapper."""
        return _ExportableVAE(self, verbose=verbose)


    @staticmethod
    def loss_function(reconstruction, observations, mean, var, beta: float = 1.0, reduction: str = "mean"):
        if reconstruction.shape != observations.shape:
            raise ValueError("reconstruction and observations must have the same shape")
        if beta < 0 or reduction not in ("mean", "sum"):
            raise ValueError("beta must be non-negative and reduction must be 'mean' or 'sum'")
        reconstruction_loss = F.mse_loss(reconstruction, observations, reduction=reduction)
        kl = 0.5 * (mean.square() + var - torch.log(var) - 1.0).sum(dim=-1)
        kl = kl.mean() if reduction == "mean" else kl.sum()
        return {"loss": reconstruction_loss + beta * kl, "reconstruction_loss": reconstruction_loss, "kl_loss": kl}


class _ExportableVAE(nn.Module):
    """Export wrapper that removes stochastic sampling and training outputs."""

    def __init__(self, model: VAE, verbose: bool = False) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(model.encoder)
        self.decoder = copy.deepcopy(model.decoder)
        self.input_dim = model.input_dim
        self.latent_dim = model.latent_dim
        self.verbose = verbose

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = torch.chunk(self.encoder(observations), 2, dim=-1)
        return self.decoder(mean)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_dim),)

    @property
    def input_names(self) -> list[str]:
        return ["observations"]

    @property
    def output_names(self) -> list[str]:
        return ["reconstruction"]
