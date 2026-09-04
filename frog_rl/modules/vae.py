from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from frog_rl.networks import MLP


class VAE(nn.Module):
	"""Variational autoencoder for vector observations.

	The encoder produces the mean and log variance of a diagonal Gaussian
	posterior. During training, :meth:`forward` samples from this posterior
	with the reparameterization trick; during evaluation it uses the mean.
	"""

	def __init__(
		self,
		input_dim: int,
		latent_dim: int,
		encoder_hidden_dims: tuple[int] | list[int] = [256, 256],
		decoder_hidden_dims: tuple[int] | list[int] | None = None,
		activation: str = "elu",
		device: str = "cpu",
	) -> None:
		
		super().__init__()

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device
		decoder_hidden_dims = encoder_hidden_dims if decoder_hidden_dims is None else decoder_hidden_dims

		encoder_output_dim = 2 * latent_dim
		self.encoder = MLP(input_dim, encoder_output_dim, encoder_hidden_dims, activation).to(device)
		self.decoder = MLP(latent_dim, input_dim, decoder_hidden_dims, activation).to(device)

	def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Encode observations into posterior mean and log variance."""
		mean, log_var = torch.chunk(self.encoder(observations), 2, dim=-1)
		return mean, log_var

	@staticmethod
	def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
		"""Sample from a diagonal Gaussian while preserving gradients."""
		if mean.shape != log_var.shape:
			raise ValueError("mean and log_var must have the same shape")
		standard_deviation = torch.exp(0.5 * log_var)
		return mean + standard_deviation * torch.randn_like(standard_deviation)

	def decode(self, latent: torch.Tensor) -> torch.Tensor:
		"""Decode latent samples into reconstructed observations."""
		return self.decoder(latent)

	def forward(
		self, observations: torch.Tensor, sample: bool = True
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return reconstruction, posterior mean, and posterior log variance."""
		_, reconstruction, mean, log_var = self.vae_forward(observations, sample)
		return reconstruction, mean, log_var

	def vae_forward(
		self, observations: torch.Tensor, sample: bool = True
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Encode, reparameterize, and decode observations for training."""
		mean, log_var = self.encode(observations)
		latent = self.reparameterize(mean, log_var) if sample else mean
		reconstruction = self.decode(latent)
		return latent, reconstruction, mean, log_var

	@staticmethod
	def loss_function(
		reconstruction: torch.Tensor,
		observations: torch.Tensor,
		mean: torch.Tensor,
		log_var: torch.Tensor,
		beta: float = 1.0,
		reduction: str = "mean",
	) -> dict[str, torch.Tensor]:
		"""Compute reconstruction, KL, and total beta-VAE losses."""
		if reconstruction.shape != observations.shape:
			raise ValueError("reconstruction and observations must have the same shape")
		if beta < 0:
			raise ValueError("beta must be non-negative")
		if reduction not in ("mean", "sum"):
			raise ValueError("reduction must be either 'mean' or 'sum'")

		reconstruction_loss = F.mse_loss(reconstruction, observations, reduction=reduction)
		kl_divergence = -0.5 * (1.0 + log_var - mean.square() - log_var.exp()).sum(dim=-1)
		if reduction == "mean":
			kl_divergence = kl_divergence.mean()
		else:
			kl_divergence = kl_divergence.sum()
		return {
			"loss": reconstruction_loss + beta * kl_divergence,
			"reconstruction_loss": reconstruction_loss,
			"kl_loss": kl_divergence,
		}

