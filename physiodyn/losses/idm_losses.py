"""Loss functions for the IDM (RR dynamics module)."""

from __future__ import annotations

import torch


def recon_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between the RR sequence and its reconstruction."""
    return torch.mean(torch.abs(x - xhat))


def temporal_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Temporal-difference consistency loss."""
    dx = x[:, 1:] - x[:, :-1]
    dx_hat = xhat[:, 1:] - xhat[:, :-1]
    return torch.mean((dx - dx_hat) ** 2)


def latent_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalty on latent activity to keep the ODE stable."""
    return torch.mean(torch.sum(z**2, dim=-1))


def freq_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Frequency-domain mismatch between real and reconstructed RR sequences."""
    x_fft = torch.abs(torch.fft.rfft(x, dim=-1))
    xhat_fft = torch.abs(torch.fft.rfft(xhat, dim=-1))
    return torch.mean((x_fft - xhat_fft) ** 2)


def total_idm_loss(x, xhat, z, lam_temp=1.0, lam_latent=1e-4, lam_freq=0.1):
    """Aggregate IDM training objective from the paper's Eq. 16-20."""
    return recon_loss(x, xhat) + lam_temp * temporal_loss(x, xhat) + lam_latent * latent_loss(z) + lam_freq * freq_loss(x, xhat)
