"""IDM: a GRU encoder -> latent Neural ODE -> GRU decoder for RR sequences."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint as odeint_impl
except ImportError:  # pragma: no cover
    odeint_impl = None


class ODEFunction(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return -self.alpha * z + self.net(z)


class IDM(nn.Module):
    """GRU + latent Neural ODE model for RR dynamics."""

    def __init__(self, input_dim: int = 1, latent_dim: int = 64, hidden_dim: int = 64, alpha: float = 0.1, noise_sigma: float = 0.01) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.noise_sigma = noise_sigma
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.z0_head = nn.Linear(hidden_dim, latent_dim)
        self.ode = ODEFunction(latent_dim, hidden_dim, alpha)
        self.decoder = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.z0_prior_mean = torch.zeros(1, latent_dim)
        self.z0_prior_std = torch.ones(1, latent_dim)

    def _latent_integrate(self, z0: torch.Tensor, length: int) -> torch.Tensor:
        if odeint_impl is not None:
            time_points = torch.linspace(0.0, 1.0, steps=length, device=z0.device, dtype=z0.dtype)
            traj = odeint_impl(self.ode, z0, time_points)
            return traj.permute(1, 0, 2)

        trajectory = [z0]
        dt = 1.0 / max(length - 1, 1)
        for _ in range(1, length):
            prev = trajectory[-1]
            next_state = prev + dt * self.ode(prev)
            trajectory.append(next_state)
        return torch.stack(trajectory, dim=1)

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = x if x.dim() == 3 else x.unsqueeze(-1)
        if self.training:
            x = x + torch.randn_like(x) * self.noise_sigma
        _, h_T = self.encoder(x)
        h_T = h_T.squeeze(0)
        z0 = self.z0_head(h_T)
        traj = self._latent_integrate(z0, x.shape[1])
        decoded, _ = self.decoder(traj)
        x_hat = self.output(decoded)
        if return_latent:
            return x_hat, traj
        return x_hat

    def fit_prior(self, z_values: torch.Tensor) -> None:
        z_values = z_values.detach().reshape(-1, self.latent_dim)
        self.z0_prior_mean = z_values.mean(dim=0, keepdim=True)
        self.z0_prior_std = z_values.std(dim=0, keepdim=True).clamp_min(1e-6)

    def generate(self, n: int, length: int, temperature: float = 1.0) -> torch.Tensor:
        mean = self.z0_prior_mean.to(device=self.z0_prior_mean.device)
        std = self.z0_prior_std.to(device=self.z0_prior_mean.device)
        z0 = torch.randn(n, self.latent_dim, device=mean.device) * std * temperature + mean
        traj = self._latent_integrate(z0, length)
        decoded, _ = self.decoder(traj)
        return self.output(decoded)


if __name__ == "__main__":
    x = torch.randn(8, 15, 1)
    model = IDM(latent_dim=64, hidden_dim=64)
    x_hat, z = model(x, return_latent=True)
    print("x_hat shape:", tuple(x_hat.shape))
    print("z trajectory shape:", tuple(z.shape))
    print("param count:", sum(p.numel() for p in model.parameters()))
