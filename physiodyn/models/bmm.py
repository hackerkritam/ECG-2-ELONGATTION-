"""Beat Morphology Module (BMM): conditional U-Net generator and projection critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BMMGenerator(nn.Module):
    """Conditional 1D U-Net generator for single ECG beats."""

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 5,
        label_embed_dim: int = 64,
        signal_length: int = 200,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.signal_length = signal_length
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + label_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, base_channels * 25),
            nn.ReLU(inplace=True),
        )

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(base_channels * 4, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True)),
            ]
        )
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(base_channels * 8 + base_channels * 4, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(base_channels * 4, base_channels * 4, 3, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv1d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv1d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True)),
            ]
        )
        self.out = nn.Conv1d(base_channels * 2, 1, 1)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        label_emb = self.label_emb(y)
        x = torch.cat([z, label_emb], dim=1)
        h = self.mlp(x).view(batch_size, 32, 25)

        skips = []
        enc = h
        for layer in self.encoders:
            enc = layer(enc)
            skips.append(enc)
            enc = self.pool(enc)
        bottleneck = self.bottleneck(enc)

        h = bottleneck
        for i, layer in enumerate(self.decoders):
            target_size = skips[-(i + 1)].shape[-1]
            h = F.interpolate(h, size=target_size, mode="linear", align_corners=False)
            h = torch.cat([h, skips[-(i + 1)]], dim=1)
            h = layer(h)

        out = self.out(h)
        out = F.interpolate(out, size=self.signal_length, mode="linear", align_corners=False)
        return out


class BMMCritic(nn.Module):
    """Projection critic for the BMM WGAN-GP formulation."""

    def __init__(self, num_classes: int = 5, latent_dim: int = 128, label_embed_dim: int = 64) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(128, 1)
        self.proj = nn.Linear(label_embed_dim, 128)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        pooled = self.pool(features).squeeze(-1)
        label_emb = self.label_emb(y)
        proj_term = torch.sum(pooled * self.proj(label_emb), dim=1, keepdim=True)
        return self.output(pooled) + proj_term


if __name__ == "__main__":
    z = torch.randn(4, 128)
    y = torch.randint(0, 5, (4,))
    gen = BMMGenerator()
    fake = gen(z, y)
    critic = BMMCritic()
    score = critic(fake, y)
    print("generator shape:", tuple(fake.shape))
    print("critic shape:", tuple(score.shape))
    print("parameter count:", sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in critic.parameters()))
