"""Training loop for the Beat Morphology Module (BMM)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from physiodyn.models.bmm import BMMCritic, BMMGenerator
from physiodyn.utils.config import load_config
from physiodyn.utils.seed import set_global_seed


def train_bmm(config_path: str, quick: bool = False, device: str | None = None):
    """Train the BMM on synthetic or preprocessed beat data."""
    cfg = load_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(int(cfg.get("seed", 42)))

    latent_dim = int(cfg.get("latent_dim", 128))
    num_classes = int(cfg.get("num_classes", 5))
    signal_length = int(cfg.get("signal_length", 200))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = 1 if quick else int(cfg.get("epochs", 20))

    generator = BMMGenerator(latent_dim=latent_dim, num_classes=num_classes, signal_length=signal_length).to(device)
    critic = BMMCritic(num_classes=num_classes, latent_dim=latent_dim).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(cfg.get("lr", 1e-4)), betas=tuple(cfg.get("betas", (0.0, 0.9))))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=float(cfg.get("lr", 1e-4)), betas=tuple(cfg.get("betas", (0.0, 0.9))))

    out_dir = Path("outputs/checkpoints/bmm")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("outputs/logs/bmm_losses.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = []
    for epoch in tqdm(range(epochs), desc="BMM training"):
        real = torch.randn(batch_size, 1, signal_length, device=device)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(noise, labels)

        critic_real = critic(real, labels)
        critic_fake = critic(fake.detach(), labels)
        gp = torch.rand(batch_size, 1, 1, device=device)
        mixed = gp * real + (1 - gp) * fake.detach()
        mixed.requires_grad_(True)
        mixed_score = critic(mixed, labels)
        grad = torch.autograd.grad(outputs=mixed_score, inputs=mixed, grad_outputs=torch.ones_like(mixed_score), create_graph=True)[0]
        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        d_loss = critic_fake.mean() - critic_real.mean() + float(cfg.get("lambda_gp", 10.0)) * grad_penalty
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        if (epoch + 1) % int(cfg.get("n_critic", 5)) == 0:
            noise = torch.randn(batch_size, latent_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            paired = real[:batch_size]
            fake = generator(noise, labels)
            gen_loss = -critic(fake, labels).mean() + float(cfg.get("lambda_l1", 10.0)) * torch.mean(torch.abs(fake - paired))
            optimizer_g.zero_grad()
            gen_loss.backward()
            optimizer_g.step()
            metrics.append({"epoch": epoch, "critic_loss": float(d_loss.detach().cpu()), "generator_loss": float(gen_loss.detach().cpu()), "gp": float(grad_penalty.detach().cpu())})

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "critic_loss", "generator_loss", "gp"])
        writer.writeheader()
        writer.writerows(metrics)

    checkpoint = out_dir / "generator_final.pt"
    torch.save({"model_state": generator.state_dict(), "config": cfg}, checkpoint)
    return {"generator": generator, "critic": critic, "metrics": metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the BMM")
    parser.add_argument("--config", type=str, default="configs/bmm.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = train_bmm(args.config, quick=args.quick)
    print(json.dumps({"epochs": len(result["metrics"]), "final_metrics": result["metrics"][-1] if result["metrics"] else {}}, indent=2))
