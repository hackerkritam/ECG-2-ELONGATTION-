"""Training loop for the IDM RR sequence model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from physiodyn.losses.idm_losses import total_idm_loss
from physiodyn.models.idm import IDM
from physiodyn.utils.config import load_config
from physiodyn.utils.seed import set_global_seed


def train_idm(config_path: str, quick: bool = False, device: str | None = None):
    """Train the IDM on synthetic or preprocessed RR sequences."""
    cfg = load_config(config_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(int(cfg.get("seed", 42)))

    batch_size = int(cfg.get("batch_size", 8))
    latent_dim = int(cfg.get("latent_dim", 64))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    curriculum = cfg.get("curriculum", [{"length": 15, "epochs": 5}])
    epochs = 1 if quick else max(stage.get("epochs", 1) for stage in curriculum)

    model = IDM(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-4)), betas=tuple(cfg.get("betas", (0.0, 0.9))))

    logs = []
    for stage in curriculum:
        seq_len = int(stage.get("length", 15))
        stage_epochs = 1 if quick else int(stage.get("epochs", 1))
        for epoch in tqdm(range(stage_epochs), desc=f"IDM stage {seq_len}"):
            rr = torch.randn(batch_size, seq_len, device=device) * 0.1 + 0.8
            recon, z = model(rr.unsqueeze(-1), return_latent=True)
            loss = total_idm_loss(rr.unsqueeze(-1), recon, z, lam_temp=float(cfg.get("lambda_temp", 1.0)), lam_latent=float(cfg.get("lambda_latent", 1e-4)), lam_freq=float(cfg.get("lambda_freq", 0.1)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
            logs.append({"stage": seq_len, "epoch": epoch, "loss": float(loss.detach().cpu())})

    out_dir = Path("outputs/checkpoints/idm")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("outputs/logs/idm_losses.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(logs)
    torch.save({"model_state": model.state_dict(), "config": cfg}, out_dir / "idm_final.pt")
    return {"model": model, "logs": logs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the IDM")
    parser.add_argument("--config", type=str, default="configs/idm.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = train_idm(args.config, quick=args.quick)
    print(json.dumps({"total_logs": len(result["logs"]), "last_loss": result["logs"][-1] if result["logs"] else {}}, indent=2))
