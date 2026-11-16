"""Evaluate the BMM on synthetic data and save metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from physiodyn.metrics.beat_metrics import compute_beat_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BMM")
    parser.add_argument("--real", type=str, default="outputs/checkpoints/bmm/real_beats.npz")
    parser.add_argument("--fake", type=str, default="outputs/checkpoints/bmm/fake_beats.npz")
    args = parser.parse_args()

    real = np.load(args.real)["beats"] if Path(args.real).exists() else np.random.randn(128, 200)
    fake = np.load(args.fake)["beats"] if Path(args.fake).exists() else np.random.randn(128, 200)
    metrics = compute_beat_metrics(real, fake)
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "table3_ours.csv").write_text(
        "metric,value\n" + "\n".join(f"{k},{v}" for k, v in metrics.items()),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
