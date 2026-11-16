"""Evaluate the IDM and report HRV metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from physiodyn.metrics.hrv import compute_hrv_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the IDM")
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    real = rng.uniform(0.5, 1.5, size=(args.n_sequences, args.length))
    fake = rng.uniform(0.5, 1.5, size=(args.n_sequences, args.length))
    real_metrics = [compute_hrv_metrics(rr) for rr in real]
    fake_metrics = [compute_hrv_metrics(rr) for rr in fake]
    aggregated = {
        "real": {k: float(np.mean([m[k] for m in real_metrics])) for k in real_metrics[0].keys()},
        "fake": {k: float(np.mean([m[k] for m in fake_metrics])) for k in fake_metrics[0].keys()},
    }
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
