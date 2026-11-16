"""Evaluate the composer by generating synthetic ECGs from RR sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from physiodyn.models.composer import compose


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the composer")
    parser.add_argument("--length", type=int, default=30)
    args = parser.parse_args()

    rr = np.random.uniform(0.6, 1.2, size=args.length)
    beats = [np.sin(np.linspace(-1.0, 1.0, 200)) for _ in range(5)]
    result = compose(rr, beats, fs=250, seed=0)
    print(json.dumps({
        "noisy_shape": list(result["noisy"].shape),
        "clean_shape": list(result["clean"].shape),
        "r_peaks_count": int(result["r_peaks"].size),
        "finite": bool(np.isfinite(result["noisy"]).all() and np.isfinite(result["clean"]).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
