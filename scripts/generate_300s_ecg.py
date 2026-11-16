"""Generate and save a 300-second synthetic ECG signal."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from physiodyn.models.composer import compose_duration


def main() -> None:
    beats = [np.sin(np.linspace(-1.0, 1.0, 200)) for _ in range(5)]
    result = compose_duration(duration_seconds=300.0, beat_waveforms=beats, seed=0)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "synthetic_ecg_300s.npz", noisy=result["noisy"], clean=result["clean"], rr=result["r_peaks"], classes=result["classes"])
    print(f"Saved 300s synthetic ECG with {result['noisy'].shape[0]} samples ({result['noisy'].shape[0] / 250.0:.2f}s).")


if __name__ == "__main__":
    main()
