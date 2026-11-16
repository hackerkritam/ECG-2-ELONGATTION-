"""HRV metrics computed on RR-interval sequences in seconds."""

from __future__ import annotations

import numpy as np


def mean_rr(rr: np.ndarray) -> float:
    return float(np.mean(rr))


def std_rr(rr: np.ndarray) -> float:
    return float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0


def rmssd(rr: np.ndarray) -> float:
    if rr.size < 2:
        return 0.0
    diff = np.diff(rr)
    return float(np.sqrt(np.mean(diff**2)))


def sdnn(rr: np.ndarray) -> float:
    return std_rr(rr)


def pnn50(rr: np.ndarray) -> float:
    if rr.size < 2:
        return 0.0
    diff = np.abs(np.diff(rr))
    return float(np.mean(diff > 0.05) * 100.0)


def sample_entropy(rr: np.ndarray, m: int = 2, r: float | None = None) -> float:
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < m + 1:
        return 0.0
    if r is None:
        r = 0.2 * np.std(rr)
    def _template(x, length):
        return np.asarray([np.all(np.abs(x[i:i + length] - x[:length]) <= r) for i in range(1, len(x) - length + 1)])
    n = rr.size
    matches_m = 0
    matches_mp1 = 0
    for i in range(n - m):
        template_m = rr[i:i + m]
        for j in range(i + 1, n - m + 1):
            if np.all(np.abs(rr[j:j + m] - template_m) <= r):
                matches_m += 1
                if np.all(np.abs(rr[j:j + m + 1] - np.append(template_m, rr[i + m])) <= r):
                    matches_mp1 += 1
    if matches_m == 0:
        return 0.0
    return float(-np.log(matches_mp1 / matches_m))


def approximate_entropy(rr: np.ndarray, m: int = 2, r: float | None = None) -> float:
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < m + 1:
        return 0.0
    if r is None:
        r = 0.2 * np.std(rr)
    def counts(seq, length):
        total = 0
        for i in range(len(rr) - length + 1):
            sub = rr[i:i + length]
            match = sum(np.max(np.abs(sub - rr[j:j + length])) <= r for j in range(len(rr) - length + 1))
            total += match
        return total
    c1 = counts(rr, m)
    c2 = counts(rr, m + 1)
    return float(np.log(c1 / max(c2, 1e-12)))


def poincare_metrics(rr: np.ndarray) -> tuple[float, float]:
    diff = np.diff(rr)
    sd1 = float(np.std(diff, ddof=1) / np.sqrt(2.0)) if diff.size > 1 else 0.0
    sdnn = std_rr(rr)
    sd2 = float(np.sqrt(2.0 * sdnn**2 - sd1**2)) if rr.size > 1 else 0.0
    return sd1, sd2


def compute_hrv_metrics(rr: np.ndarray) -> dict[str, float]:
    """Compute HRV metrics in seconds for a RR interval sequence."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size == 0:
        return {
            "Mean RR": 0.0,
            "STD": 0.0,
            "Min": 0.0,
            "Max": 0.0,
            "RMSSD": 0.0,
            "SDNN": 0.0,
            "pNN50": 0.0,
            "Sample Entropy": 0.0,
            "Approximate Entropy": 0.0,
            "SD1": 0.0,
            "SD2": 0.0,
        }
    sd1, sd2 = poincare_metrics(rr)
    return {
        "Mean RR": mean_rr(rr),
        "STD": std_rr(rr),
        "Min": float(np.min(rr)),
        "Max": float(np.max(rr)),
        "RMSSD": rmssd(rr),
        "SDNN": sdnn(rr),
        "pNN50": pnn50(rr),
        "Sample Entropy": sample_entropy(rr),
        "Approximate Entropy": approximate_entropy(rr),
        "SD1": sd1,
        "SD2": sd2,
    }
