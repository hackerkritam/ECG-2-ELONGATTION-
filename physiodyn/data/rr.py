"""RR-interval extraction and windowing utilities."""

from __future__ import annotations

import numpy as np


def compute_rr_series(r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """Compute RR intervals in seconds from R-peak indices."""
    if len(r_peaks) < 2:
        return np.empty((0,), dtype=np.float64)
    rr = np.diff(r_peaks, axis=0) / float(fs)
    return rr.astype(np.float64)


def standardize_window(rr: np.ndarray, length: int, pad_value: float | None = None) -> np.ndarray:
    """Standardize an RR series to a fixed-length window by truncation or edge padding."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size >= length:
        return rr[:length]
    padded = np.empty(length, dtype=np.float64)
    pad_value = rr[-1] if pad_value is None else pad_value
    padded[: rr.size] = rr
    padded[rr.size:] = pad_value
    return padded


def make_rr_windows(rr_values: np.ndarray, lengths: list[int] | tuple[int, ...] = (15, 30, 200), non_overlapping: bool = True):
    """Create non-overlapping RR windows for the IDM curriculum."""
    windows = {length: [] for length in lengths}
    rr = np.asarray(rr_values, dtype=np.float64)
    for length in lengths:
        idx = 0
        while idx + length <= rr.size:
            seq = rr[idx:idx + length]
            windows[length].append(seq)
            idx += length if non_overlapping else 1
    return {length: np.asarray(windows[length], dtype=np.float64) for length in lengths}


def split_records(records: list[str], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 0):
    """Split a list of record IDs into train/val/test sets without leakage."""
    rng = np.random.default_rng(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test
