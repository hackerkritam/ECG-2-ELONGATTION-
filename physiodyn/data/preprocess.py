"""Preprocessing utilities for MIT-BIH ECG signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

try:
    import wfdb
except ImportError:  # pragma: no cover
    wfdb = None


def resample_signal(sig: np.ndarray, original_fs: int, target_fs: int = 250) -> np.ndarray:
    """Resample a 1D ECG signal using a rational polyphase resampling."""
    if original_fs == target_fs:
        return sig.astype(np.float64)
    return signal.resample_poly(sig.astype(np.float64), target_fs, original_fs)


def bandpass_filter(sig: np.ndarray, fs: int = 250, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter with zero-phase filtering."""
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, sig)


def zscore_normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal."""
    mu = float(np.mean(sig))
    sigma = float(np.std(sig))
    if sigma == 0.0:
        return sig.astype(np.float64)
    return (sig - mu) / sigma


def load_record(record_path: str | Path):
    """Load a WFDB record and return the signal and metadata."""
    if wfdb is None:
        raise ImportError("wfdb is required to load records")
    record_path = str(record_path)
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.fs, record


def preprocess_record(record_path: str | Path, target_fs: int = 250):
    """Resample, bandpass, z-score normalize, and return a processed signal and fs."""
    signal_data, original_fs, metadata = load_record(record_path)
    lead = signal_data[:, 0] if signal_data.ndim > 1 else signal_data
    resampled = resample_signal(lead, original_fs, target_fs)
    filtered = bandpass_filter(resampled, fs=target_fs)
    normalized = zscore_normalize(filtered)
    return normalized, target_fs, metadata
