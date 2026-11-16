"""Beat extraction utilities for the BMM."""

from __future__ import annotations

from collections import Counter

import numpy as np

AAMI_CLASS_MAP = {
    0: {"N", "L", "R", "e", "j"},
    1: {"A", "a", "J", "S"},
    2: {"V", "E"},
    3: {"F"},
    4: {"/", "f", "Q"},
}


def _class_from_symbol(symbol: str) -> int:
    symbol = str(symbol)
    for idx, labels in AAMI_CLASS_MAP.items():
        if symbol in labels:
            return idx
    return 0


def extract_beats(signal: np.ndarray, r_peaks: np.ndarray, fs: int = 250, target_length: int = 200, annotation_symbols=None):
    """Extract target-length windows centered around annotated R-peaks."""
    beats: list[np.ndarray] = []
    labels: list[int] = []

    annotations = annotation_symbols if annotation_symbols is not None else ["N"] * len(r_peaks)
    for peak, annotation in zip(r_peaks, annotations):
        peak = int(peak)
        if peak < target_length // 2 or peak > len(signal) - target_length // 2:
            continue
        start = int(peak - target_length / 2)
        end = int(peak + target_length / 2)
        window = signal[start:end]
        if window.shape[0] != target_length:
            continue
        beats.append(window)
        labels.append(_class_from_symbol(annotation))

    if not beats:
        return np.empty((0, target_length), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.stack(beats), np.asarray(labels, dtype=np.int64)


def summarize_class_counts(labels: np.ndarray):
    counts = Counter(int(label) for label in labels)
    return {str(k): v for k, v in sorted(counts.items())}
