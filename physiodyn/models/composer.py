"""Composer module that synthesizes ECG traces from RR intervals and beat templates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def _raised_cosine(length: int, crossfade: int) -> np.ndarray:
    window = np.ones(length, dtype=np.float64)
    if crossfade <= 0:
        return window
    fade = np.linspace(0.0, 1.0, crossfade, endpoint=True)
    fade = 0.5 - 0.5 * np.cos(np.pi * fade)
    window[:crossfade] = fade
    window[-crossfade:] = fade[::-1]
    return window


def generate_rr_sequence_for_duration(duration_seconds: float, fs: int = 250, seed: int = 0, rr_mean: float = 0.8, rr_std: float = 0.25) -> np.ndarray:
    """Generate an RR series whose total duration matches the requested signal length in seconds."""
    if duration_seconds <= 0:
        return np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    rr = []
    total = 0.0
    while total < duration_seconds:
        value = float(np.clip(rng.normal(rr_mean, rr_std), 0.35, 1.75))
        rr.append(value)
        total += value
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size > 0 and total > duration_seconds:
        rr[-1] = max(0.35, rr[-1] - (total - duration_seconds))
    return rr


def _warp_beat(beat: np.ndarray, rr_seconds: float, fs: int = 250) -> np.ndarray:
    """Resample a beat template to the target RR interval while keeping the R-peak centered."""
    beat = np.asarray(beat, dtype=np.float64)
    target_length = max(1, int(round(rr_seconds * fs)))
    if beat.size == 0:
        return np.zeros(target_length, dtype=np.float64)
    if target_length == beat.size:
        return beat.copy()
    x_old = np.linspace(0.0, 1.0, beat.size)
    x_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_new, x_old, beat)


def compose_duration(
    duration_seconds: float = 300.0,
    fs: int = 250,
    beat_waveforms: list[np.ndarray] | np.ndarray | None = None,
    seed: int = 0,
    rr_mean: float = 0.8,
    rr_std: float = 0.25,
    respiration_freq_range: tuple[float, float] = (0.2, 0.3),
    respiration_amp: float = 0.05,
    baseline_freq_range: tuple[float, float] = (0.05, 0.5),
    baseline_amp: float = 0.1,
    muscle_noise_std: float = 0.02,
    bandpass: tuple[float, float] = (0.5, 40.0),
) -> dict[str, Any]:
    """Generate a synthetic ECG with a specified duration in seconds."""
    if beat_waveforms is None:
        beat_waveforms = [np.sin(np.linspace(-1.0, 1.0, 200)) for _ in range(5)]
    rr_seconds = generate_rr_sequence_for_duration(duration_seconds, fs=fs, seed=seed, rr_mean=rr_mean, rr_std=rr_std)
    result = compose(rr_seconds, beat_waveforms, fs=fs, seed=seed, respiration_freq_range=respiration_freq_range, respiration_amp=respiration_amp, baseline_freq_range=baseline_freq_range, baseline_amp=baseline_amp, muscle_noise_std=muscle_noise_std, bandpass=bandpass)
    result["duration_seconds"] = float(duration_seconds)
    return result


def compose(
    rr_seconds: np.ndarray,
    beat_waveforms: list[np.ndarray] | np.ndarray,
    fs: int = 250,
    class_probs: dict[str, float] | None = None,
    seed: int = 0,
    crossfade_length: int = 10,
    respiration_freq_range: tuple[float, float] = (0.2, 0.3),
    respiration_amp: float = 0.05,
    baseline_freq_range: tuple[float, float] = (0.05, 0.5),
    baseline_amp: float = 0.1,
    muscle_noise_std: float = 0.02,
    bandpass: tuple[float, float] = (0.5, 40.0),
) -> dict[str, Any]:
    """Compose a synthetic ECG from RR intervals and beat templates."""
    rr_seconds = np.asarray(rr_seconds, dtype=np.float64)
    if rr_seconds.size == 0:
        return {"noisy": np.array([], dtype=np.float64), "clean": np.array([], dtype=np.float64), "r_peaks": np.array([], dtype=np.float64), "classes": np.array([], dtype=np.int64)}

    rng = np.random.default_rng(seed)
    if class_probs is None:
        class_probs = {"N": 0.8, "S": 0.1, "V": 0.08, "F": 0.015, "Q": 0.005}
    if isinstance(beat_waveforms, np.ndarray):
        beat_waveforms = [beat_waveforms[i] for i in range(beat_waveforms.shape[0])]

    t = np.cumsum(np.concatenate(([0.0], rr_seconds)))
    total_samples = int(np.ceil(t[-1] * fs)) + 64
    signal_buffer = np.zeros(total_samples, dtype=np.float64)
    classes = []
    r_peaks = []

    for idx, rr in enumerate(rr_seconds):
        classes.append(int(rng.integers(0, 5)))
        beat_idx = int(rng.integers(0, len(beat_waveforms)))
        beat = np.asarray(beat_waveforms[beat_idx], dtype=np.float64).copy()
        warped = _warp_beat(beat, float(rr), fs=fs)
        peak_index = int(np.round((t[idx + 1]) * fs))
        start = max(0, peak_index - warped.size // 2)
        end = min(total_samples, start + warped.size)
        signal_buffer[start:end] += warped[: end - start]
        r_peaks.append(peak_index)

    time_axis = np.arange(total_samples, dtype=np.float64) / float(fs)
    respiration = respiration_amp * np.sin(2 * np.pi * rng.uniform(respiration_freq_range[0], respiration_freq_range[1]) * time_axis + rng.uniform(0.0, 2 * np.pi))
    baseline = baseline_amp * np.sin(2 * np.pi * rng.uniform(baseline_freq_range[0], baseline_freq_range[1]) * time_axis + rng.uniform(0.0, 2 * np.pi))
    muscle_noise = rng.normal(0.0, muscle_noise_std, size=total_samples)
    noisy = signal_buffer + respiration + baseline + muscle_noise
    noisy = np.asarray(noisy, dtype=np.float64)

    nyquist = 0.5 * fs
    low = bandpass[0] / nyquist
    high = bandpass[1] / nyquist
    b, a = signal.butter(4, [low, high], btype="bandpass")
    clean = signal.filtfilt(b, a, signal_buffer)
    clean = np.asarray(clean, dtype=np.float64)

    return {
        "noisy": noisy,
        "clean": clean,
        "r_peaks": np.asarray(r_peaks, dtype=np.float64),
        "classes": np.asarray(classes, dtype=np.int64),
    }


if __name__ == "__main__":
    rr = np.array([0.8, 0.9, 1.0, 0.8], dtype=np.float64)
    beats = [np.sin(np.linspace(-1.0, 1.0, 200)) for _ in range(5)]
    result = compose(rr, beats, seed=0)
    print(result["noisy"].shape, result["clean"].shape)

    long_result = compose_duration(duration_seconds=300.0, beat_waveforms=beats, seed=0)
    print("300s output length:", long_result["noisy"].shape[0], "samples")
