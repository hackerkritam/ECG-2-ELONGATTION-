"""Beat-level metrics comparing real and synthetic beats."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

try:
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover
    fastdtw = None


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def prd(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.sum(x**2)
    if denom == 0:
        return 0.0
    return 100.0 * np.sqrt(np.sum((x - y) ** 2) / denom)


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    if fastdtw is None:
        return float(np.mean(np.abs(x - y)))
    distance, _ = fastdtw(x, y)
    return float(distance)


def frechet_distance(real: np.ndarray, fake: np.ndarray) -> float:
    """Simplified Frechet distance using mean and covariance."""
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.size == 0 or fake.size == 0:
        return 0.0
    mu_r = real.mean(axis=0)
    mu_f = fake.mean(axis=0)
    sigma_r = np.cov(real, rowvar=False)
    sigma_f = np.cov(fake, rowvar=False)
    if np.ndim(sigma_r) == 0:
        sigma_r = np.array([[float(sigma_r)]])
    if np.ndim(sigma_f) == 0:
        sigma_f = np.array([[float(sigma_f)]])
    diff = mu_r - mu_f
    cov_prod = sigma_r @ sigma_f
    eigvals = np.linalg.eigvalsh(cov_prod)
    sqrt_term = np.maximum(eigvals, 0.0)
    return float(np.sqrt(diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.sum(np.sqrt(sqrt_term))))


def mmd_rbf(real: np.ndarray, fake: np.ndarray, bandwidth: float | None = None) -> float:
    """Compute an MMD with a median-heuristic RBF kernel."""
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.size == 0 or fake.size == 0:
        return 0.0
    if bandwidth is None:
        x = np.concatenate([real, fake], axis=0)
        distances = cdist(x, x)
        nonzero = distances[distances > 0]
        median = float(np.median(nonzero)) if nonzero.size else 1.0
        bandwidth = median / 2.0
    gram = np.exp(-cdist(real, real) ** 2 / (2 * bandwidth**2))
    gram2 = np.exp(-cdist(fake, fake) ** 2 / (2 * bandwidth**2))
    cross = np.exp(-cdist(real, fake) ** 2 / (2 * bandwidth**2))
    mmd = np.mean(gram) + np.mean(gram2) - 2 * np.mean(cross)
    return float(mmd)


def compute_beat_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    """Compare real and generated beats using nearest-neighbor pairing."""
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(1, -1)
    if generated.ndim == 1:
        generated = generated.reshape(1, -1)
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated beat widths must match.")
    pairwise = cdist(generated, real)
    nearest = pairwise.argmin(axis=1)
    paired_real = real[nearest]
    metrics = {
        "RMSE": float(np.mean([rmse(generated[i], paired_real[i]) for i in range(generated.shape[0])])),
        "MSE": float(np.mean([mse(generated[i], paired_real[i]) for i in range(generated.shape[0])])),
        "MAE": float(np.mean([mae(generated[i], paired_real[i]) for i in range(generated.shape[0])])),
        "PRD": float(np.mean([prd(real[i], generated[i]) for i in range(generated.shape[0])])),
        "DTW": float(np.mean([dtw_distance(generated[i], paired_real[i]) for i in range(generated.shape[0])])),
        "FD": frechet_distance(real, generated),
        "MMD": mmd_rbf(real, generated),
    }
    return metrics
