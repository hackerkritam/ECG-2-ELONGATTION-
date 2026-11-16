"""Global seeding helpers for deterministic experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, PyTorch, and CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
