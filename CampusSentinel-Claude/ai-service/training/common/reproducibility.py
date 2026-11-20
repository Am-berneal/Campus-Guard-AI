"""Reproducibility helpers shared by every training script.

Setting seeds makes weight initialisation, data shuffling and augmentation
repeatable, so two runs with the same seed and data can be compared fairly.

GPU note: some cuDNN convolution algorithms are non-deterministic. With
deterministic=True we ask PyTorch to avoid them. That makes runs repeatable
on the same hardware, at a small speed cost. Results can still differ
slightly between different GPUs (e.g. Colab T4 vs. a local RTX 4050).
"""

from __future__ import annotations

import os
import random

import torch

try:  # NumPy is optional in Stage 1; data pipelines will use it later.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy (if installed) and PyTorch (CPU + all GPUs)."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    # benchmark=True lets cuDNN pick the fastest algorithm per input shape,
    # but that choice can vary between runs. Disable it when determinism matters.
    torch.backends.cudnn.benchmark = not deterministic


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the CUDA device if available (and wanted), otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
