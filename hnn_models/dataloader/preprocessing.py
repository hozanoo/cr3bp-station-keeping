# hnn_models/dataloader/preprocessing.py
"""
Preprocessing utilities for HNN training data.

This module provides:
- central-difference acceleration estimation from velocity samples
- simple mean/std standardization helpers
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def central_difference(
    values: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute time derivatives using a central-difference scheme.

    Parameters
    ----------
    values:
        Array of shape (N, D) containing sampled values v(t)
        for one episode (for example velocities).
    times:
        Array of shape (N,) with monotonically increasing time stamps.

    Returns
    -------
    derivatives:
        Array of shape (N, D) with central-difference derivatives.
        Endpoints are filled by copying the nearest interior derivative.
    """
    if values.ndim != 2:
        raise ValueError(f"values must have shape (N, D), got {values.shape}")
    if times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ValueError(
            f"times must be (N,) and aligned with values, "
            f"got times={times.shape}, values={values.shape}"
        )

    n_steps, dim = values.shape
    if n_steps < 3:
        raise ValueError("Need at least 3 samples for central differences.")

    derivatives = np.empty_like(values)

    dt = times[2:] - times[:-2]
    if np.any(dt <= 0.0):
        raise ValueError("Time stamps must be strictly increasing per episode.")

    dv = values[2:, :] - values[:-2, :]
    center = dv / dt[:, None]

    derivatives[1:-1, :] = center
    derivatives[0, :] = derivatives[1, :]
    derivatives[-1, :] = derivatives[-2, :]

    return derivatives


def compute_standardization_stats(
    data: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for feature standardization.

    Parameters
    ----------
    data:
        Array of shape (N, D) with features.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    mean:
        Feature-wise mean of shape (D,).
    std:
        Feature-wise standard deviation of shape (D,), clipped at ``eps``.
    """
    if data.ndim != 2:
        raise ValueError(f"data must have shape (N, D), got {data.shape}")

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, eps)
    return mean, std
