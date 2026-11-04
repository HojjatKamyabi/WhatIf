"""Scores for an Outcome Distribution against what actually happened."""

from __future__ import annotations

import numpy as np


def crps(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score of sample ensembles; lower is better.

    `samples` has the ensemble on axis 0; `actual` matches the remaining axes.
    Uses CRPS = E|X - y| - 0.5 E|X - X'| over the empirical distribution.
    """
    m = samples.shape[0]
    spread_to_actual = np.abs(samples - actual).mean(axis=0)
    ordered = np.sort(samples, axis=0)
    rank_weights = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (samples.ndim - 1))
    mean_pair_distance = 2.0 * (rank_weights * ordered).sum(axis=0) / (m * m)
    return spread_to_actual - 0.5 * mean_pair_distance


def covered(samples: np.ndarray, actual: np.ndarray, interval: float = 0.8) -> np.ndarray:
    """Whether each actual value falls inside the central `interval` of its ensemble."""
    tail = (1.0 - interval) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    return (actual >= low) & (actual <= high)
