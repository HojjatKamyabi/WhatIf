"""Baselines: simple forecasts every Backtest must beat."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    """Anything that turns Poll Average history into an Outcome Distribution.

    Given `history` (weeks x parties, all weeks before the start week), returns samples
    shaped (runs, horizon, parties), each row summing to 100.
    """

    name: str

    def forecast(self, history: np.ndarray, horizon: int, runs: int, rng: np.random.Generator) -> np.ndarray: ...


class NaiveBaseline:
    """Naïve "same as last week", with uncertainty from how much the Poll Average usually moves.

    Each Run starts from the last week of history and takes a random weekly step per party,
    sized by the standard deviation of that party's recent weekly changes.
    """

    name = "naive"

    def __init__(self, lookback_weeks: int = 52):
        self.lookback_weeks = lookback_weeks

    def forecast(self, history: np.ndarray, horizon: int, runs: int, rng: np.random.Generator) -> np.ndarray:
        recent = history[-(self.lookback_weeks + 1):]
        step_size = np.diff(recent, axis=0).std(axis=0)
        steps = rng.normal(0.0, step_size, size=(runs, horizon, history.shape[1]))
        return _normalise(history[-1] + np.cumsum(steps, axis=1))


class StatisticalBaseline:
    """A damped-trend forecast per party, the standard way to extrapolate a poll trend.

    For each party, fits Holt's damped trend smoothing (a level that follows the polls and
    a trend that fades out) to the recent Poll Average by grid search on one-step-ahead
    error, then simulates Runs by feeding random one-step errors back through the model.
    """

    name = "statistical"

    ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)  # how fast the level follows new polls
    BETAS = (0.0, 0.02, 0.05, 0.1, 0.2)  # how fast the trend reacts
    PHIS = (0.8, 0.9, 0.95, 0.98)  # how slowly the trend fades out

    def __init__(self, lookback_weeks: int = 104):
        self.lookback_weeks = lookback_weeks

    def forecast(self, history: np.ndarray, horizon: int, runs: int, rng: np.random.Generator) -> np.ndarray:
        recent = history[-self.lookback_weeks:]
        parties = history.shape[1]
        samples = np.empty((runs, horizon, parties))
        for j in range(parties):
            fit = _fit_damped_trend(recent[:, j], self.ALPHAS, self.BETAS, self.PHIS)
            samples[:, :, j] = fit.simulate(horizon, runs, rng)
        return _normalise(samples)


class _DampedTrend:
    def __init__(self, alpha, beta, phi, level, trend, sigma):
        self.alpha, self.beta, self.phi = alpha, beta, phi
        self.level, self.trend, self.sigma = level, trend, sigma

    def simulate(self, horizon: int, runs: int, rng: np.random.Generator) -> np.ndarray:
        level = np.full(runs, self.level)
        trend = np.full(runs, self.trend)
        paths = np.empty((runs, horizon))
        for h in range(horizon):
            expected = level + self.phi * trend
            error = rng.normal(0.0, self.sigma, runs)
            paths[:, h] = expected + error
            level = expected + self.alpha * error
            trend = self.phi * trend + self.beta * error
        return paths


def _fit_damped_trend(series: np.ndarray, alphas, betas, phis) -> _DampedTrend:
    best = None
    for alpha in alphas:
        for beta in betas:
            for phi in phis:
                level, trend = series[0], 0.0
                errors = np.empty(len(series) - 1)
                for t, value in enumerate(series[1:]):
                    expected = level + phi * trend
                    errors[t] = value - expected
                    level = expected + alpha * errors[t]
                    trend = phi * trend + beta * errors[t]
                sse = float(errors @ errors)
                if best is None or sse < best[0]:
                    best = (sse, _DampedTrend(alpha, beta, phi, level, trend, float(errors.std())))
    return best[1]


def _normalise(samples: np.ndarray) -> np.ndarray:
    """Clip negative shares to zero and rescale every week to sum to 100."""
    samples = np.clip(samples, 0.0, None)
    return 100.0 * samples / samples.sum(axis=2, keepdims=True)
