"""The Backtest runner: WhatIf's main seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from whatif.baselines import Forecaster
from whatif.poll_average import PollAverage, week_of
from whatif.polls import PARTIES
from whatif.scoring import covered, crps
from whatif.store import MissingDataError

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
COVERAGE_INTERVAL = 0.8


@dataclass(frozen=True)
class BacktestConfig:
    start_week: date
    horizon: int = 12
    runs: int = 200
    seed: int = 0
    min_history_weeks: int = 8

    def __post_init__(self):
        if self.start_week != week_of(self.start_week):
            raise ValueError(f"start_week must be a Monday, got {self.start_week}")
        if self.horizon < 1 or self.runs < 2:
            raise ValueError("horizon must be at least 1 and runs at least 2")


@dataclass(frozen=True)
class ForecastScores:
    """How one forecaster's Outcome Distribution scored over the Horizon."""

    crps_by_party: dict[str, float]
    crps_mean: float
    coverage_by_party: dict[str, float]
    coverage: float
    quantiles: dict[str, list[list[float]]]  # quantile -> [week][party]


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    weeks: list[date]
    actual: list[list[float]]  # [week][party]
    carried_weeks: list[date]
    scores: dict[str, ForecastScores]
    data_versions: dict[str, str] = field(default_factory=dict)


def run_backtest(
    poll_average: PollAverage,
    config: BacktestConfig,
    forecasters: Sequence[Forecaster],
    data_versions: dict[str, str] | None = None,
) -> BacktestResult:
    """Run every forecaster from the start week and score it against the real Poll Average.

    History is every Poll Average week before `start_week`; the Horizon is `start_week`
    and the weeks after it. Refuses to start when either is not covered by the data.
    """
    history, actual, weeks = _split(poll_average, config)
    carried = [w for w in weeks if poll_average.carried[poll_average.index_of(w)]]

    scores = {}
    for index, forecaster in enumerate(forecasters):
        rng = np.random.default_rng([config.seed, index])
        samples = forecaster.forecast(history, config.horizon, config.runs, rng)
        scores[forecaster.name] = _score(samples, actual)

    return BacktestResult(
        config=config,
        weeks=weeks,
        actual=actual.tolist(),
        carried_weeks=carried,
        scores=scores,
        data_versions=dict(data_versions or {}),
    )


def _split(poll_average: PollAverage, config: BacktestConfig):
    weeks = [config.start_week + timedelta(weeks=i) for i in range(config.horizon)]
    first, last = poll_average.weeks[0], poll_average.weeks[-1]
    if weeks[-1] > last:
        raise MissingDataError(
            f"Poll Average ends {last}, but the Horizon runs to {weeks[-1]}."
        )
    history_weeks = (config.start_week - first).days // 7
    if history_weeks < config.min_history_weeks:
        raise MissingDataError(
            f"Only {max(history_weeks, 0)} weeks of Poll Average before {config.start_week}; "
            f"need {config.min_history_weeks}."
        )
    start = poll_average.index_of(config.start_week)
    return (
        poll_average.shares[:start],
        poll_average.shares[start : start + config.horizon],
        weeks,
    )


def _score(samples: np.ndarray, actual: np.ndarray) -> ForecastScores:
    by_cell = crps(samples, actual)  # [week][party]
    hits = covered(samples, actual, COVERAGE_INTERVAL)
    quantiles = np.quantile(samples, QUANTILES, axis=0)
    return ForecastScores(
        crps_by_party={p: float(by_cell[:, j].mean()) for j, p in enumerate(PARTIES)},
        crps_mean=float(by_cell.mean()),
        coverage_by_party={p: float(hits[:, j].mean()) for j, p in enumerate(PARTIES)},
        coverage=float(hits.mean()),
        quantiles={str(q): quantiles[i].round(3).tolist() for i, q in enumerate(QUANTILES)},
    )
