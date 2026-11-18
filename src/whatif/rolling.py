"""Rolling-window Backtests: the same Backtest from many start weeks, summarised in one place."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from whatif.backtest import (
    DEFAULT_CALIBRATION_PERIOD,
    BacktestConfig,
    BacktestResult,
    CalibrationPeriod,
    run_backtest,
)
from whatif.baselines import BASELINE_NAMES, Forecaster
from whatif.poll_average import PollAverage, week_of
from whatif.data.store import MissingDataError


@dataclass(frozen=True)
class WindowStep:
    """What one start week of a window runs: its forecasters, and what to record about them."""

    forecasters: Sequence[Forecaster]
    data_versions: dict[str, str] = field(default_factory=dict)
    settings: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecasterSummary:
    """One forecaster's scores over the Backtests of a window that count for it."""

    backtests: int  # how many Backtests count: for the Twin, none from the Calibration Period
    crps_mean: float  # mean of each Backtest's mean CRPS (lower is better)
    coverage: float  # mean share of actual values inside the 80% interval (0.8 is right)
    lowest_crps: int  # Backtests, outside the Calibration Period, in which it had the lowest CRPS


@dataclass(frozen=True)
class RollingResult:
    backtests: list[BacktestResult]
    summary: dict[str, ForecasterSummary]
    calibration_period: CalibrationPeriod


def rolling_start_weeks(
    average: PollAverage,
    first: date,
    every: int,
    horizon: int,
    last: date | None = None,
    min_history_weeks: int = BacktestConfig.min_history_weeks,
) -> list[date]:
    """Start weeks from `first`, every `every` weeks, up to the last one whose Horizon is in the data."""
    first = week_of(first)
    earliest = average.weeks[0] + timedelta(weeks=min_history_weeks)
    if first < earliest:
        raise MissingDataError(
            f"A window starting {first} has too little history: the Poll Average starts {average.weeks[0]}, "
            f"and a Backtest needs {min_history_weeks} weeks before its start. Start at {earliest} or later."
        )
    last_full = average.weeks[-1] - timedelta(weeks=horizon - 1)
    last = last_full if last is None else min(week_of(last), last_full)
    if first > last:
        raise MissingDataError(f"No start week from {first} has a full {horizon}-week Horizon in the Poll Average.")
    return [first + timedelta(weeks=n) for n in range(0, (last - first).days // 7 + 1, every)]


def run_rolling(
    average: PollAverage,
    starts: Sequence[date],
    step_for: Callable[[date], WindowStep | Sequence[Forecaster]],
    horizon: int,
    runs: int,
    seed: int,
    calibration_period: CalibrationPeriod = DEFAULT_CALIBRATION_PERIOD,
    election_results: dict[date, dict[str, float]] | None = None,
) -> RollingResult:
    """One Backtest per start week, each exactly as `run_backtest` would run it alone with the same seed.

    The summary scores each forecaster over the window. A Backtest starting in the Calibration
    Period counts for the Baselines but never for the Twin (or any forecaster that isn't a
    Baseline), so tuning on those weeks can't carry hindsight into the Twin's scores.
    """
    backtests = []
    for start in starts:
        step = step_for(start)
        if not isinstance(step, WindowStep):
            step = WindowStep(forecasters=step)
        backtests.append(
            run_backtest(
                average,
                BacktestConfig(start_week=start, horizon=horizon, runs=runs, seed=seed),
                step.forecasters,
                data_versions=step.data_versions,
                settings=step.settings,
                calibration_period=calibration_period,
                election_results=election_results,
            )
        )
    return RollingResult(backtests, _summarise(backtests), calibration_period)


def _summarise(backtests: list[BacktestResult]) -> dict[str, ForecasterSummary]:
    names = list(dict.fromkeys(name for b in backtests for name in b.scores))
    scored = [b for b in backtests if not b.calibration]
    lowest = {name: 0 for name in names}
    for backtest in scored:
        lowest[min(backtest.scores, key=lambda name: backtest.scores[name].crps_mean)] += 1

    summary = {}
    for name in names:
        counted = [b.scores[name] for b in (backtests if name in BASELINE_NAMES else scored) if name in b.scores]
        summary[name] = ForecasterSummary(
            backtests=len(counted),
            crps_mean=float(np.mean([s.crps_mean for s in counted])) if counted else float("nan"),
            coverage=float(np.mean([s.coverage for s in counted])) if counted else float("nan"),
            lowest_crps=lowest[name],
        )
    return summary
