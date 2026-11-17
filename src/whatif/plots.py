"""Plots of Backtest results, saved as PNG files.

Uses matplotlib's Figure API directly (no pyplot), so no display or GUI backend is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from whatif.backtest import BacktestResult
from whatif.baselines import BASELINE_NAMES
from whatif.polls import PARTIES
from whatif.rolling import RollingResult

COLOURS = {"naive": "#8c8c8c", "statistical": "#3a6fb0", "twin": "#d9822b"}
ACTUAL = "#1a1a1a"
ELECTION = "#b03a48"


def _colour(name: str) -> str:
    return COLOURS.get(name, COLOURS["twin"] if name.startswith("twin") else "#6a3d9a")


def plot_backtest(result: BacktestResult, path: Path) -> Path:
    """Every party's Outcome Distributions over the Horizon against the real Poll Average.

    Each forecaster is drawn as its median and its 80% interval (10th to 90th percentile).
    A federal election held during the Horizon is marked with its result.
    """
    weeks = np.array(result.weeks, dtype="datetime64[D]")
    actual = np.array(result.actual)
    figure = Figure(figsize=(14, 6.5), layout="constrained")
    axes = figure.subplots(2, 4, sharex=True).ravel()
    for j, (axis, party) in enumerate(zip(axes, PARTIES)):
        for name, scores in result.scores.items():
            q = {k: np.array(v)[:, j] for k, v in scores.quantiles.items()}
            colour = _colour(name)
            axis.fill_between(weeks, q["0.1"], q["0.9"], color=colour, alpha=0.15, linewidth=0)
            axis.plot(weeks, q["0.5"], color=colour, linewidth=1.5, label=f"{name} (median, 80% interval)")
        axis.plot(weeks, actual[:, j], color=ACTUAL, linewidth=2, marker="o", markersize=3, label="Poll Average")
        for election in result.elections:
            day = np.datetime64(election["day"], "D")
            axis.axvline(day, color=ELECTION, linewidth=0.8, linestyle=":")
            axis.plot([day], [election["shares"][party]], marker="D", color=ELECTION, label="election result")
        axis.set_title(party, fontsize=10)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("share (%)")
    axes[4].set_ylabel("share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=len(unique), fontsize=8)
    calibration = " (Calibration Period: Twin not scored)" if result.calibration else ""
    figure.suptitle(
        f"Backtest from {result.config.start_week}, {result.config.horizon} weeks, "
        f"{result.config.runs} Runs, seed {result.config.seed}{calibration}",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    return path


def plot_rolling(rolling: RollingResult, path: Path, elections: Sequence[date] = ()) -> Path:
    """Mean CRPS per start week for every forecaster, with federal elections marked.

    Start weeks in the Calibration Period are shaded, and the Twin's scores there are left out,
    as they are from the summary.
    """
    starts = np.array([b.config.start_week for b in rolling.backtests], dtype="datetime64[D]")
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    axis = figure.subplots()
    names = list(rolling.summary)
    for name in names:
        crps = np.array(
            [
                b.scores[name].crps_mean if name in b.scores and (name in BASELINE_NAMES or not b.calibration) else np.nan
                for b in rolling.backtests
            ]
        )
        row = rolling.summary[name]
        axis.plot(
            starts, crps, color=_colour(name), marker="o", markersize=3, linewidth=1.3,
            label=f"{name}: mean {row.crps_mean:.3f} over {row.backtests}",
        )
    period = rolling.calibration_period
    if any(b.calibration for b in rolling.backtests):
        axis.axvspan(
            np.datetime64(period.first, "D"), np.datetime64(period.last, "D"),
            color="#999999", alpha=0.15, label="Calibration Period (Twin not scored)",
        )
    first, last = (starts.min(), starts.max()) if len(starts) else (None, None)
    for i, day in enumerate(elections):
        when = np.datetime64(day, "D")
        if first is not None and first <= when <= last + np.timedelta64(7 * rolling.backtests[-1].config.horizon, "D"):
            axis.axvline(when, color=ELECTION, linewidth=1, linestyle=":", label="federal election" if i == 0 else None)
    axis.set_ylabel("mean CRPS over the Horizon (lower is better)")
    axis.set_xlabel("start week")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper left")
    horizon = rolling.backtests[0].config.horizon if rolling.backtests else 0
    figure.suptitle(f"Rolling-window Backtests: {len(rolling.backtests)} start weeks, {horizon}-week Horizon", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    return path
