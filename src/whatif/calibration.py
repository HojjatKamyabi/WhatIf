"""Calibration: tune Voter reactions and Stand-in rules on the Calibration Period, never on scored weeks.

The search is a grid in two stages, scored by the mean CRPS over Backtests whose whole Horizon lies
inside the Calibration Period:

1. the Voter reactions to the economy and to Events, with no Party Actors;
2. the reaction to Party Actor moves and the Stand-in's rules, with stage 1's winner fixed.

Every grid point sees the same random stream, so points differ only by their values. Ties go to
the point that changes least from the untuned values. The winner is scored again at more Runs,
next to the untuned Twin and both Baselines, and saved with the period it was tuned on.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from whatif.actors import StandInParameters
from whatif.backtest import DEFAULT_CALIBRATION_PERIOD, BacktestConfig, CalibrationPeriod, run_backtest
from whatif.baselines import Forecaster, NaiveBaseline, StatisticalBaseline
from whatif.data.store import MissingDataError
from whatif.poll_average import PollAverage
from whatif.twin import FIRST_GUESS_REACTIONS, TwinParameters

# The values a calibration tunes, and so the values a calibration file sets.
TWIN_VALUES = ("inflation_reaction", "unemployment_reaction", "event_reaction", "move_reaction")
STAND_IN_VALUES = ("rebut_threshold", "min_attention")


@dataclass(frozen=True)
class Grid:
    """The values each stage tries. `reactions`: Twin parameters tuned without Party Actors.
    `actors`: `move_reaction` and the Stand-in's parameters, tuned with Stand-ins."""

    reactions: Mapping[str, tuple[float, ...]]
    actors: Mapping[str, tuple[float, ...]]

    def describe(self) -> dict:
        return {"reactions": {k: list(v) for k, v in self.reactions.items()},
                "actors": {k: list(v) for k, v in self.actors.items()}}


def _scaled(name: str, factors: Sequence[float]) -> tuple[float, ...]:
    return tuple(f * FIRST_GUESS_REACTIONS[name] for f in factors)


# Multiples of the researcher's first guesses. The Event reaction is also tried negative: in the
# 2025 campaign heavily covered parties rose despite bad news, so the rule's direction is in question.
DEFAULT_GRID = Grid(
    reactions={
        "inflation_reaction": _scaled("inflation_reaction", (0, 0.5, 1, 2)),
        "unemployment_reaction": _scaled("unemployment_reaction", (0, 0.5, 1, 2)),
        "event_reaction": _scaled("event_reaction", (-1, -0.5, 0, 0.5, 1, 2)),
    },
    actors={
        "move_reaction": _scaled("move_reaction", (0, 0.5, 1, 2)),
        "rebut_threshold": (3.0, 6.0, 12.0),
        "min_attention": (0.5, 1.0, 2.0),
    },
)


def points(axes: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    """Every combination of the axes' values, the first axis changing fastest."""
    names = list(axes)
    combos = itertools.product(*(axes[n] for n in reversed(names)))
    return [dict(zip(names, reversed(combo))) for combo in combos]


def calibration_start_weeks(
    average: PollAverage,
    period: CalibrationPeriod = DEFAULT_CALIBRATION_PERIOD,
    every: int = 4,
    horizon: int = 12,
    min_history_weeks: int = BacktestConfig.min_history_weeks,
    after: date | None = None,
) -> list[date]:
    """Start weeks every `every` weeks whose whole Horizon lies inside the period, with enough history.

    With `after`, only start weeks later than it are kept, still on the same `every`-week rhythm:
    a Twin is built from the last federal election before its start week, so none can start
    before the first one on record.
    """
    first = max(period.first, average.weeks[0] + timedelta(weeks=min_history_weeks))
    last = min(period.last, average.weeks[-1]) - timedelta(weeks=horizon - 1)
    if first > last:
        raise MissingDataError(
            f"No {horizon}-week Horizon fits inside the Calibration Period ({period.first} to {period.last}) "
            f"with {min_history_weeks} weeks of polls before it."
        )
    starts = [first + timedelta(weeks=n) for n in range(0, (last - first).days // 7 + 1, every)]
    kept = [s for s in starts if after is None or s > after]
    if not kept:
        raise MissingDataError(
            f"No start week inside the Calibration Period ({period.first} to {period.last}) comes after {after}."
        )
    return kept


@dataclass(frozen=True)
class GridScore:
    """One grid point: its values, and the Twin's mean CRPS and coverage over the Backtests."""

    values: dict[str, float]
    crps_mean: float
    coverage: float
    backtests: list[float]  # each Backtest's mean CRPS, in start-week order


@dataclass(frozen=True)
class Calibration:
    period: CalibrationPeriod
    starts: list[date]
    horizon: int
    runs: int
    final_runs: int
    seed: int
    grid: Grid
    reactions: list[GridScore]  # stage 1, every point
    actors: list[GridScore]  # stage 2, every point
    twin: TwinParameters  # tuned, with tuned_on set
    stand_in: StandInParameters  # tuned, with tuned_on set
    final: dict[str, dict] = field(default_factory=dict)  # name -> backtests, crps_mean, coverage

    @property
    def reactions_best(self) -> GridScore:
        return self.reactions[_best(self.reactions, _untuned_values())]

    @property
    def actors_best(self) -> GridScore:
        return self.actors[_best(self.actors, _untuned_values())]


# make_twin(start week, Twin parameters, Stand-in parameters or None for no Party Actors, name)
MakeTwin = Callable[[date, TwinParameters, StandInParameters | None, str], Forecaster]
Progress = Callable[[str, int, int, GridScore], None]


def calibrate(
    average: PollAverage,
    starts: Sequence[date],
    make_twin: MakeTwin,
    period: CalibrationPeriod = DEFAULT_CALIBRATION_PERIOD,
    grid: Grid = DEFAULT_GRID,
    horizon: int = 12,
    runs: int = 50,
    final_runs: int = 200,
    seed: int = 0,
    base: TwinParameters | None = None,
    election_results: dict[date, dict[str, float]] | None = None,
    progress: Progress | None = None,
) -> Calibration:
    """Search `grid` on Backtests from `starts`, all of which must score only weeks inside `period`."""
    base = base or TwinParameters()
    outside = [s for s in starts if s < period.first or s + timedelta(weeks=horizon - 1) > period.last]
    if outside:
        raise ValueError(
            f"Backtests from {', '.join(map(str, outside))} score weeks outside the Calibration Period "
            f"({period.first} to {period.last}); tuning may only see weeks inside it."
        )
    configs = [BacktestConfig(start_week=s, horizon=horizon, runs=runs, seed=seed) for s in starts]

    def score(parameters: TwinParameters, stand_in: StandInParameters | None, values: dict) -> GridScore:
        crps, coverage = [], []
        for config in configs:
            twin = make_twin(config.start_week, parameters, stand_in, "twin")
            result = run_backtest(average, config, [twin], calibration_period=period)
            crps.append(result.scores["twin"].crps_mean)
            coverage.append(result.scores["twin"].coverage)
        return GridScore(dict(values), float(np.mean(crps)), float(np.mean(coverage)), crps)

    untuned = _untuned_values(base)
    reactions, grid_points = [], points(grid.reactions)
    for i, values in enumerate(grid_points):
        reactions.append(score(replace(base, **values), None, values))
        if progress:
            progress("reactions", i + 1, len(grid_points), reactions[-1])
    twin = replace(base, **reactions[_best(reactions, untuned)].values)

    actors, grid_points = [], points(grid.actors)
    for i, values in enumerate(grid_points):
        twin_values = {k: v for k, v in values.items() if k in TWIN_VALUES}
        stand_in_values = {k: v for k, v in values.items() if k in STAND_IN_VALUES}
        actors.append(score(replace(twin, **twin_values), StandInParameters(**stand_in_values), values))
        if progress:
            progress("actors", i + 1, len(grid_points), actors[-1])
    best = actors[_best(actors, untuned)].values
    twin = replace(twin, **{k: v for k, v in best.items() if k in TWIN_VALUES}, tuned_on=period)
    stand_in = StandInParameters(**{k: v for k, v in best.items() if k in STAND_IN_VALUES}, tuned_on=period)

    final = _final(average, starts, make_twin, base, twin, stand_in, horizon, final_runs, seed, period,
                   election_results)
    return Calibration(period, list(starts), horizon, runs, final_runs, seed, grid, reactions, actors,
                       twin, stand_in, final)


def _untuned_values(base: TwinParameters | None = None) -> dict[str, float]:
    base = base or TwinParameters()
    stand_in = StandInParameters()
    return {k: getattr(base, k) for k in TWIN_VALUES} | {k: getattr(stand_in, k) for k in STAND_IN_VALUES}


def _best(scores: Sequence[GridScore], untuned: Mapping[str, float]) -> int:
    """The lowest mean CRPS; on a tie, the point changing fewest values from untuned, then the first."""
    def key(i: int):
        changed = sum(v != untuned.get(k) for k, v in scores[i].values.items())
        return (scores[i].crps_mean, changed, i)

    return min(range(len(scores)), key=key)


def _final(average, starts, make_twin, base, twin, stand_in, horizon, runs, seed, period, elections) -> dict:
    """The winner at more Runs, next to the untuned Twin and both Baselines, on the same Backtests."""
    by_name: dict[str, list] = {}
    for start in starts:
        forecasters = [
            NaiveBaseline(),
            StatisticalBaseline(),
            make_twin(start, replace(base, tuned_on=None), None, "twin:untuned"),
            make_twin(start, twin, None, "twin:calibrated"),
            make_twin(start, twin, stand_in, "twin:calibrated+stand-ins"),
        ]
        config = BacktestConfig(start_week=start, horizon=horizon, runs=runs, seed=seed)
        result = run_backtest(average, config, forecasters, shared_noise=True, calibration_period=period,
                              election_results=elections)
        for name, scores in result.scores.items():
            by_name.setdefault(name, []).append(scores)
    return {
        name: {"backtests": len(rows), "crps_mean": float(np.mean([r.crps_mean for r in rows])),
               "coverage": float(np.mean([r.coverage for r in rows]))}
        for name, rows in by_name.items()
    }


# --- Saving and loading ---------------------------------------------------------------------------


def calibration_record(calibration: Calibration, data_versions: Mapping[str, str]) -> dict:
    period = calibration.period
    return {
        "tuned_on": [str(period.first), str(period.last)],
        "tuned": {
            "twin": {k: getattr(calibration.twin, k) for k in TWIN_VALUES},
            "stand_in": {k: getattr(calibration.stand_in, k) for k in STAND_IN_VALUES},
        },
        "starts": [str(s) for s in calibration.starts],
        "horizon": calibration.horizon,
        "runs": calibration.runs,
        "final_runs": calibration.final_runs,
        "seed": calibration.seed,
        "grid": calibration.grid.describe(),
        "search": {
            "reactions": [asdict(s) for s in calibration.reactions],
            "actors": [asdict(s) for s in calibration.actors],
        },
        "final": calibration.final,
        "data_versions": dict(data_versions),
    }


def save_calibration(
    calibration: Calibration, directory: Path, data_versions: Mapping[str, str], created: datetime | None = None
) -> Path:
    created = created or datetime.now(timezone.utc)
    period = calibration.period
    record = {"created_at": created.isoformat()} | calibration_record(calibration, data_versions)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"calibration-{period.first}-to-{period.last}-seed{calibration.seed}-{created:%Y%m%dT%H%M%SZ}.json"
    )
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


class CalibrationFileError(ValueError):
    """A calibration file that is missing or not one `whatif calibrate` wrote."""


def load_calibrated(path: Path) -> tuple[TwinParameters, StandInParameters]:
    """The tuned values of a saved calibration, carrying the period they were tuned on."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        first, last = (date.fromisoformat(d) for d in record["tuned_on"])
        tuned = record["tuned"]
        twin_values, stand_in_values = dict(tuned["twin"]), dict(tuned["stand_in"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CalibrationFileError(f"Cannot read calibrated values from {path}: {error}") from error
    period = CalibrationPeriod(first, last)
    unknown = (set(twin_values) - set(TWIN_VALUES)) | (set(stand_in_values) - set(STAND_IN_VALUES))
    if unknown:
        raise CalibrationFileError(f"{path} sets values a calibration does not tune: {', '.join(sorted(unknown))}.")
    tuned = {"twin": twin_values, "stand_in": stand_in_values}
    return (
        TwinParameters(**tuned["twin"], tuned_on=period),
        StandInParameters(**tuned["stand_in"], tuned_on=period),
    )
