"""The `whatif` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from whatif.actors import StandIn, StandInParameters
from whatif.backtest import (
    DEFAULT_CALIBRATION_PERIOD,
    BacktestConfig,
    CalibrationPeriod,
    require_tuned_outside,
    run_backtest,
)
from whatif.baselines import NaiveBaseline, StatisticalBaseline
from whatif.population import SyntheticPopulation, build_synthetic_population, fit_report
from whatif.poll_average import PollAverage, week_of, weekly_poll_average
from whatif.data.adapters import LicenceNotRecordedError, fetch, latest, snapshots
from whatif.boundary import WeeklyBoundary, build_boundary
from whatif.data.eurostat import INFLATION, UNEMPLOYMENT, Economy
from whatif.data.elections import ELECTIONS, election_before, national_shares
from whatif.events import GDELT, PAGEVIEWS, event_log, page_view_log
from whatif.data.polls import DAWUM, PARTIES, TRACKED_PARTIES, parse_dawum
from whatif.plots import plot_backtest, plot_rolling
from whatif.results import save_result
from whatif.rolling import WindowStep, rolling_start_weeks, run_rolling
from whatif.scenarios import NO_INTERVENTION, Scenario, ScenarioError, require_reactions, scenario_named
from whatif.twin import FIRST_GUESS_REACTIONS, Twin, TwinParameters
from whatif.data.store import MissingDataError


def parse_week(text: str) -> date:
    """Accept an ISO week (2025-W10) or any date (2025-03-05) and return its Monday."""
    if "-W" in text.upper():
        year, week = text.upper().split("-W")
        return date.fromisocalendar(int(year), int(week), 1)
    return week_of(date.fromisoformat(text))


def parse_day(text: str) -> date:
    """Accept a date (2025-03-05), or an ISO week (2025-W10) meaning its Monday."""
    if "-W" in text.upper():
        return parse_week(text)
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whatif", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch-polls", help="download national polls from DAWUM into the data store")

    commands.add_parser(
        "fetch-elections", help="download federal election results and Constituency structural data"
    )

    commands.add_parser("fetch-economy", help="download monthly inflation and unemployment for Germany")

    events = commands.add_parser(
        "fetch-events", help="download recorded Events for Germany from GDELT, by date range only"
    )
    events.add_argument("--start", type=parse_day, required=True, help="first day, e.g. 2025-01-06")
    events.add_argument("--end", type=parse_day, required=True, help="last day (inclusive)")

    attention = commands.add_parser("fetch-attention", help="download daily Wikipedia page views per party and topic")
    attention.add_argument("--start", type=parse_day, default=date(2016, 1, 1), help="first day (default 2016-01-01)")
    attention.add_argument(
        "--end", type=parse_day, default=date.today() - timedelta(days=3), help="last day (default 3 days ago)"
    )

    population = commands.add_parser(
        "population", help="build the Synthetic Population for a start week and print its fit report"
    )
    population.add_argument("--start", type=parse_week, required=True, help="start week, e.g. 2025-W10")
    population.add_argument("--scale", type=int, default=1000, help="eligible voters per Voter (default 1000)")
    population.add_argument("--seed", type=int, default=0)

    show = commands.add_parser("poll-average", help="print the weekly Poll Average")
    show.add_argument("--last", type=int, default=10, help="number of most recent weeks to show")

    backtest = commands.add_parser("backtest", help="run a Backtest from a past start week")
    _twin_options(backtest)
    backtest.add_argument(
        "--scenario", default=NO_INTERVENTION.name, help="name of the Scenario to run under (default no-intervention)"
    )

    compare = commands.add_parser(
        "compare", help="run two Scenarios from the same start week and show them side by side"
    )
    _twin_options(compare)
    compare.add_argument("first", help="the Scenario to compare against, e.g. no-intervention")
    compare.add_argument("second", help="the Scenario to compare")

    rolling = commands.add_parser(
        "rolling", help="run Backtests from many start weeks and summarise the Twin against both Baselines"
    )
    rolling.add_argument("--first", type=parse_week, default=parse_week("2020-W01"), help="first start week (default 2020-W01)")
    rolling.add_argument("--last", type=parse_week, help="last start week (default: the last with a full Horizon)")
    rolling.add_argument("--every", type=int, default=4, help="weeks between start weeks (default 4)")
    rolling.add_argument("--horizon", type=int, default=12, help="weeks ahead (default 12)")
    rolling.add_argument("--runs", type=int, default=200)
    rolling.add_argument("--seed", type=int, default=0)
    rolling.add_argument("--scale", type=int, default=10000, help="eligible voters per Voter (default 10000)")
    rolling.add_argument("--results-dir", type=Path, default=Path("results"))
    rolling.add_argument("--calibration-period", type=CalibrationPeriod.parse, default=DEFAULT_CALIBRATION_PERIOD,
                         help="ISO weeks whose Twin scores never count, e.g. 2017-W01:2019-W52 (the default)")

    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-polls":
            snapshot = fetch(DAWUM, args.data_dir)
            print(f"Saved {snapshot.path} ({snapshot.metadata['attribution']})")
        elif args.command == "fetch-elections":
            for election in ELECTIONS:
                for adapter in (election.results, election.structure):
                    snapshot = fetch(adapter, args.data_dir)
                    print(f"Saved {snapshot.path}")
        elif args.command == "fetch-economy":
            for adapter in (INFLATION, UNEMPLOYMENT):
                snapshot = fetch(adapter, args.data_dir)
                print(f"Saved {snapshot.path} ({snapshot.metadata['attribution']})")
        elif args.command in ("fetch-events", "fetch-attention") and args.end >= date.today() - timedelta(days=1):
            print("error: --end must be a day that is fully published, at the latest the day before yesterday.", file=sys.stderr)
            return 1
        elif args.command == "fetch-events":
            snapshot = fetch(GDELT, args.data_dir, start=args.start, end=args.end)
            print(
                f"Saved {snapshot.path}: {snapshot.metadata['files']} GDELT files, "
                f"{snapshot.metadata['missing_files']} never published ({snapshot.metadata['attribution']})"
            )
        elif args.command == "fetch-attention":
            snapshot = fetch(PAGEVIEWS, args.data_dir, start=args.start, end=args.end)
            print(f"Saved {snapshot.path} ({snapshot.metadata['attribution']})")
        elif args.command == "population":
            _population(args)
        elif args.command == "poll-average":
            average, _ = _load_poll_average(args.data_dir)
            _print_poll_average(average, args.last)
        elif args.command == "backtest":
            _backtest(args)
        elif args.command == "compare":
            _compare(args)
        elif args.command == "rolling":
            _rolling(args)
    except (MissingDataError, LicenceNotRecordedError, ScenarioError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _twin_options(command) -> None:
    command.add_argument("--start", type=parse_week, required=True, help="start week, e.g. 2025-W10")
    command.add_argument("--horizon", type=int, default=12, help="weeks ahead (default 12)")
    command.add_argument("--runs", type=int, default=200)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--scale", type=int, default=10000, help="eligible voters per Voter (default 10000)")
    command.add_argument("--results-dir", type=Path, default=Path("results"))
    command.add_argument(
        "--first-guess-reactions",
        action="store_true",
        help="let Voters react to the economy and Events with the uncalibrated first-guess strengths "
        "(needs fetch-economy, fetch-events and fetch-attention; for exploring, not for reported scores)",
    )
    command.add_argument(
        "--scenario-file", type=Path, default=Path("scenarios.toml"), help="where Scenarios are defined"
    )
    command.add_argument("--calibration-period", type=CalibrationPeriod.parse, default=DEFAULT_CALIBRATION_PERIOD,
                         help="ISO weeks whose Twin scores never count, e.g. 2017-W01:2019-W52 (the default)")
    command.add_argument(
        "--party-actors", choices=("none", "stand-in"), default="none",
        help="Party Actors choosing a move each week: none (default) or the rule-based stand-in "
        "(needs fetch-economy, fetch-events and fetch-attention)",
    )


def _population(args) -> None:
    election = election_before(args.start)
    results = election.results.parse(latest(election.results, args.data_dir).raw())
    structure = election.structure.parse(latest(election.structure, args.data_dir).raw())
    population = build_synthetic_population(results, structure, scale=args.scale, seed=args.seed)
    print(
        f"Synthetic Population for {args.start}: federal election of {election.day}, "
        f"{len(population):,} Voters at one per {args.scale:,} eligible voters, seed {args.seed}"
    )
    print(fit_report(population).format())


def _load_poll_average(data_dir: Path) -> tuple[PollAverage, str]:
    snapshot = latest(DAWUM, data_dir)
    return weekly_poll_average(parse_dawum(snapshot.load())), snapshot.version


def _print_poll_average(average: PollAverage, last: int) -> None:
    print("week        " + "".join(f"{p:>9}" for p in PARTIES))
    for i in range(max(0, len(average.weeks) - last), len(average.weeks)):
        mark = " *" if average.carried[i] else ""
        print(f"{average.weeks[i]}  " + "".join(f"{v:9.1f}" for v in average.shares[i]) + mark)
    print("* no poll that week; carried forward")


@dataclass
class _TwinSetup:
    """Everything a Twin needs for one start week, before a Scenario is chosen."""

    average: PollAverage
    config: BacktestConfig
    population: SyntheticPopulation
    parameters: TwinParameters
    boundary: WeeklyBoundary | None  # as recorded; None when no Voter rule or Party Actor uses it
    data_versions: dict[str, str]
    settings: dict
    party_actors: str = "none"
    stand_in: StandInParameters = field(default_factory=StandInParameters)

    def twin(self, scenario: Scenario, seed: int, name: str = "twin") -> Twin:
        require_reactions(scenario, self.parameters)
        start, horizon = self.config.start_week, self.config.horizon
        boundary = None if self.boundary is None else scenario.apply(self.boundary, start)
        actors = {p: StandIn(self.stand_in) for p in TRACKED_PARTIES} if self.party_actors == "stand-in" else None
        return Twin(
            self.population, self.parameters, seed=seed, boundary=boundary, name=name,
            actors=actors, forced_moves=scenario.forced_moves(start, horizon),
        )

    def twin_settings(self, scenario: Scenario) -> dict:
        return self.settings | {"scenario": scenario.describe()}


def _setup(args) -> _TwinSetup:
    average, version = _load_poll_average(args.data_dir)
    config = BacktestConfig(start_week=args.start, horizon=args.horizon, runs=args.runs, seed=args.seed)

    election = election_before(args.start)
    results_snapshot = latest(election.results, args.data_dir)
    structure_snapshot = latest(election.structure, args.data_dir)
    population = build_synthetic_population(
        election.results.parse(results_snapshot.raw()),
        election.structure.parse(structure_snapshot.raw()),
        scale=args.scale,
        seed=args.seed,
    )
    parameters = TwinParameters(**(FIRST_GUESS_REACTIONS if args.first_guess_reactions else {}))
    stand_in = StandInParameters()
    for tuned in (parameters.tuned_on, stand_in.tuned_on):
        try:
            require_tuned_outside(tuned, config)
        except ValueError as error:
            raise ScenarioError(str(error)) from error
    data_versions = {
        "dawum": version,
        election.results.name: results_snapshot.version,
        election.structure.name: structure_snapshot.version,
    }

    boundary = None
    if parameters.reacts_to_boundary or args.party_actors != "none":  # only then is Boundary data needed
        inflation, unemployment = latest(INFLATION, args.data_dir), latest(UNEMPLOYMENT, args.data_dir)
        economy = Economy(INFLATION.parse(inflation.raw()), UNEMPLOYMENT.parse(unemployment.raw()))
        events, views = event_log(args.data_dir), page_view_log(args.data_dir)
        boundary = build_boundary(
            config.start_week, config.horizon, economy, events, views, parameters.event_baseline_weeks
        )
        data_versions |= {
            INFLATION.name: inflation.version,
            UNEMPLOYMENT.name: unemployment.version,
            GDELT.name: "+".join(s.version for s in snapshots(GDELT, args.data_dir)),
            PAGEVIEWS.name: latest(PAGEVIEWS, args.data_dir).version,
        }
    settings = {
        **asdict(parameters),
        "scale": args.scale,
        "voters": len(population),
        "election_day": election.day.isoformat(),
        "party_actors": args.party_actors,
    }
    if args.party_actors != "none":
        settings |= {"decision_mode": "shared", "stand_in": asdict(stand_in)}
    return _TwinSetup(
        average, config, population, parameters, boundary, data_versions, settings, args.party_actors, stand_in
    )


def _backtest(args) -> None:
    scenario = scenario_named(args.scenario_file, args.scenario)
    setup = _setup(args)
    twin = setup.twin(scenario, seed=args.seed)
    config = setup.config
    result = run_backtest(
        setup.average,
        config,
        [NaiveBaseline(), StatisticalBaseline(), twin],
        data_versions=setup.data_versions,
        settings={"twin": setup.twin_settings(scenario)},
        calibration_period=args.calibration_period,
        election_results=_election_results(args.data_dir, config.start_week, config.horizon),
    )
    path = save_result(result, args.results_dir)
    plot_backtest(result, path.with_suffix(".png"))

    print(
        f"Backtest from {config.start_week}, {config.horizon} weeks, {config.runs} Runs, seed {config.seed}, "
        f"Scenario {scenario.name}"
    )
    if result.carried_weeks:
        print(f"Note: {len(result.carried_weeks)} Horizon weeks had no poll and were carried forward")
    if result.calibration:
        print("Note: starts in the Calibration Period, so the Twin's score here is not a Twin result")
    for election in result.elections:
        print(f"Federal election on {election['day']} during the Horizon; its result is in the plot and the file")
    print(f"{'forecast':<12}{'CRPS':>8}{'80% coverage':>14}")
    for name, scores in result.scores.items():
        print(f"{name:<12}{scores.crps_mean:8.3f}{scores.coverage:14.0%}")
    print(f"Saved {path} and its plot")


def _compare(args) -> None:
    """Two Scenarios from the same start week: each one's Outcome Distribution per party, side by side."""
    first, second = (scenario_named(args.scenario_file, name) for name in (args.first, args.second))
    setup = _setup(args)
    twins = [setup.twin(s, seed=args.seed, name=f"twin:{s.name}") for s in (first, second)]
    config = setup.config
    result = run_backtest(
        setup.average,
        config,
        twins,
        data_versions=setup.data_versions,
        settings={twin.name: setup.twin_settings(s) for twin, s in zip(twins, (first, second))},
        calibration_period=args.calibration_period,
        shared_noise=True,  # both Scenarios see the same polling noise, so differences are the Scenario's
    )
    path = save_result(result, args.results_dir, kind="compare")

    a, b = (result.scores[t.name] for t in twins)
    print(
        f"Scenarios from {config.start_week}, {config.horizon} weeks, {config.runs} Runs, seed {config.seed}. "
        f"Median and 80% interval (%) in the last Horizon week ({result.weeks[-1]}):"
    )
    print(f"{'party':<9}{first.name[:24]:>26}{second.name[:24]:>26}{'difference':>12}")
    for j, party in enumerate(PARTIES):
        cells = [
            f"{x.quantiles['0.5'][-1][j]:5.1f} [{x.quantiles['0.1'][-1][j]:4.1f}-{x.quantiles['0.9'][-1][j]:4.1f}]"
            for x in (a, b)
        ]
        difference = b.quantiles["0.5"][-1][j] - a.quantiles["0.5"][-1][j]
        print(f"{party:<9}{cells[0]:>26}{cells[1]:>26}{difference:>+12.1f}")
    print(f"Every week's distribution is in the saved result. Saved {path}")


def _election_results(data_dir: Path, first: date, horizon_or_last) -> dict[date, dict[str, float]]:
    """National results of the federal elections held from `first` to the end of the window.

    `horizon_or_last` is a Horizon in weeks, or the last day of the window.
    """
    last = horizon_or_last if isinstance(horizon_or_last, date) else first + timedelta(weeks=horizon_or_last, days=-1)
    return {
        e.day: national_shares(e.results.parse(latest(e.results, data_dir).raw()))
        for e in ELECTIONS
        if first <= e.day <= last
    }


def _rolling(args) -> None:
    average, version = _load_poll_average(args.data_dir)
    starts = rolling_start_weeks(average, args.first, args.every, args.horizon, last=args.last)
    window_end = starts[-1] + timedelta(weeks=args.horizon, days=-1)
    elections = _election_results(args.data_dir, starts[0], window_end)
    parameters = TwinParameters()
    populations: dict[date, tuple] = {}

    def step(start: date) -> WindowStep:
        election = election_before(start)
        if election.day not in populations:
            results, structure = latest(election.results, args.data_dir), latest(election.structure, args.data_dir)
            population = build_synthetic_population(
                election.results.parse(results.raw()), election.structure.parse(structure.raw()),
                scale=args.scale, seed=args.seed,
            )
            populations[election.day] = (population, {
                election.results.name: results.version, election.structure.name: structure.version,
            })
        population, versions = populations[election.day]
        return WindowStep(
            forecasters=[NaiveBaseline(), StatisticalBaseline(), Twin(population, parameters, seed=args.seed)],
            data_versions={"dawum": version, **versions},
            settings={"twin": {
                **asdict(parameters), "scale": args.scale, "voters": len(population),
                "election_day": election.day.isoformat(), "scenario": NO_INTERVENTION.describe(),
                "party_actors": "none",
            }},
        )

    result = run_rolling(
        average, starts, step, horizon=args.horizon, runs=args.runs, seed=args.seed,
        calibration_period=args.calibration_period, election_results=elections,
    )

    created = datetime.now(timezone.utc)
    run_dir = args.results_dir / (
        f"rolling-{starts[0]}-to-{starts[-1]}-every{args.every}-seed{args.seed}-{created:%Y%m%dT%H%M%SZ}"
    )
    files = []
    for backtest in result.backtests:
        path = save_result(backtest, run_dir)
        plot_backtest(backtest, path.with_suffix(".png"))
        files.append(path.name)
    period = result.calibration_period
    (run_dir / "summary.json").write_text(json.dumps({
        "created_at": created.isoformat(),
        "window": {"first": str(starts[0]), "last": str(starts[-1]), "every": args.every,
                   "horizon": args.horizon, "runs": args.runs, "seed": args.seed, "scale": args.scale},
        "calibration_period": [str(period.first), str(period.last)],
        "summary": {name: asdict(row) for name, row in result.summary.items()},
        "backtests": files,
        "calibration_backtests": [f for f, b in zip(files, result.backtests) if b.calibration],
    }, indent=2), encoding="utf-8")
    plot_rolling(result, run_dir / "summary.png", elections=[e.day for e in ELECTIONS])

    calibrating = sum(b.calibration for b in result.backtests)
    print(
        f"{len(starts)} Backtests from {starts[0]} to {starts[-1]}, every {args.every} weeks, "
        f"{args.horizon}-week Horizon, {args.runs} Runs, seed {args.seed}"
    )
    if calibrating:
        print(f"{calibrating} start in the Calibration Period ({period.first} to {period.last}): "
              "Baselines scored there, the Twin not")
    print(f"{'forecast':<12}{'Backtests':>10}{'mean CRPS':>11}{'80% coverage':>14}{'lowest CRPS':>13}")
    for name, row in result.summary.items():
        print(f"{name:<12}{row.backtests:>10}{row.crps_mean:11.3f}{row.coverage:14.0%}{row.lowest_crps:>13}")
    twin, naive, statistical = (result.summary.get(n) for n in ("twin", "naive", "statistical"))
    if twin and naive and statistical and twin.backtests:
        print(
            f"Over {twin.backtests} Backtests, the Twin's mean CRPS was {twin.crps_mean:.3f} against "
            f"{naive.crps_mean:.3f} (naive) and {statistical.crps_mean:.3f} (statistical); "
            f"it had the lowest CRPS in {twin.lowest_crps} of {twin.backtests}."
        )
    print(f"Saved {len(files)} Backtests, their plots, summary.json and summary.png in {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
