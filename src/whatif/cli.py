"""The `whatif` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from whatif.backtest import BacktestConfig, run_backtest
from whatif.baselines import NaiveBaseline, StatisticalBaseline
from whatif.population import build_synthetic_population, fit_report
from whatif.poll_average import PollAverage, week_of, weekly_poll_average
from whatif.adapters import LicenceNotRecordedError, fetch, latest
from whatif.elections import ELECTIONS, election_before
from whatif.polls import DAWUM, PARTIES, parse_dawum
from whatif.results import save_result
from whatif.twin import Twin, TwinParameters
from whatif.store import MissingDataError


def parse_week(text: str) -> date:
    """Accept an ISO week (2025-W10) or any date (2025-03-05) and return its Monday."""
    if "-W" in text.upper():
        year, week = text.upper().split("-W")
        return date.fromisocalendar(int(year), int(week), 1)
    return week_of(date.fromisoformat(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whatif", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch-polls", help="download national polls from DAWUM into the data store")

    commands.add_parser(
        "fetch-elections", help="download federal election results and Constituency structural data"
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
    backtest.add_argument("--start", type=parse_week, required=True, help="start week, e.g. 2025-W10")
    backtest.add_argument("--horizon", type=int, default=12, help="weeks ahead (default 12)")
    backtest.add_argument("--runs", type=int, default=200)
    backtest.add_argument("--seed", type=int, default=0)
    backtest.add_argument("--scale", type=int, default=10000, help="eligible voters per Voter (default 10000)")
    backtest.add_argument("--results-dir", type=Path, default=Path("results"))

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
        elif args.command == "population":
            _population(args)
        elif args.command == "poll-average":
            average, _ = _load_poll_average(args.data_dir)
            _print_poll_average(average, args.last)
        elif args.command == "backtest":
            _backtest(args)
    except (MissingDataError, LicenceNotRecordedError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


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


def _backtest(args) -> None:
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
    parameters = TwinParameters()
    twin = Twin(population, parameters, seed=args.seed)

    result = run_backtest(
        average,
        config,
        [NaiveBaseline(), StatisticalBaseline(), twin],
        data_versions={
            "dawum": version,
            election.results.name: results_snapshot.version,
            election.structure.name: structure_snapshot.version,
        },
        settings={
            "twin": {
                **asdict(parameters),
                "scale": args.scale,
                "voters": len(population),
                "election_day": election.day.isoformat(),
            }
        },
    )
    path = save_result(result, args.results_dir)

    print(f"Backtest from {config.start_week}, {config.horizon} weeks, {config.runs} Runs, seed {config.seed}")
    if result.carried_weeks:
        print(f"Note: {len(result.carried_weeks)} Horizon weeks had no poll and were carried forward")
    print(f"{'forecast':<12}{'CRPS':>8}{'80% coverage':>14}")
    for name, scores in result.scores.items():
        print(f"{name:<12}{scores.crps_mean:8.3f}{scores.coverage:14.0%}")
    print(f"Saved {path}")


if __name__ == "__main__":
    sys.exit(main())
