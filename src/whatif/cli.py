"""The `whatif` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from whatif import calibration
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
from whatif.calibration import (
    CalibrationFileError,
    calibrate,
    calibration_start_weeks,
    load_calibrated,
    save_calibration,
)
from whatif.data.eurostat import INFLATION, UNEMPLOYMENT, Economy
from whatif.data.elections import ELECTIONS, election_before, national_shares
from whatif.events import GDELT, PAGEVIEWS, event_log, page_view_log
from whatif.data.polls import DAWUM, PARTIES, TRACKED_PARTIES, parse_dawum
from whatif.llm import (
    LanguageModelActor,
    Masker,
    ModelConfig,
    ModelConfigError,
    OpenAICompatibleClient,
    ResponseCache,
    UsageLedger,
    cutoff_label,
    hindsight_leak_flag,
    load_models,
)
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
    _reaction_options(rolling)
    rolling.add_argument(
        "--party-actors", choices=("none", "stand-in"), default="none",
        help="Party Actors choosing a move each week: none (default) or the rule-based stand-in",
    )

    tune = commands.add_parser(
        "calibrate",
        help="tune Voter reactions and Stand-in rules on Backtests inside the Calibration Period only",
    )
    tune.add_argument("--calibration-period", type=CalibrationPeriod.parse, default=DEFAULT_CALIBRATION_PERIOD,
                      help="ISO weeks to tune on, e.g. 2017-W01:2019-W52 (the default)")
    tune.add_argument("--every", type=int, default=4, help="weeks between start weeks (default 4)")
    tune.add_argument("--horizon", type=int, default=12, help="weeks ahead (default 12)")
    tune.add_argument("--runs", type=int, default=50, help="Runs per Backtest during the search (default 50)")
    tune.add_argument("--final-runs", type=int, default=200, help="Runs for scoring the winner again (default 200)")
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--scale", type=int, default=10000, help="eligible voters per Voter (default 10000)")
    tune.add_argument("--results-dir", type=Path, default=Path("results"))

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
        elif args.command == "calibrate":
            _calibrate(args)
    except (MissingDataError, LicenceNotRecordedError, ScenarioError, ModelConfigError, CalibrationFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _reaction_options(command) -> None:
    reactions = command.add_mutually_exclusive_group()
    reactions.add_argument(
        "--first-guess-reactions",
        action="store_true",
        help="let Voters react to the economy and Events with the uncalibrated first-guess strengths "
        "(needs fetch-economy, fetch-events and fetch-attention; for exploring, not for reported scores)",
    )
    reactions.add_argument(
        "--calibrated", type=Path, metavar="FILE",
        help="Voter reactions and Stand-in rules from a calibration file saved by `whatif calibrate`; "
        "refused for Backtests scoring weeks they were tuned on",
    )


def _parameters(args) -> tuple[TwinParameters, StandInParameters, dict]:
    """The Twin's and Stand-ins' parameters: calibrated, first guesses, or untuned; and what to record."""
    if getattr(args, "calibrated", None):
        twin, stand_in = load_calibrated(args.calibrated)
        return twin, stand_in, {"calibrated": str(args.calibrated)}
    first_guesses = FIRST_GUESS_REACTIONS if getattr(args, "first_guess_reactions", False) else {}
    return TwinParameters(**first_guesses), StandInParameters(), {}


def _refuse_tuned_weeks(config: BacktestConfig, *parameters) -> None:
    for tuned in (p.tuned_on for p in parameters):
        try:
            require_tuned_outside(tuned, config)
        except ValueError as error:
            raise ScenarioError(str(error)) from error


@dataclass
class _BoundaryData:
    """The economy, recorded Events and page views, loaded once for any number of Backtests."""

    economy: Economy
    events: object  # EventLog
    views: object  # PageViewLog
    versions: dict[str, str]

    @classmethod
    def load(cls, data_dir: Path) -> _BoundaryData:
        inflation, unemployment = latest(INFLATION, data_dir), latest(UNEMPLOYMENT, data_dir)
        return cls(
            economy=Economy(INFLATION.parse(inflation.raw()), UNEMPLOYMENT.parse(unemployment.raw())),
            events=event_log(data_dir),
            views=page_view_log(data_dir),
            versions={
                INFLATION.name: inflation.version,
                UNEMPLOYMENT.name: unemployment.version,
                GDELT.name: "+".join(s.version for s in snapshots(GDELT, data_dir)),
                PAGEVIEWS.name: latest(PAGEVIEWS, data_dir).version,
            },
        )

    def boundary(self, start: date, horizon: int, baseline_weeks: int) -> WeeklyBoundary:
        return build_boundary(start, horizon, self.economy, self.events, self.views, baseline_weeks)


class _Populations:
    """The Synthetic Population of the federal election before each start week, built once each."""

    def __init__(self, data_dir: Path, scale: int, seed: int):
        self.data_dir, self.scale, self.seed = data_dir, scale, seed
        self._built: dict[date, tuple[SyntheticPopulation, dict[str, str]]] = {}

    def __call__(self, start: date) -> tuple[SyntheticPopulation, dict[str, str], date]:
        election = election_before(start)
        if election.day not in self._built:
            results, structure = latest(election.results, self.data_dir), latest(election.structure, self.data_dir)
            population = build_synthetic_population(
                election.results.parse(results.raw()), election.structure.parse(structure.raw()),
                scale=self.scale, seed=self.seed,
            )
            self._built[election.day] = (population, {
                election.results.name: results.version, election.structure.name: structure.version,
            })
        population, versions = self._built[election.day]
        return population, versions, election.day


def _twin_options(command) -> None:
    command.add_argument("--start", type=parse_week, required=True, help="start week, e.g. 2025-W10")
    command.add_argument("--horizon", type=int, default=12, help="weeks ahead (default 12)")
    command.add_argument(
        "--runs", type=int, default=None,
        help="Runs (default 200; 20 for language-model Party Actors with per-Run decisions)",
    )
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--scale", type=int, default=10000, help="eligible voters per Voter (default 10000)")
    command.add_argument("--results-dir", type=Path, default=Path("results"))
    _reaction_options(command)
    command.add_argument(
        "--scenario-file", type=Path, default=Path("scenarios.toml"), help="where Scenarios are defined"
    )
    command.add_argument("--calibration-period", type=CalibrationPeriod.parse, default=DEFAULT_CALIBRATION_PERIOD,
                         help="ISO weeks whose Twin scores never count, e.g. 2017-W01:2019-W52 (the default)")
    command.add_argument(
        "--party-actors", choices=("none", "stand-in", "language-model"), default="none",
        help="Party Actors choosing a move each week: none (default), the rule-based stand-in, or a "
        "language model (always repeated with stand-ins); both need fetch-economy, fetch-events and "
        "fetch-attention",
    )
    command.add_argument("--model", help="the model's name in the models file, for --party-actors language-model")
    command.add_argument("--models-file", type=Path, default=Path("models.toml"),
                         help="where models are defined (default models.toml; see models.example.toml)")
    command.add_argument(
        "--decision-mode", choices=("shared", "per-run"), default="shared",
        help="shared (default): Party Actors decide once per week for all Runs; per-run: every Run decides anew",
    )
    command.add_argument("--cutoff-margin-months", type=int, default=3,
                         help="safety margin after a model's Training Cutoff for the After label (default 3)")


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
    decision_mode: str = "shared"
    model: ModelConfig | None = None  # for language-model Party Actors
    masker: Masker | None = None
    ledger: UsageLedger | None = None
    cache_dir: Path | None = None
    cutoff_margin_months: int = 3

    def actors(self, kind: str, seed: int) -> dict | None:
        if kind == "stand-in":
            return {p: StandIn(self.stand_in) for p in TRACKED_PARTIES}
        if kind == "language-model":
            client = OpenAICompatibleClient(self.model)
            cache = ResponseCache(self.cache_dir)
            return {
                p: LanguageModelActor(client, self.model, cache, self.masker, self.config.horizon, seed, self.ledger)
                for p in TRACKED_PARTIES
            }
        return None

    def twin(self, scenario: Scenario, seed: int, name: str = "twin", kind: str | None = None) -> Twin:
        require_reactions(scenario, self.parameters)
        start, horizon = self.config.start_week, self.config.horizon
        boundary = None if self.boundary is None else scenario.apply(self.boundary, start)
        return Twin(
            self.population, self.parameters, seed=seed, boundary=boundary, name=name,
            actors=self.actors(kind or self.party_actors, seed), forced_moves=scenario.forced_moves(start, horizon),
            decision_mode=self.decision_mode,
        )

    def twins(self, scenario: Scenario, seed: int) -> list[Twin]:
        """The Twins a Backtest scores: a language-model run is always repeated with Stand-ins."""
        if self.party_actors == "language-model":
            return [
                self.twin(scenario, seed, name="twin:stand-in", kind="stand-in"),
                self.twin(scenario, seed, name="twin:language-model", kind="language-model"),
            ]
        return [self.twin(scenario, seed)]

    def cutoff_label(self) -> str:
        first = self.config.start_week
        last = first + timedelta(weeks=self.config.horizon, days=-1)
        return cutoff_label(first, last, self.model.training_cutoff, self.cutoff_margin_months)

    def twin_settings(self, scenario: Scenario, kind: str | None = None) -> dict:
        kind = kind or self.party_actors
        settings = self.settings | {"scenario": scenario.describe(), "party_actors": kind}
        if kind != "none":
            settings |= {"decision_mode": self.decision_mode, "stand_in": asdict(self.stand_in)}
        if kind == "language-model":
            settings |= {
                "model": self.model.describe(),
                "cutoff_margin_months": self.cutoff_margin_months,
                "cutoff_label": self.cutoff_label(),
                "masking": self.masker.describe(),
                "prompts": "masked",
            }
        return settings


def _setup(args) -> _TwinSetup:
    language_model = args.party_actors == "language-model"
    model = _model(args) if language_model else None
    average, version = _load_poll_average(args.data_dir)
    runs = args.runs or (20 if language_model and args.decision_mode == "per-run" else 200)
    config = BacktestConfig(start_week=args.start, horizon=args.horizon, runs=runs, seed=args.seed)

    election = election_before(args.start)
    results_snapshot = latest(election.results, args.data_dir)
    structure_snapshot = latest(election.structure, args.data_dir)
    population = build_synthetic_population(
        election.results.parse(results_snapshot.raw()),
        election.structure.parse(structure_snapshot.raw()),
        scale=args.scale,
        seed=args.seed,
    )
    parameters, stand_in, provenance = _parameters(args)
    _refuse_tuned_weeks(config, parameters, stand_in)
    data_versions = {
        "dawum": version,
        election.results.name: results_snapshot.version,
        election.structure.name: structure_snapshot.version,
    }

    boundary = None
    if parameters.reacts_to_boundary or args.party_actors != "none":  # only then is Boundary data needed
        data = _BoundaryData.load(args.data_dir)
        boundary = data.boundary(config.start_week, config.horizon, parameters.event_baseline_weeks)
        data_versions |= data.versions
    settings = {
        **asdict(parameters),
        **provenance,
        "scale": args.scale,
        "voters": len(population),
        "election_day": election.day.isoformat(),
    }
    return _TwinSetup(
        average, config, population, parameters, boundary, data_versions, settings, args.party_actors, stand_in,
        decision_mode=args.decision_mode,
        model=model,
        masker=Masker(seed=args.seed) if language_model else None,
        ledger=UsageLedger(model.input_price, model.output_price) if language_model else None,
        cache_dir=args.data_dir / "llm-cache",
        cutoff_margin_months=args.cutoff_margin_months,
    )


def _model(args) -> ModelConfig:
    if not args.model:
        raise ModelConfigError("--party-actors language-model needs --model NAME, from the models file.")
    try:
        models = load_models(args.models_file)
    except (FileNotFoundError, ValueError) as error:
        raise ModelConfigError(str(error)) from error
    if args.model not in models:
        raise ModelConfigError(f"No model {args.model!r} in {args.models_file}. Known: {', '.join(models) or 'none'}.")
    return models[args.model]


def _backtest(args) -> None:
    scenario = scenario_named(args.scenario_file, args.scenario)
    setup = _setup(args)
    twins = setup.twins(scenario, seed=args.seed)
    kinds = {"twin:stand-in": "stand-in", "twin:language-model": "language-model"}
    config = setup.config
    result = run_backtest(
        setup.average,
        config,
        [NaiveBaseline(), StatisticalBaseline(), *twins],
        data_versions=setup.data_versions,
        settings={t.name: setup.twin_settings(scenario, kinds.get(t.name)) for t in twins},
        calibration_period=args.calibration_period,
        election_results=_election_results(args.data_dir, config.start_week, config.horizon),
        shared_noise=len(twins) > 1,  # the two arms see the same polling noise
    )
    if setup.ledger is not None:
        result.settings["twin:language-model"]["usage"] = setup.ledger.describe()
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
    print(f"{'forecast':<22}{'CRPS':>8}{'80% coverage':>14}")
    for name, scores in result.scores.items():
        print(f"{name:<22}{scores.crps_mean:8.3f}{scores.coverage:14.0%}")
    if setup.model is not None:
        model, usage = setup.model, setup.ledger
        print(
            f"Model {model.model} ({model.quantization}) at {model.base_url}, Training Cutoff "
            f"{model.training_cutoff}; Cutoff Label: {setup.cutoff_label()} "
            f"(safety margin {setup.cutoff_margin_months} months); {setup.decision_mode} decisions"
        )
        print(
            f"Language-model cost: ${usage.cost_usd:.4f} for {usage.calls} calls "
            f"({usage.prompt_tokens:,} prompt and {usage.completion_tokens:,} completion tokens), "
            f"{usage.cache_hits} answers from the cache, {usage.invalid_answers} invalid answers"
        )
        scores = result.scores
        if hindsight_leak_flag(
            scores["twin:language-model"].crps_mean, scores["twin:stand-in"].crps_mean, scores["statistical"].crps_mean
        ):
            print("Possible Hindsight Leak: only the language-model Twin beats the statistical Baseline.")
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
    if setup.ledger is not None:
        for twin in twins:
            result.settings[twin.name]["usage"] = setup.ledger.describe()  # one ledger for both Scenarios
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
    parameters, stand_in, provenance = _parameters(args)
    populations = _Populations(args.data_dir, args.scale, args.seed)
    needs_boundary = parameters.reacts_to_boundary or args.party_actors != "none"
    data = _BoundaryData.load(args.data_dir) if needs_boundary else None

    def step(start: date) -> WindowStep:
        _refuse_tuned_weeks(BacktestConfig(start_week=start, horizon=args.horizon, runs=args.runs), parameters, stand_in)
        population, versions, election_day = populations(start)
        boundary = data.boundary(start, args.horizon, parameters.event_baseline_weeks) if data else None
        actors = {p: StandIn(stand_in) for p in TRACKED_PARTIES} if args.party_actors == "stand-in" else None
        settings = {
            **asdict(parameters), **provenance, "scale": args.scale, "voters": len(population),
            "election_day": election_day.isoformat(), "scenario": NO_INTERVENTION.describe(),
            "party_actors": args.party_actors,
        }
        if actors:
            settings |= {"decision_mode": "shared", "stand_in": asdict(stand_in)}
        return WindowStep(
            forecasters=[
                NaiveBaseline(), StatisticalBaseline(),
                Twin(population, parameters, seed=args.seed, boundary=boundary, actors=actors),
            ],
            data_versions={"dawum": version, **versions, **(data.versions if data else {})},
            settings={"twin": settings},
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
                   "horizon": args.horizon, "runs": args.runs, "seed": args.seed, "scale": args.scale,
                   "party_actors": args.party_actors,
                   "calibrated": str(args.calibrated) if args.calibrated else None,
                   "first_guess_reactions": args.first_guess_reactions},
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



def _calibrate(args) -> None:
    average, version = _load_poll_average(args.data_dir)
    period = args.calibration_period
    first_election = min(e.day for e in ELECTIONS)
    every_start = calibration_start_weeks(average, period, every=args.every, horizon=args.horizon)
    starts = calibration_start_weeks(average, period, every=args.every, horizon=args.horizon, after=first_election)
    skipped = len(every_start) - len(starts)
    base = TwinParameters()
    populations = _Populations(args.data_dir, args.scale, args.seed)
    data = _BoundaryData.load(args.data_dir)
    # Every Boundary up front, so missing data is reported before the search, not hours into it.
    boundaries = {start: data.boundary(start, args.horizon, base.event_baseline_weeks) for start in starts}
    data_versions = {"dawum": version, **data.versions}
    for start in starts:
        data_versions |= populations(start)[1]

    def make_twin(start, parameters, stand_in, name):
        population = populations(start)[0]
        actors = None if stand_in is None else {p: StandIn(stand_in) for p in TRACKED_PARTIES}
        return Twin(population, parameters, seed=args.seed, boundary=boundaries[start], name=name, actors=actors)

    print(
        f"Calibrating on {len(starts)} Backtests from {starts[0]} to {starts[-1]}, every {args.every} weeks, "
        f"{args.horizon}-week Horizon, inside the Calibration Period ({period.first} to {period.last}); "
        f"{args.runs} Runs each, seed {args.seed}"
    )
    if skipped:
        print(f"  skipped {skipped} start week{'s' if skipped > 1 else ''} before the first registered "
              f"federal election ({first_election}): no Voters can be built for them")

    def progress(stage, done, total, score):
        values = ", ".join(f"{k} {v:g}" for k, v in score.values.items())
        print(f"  {stage} {done}/{total}: mean CRPS {score.crps_mean:.4f} ({values})", flush=True)

    result = calibrate(
        average, starts, make_twin, period=period, grid=calibration.DEFAULT_GRID, horizon=args.horizon,
        runs=args.runs, final_runs=args.final_runs, seed=args.seed, base=base,
        election_results=_election_results(args.data_dir, starts[0], starts[-1] + timedelta(weeks=args.horizon)),
        progress=progress,
    )
    path = save_calibration(result, args.results_dir, data_versions)

    tuned = {k: getattr(result.twin, k) for k in calibration.TWIN_VALUES}
    tuned |= {k: getattr(result.stand_in, k) for k in calibration.STAND_IN_VALUES}
    print("Tuned: " + ", ".join(f"{k} {v:g}" for k, v in tuned.items()))
    print(f"Scored again at {args.final_runs} Runs on the same {len(starts)} Backtests (in-sample, so optimistic):")
    print(f"{'forecast':<28}{'mean CRPS':>11}{'80% coverage':>14}")
    for name, row in result.final.items():
        print(f"{name:<28}{row['crps_mean']:11.3f}{row['coverage']:14.0%}")
    print(f"Saved {path}; use it with --calibrated on Backtests after {period.last}")


if __name__ == "__main__":
    sys.exit(main())
