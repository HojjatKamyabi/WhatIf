"""Scenarios: dated interventions (injected Events, indicator overrides) and running Backtests under them."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from whatif.backtest import BacktestConfig, run_backtest
from whatif.boundary import WeeklyBoundary
from whatif.elections import parse_results, parse_structure
from whatif.events import TOPICS
from whatif.poll_average import weekly_poll_average
from whatif.polls import PARTIES
from whatif.population import build_synthetic_population
from whatif.scenarios import NO_INTERVENTION, ScenarioError, load_scenarios
from whatif.twin import FIRST_GUESS_REACTIONS, Twin, TwinParameters
from tests.conftest import FIRST_WEEK

START = FIRST_WEEK + timedelta(weeks=20)  # 2024-05-20, a Monday
HORIZON = 6
ELECTION_FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"

SCENARIOS = """
[scenario.party-b-scandal]
description = "A scandal hits Party B in week 3"

[[scenario.party-b-scandal.event]]
week = 3
party = "AfD"
topic = "other"
shock = 20.0

[scenario.unemployment-spike]
description = "Unemployment jumps to 6% for three weeks from 2024-06-03"

[[scenario.unemployment-spike.indicator]]
indicator = "unemployment"
from = 2024-06-03
until = 2024-06-23
value = 6.0
"""


def scenario_file(tmp_path: Path, text: str = SCENARIOS) -> Path:
    path = tmp_path / "scenarios.toml"
    path.write_text(text, encoding="utf-8")
    return path


def quiet() -> WeeklyBoundary:
    return WeeklyBoundary.quiet(HORIZON, government=frozenset({"SPD", "Greens", "FDP"}), economy=(2.0, 3.5))


# --- Defining Scenarios ---------------------------------------------------------------------------


def test_no_intervention_exists_without_any_file_and_changes_nothing(tmp_path):
    scenarios = load_scenarios(tmp_path / "missing.toml")
    assert scenarios["no-intervention"] is NO_INTERVENTION
    applied = NO_INTERVENTION.apply(quiet(), START)
    assert applied.event_shock == pytest.approx(quiet().event_shock)
    assert applied.economy == pytest.approx(quiet().economy)


def test_scenarios_are_read_by_name_from_a_config_file(tmp_path):
    scenarios = load_scenarios(scenario_file(tmp_path))
    assert set(scenarios) == {"no-intervention", "party-b-scandal", "unemployment-spike"}
    scandal = scenarios["party-b-scandal"]
    assert scandal.description == "A scandal hits Party B in week 3"
    assert scandal.injects_events and not scandal.overrides_indicators


def test_an_injected_event_adds_its_shock_in_its_week_only(tmp_path):
    applied = load_scenarios(scenario_file(tmp_path))["party-b-scandal"].apply(quiet(), START)
    afd, other = PARTIES.index("AfD"), TOPICS.index("other")
    assert applied.event_shock[2, afd, other] == 20.0  # week 3 is the third Horizon week
    applied.event_shock[2, afd, other] = 0.0
    assert not applied.event_shock.any()


def test_an_indicator_override_replaces_the_recorded_value_only_for_its_dates(tmp_path):
    recorded = quiet()
    applied = load_scenarios(scenario_file(tmp_path))["unemployment-spike"].apply(recorded, START)
    # Row 0 is the week before the start; rows 1.. are Horizon weeks starting 2024-05-20.
    # 2024-06-03, -10 and -17 are Horizon weeks 3, 4 and 5.
    assert applied.economy[:, 1] == pytest.approx([3.5, 3.5, 3.5, 6.0, 6.0, 6.0, 3.5])
    assert applied.economy[:, 0] == pytest.approx(recorded.economy[:, 0])  # inflation untouched


def test_an_event_can_be_placed_by_date(tmp_path):
    text = '[[scenario.dated.event]]\ndate = 2024-06-05\nparty = "SPD"\ntopic = "economy"\nshock = -5.0\n'
    applied = load_scenarios(scenario_file(tmp_path, text))["dated"].apply(quiet(), START)
    assert applied.event_shock[2, PARTIES.index("SPD"), TOPICS.index("economy")] == -5.0  # good news


def test_an_intervention_outside_the_horizon_is_refused(tmp_path):
    scandal = load_scenarios(scenario_file(tmp_path))["party-b-scandal"]
    with pytest.raises(ScenarioError, match="outside the Horizon"):
        scandal.apply(WeeklyBoundary.quiet(2, government=frozenset({"SPD"})), START)


@pytest.mark.parametrize(
    "text, message",
    [
        ('[[scenario.x.event]]\nweek = 1\nparty = "Pirates"\ntopic = "other"\nshock = 1.0\n', "Pirates"),
        ('[[scenario.x.event]]\nweek = 1\nparty = "AfD"\ntopic = "weather"\nshock = 1.0\n', "weather"),
        ('[[scenario.x.event]]\nparty = "AfD"\ntopic = "other"\nshock = 1.0\n', "week or a date"),
        ('[[scenario.x.indicator]]\nindicator = "gdp"\nfrom = 2024-06-03\nuntil = 2024-06-09\nvalue = 1.0\n', "gdp"),
        ('[scenario.no-intervention]\ndescription = "mine"\n', "no-intervention"),
    ],
)
def test_mistakes_in_a_scenario_file_are_reported(tmp_path, text, message):
    with pytest.raises(ScenarioError, match=message):
        load_scenarios(scenario_file(tmp_path, text))


# --- Running under a Scenario, through the Backtest runner --------------------------------------


def twin_under(scenario_name: str, tmp_path: Path, **parameters) -> Twin:
    results = parse_results((ELECTION_FIXTURES / "kerg2_2025.csv").read_bytes())
    structure = parse_structure((ELECTION_FIXTURES / "strukturdaten_2025.csv").read_bytes())
    population = build_synthetic_population(results, structure, scale=1000, seed=0)
    scenario = load_scenarios(scenario_file(tmp_path))[scenario_name]
    boundary = scenario.apply(quiet(), START)
    params = TwinParameters(**({"influence": 0.0, "poll_sample": 100_000} | FIRST_GUESS_REACTIONS | parameters))
    return Twin(population, params, seed=0, boundary=boundary, name=f"twin:{scenario_name}")


def test_an_injected_scandal_for_party_b_lowers_its_share_against_no_intervention(poll_series, tmp_path):
    result = run_backtest(
        weekly_poll_average(poll_series),
        BacktestConfig(start_week=START, horizon=HORIZON, runs=100),
        [twin_under("no-intervention", tmp_path), twin_under("party-b-scandal", tmp_path)],
    )
    afd = PARTIES.index("AfD")
    usual = np.array(result.scores["twin:no-intervention"].quantiles["0.5"])[:, afd]
    scandal = np.array(result.scores["twin:party-b-scandal"].quantiles["0.5"])[:, afd]
    assert scandal[:2] == pytest.approx(usual[:2], abs=0.1)  # nothing before week 3
    assert (scandal[2:] < usual[2:] - 0.2).all()  # lower from the scandal on


def test_the_twin_carries_its_scenario_name_into_the_scores(poll_series, tmp_path):
    result = run_backtest(
        weekly_poll_average(poll_series),
        BacktestConfig(start_week=START, horizon=HORIZON, runs=20),
        [twin_under("party-b-scandal", tmp_path)],
    )
    assert list(result.scores) == ["twin:party-b-scandal"]


def test_scenarios_compared_with_shared_polling_noise_differ_only_by_their_interventions(poll_series, tmp_path):
    same = [twin_under("no-intervention", tmp_path), twin_under("no-intervention", tmp_path)]
    same[1].name = "twin:again"
    result = run_backtest(
        weekly_poll_average(poll_series),
        BacktestConfig(start_week=START, horizon=HORIZON, runs=50),
        same,
        shared_noise=True,
    )
    assert result.scores["twin:no-intervention"].quantiles == result.scores["twin:again"].quantiles
