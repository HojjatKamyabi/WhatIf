"""Party Actors: one move per party per week from a fixed menu, rule-based Stand-ins, forced moves."""

from dataclasses import fields, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FIRST_WEEK
from tests.test_events import collected_events
from whatif.actors import (
    ISSUE_OWNERSHIP,
    MENU,
    Decision,
    EventSummary,
    Move,
    StandIn,
    StandInParameters,
    StateSummary,
)
from whatif.backtest import BacktestConfig, CalibrationPeriod, require_tuned_outside, run_backtest
from whatif.boundary import WeeklyBoundary, event_summaries
from whatif.data.elections import parse_results, parse_structure
from whatif.data.gdelt import EventLog
from whatif.data.polls import PARTIES, TRACKED_PARTIES
from whatif.events import TOPICS
from whatif.poll_average import weekly_poll_average
from whatif.population import build_synthetic_population
from whatif.scenarios import ScenarioError, load_scenarios, require_reactions
from whatif.twin import FIRST_GUESS_REACTIONS, Twin, TwinParameters

ELECTION_FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"
HORIZON = 6
START = FIRST_WEEK + timedelta(weeks=20)
LATEST = np.array([30.0, 20.0, 15.0, 12.0, 8.0, 5.0, 4.0, 6.0])
AFD = PARTIES.index("AfD")


# --- The move menu and the Stand-in (the decision-maker seam) ------------------------------------


def test_the_menu_has_five_emphases_a_rebuttal_and_holding_still():
    assert [str(m) for m in MENU] == [
        "emphasise economy", "emphasise migration", "emphasise climate", "emphasise security",
        "emphasise social", "rebut", "hold",
    ]
    assert all(Move.parse(str(m)) == m for m in MENU)
    with pytest.raises(ValueError):
        Move.parse("emphasise weather")


def state(party="AfD", events=(), attention=None, support=None) -> StateSummary:
    return StateSummary(
        week=1,
        party=party,
        support=support or dict(zip(PARTIES, LATEST.tolist())),
        events=tuple(events),
        topic_attention=attention or {t: 1.0 for t in TOPICS if t != "other"},
    )


def test_the_stand_in_rebuts_bad_news_above_its_threshold():
    bad = EventSummary(party="AfD", topic="migration", tone=-5.0, size=3.0)  # 15 > the threshold
    decision = StandIn().decide(state(events=[bad]), MENU)
    assert decision.move == Move("rebut") and decision.reason


def test_otherwise_the_stand_in_emphasises_its_own_topic_with_the_most_attention():
    attention = {"economy": 1.1, "migration": 2.0, "climate": 3.0, "security": 1.5, "social": 1.0}
    decision = StandIn().decide(state(attention=attention), MENU)
    owned = ISSUE_OWNERSHIP["AfD"]
    assert decision.move == Move("emphasise", max(owned, key=attention.get))  # climate isn't AfD's
    assert decision.move.topic in owned


def test_the_stand_in_holds_when_none_of_its_topics_draws_attention():
    attention = {t: 0.5 for t in TOPICS if t != "other"}
    assert StandIn().decide(state(attention=attention), MENU).move == Move("hold")


def test_the_stand_in_returns_a_valid_move_for_every_state():
    rng = np.random.default_rng(0)
    for _ in range(300):
        party = rng.choice(TRACKED_PARTIES)
        support = dict(zip(PARTIES, rng.dirichlet(np.ones(len(PARTIES))) * 100))
        events = [
            EventSummary(rng.choice(TRACKED_PARTIES), rng.choice(TOPICS), float(rng.normal(-2, 4)), float(rng.exponential(2)))
            for _ in range(rng.integers(0, 12))
        ]
        attention = {t: float(rng.exponential(1.5)) for t in TOPICS if t != "other"}
        decision = StandIn().decide(state(party, events, attention, support), MENU)
        assert isinstance(decision, Decision) and decision.move in MENU and decision.reason


def test_a_state_summary_holds_only_structured_fields_and_no_names_from_the_source_records():
    # The fixture's records come from articles naming Merz, Scholz and others.
    records = collected_events()
    day = date(2025, 2, 10)  # a Monday
    summaries = event_summaries(EventLog(records, ((day - timedelta(weeks=8), day + timedelta(days=6)),)), day, 1, 8)
    summary = state(events=summaries[0])
    assert {f.name for f in fields(StateSummary)} == {"week", "party", "support", "events", "topic_attention"}
    assert {f.name for f in fields(EventSummary)} == {"party", "topic", "tone", "size"}
    assert {e.party for e in summary.events} == {"CDU/CSU", "Greens", "AfD", "SPD"}
    text = repr(summary).lower()
    for name in ("merz", "scholz", "leyen", "habeck", "euronews", "http"):
        assert name not in text


# --- Party Actors in the Twin (the Backtest seam, with a scripted fake) ---------------------------


class Scripted:
    """A fake decision-maker: always the same move, and it records every state it was shown."""

    name = "scripted"

    def __init__(self, move: str):
        self.move, self.seen = Move.parse(move), []

    def decide(self, summary: StateSummary, menu) -> Decision:
        self.seen.append(summary)
        return Decision(self.move, f"scripted: {self.move}")


@pytest.fixture(scope="module")
def population():
    results = parse_results((ELECTION_FIXTURES / "kerg2_2025.csv").read_bytes())
    structure = parse_structure((ELECTION_FIXTURES / "strukturdaten_2025.csv").read_bytes())
    return build_synthetic_population(results, structure, scale=100, seed=0)


def quiet() -> WeeklyBoundary:
    return WeeklyBoundary.quiet(HORIZON, government=frozenset({"SPD", "Greens"}))


def twin(population, actors=None, forced=None, boundary=None, **parameters) -> Twin:
    parameters = {"influence": 0.0, **FIRST_GUESS_REACTIONS} | parameters
    return Twin(
        population, TwinParameters(**parameters), seed=0, boundary=boundary or quiet(),
        actors=actors, forced_moves=forced,
    )


def everyone(move: str) -> dict:
    return {party: Scripted(move) for party in TRACKED_PARTIES}


def test_voters_react_to_what_a_party_emphasises(population):
    actors = everyone("hold") | {"AfD": Scripted("emphasise migration")}
    holding = twin(population, everyone("hold")).run(LATEST, HORIZON)
    emphasising = twin(population, actors).run(LATEST, HORIZON)
    assert emphasising.shares[-1, AFD] > holding.shares[-1, AFD] + 0.05
    assert holding.shares == pytest.approx(np.tile(LATEST, (HORIZON, 1)), abs=1e-9)  # holding changes nothing


def test_with_move_reactions_off_moves_change_nothing(population):
    actors = everyone("emphasise migration")
    run = twin(population, actors, move_reaction=0.0).run(LATEST, HORIZON)
    assert run.shares == pytest.approx(np.tile(LATEST, (HORIZON, 1)), abs=1e-9)


def test_a_rebuttal_softens_the_week_s_bad_news(population):
    shock = np.zeros((HORIZON, len(PARTIES), len(TOPICS)))
    shock[2, AFD, TOPICS.index("migration")] = 10.0
    boundary = replace(quiet(), event_shock=shock)
    silent = twin(population, everyone("hold"), boundary=boundary).run(LATEST, HORIZON).shares[-1, AFD]
    rebutted = twin(population, everyone("hold") | {"AfD": Scripted("rebut")}, boundary=boundary).run(LATEST, HORIZON)
    assert silent < rebutted.shares[-1, AFD] < LATEST[AFD]


def test_party_actors_see_support_and_events_but_never_raw_voter_data(population):
    actors = everyone("hold")
    twin(population, actors).run(LATEST, HORIZON)
    seen = actors["AfD"].seen
    assert len(seen) == HORIZON and [s.week for s in seen] == list(range(1, HORIZON + 1))
    for summary in seen:
        assert isinstance(summary, StateSummary) and summary.party == "AfD"
        assert set(summary.support) == set(PARTIES)  # national shares only, one per party
        assert sum(summary.support.values()) == pytest.approx(100.0)
        assert all(isinstance(v, float) for v in summary.support.values())


def test_a_forced_move_overrides_the_actor_s_own_decision(population):
    actors = everyone("hold") | {"AfD": Scripted("emphasise migration")}
    run = twin(population, actors, forced={(2, "AfD"): Move("hold")}).run(LATEST, HORIZON)
    week3 = [d for d in run.decisions if d["week"] == 3 and d["party"] == "AfD"][0]
    assert week3["move"] == "hold" and week3["forced"] is True
    assert week3["proposed"] == "emphasise migration"  # what the actor wanted is kept for the record
    week2 = [d for d in run.decisions if d["week"] == 2 and d["party"] == "AfD"][0]
    assert week2["move"] == "emphasise migration" and week2["forced"] is False


def test_every_decision_is_logged_in_the_saved_result_with_its_inputs_and_reason(poll_series, population):
    stand_ins = {party: StandIn() for party in TRACKED_PARTIES}
    result = run_backtest(
        weekly_poll_average(poll_series),
        BacktestConfig(start_week=START, horizon=HORIZON, runs=20),
        [twin(population, stand_ins)],
    )
    log = result.logs["twin"]
    assert len(log) == HORIZON * len(TRACKED_PARTIES)
    first = log[0]
    assert set(first) == {"week", "party", "move", "reason", "forced", "proposed", "state"}
    assert set(first["state"]) == {"support", "events", "topic_attention"}
    assert first["move"] in {str(m) for m in MENU} and first["reason"]


def test_a_twin_without_party_actors_logs_nothing(poll_series, population):
    result = run_backtest(
        weekly_poll_average(poll_series), BacktestConfig(start_week=START, horizon=HORIZON, runs=20), [twin(population)]
    )
    assert result.logs == {}


# --- Forced moves in Scenarios ----------------------------------------------------------------------


def scenario(tmp_path, text):
    path = tmp_path / "scenarios.toml"
    path.write_text(text, encoding="utf-8")
    return load_scenarios(path)


def test_a_scenario_forces_a_move_for_a_party_and_week(tmp_path):
    loaded = scenario(tmp_path, '[[scenario.quiet-afd.move]]\nweek = 3\nparty = "AfD"\nmove = "hold"\n')["quiet-afd"]
    assert loaded.forced_moves(START, HORIZON) == {(2, "AfD"): Move("hold")}
    assert loaded.describe()["moves"] == [{"week": 3, "party": "AfD", "move": "hold"}]


@pytest.mark.parametrize(
    "text, message",
    [
        ('[[scenario.x.move]]\nweek = 3\nparty = "AfD"\nmove = "dance"\n', "dance"),
        ('[[scenario.x.move]]\nweek = 3\nparty = "Others"\nmove = "hold"\n', "Others"),
    ],
)
def test_mistakes_in_forced_moves_are_reported(tmp_path, text, message):
    with pytest.raises(ScenarioError, match=message):
        scenario(tmp_path, text)


def test_a_forced_move_outside_the_horizon_is_refused(tmp_path):
    loaded = scenario(tmp_path, '[[scenario.x.move]]\nweek = 9\nparty = "AfD"\nmove = "hold"\n')["x"]
    with pytest.raises(ScenarioError, match="outside the Horizon"):
        loaded.forced_moves(START, HORIZON)


def test_a_forced_move_voters_cannot_react_to_is_refused(tmp_path):
    loaded = scenario(tmp_path, '[[scenario.x.move]]\nweek = 1\nparty = "AfD"\nmove = "emphasise migration"\n')["x"]
    with pytest.raises(ScenarioError, match="move_reaction"):
        require_reactions(loaded, TwinParameters())


# --- Tuning stays inside the Calibration Period --------------------------------------------------


def test_stand_in_parameters_record_that_they_are_untuned():
    assert StandInParameters().tuned_on is None
    assert TwinParameters().tuned_on is None


def test_parameters_tuned_on_weeks_a_backtest_scores_are_refused():
    config = BacktestConfig(start_week=date(2021, 5, 17), horizon=12)
    with pytest.raises(ValueError, match="tuned on"):
        require_tuned_outside(CalibrationPeriod(date(2021, 1, 4), date(2021, 12, 27)), config)
    require_tuned_outside(CalibrationPeriod(date(2017, 1, 2), date(2019, 12, 23)), config)  # fine
    require_tuned_outside(None, config)  # untuned: fine
