"""The Twin run directly: its weekly national shares, starting map and Neighbour Disagreement.

Most Twin behaviour is tested through the Backtest runner (tests/test_backtest.py); this seam
exists for what national shares can't show.
"""

from pathlib import Path

import numpy as np
import pytest

from whatif.elections import parse_results, parse_structure
from whatif.polls import PARTIES
from whatif.population import build_synthetic_population
from whatif.twin import Twin, TwinParameters

FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"

# A latest Poll Average to start from: CDU/CSU, AfD, SPD, Greens, Left, BSW, FDP, Others.
LATEST = np.array([30.0, 20.0, 15.0, 12.0, 8.0, 5.0, 4.0, 6.0])


def population(year: int = 2025, scale: int = 100):
    results = parse_results((FIXTURES / f"kerg2_{year}.csv").read_bytes())
    structure = parse_structure((FIXTURES / f"strukturdaten_{year}.csv").read_bytes())
    return build_synthetic_population(results, structure, scale=scale, seed=0)


def election_shares(year: int, constituency: int) -> dict[str, float]:
    result = {r.constituency: r for r in parse_results((FIXTURES / f"kerg2_{year}.csv").read_bytes())}[constituency]
    return {p: 100 * result.votes[p] / result.valid for p in PARTIES}


def test_a_run_starts_at_the_latest_poll_average_nationally():
    run = Twin(population(), seed=0).run(LATEST, horizon=0)
    assert run.start == pytest.approx(LATEST, abs=0.01)


def test_the_start_keeps_the_election_map_by_a_uniform_swing():
    run = Twin(population(), seed=0).run(LATEST, horizon=0)
    flensburg, altoetting = run.start_by_constituency[1], run.start_by_constituency[211]
    for party in ("CDU/CSU", "SPD", "Greens"):
        # The gap between two Constituencies is the gap in the election, whatever the national swing.
        election_gap = election_shares(2025, 1)[party] - election_shares(2025, 211)[party]
        assert flensburg[party] - altoetting[party] == pytest.approx(election_gap, abs=0.1)


def test_a_party_that_did_not_stand_starts_at_the_same_share_everywhere():
    run = Twin(population(year=2021), seed=0).run(LATEST, horizon=0)  # no BSW in 2021
    for shares in run.start_by_constituency.values():
        assert shares["BSW"] == pytest.approx(5.0, abs=0.05)


def test_with_influence_switched_off_connected_voters_do_not_converge():
    run = Twin(population(), TwinParameters(influence=0.0), seed=0).run(LATEST, horizon=6)
    assert len(run.disagreement) == 7  # the start plus every week
    assert run.disagreement == pytest.approx([run.disagreement[0]] * 7)
    assert run.disagreement[0] > 0
    assert run.shares == pytest.approx(np.tile(LATEST, (6, 1)), abs=1e-9)  # nothing else moves Voters yet


def test_with_influence_connected_voters_drift_towards_each_other_every_week():
    run = Twin(population(), TwinParameters(influence=0.1), seed=0).run(LATEST, horizon=6)
    assert all(later < earlier for earlier, later in zip(run.disagreement, run.disagreement[1:]))


def test_weekly_shares_cover_every_party_and_sum_to_100():
    run = Twin(population(), seed=0).run(LATEST, horizon=4)
    assert run.shares.shape == (4, len(PARTIES))
    assert run.shares.sum(axis=1) == pytest.approx([100.0] * 4)


# --- The Boundary reaching Voters: the economy and recorded Events ---------------------------

from dataclasses import replace  # noqa: E402

from whatif.boundary import WeeklyBoundary  # noqa: E402
from whatif.events import TOPICS  # noqa: E402

HORIZON = 6
GOVERNMENT = frozenset({"SPD", "Greens"})


def quiet(**changes) -> WeeklyBoundary:
    """Nothing happens: flat economy, no Event above the usual level, usual attention."""
    return replace(WeeklyBoundary.quiet(HORIZON, government=GOVERNMENT), **changes)


def twin_run(boundary: WeeklyBoundary, **parameters):
    parameters = {"influence": 0.0} | parameters
    return Twin(population(), TwinParameters(**parameters), seed=0, boundary=boundary).run(LATEST, HORIZON)


def government_share(shares) -> float:
    return sum(shares[p] for p in GOVERNMENT)


def as_dict(row) -> dict[str, float]:
    return dict(zip(PARTIES, row))


def test_a_quiet_boundary_changes_nothing():
    run = twin_run(quiet())
    assert run.shares == pytest.approx(np.tile(LATEST, (HORIZON, 1)), abs=1e-9)


def test_an_unemployment_rise_costs_the_governing_parties_votes():
    economy = quiet().economy.copy()
    economy[3:, 1] += 1.0  # unemployment one point higher from week 3 on
    flat, risen = twin_run(quiet()), twin_run(quiet(economy=economy))
    assert government_share(as_dict(risen.shares[-1])) < government_share(as_dict(flat.shares[-1])) - 0.5
    assert risen.shares[:2] == pytest.approx(flat.shares[:2])  # nothing moves before the rise
    assert risen.shares[-1].sum() == pytest.approx(100.0)


def test_an_unemployment_rise_hits_voters_where_unemployment_is_already_high_hardest():
    economy = quiet().economy.copy()
    economy[1:, 1] += 1.0
    flat, risen = twin_run(quiet()), twin_run(quiet(economy=economy))

    def relative_loss(constituency: int) -> float:
        before = government_share(flat.end_by_constituency[constituency])
        return (before - government_share(risen.end_by_constituency[constituency])) / before

    # Flensburg (6.3% unemployment in the structural data) against Altötting (about 3%).
    assert relative_loss(1) > relative_loss(211)


def test_falling_inflation_rewards_the_governing_parties():
    economy = quiet().economy.copy()
    economy[2:, 0] -= 2.0
    assert government_share(as_dict(twin_run(quiet(economy=economy)).shares[-1])) > government_share(
        as_dict(twin_run(quiet()).shares[-1])
    )


def afd_scandal(size: float = 5.0, week: int = 2) -> np.ndarray:
    shock = np.zeros((HORIZON, len(PARTIES), len(TOPICS)))
    shock[week, PARTIES.index("AfD"), TOPICS.index("migration")] = size
    return shock


def test_a_negative_event_involving_a_party_lowers_its_share():
    without, scandal = twin_run(quiet()), twin_run(quiet(event_shock=afd_scandal()))
    afd = PARTIES.index("AfD")
    assert scandal.shares[-1, afd] < without.shares[-1, afd] - 0.1
    assert scandal.shares[:2, afd] == pytest.approx(without.shares[:2, afd])  # before the Event
    others = [j for j in range(len(PARTIES)) if j != afd]
    assert scandal.shares[-1, others].sum() > without.shares[-1, others].sum()


def test_more_attention_makes_the_same_event_move_more_voters():
    afd = PARTIES.index("AfD")
    attention = quiet().party_attention.copy()
    attention[:, afd] = 3.0
    usual = twin_run(quiet(event_shock=afd_scandal())).shares[-1, afd]
    watched = twin_run(quiet(event_shock=afd_scandal(), party_attention=attention)).shares[-1, afd]
    assert watched < usual


def test_an_event_on_a_topic_no_voter_cares_about_moves_nobody():
    priorities = {band: tuple(0.0 if t == "migration" else 1.0 for t in TOPICS) for band in TwinParameters().topic_priorities}
    run = twin_run(quiet(event_shock=afd_scandal()), topic_priorities=priorities)
    assert run.shares == pytest.approx(np.tile(LATEST, (HORIZON, 1)), abs=1e-9)


def test_reaction_strengths_are_named_parameters():
    economy = quiet().economy.copy()
    economy[1:, 1] += 1.0
    off = twin_run(quiet(economy=economy, event_shock=afd_scandal()), unemployment_reaction=0.0, event_reaction=0.0)
    assert off.shares == pytest.approx(np.tile(LATEST, (HORIZON, 1)), abs=1e-9)


def test_the_boundary_must_cover_the_horizon():
    with pytest.raises(ValueError, match="Boundary"):
        Twin(population(), seed=0, boundary=WeeklyBoundary.quiet(2, government=GOVERNMENT)).run(LATEST, 3)
