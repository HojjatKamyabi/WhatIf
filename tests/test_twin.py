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
