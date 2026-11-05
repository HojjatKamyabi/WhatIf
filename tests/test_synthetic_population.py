"""The Synthetic Population, seen through its per-Constituency summary and fit report."""

from pathlib import Path

import pytest

from whatif.elections import parse_results, parse_structure
from whatif.population import build_synthetic_population, fit_report
from whatif.store import MissingDataError

FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"


def inputs(year: int = 2025):
    return (
        parse_results((FIXTURES / f"kerg2_{year}.csv").read_bytes()),
        parse_structure((FIXTURES / f"strukturdaten_{year}.csv").read_bytes()),
    )


def test_starting_preferences_reproduce_the_party_list_vote_by_exact_quota():
    results, structure = inputs()
    population = build_synthetic_population(results, structure, scale=1000, seed=0)

    flensburg = population.summary()[1]
    assert flensburg.voters == 232  # 232,131 eligible voters at one Voter per 1,000
    # Largest remainder over 232 Voters, worked by hand from the valid party-list votes (191,894):
    # quotas CDU/CSU 57.65, AfD 35.39, SPD 36.94, Greens 36.99, Left 19.24, BSW 7.32, FDP 8.54,
    # Others 29.93; floors sum to 227, and the 5 left go to Greens, SPD, Others, CDU/CSU, FDP.
    assert flensburg.preferences == {
        "CDU/CSU": 58,
        "AfD": 35,
        "SPD": 37,
        "Greens": 37,
        "Left": 19,
        "BSW": 7,
        "FDP": 9,
        "Others": 30,
    }
    assert flensburg.turnout == pytest.approx(192882 / 232131)


def test_voters_are_adults_allocated_to_the_constituency_age_bands_by_exact_quota():
    results, structure = inputs()
    flensburg = build_synthetic_population(results, structure, scale=1000, seed=0).summary()[1]

    # Adult bands 7.7 / 12.3 / 32.1 / 19.2 / 11.7 (sum 83.0) over 232 Voters, worked by hand:
    # quotas 21.52, 34.38, 89.73, 53.67, 32.70; floors sum to 229, and the 3 left go to the
    # largest remainders: 35-59 (.73), 75+ (.70), 60-74 (.67).
    assert flensburg.ages == {"18-24": 21, "25-34": 34, "35-59": 90, "60-74": 54, "75+": 33}


def test_voters_carry_their_constituency_circumstances():
    results, structure = inputs()
    flensburg = build_synthetic_population(results, structure, scale=1000, seed=0).summary()[1]
    assert flensburg.land == "Schleswig-Holstein"
    assert flensburg.population_density == 141.5
    assert flensburg.disposable_income == 23052
    assert flensburg.unemployment_rate == 6.3


def test_building_is_reproducible_from_a_seed():
    results, structure = inputs()
    first = build_synthetic_population(results, structure, scale=100, seed=7).summary()
    again = build_synthetic_population(results, structure, scale=100, seed=7).summary()
    assert first == again
    # The seed only decides which Voters share which party and age band; the per-Constituency
    # counts are fixed by quota, so a different seed gives the same summary.


def test_a_constituency_without_structural_data_is_refused():
    results, structure = inputs()
    with pytest.raises(MissingDataError, match="211"):
        build_synthetic_population(results, [s for s in structure if s.constituency != 211], scale=1000, seed=0)


def test_fit_report_shows_how_far_each_constituency_is_from_its_sources():
    results, structure = inputs()
    report = fit_report(build_synthetic_population(results, structure, scale=1000, seed=0))

    flensburg = next(row for row in report.rows if row.constituency == 1)
    assert flensburg.name == "Flensburg – Schleswig"
    assert flensburg.voters == 232
    # Worst party: FDP, 9 of 232 Voters (3.879%) against 7,067 of 191,894 valid votes (3.683%).
    assert flensburg.party_gap == pytest.approx(0.197, abs=0.001)
    assert {row.constituency for row in report.rows} == {1, 2, 211}


def test_every_constituency_matches_its_result_within_one_voter():
    results, structure = inputs()
    report = fit_report(build_synthetic_population(results, structure, scale=1000, seed=0))
    for row in report.rows:
        assert row.party_gap <= 100 / row.voters
        assert row.party_within_tolerance
    assert report.all_parties_within_tolerance


def test_every_constituency_matches_its_age_bands_within_one_voter():
    results, structure = inputs()
    report = fit_report(build_synthetic_population(results, structure, scale=1000, seed=0))
    for row in report.rows:
        assert row.age_gap <= 100 / row.voters
        assert row.age_within_tolerance
    assert report.all_ages_within_tolerance


def test_the_fit_report_reads_as_a_table():
    results, structure = inputs()
    text = fit_report(build_synthetic_population(results, structure, scale=1000, seed=0)).format()
    assert "Flensburg – Schleswig" in text
    assert "party gap" in text
    assert "age gap" in text
