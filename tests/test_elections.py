"""Federal election results and Constituency structural data from the Federal Returning Officer.

Expected values are read off the saved sample files by hand, not recomputed by the parser.
"""

from datetime import date
from pathlib import Path

import pytest

from whatif.elections import ELECTIONS, election_before, parse_results, parse_structure
from whatif.sources import ADAPTERS
from whatif.store import MissingDataError

FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"


def structure(year: int):
    return {s.constituency: s for s in parse_structure((FIXTURES / f"strukturdaten_{year}.csv").read_bytes())}


def results(year: int):
    return {r.constituency: r for r in parse_results((FIXTURES / f"kerg2_{year}.csv").read_bytes())}


def test_party_list_votes_per_constituency_on_the_fixed_party_list():
    flensburg = results(2025)[1]
    assert flensburg.name == "Flensburg – Schleswig"
    assert flensburg.election_day == date(2025, 2, 23)
    assert flensburg.eligible == 232131
    assert flensburg.voted == 192882
    assert flensburg.valid == 191894
    assert flensburg.votes == {
        "CDU/CSU": 47686,
        "AfD": 29272,
        "SPD": 30558,
        "Greens": 30595,
        "Left": 15912,
        "BSW": 6050,
        "FDP": 7067,
        # SSW 20575, Die PARTEI 1212, FREIE WÄHLER 1137, Volt 1449, MLPD 53, BÜNDNIS DEUTSCHLAND 328
        "Others": 24754,
    }


def test_csu_counts_as_cdu_csu_and_the_2017_turnout_label_is_read():
    altoetting = results(2017)[212]
    assert altoetting.election_day == date(2017, 9, 24)
    assert altoetting.eligible == 168221
    assert altoetting.voted == 126786  # labelled "Wähler" in 2017, "Wählende" later
    assert altoetting.valid == 125910
    assert altoetting.votes["CDU/CSU"] == 55417
    assert altoetting.votes["Left"] == 6659  # "DIE LINKE" in 2017
    assert altoetting.votes["BSW"] == 0
    assert altoetting.votes["Others"] == 10176


def test_every_sample_year_parses_three_constituencies_whose_votes_add_up():
    for year in (2017, 2021, 2025):
        parsed = results(year)
        assert len(parsed) == 3
        for result in parsed.values():
            assert sum(result.votes.values()) == result.valid
            assert result.valid < result.voted <= result.eligible


def test_structural_data_per_constituency():
    flensburg = structure(2025)[1]
    assert flensburg.name == "Flensburg – Schleswig"
    assert flensburg.land == "Schleswig-Holstein"
    assert flensburg.population_density == 141.5
    assert flensburg.age_shares == {
        "under 18": 16.9,
        "18-24": 7.7,
        "25-34": 12.3,
        "35-59": 32.1,
        "60-74": 19.2,
        "75+": 11.7,
    }
    assert flensburg.disposable_income == 23052
    assert flensburg.unemployment_rate == 6.3


def test_state_and_national_totals_are_not_constituencies():
    assert set(structure(2025)) == {1, 2, 211}


def test_older_structural_files_with_other_encodings_and_column_orders_parse():
    altoetting = structure(2017)[212]  # 2017 is CP 1252 and has income in another column
    assert altoetting.name == "Altötting"
    assert altoetting.population_density == 160.4
    assert altoetting.age_shares["75+"] == 10.5
    assert altoetting.disposable_income == 22185
    assert altoetting.unemployment_rate == 3.9
    assert set(structure(2021)) == {1, 2, 212}  # zero-padded numbers in 2021


def test_the_2017_2021_and_2025_elections_are_registered_with_both_adapters():
    assert [e.day for e in ELECTIONS] == [date(2017, 9, 24), date(2021, 9, 26), date(2025, 2, 23)]
    for election in ELECTIONS:
        assert election.results in ADAPTERS
        assert election.structure in ADAPTERS
        assert {r.election_day for r in election.results.parse(election.results.fixture.read_bytes())} == {election.day}


@pytest.mark.parametrize(
    ("start_week", "election_day"),
    [
        (date(2025, 2, 17), date(2021, 9, 26)),  # the week before the 2025 election
        (date(2025, 2, 24), date(2025, 2, 23)),  # the Monday after it
        (date(2019, 6, 3), date(2017, 9, 24)),
    ],
)
def test_a_backtest_uses_the_last_federal_election_before_its_start_week(start_week, election_day):
    assert election_before(start_week).day == election_day


def test_a_start_week_before_any_registered_election_is_refused():
    with pytest.raises(MissingDataError):
        election_before(date(2017, 9, 18))
