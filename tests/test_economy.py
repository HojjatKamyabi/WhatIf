"""Monthly inflation and unemployment for Germany, and who governs, as Boundary series."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from whatif.boundary import governing_parties
from whatif.economy import INFLATION, UNEMPLOYMENT, Economy, MonthlyValue, parse_eurostat
from whatif.store import MissingDataError

FIXTURES = Path(__file__).parent / "fixtures" / "eurostat"


def economy() -> Economy:
    return Economy(
        inflation=parse_eurostat((FIXTURES / "inflation_de.json").read_bytes()),
        unemployment=parse_eurostat((FIXTURES / "unemployment_de.json").read_bytes()),
    )


def test_inflation_is_read_month_by_month_from_eurostat():
    values = parse_eurostat((FIXTURES / "inflation_de.json").read_bytes())
    assert all(isinstance(v, MonthlyValue) for v in values)
    assert [v.month for v in values] == [date(2024, m, 1) for m in (10, 11, 12)] + [date(2025, m, 1) for m in (1, 2, 3)]
    assert values[0].value == 2.4
    assert values[2].value == 2.8
    assert values[-1].value == 2.3


def test_unemployment_is_read_month_by_month_from_eurostat():
    values = {v.month: v.value for v in parse_eurostat((FIXTURES / "unemployment_de.json").read_bytes())}
    assert values[date(2024, 10, 1)] == 3.4
    assert values[date(2025, 2, 1)] == 3.6


def test_each_week_takes_the_month_its_monday_falls_in_plus_the_week_before_the_start():
    weekly = economy().weekly(date(2025, 1, 6), horizon=4)
    assert weekly.shape == (5, 2)  # the week before the start, then every Horizon week
    assert weekly[0] == pytest.approx([2.8, 3.5])  # Monday 2024-12-30 is in December
    assert weekly[1:] == pytest.approx(np.tile([2.8, 3.5], (4, 1)))


def test_a_month_without_data_is_refused_and_says_what_to_fetch():
    with pytest.raises(MissingDataError, match="fetch-economy"):
        economy().weekly(date(2025, 3, 17), horizon=4)  # runs into April 2025


def test_the_adapters_are_named_after_their_series():
    assert INFLATION.name == "eurostat-inflation"
    assert UNEMPLOYMENT.name == "eurostat-unemployment"
    assert INFLATION.fetch_command == UNEMPLOYMENT.fetch_command == "fetch-economy"


@pytest.mark.parametrize(
    "day, parties",
    [
        (date(2019, 6, 1), {"CDU/CSU", "SPD"}),
        (date(2023, 6, 1), {"SPD", "Greens", "FDP"}),
        (date(2024, 12, 1), {"SPD", "Greens"}),
        (date(2025, 6, 1), {"CDU/CSU", "SPD"}),
    ],
)
def test_governing_parties_follow_the_federal_cabinets(day, parties):
    assert governing_parties(day) == frozenset(parties)
