from datetime import date

import pytest

from whatif.poll_average import weekly_poll_average
from whatif.polls import PARTIES
from tests.conftest import FIRST_WEEK, make_poll


def share(average, week_index, party):
    return average.shares[week_index, PARTIES.index(party)]


def test_polls_in_a_week_are_weighted_by_sample_size():
    average = weekly_poll_average([
        make_poll(0, {"SPD": 30, "Others": 70}, sample_size=3000),
        make_poll(0, {"SPD": 10, "Others": 90}, sample_size=1000),
    ])
    assert share(average, 0, "SPD") == pytest.approx(25)


def test_missing_sample_size_uses_the_default_weight():
    average = weekly_poll_average([
        make_poll(0, {"SPD": 30, "Others": 70}, sample_size=None),
        make_poll(0, {"SPD": 10, "Others": 90}, sample_size=1000),
    ], default_sample_size=1000)
    assert share(average, 0, "SPD") == pytest.approx(20)


def test_weeks_without_polls_are_carried_forward_and_flagged():
    average = weekly_poll_average([
        make_poll(0, {"SPD": 30, "Others": 70}),
        make_poll(3, {"SPD": 20, "Others": 80}),
    ])
    assert average.weeks[0] == FIRST_WEEK
    assert len(average.weeks) == 4
    assert average.carried.tolist() == [False, True, True, False]
    assert share(average, 2, "SPD") == pytest.approx(30)
    assert share(average, 3, "SPD") == pytest.approx(20)


def test_a_party_is_averaged_only_over_polls_that_reported_it():
    # Poll B folds BSW into Others; BSW should come from poll A alone, not be halved.
    average = weekly_poll_average([
        make_poll(0, {"BSW": 4, "SPD": 16, "Others": 80}, institute="A"),
        make_poll(0, {"SPD": 16, "Others": 84}, institute="B", unreported=("BSW",)),
    ])
    assert share(average, 0, "BSW") == pytest.approx(4)
    assert share(average, 0, "SPD") == pytest.approx(16)
    assert share(average, 0, "Others") == pytest.approx(80)


def test_a_party_no_poll_reported_keeps_last_week_value():
    average = weekly_poll_average([
        make_poll(0, {"BSW": 4, "SPD": 16, "Others": 80}),
        make_poll(1, {"SPD": 20, "Others": 80}, unreported=("BSW",)),
    ])
    assert share(average, 1, "BSW") == pytest.approx(4)
    assert share(average, 1, "SPD") == pytest.approx(20)
    assert average.carried.tolist() == [False, False]


def test_every_week_sums_to_100(poll_series):
    average = weekly_poll_average(poll_series)
    assert average.shares.sum(axis=1) == pytest.approx([100] * len(average.weeks))


def test_weeks_start_on_monday(poll_series):
    average = weekly_poll_average(poll_series)
    assert all(week.weekday() == 0 for week in average.weeks)
    assert average.index_of(date(2024, 1, 8)) == 1
