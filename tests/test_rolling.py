"""Rolling-window Backtests: many start weeks, one summary, the Calibration Period kept out, plots."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from whatif.backtest import DEFAULT_CALIBRATION_PERIOD, BacktestConfig, CalibrationPeriod, run_backtest
from whatif.baselines import NaiveBaseline, StatisticalBaseline
from whatif.elections import national_shares, parse_results, parse_structure
from whatif.plots import plot_backtest, plot_rolling
from whatif.poll_average import weekly_poll_average
from whatif.polls import PARTIES
from whatif.population import build_synthetic_population
from whatif.rolling import rolling_start_weeks, run_rolling
from whatif.store import MissingDataError
from whatif.twin import Twin
from tests.conftest import FIRST_WEEK

ELECTION_FIXTURES = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"
HORIZON = 6
NO_CALIBRATION = CalibrationPeriod(date(2000, 1, 3), date(2000, 1, 3))  # a period no fixture week is in


def week(n: int) -> date:
    return FIRST_WEEK + timedelta(weeks=n)


@pytest.fixture(scope="module")
def twin() -> Twin:
    results = parse_results((ELECTION_FIXTURES / "kerg2_2025.csv").read_bytes())
    structure = parse_structure((ELECTION_FIXTURES / "strukturdaten_2025.csv").read_bytes())
    return Twin(build_synthetic_population(results, structure, scale=1000, seed=0), seed=0)


def forecasters(twin):
    return lambda start: [NaiveBaseline(), StatisticalBaseline(), twin]


def rolling(poll_series, twin, calibration_period=NO_CALIBRATION, **window):
    average = weekly_poll_average(poll_series)
    starts = rolling_start_weeks(average, first=window.get("first", week(8)), every=4, horizon=HORIZON)
    return run_rolling(
        average, starts, forecasters(twin), horizon=HORIZON, runs=50, seed=3, calibration_period=calibration_period
    )


# --- The window ---------------------------------------------------------------------------------


def test_start_weeks_run_every_few_weeks_up_to_the_last_full_horizon(poll_series):
    average = weekly_poll_average(poll_series)  # 40 weeks
    starts = rolling_start_weeks(average, first=week(8), every=4, horizon=HORIZON)
    assert starts == [week(n) for n in (8, 12, 16, 20, 24, 28, 32)]  # week 36 + 6 would pass the data


def test_a_window_starting_without_enough_history_is_refused(poll_series):
    with pytest.raises(MissingDataError, match="history"):
        rolling_start_weeks(weekly_poll_average(poll_series), first=week(2), every=4, horizon=HORIZON)


def test_a_window_with_no_full_horizon_is_refused(poll_series):
    with pytest.raises(MissingDataError, match="Horizon"):
        rolling_start_weeks(weekly_poll_average(poll_series), first=week(36), every=4, horizon=HORIZON)


# --- The summary ----------------------------------------------------------------------------------


def test_the_summary_scores_the_twin_and_both_baselines_over_the_window(poll_series, twin):
    result = rolling(poll_series, twin)
    assert len(result.backtests) == 7
    assert set(result.summary) == {"naive", "statistical", "twin"}
    for name, row in result.summary.items():
        assert row.backtests == 7
        assert row.crps_mean == pytest.approx(np.mean([b.scores[name].crps_mean for b in result.backtests]))
        assert row.coverage == pytest.approx(np.mean([b.scores[name].coverage for b in result.backtests]))
    assert sum(row.lowest_crps for row in result.summary.values()) == 7  # one forecaster is best per start week


def test_each_backtest_in_the_window_is_reproducible_on_its_own(poll_series, twin):
    result = rolling(poll_series, twin)
    alone = run_backtest(
        weekly_poll_average(poll_series),
        BacktestConfig(start_week=week(16), horizon=HORIZON, runs=50, seed=3),
        forecasters(twin)(week(16)),
        calibration_period=NO_CALIBRATION,
    )
    assert result.backtests[2].config.start_week == week(16)
    assert result.backtests[2].scores == alone.scores


# --- The Calibration Period --------------------------------------------------------------------


def test_twin_scores_from_the_calibration_period_never_reach_the_summary(poll_series, twin):
    period = CalibrationPeriod(week(8), week(12))  # covers the first two start weeks
    result = rolling(poll_series, twin, calibration_period=period)
    assert [b.calibration for b in result.backtests] == [True, True] + [False] * 5
    assert result.summary["twin"].backtests == 5
    scored = [b.scores["twin"].crps_mean for b in result.backtests[2:]]
    assert result.summary["twin"].crps_mean == pytest.approx(np.mean(scored))
    assert result.summary["naive"].backtests == result.summary["statistical"].backtests == 7  # Baselines still reported
    assert sum(row.lowest_crps for row in result.summary.values()) == 5  # head-to-head only where all are scored


def test_a_single_backtest_starting_in_the_calibration_period_is_marked(poll_series):
    average = weekly_poll_average(poll_series)
    config = BacktestConfig(start_week=week(20), horizon=HORIZON, runs=20)
    inside = run_backtest(average, config, [NaiveBaseline()], calibration_period=CalibrationPeriod(week(20), week(30)))
    outside = run_backtest(average, config, [NaiveBaseline()], calibration_period=NO_CALIBRATION)
    assert inside.calibration and not outside.calibration


def test_the_default_calibration_period_is_2017_to_2019():
    assert DEFAULT_CALIBRATION_PERIOD.contains(date(2017, 1, 2))
    assert DEFAULT_CALIBRATION_PERIOD.contains(date(2019, 12, 23))
    assert not DEFAULT_CALIBRATION_PERIOD.contains(date(2020, 1, 6))


def test_a_calibration_period_is_read_from_iso_weeks():
    assert CalibrationPeriod.parse("2017-W01:2019-W52") == DEFAULT_CALIBRATION_PERIOD


# --- Elections inside a Horizon ----------------------------------------------------------------


def test_national_shares_add_up_the_party_list_vote_over_constituencies():
    shares = national_shares(parse_results((ELECTION_FIXTURES / "kerg2_2025.csv").read_bytes()))
    # Constituencies 1, 2 and 211 of the fixture: 147,380 CDU/CSU votes of 482,059 valid.
    assert shares["CDU/CSU"] == pytest.approx(30.573, abs=1e-3)
    assert shares["BSW"] == pytest.approx(3.219, abs=1e-3)
    assert sum(shares.values()) == pytest.approx(100.0)


def test_an_election_inside_the_horizon_is_reported_next_to_the_poll_average(poll_series):
    average = weekly_poll_average(poll_series)
    config = BacktestConfig(start_week=week(20), horizon=HORIZON, runs=20)
    shares = dict(zip(PARTIES, [30.0, 20.0, 15.0, 12.0, 8.0, 5.0, 4.0, 6.0]))
    elections = {week(22) + timedelta(days=6): shares, week(40): shares}  # a Sunday inside; one after the Horizon
    result = run_backtest(average, config, [NaiveBaseline()], election_results=elections)
    assert result.elections == [{"day": week(22) + timedelta(days=6), "shares": shares}]


# --- Plots ------------------------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n"


def test_a_plot_is_saved_per_backtest(poll_series, twin, tmp_path):
    result = rolling(poll_series, twin).backtests[0]
    path = plot_backtest(result, tmp_path / "backtest.png")
    assert path.read_bytes().startswith(PNG)


def test_the_summary_plot_shows_crps_per_start_week(poll_series, twin, tmp_path):
    result = rolling(poll_series, twin, calibration_period=CalibrationPeriod(week(8), week(12)))
    path = plot_rolling(result, tmp_path / "summary.png", elections=[week(18)])
    assert path.read_bytes().startswith(PNG)
