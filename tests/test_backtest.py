import json
from datetime import date, timedelta

import numpy as np
import pytest

from whatif.backtest import BacktestConfig, run_backtest
from whatif.baselines import NaiveBaseline, StatisticalBaseline
from whatif.poll_average import weekly_poll_average
from whatif.polls import PARTIES
from whatif.results import save_result
from whatif.store import MissingDataError
from tests.conftest import FIRST_WEEK, make_poll

START = FIRST_WEEK + timedelta(weeks=20)


def backtest(polls, **config):
    return run_backtest(
        weekly_poll_average(polls),
        BacktestConfig(start_week=config.pop("start_week", START), **config),
        [NaiveBaseline(), StatisticalBaseline()],
        data_versions={"dawum": "fixture"},
    )


def test_same_seed_gives_identical_result(poll_series):
    first = backtest(poll_series, seed=7)
    second = backtest(poll_series, seed=7)
    assert first.scores == second.scores


def test_different_seed_gives_different_runs(poll_series):
    assert backtest(poll_series, seed=1).scores != backtest(poll_series, seed=2).scores


def test_scores_cover_every_party_and_week(poll_series):
    result = backtest(poll_series, horizon=6)
    naive = result.scores["naive"]
    assert len(result.weeks) == len(result.actual) == 6
    assert set(naive.crps_by_party) == set(PARTIES)
    assert 0 <= naive.coverage <= 1
    assert len(naive.quantiles["0.5"]) == 6


def test_actual_is_the_real_poll_average_over_the_horizon(poll_series):
    average = weekly_poll_average(poll_series)
    result = backtest(poll_series, horizon=3)
    start = average.index_of(START)
    assert np.allclose(result.actual, average.shares[start : start + 3])


def test_refuses_when_horizon_runs_past_the_data(poll_series):
    with pytest.raises(MissingDataError, match="Horizon"):
        backtest(poll_series, start_week=FIRST_WEEK + timedelta(weeks=35), horizon=12)


def test_refuses_without_enough_history(poll_series):
    with pytest.raises(MissingDataError, match="weeks of Poll Average"):
        backtest(poll_series, start_week=FIRST_WEEK + timedelta(weeks=3))


@pytest.mark.parametrize("baseline", ["naive", "statistical"])
def test_baseline_ignores_data_after_the_start_week(poll_series, baseline):
    jumped = [
        make_poll(i, {"SPD": 60, "Others": 40}) if i >= 20 else poll
        for i, poll in enumerate(poll_series)
    ]
    original = backtest(poll_series, horizon=4).scores[baseline].quantiles
    with_future_jump = backtest(jumped, horizon=4).scores[baseline].quantiles
    assert original == with_future_jump


@pytest.mark.parametrize("baseline", ["naive", "statistical"])
def test_baselines_forecast_every_week_with_shares_summing_to_100(poll_series, baseline):
    history = weekly_poll_average(poll_series).shares[:20]
    forecaster = {"naive": NaiveBaseline(), "statistical": StatisticalBaseline()}[baseline]
    samples = forecaster.forecast(history, 12, 100, np.random.default_rng(0))
    assert samples.shape == (100, 12, len(PARTIES))
    assert samples.min() >= 0
    assert samples.sum(axis=2) == pytest.approx(np.full((100, 12), 100.0))


def test_statistical_baseline_beats_naive_on_a_steady_trend():
    # AfD rises 0.3 points a week at the expense of CDU/CSU; little noise.
    rng = np.random.default_rng(5)
    polls = [
        make_poll(i, {"CDU/CSU": 32 - 0.3 * i + rng.normal(0, 0.2), "AfD": 15 + 0.3 * i + rng.normal(0, 0.2),
                      "SPD": 16, "Greens": 12, "Others": 25})
        for i in range(40)
    ]
    scores = backtest(polls, horizon=12).scores
    assert scores["statistical"].crps_mean < scores["naive"].crps_mean


def test_output_shows_both_baselines_side_by_side(poll_series):
    assert set(backtest(poll_series).scores) == {"naive", "statistical"}


def test_carried_forward_horizon_weeks_are_reported(poll_series):
    gappy = [poll for i, poll in enumerate(poll_series) if i not in (22, 23)]
    result = backtest(gappy, horizon=6)
    assert result.carried_weeks == [START + timedelta(weeks=2), START + timedelta(weeks=3)]


def test_start_week_must_be_a_monday():
    with pytest.raises(ValueError):
        BacktestConfig(start_week=date(2024, 5, 8))


def test_saved_result_is_self_contained(poll_series, tmp_path):
    result = backtest(poll_series, seed=3, horizon=4)
    saved = json.loads(save_result(result, tmp_path).read_text(encoding="utf-8"))
    assert saved["config"]["seed"] == 3
    assert saved["config"]["start_week"] == START.isoformat()
    assert saved["data_versions"] == {"dawum": "fixture"}
    assert saved["parties"] == list(PARTIES)
    assert saved["scores"]["naive"]["crps_mean"] == pytest.approx(result.scores["naive"].crps_mean)
