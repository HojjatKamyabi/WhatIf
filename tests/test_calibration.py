"""Calibration: Voter reactions and Stand-in rules are tuned on the Calibration Period only."""

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pytest

from whatif.actors import StandInParameters
from whatif.backtest import BacktestConfig, CalibrationPeriod, require_tuned_outside
from whatif.calibration import (
    DEFAULT_GRID,
    Grid,
    calibrate,
    calibration_start_weeks,
    load_calibrated,
    points,
    save_calibration,
)
from whatif.poll_average import weekly_poll_average
from whatif.data.store import MissingDataError
from whatif.twin import FIRST_GUESS_REACTIONS, TwinParameters
from tests.conftest import FIRST_WEEK

HORIZON = 4


def week(n: int) -> date:
    return FIRST_WEEK + timedelta(weeks=n)


PERIOD = CalibrationPeriod(week(0), week(27))  # the poll series runs 40 weeks

SMALL_GRID = Grid(
    reactions={"inflation_reaction": (0.0, 0.01), "unemployment_reaction": (0.0, 0.03), "event_reaction": (-0.002, 0.0, 0.002)},
    actors={"move_reaction": (0.0, 0.01, 0.02), "rebut_threshold": (6.0, 12.0), "min_attention": (0.5, 1.0)},
)
TARGET = {  # the values the fake Twin forecasts best with
    "inflation_reaction": 0.01, "unemployment_reaction": 0.0, "event_reaction": -0.002,
    "move_reaction": 0.02, "rebut_threshold": 12.0, "min_attention": 0.5,
}
SCALE = {"inflation_reaction": 0.01, "unemployment_reaction": 0.03, "event_reaction": 0.002,
         "move_reaction": 0.01, "rebut_threshold": 6.0, "min_attention": 1.0}


@dataclass
class FakeTwin:
    """Forecasts last week's polls, with more spread the further its parameters are from TARGET."""

    parameters: TwinParameters
    stand_in: StandInParameters | None
    name: str = "twin"

    def forecast(self, history, horizon, runs, rng):
        values = {k: getattr(self.parameters, k) for k in ("inflation_reaction", "unemployment_reaction",
                                                          "event_reaction", "move_reaction")}
        if self.stand_in is not None:
            values |= {"rebut_threshold": self.stand_in.rebut_threshold, "min_attention": self.stand_in.min_attention}
        else:  # without Stand-ins, the Party Actor values can't matter
            values |= {k: TARGET[k] for k in ("move_reaction", "rebut_threshold", "min_attention")}
        distance = sum(abs(values[k] - TARGET[k]) / SCALE[k] for k in TARGET)
        noise = rng.normal(0.0, 0.5 + 3.0 * distance, (runs, horizon, history.shape[1]))
        return history[-1] + noise


def fake_twin(start, parameters, stand_in, name):
    return FakeTwin(parameters, stand_in, name)


def run(poll_series, grid=SMALL_GRID, make_twin=fake_twin, **options):
    average = weekly_poll_average(poll_series)
    starts = calibration_start_weeks(average, PERIOD, every=4, horizon=HORIZON)
    return calibrate(average, starts, make_twin, period=PERIOD, grid=grid, horizon=HORIZON,
                     runs=40, final_runs=60, seed=1, **options)


# --- Which Backtests tune ---------------------------------------------------------------------------


def test_calibration_backtests_start_every_few_weeks_with_their_whole_horizon_inside_the_period(poll_series):
    average = weekly_poll_average(poll_series)
    starts = calibration_start_weeks(average, PERIOD, every=4, horizon=HORIZON)
    assert starts[0] == week(8)  # the first week with 8 weeks of polls before it
    assert starts == [week(n) for n in (8, 12, 16, 20, 24)]  # week 24's Horizon ends in week 27, the last
    assert all(s + timedelta(weeks=HORIZON - 1) <= PERIOD.last for s in starts)


def test_start_weeks_before_the_first_registered_election_are_skipped_keeping_the_4_week_rhythm(poll_series):
    average = weekly_poll_average(poll_series)
    starts = calibration_start_weeks(average, PERIOD, every=4, horizon=HORIZON, after=week(12))
    assert starts == [week(n) for n in (16, 20, 24)]  # week 12 itself is the election week, so not after it


def test_a_period_too_short_for_one_horizon_is_refused(poll_series):
    with pytest.raises(MissingDataError, match="Calibration Period"):
        calibration_start_weeks(weekly_poll_average(poll_series), CalibrationPeriod(week(9), week(10)),
                                every=4, horizon=HORIZON)


@pytest.mark.parametrize("start", [week(26), week(30)], ids=["horizon-leaves-the-period", "starts-after-it"])
def test_calibrate_refuses_any_backtest_scoring_weeks_outside_the_period(poll_series, start):
    average = weekly_poll_average(poll_series)
    with pytest.raises(ValueError, match="outside the Calibration Period"):
        calibrate(average, [week(8), start], fake_twin, period=PERIOD, grid=SMALL_GRID, horizon=HORIZON,
                  runs=10, final_runs=10)


# --- The search -------------------------------------------------------------------------------------


def test_grid_points_cover_every_combination_in_order():
    assert points({"a": (1, 2), "b": (3,)}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_the_default_grid_scales_the_first_guesses_and_tries_the_event_rule_both_ways():
    reactions, actors = DEFAULT_GRID.reactions, DEFAULT_GRID.actors
    assert reactions["inflation_reaction"] == pytest.approx(tuple(f * 0.01 for f in (0, 0.5, 1, 2)))
    assert reactions["unemployment_reaction"] == pytest.approx(tuple(f * 0.03 for f in (0, 0.5, 1, 2)))
    assert reactions["event_reaction"] == pytest.approx(tuple(f * 0.002 for f in (-1, -0.5, 0, 0.5, 1, 2)))
    assert actors["move_reaction"] == pytest.approx(tuple(f * FIRST_GUESS_REACTIONS["move_reaction"] for f in (0, 0.5, 1, 2)))
    assert actors["rebut_threshold"] == (3.0, 6.0, 12.0)
    assert actors["min_attention"] == (0.5, 1.0, 2.0)
    assert len(points(reactions)) == 96 and len(points(actors)) == 36


def test_stage_one_tunes_the_reactions_without_party_actors_and_stage_two_the_actors_on_top(poll_series):
    calls = []

    def recording(start, parameters, stand_in, name):
        calls.append((parameters, stand_in, name))
        return fake_twin(start, parameters, stand_in, name)

    result = run(poll_series, make_twin=recording)
    starts = 5
    stage_one = [c for c in calls[: 12 * starts]]
    assert all(stand_in is None for _, stand_in, _ in stage_one)
    stage_two = calls[12 * starts : (12 + 12) * starts]
    assert all(stand_in is not None for _, stand_in, _ in stage_two)
    for parameters, _, _ in stage_two:  # stage one's winner stays fixed
        assert (parameters.inflation_reaction, parameters.unemployment_reaction, parameters.event_reaction) == (
            0.01, 0.0, -0.002)
    assert len(result.reactions) == 12 and len(result.actors) == 12


def test_the_lowest_mean_crps_wins_each_stage(poll_series):
    result = run(poll_series)
    best = min(result.reactions, key=lambda s: s.crps_mean)
    assert result.reactions_best == best
    for key in ("inflation_reaction", "unemployment_reaction", "event_reaction", "move_reaction"):
        assert getattr(result.twin, key) == TARGET[key]
    assert (result.stand_in.rebut_threshold, result.stand_in.min_attention) == (12.0, 0.5)


def test_every_grid_point_scores_the_mean_of_its_backtests(poll_series):
    result = run(poll_series)
    for score in result.reactions + result.actors:
        assert len(score.backtests) == len(result.starts)
        assert score.crps_mean == pytest.approx(np.mean(score.backtests))


def test_ties_go_to_the_point_that_changes_least_from_untuned(poll_series):
    ignores = lambda start, parameters, stand_in, name: FakeTwin(TwinParameters(**{
        k: TARGET[k] for k in ("inflation_reaction", "unemployment_reaction", "event_reaction", "move_reaction")
    }), StandInParameters(12.0, 0.5), name)  # every point forecasts alike
    result = run(poll_series, make_twin=ignores)
    assert (result.twin.inflation_reaction, result.twin.unemployment_reaction, result.twin.event_reaction,
            result.twin.move_reaction) == (0.0, 0.0, 0.0, 0.0)
    assert (result.stand_in.rebut_threshold, result.stand_in.min_attention) == (6.0, 1.0)  # the defaults


def test_every_backtest_uses_the_same_random_stream_so_points_differ_only_by_their_values(poll_series):
    same = run(poll_series)
    again = run(poll_series)
    assert [s.crps_mean for s in same.reactions] == [s.crps_mean for s in again.reactions]


def test_the_winner_is_scored_again_at_more_runs_against_untuned_and_the_baselines(poll_series):
    result = run(poll_series)
    assert set(result.final) == {"naive", "statistical", "twin:untuned", "twin:calibrated",
                                 "twin:calibrated+stand-ins"}
    for row in result.final.values():
        assert row["backtests"] == len(result.starts)
    assert result.final_runs == 60


# --- Provenance ---------------------------------------------------------------------------------------


def test_tuned_values_carry_the_period_they_were_tuned_on(poll_series):
    result = run(poll_series)
    assert result.twin.tuned_on == PERIOD
    assert result.stand_in.tuned_on == PERIOD


def test_the_saved_calibration_records_the_search_and_loads_back(poll_series, tmp_path):
    result = run(poll_series)
    path = save_calibration(result, tmp_path, data_versions={"dawum": "v1"})
    import json

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tuned_on"] == [str(PERIOD.first), str(PERIOD.last)]
    assert saved["starts"] == [str(s) for s in result.starts]
    assert saved["grid"]["reactions"]["event_reaction"] == [-0.002, 0.0, 0.002]
    assert len(saved["search"]["reactions"]) == 12 and len(saved["search"]["actors"]) == 12
    assert {"runs", "final_runs", "seed", "horizon", "data_versions", "final"} <= set(saved)

    twin, stand_in = load_calibrated(path)
    assert twin.event_reaction == -0.002 and twin.move_reaction == 0.02
    assert stand_in == StandInParameters(12.0, 0.5, tuned_on=PERIOD)
    assert twin.tuned_on == PERIOD
    assert replace(twin, tuned_on=None) == TwinParameters(**{
        k: TARGET[k] for k in ("inflation_reaction", "unemployment_reaction", "event_reaction", "move_reaction")
    })


def test_loaded_values_are_refused_for_backtests_scoring_tuned_weeks(poll_series, tmp_path):
    twin, _ = load_calibrated(save_calibration(run(poll_series), tmp_path, data_versions={}))
    with pytest.raises(ValueError, match="tune on the Calibration Period only"):
        require_tuned_outside(twin.tuned_on, BacktestConfig(start_week=week(25), horizon=HORIZON, runs=10))
    require_tuned_outside(twin.tuned_on, BacktestConfig(start_week=week(28), horizon=HORIZON, runs=10))


# --- The commands -------------------------------------------------------------------------------------

import json  # noqa: E402

from whatif import calibration  # noqa: E402
from whatif.cli import main  # noqa: E402
from tests.test_cli import write_boundary_store, write_election_store, write_store  # noqa: E402

TINY_GRID = Grid(
    reactions={"inflation_reaction": (0.0, 0.01), "unemployment_reaction": (0.0,), "event_reaction": (0.0, 0.002)},
    actors={"move_reaction": (0.0, 0.01), "rebut_threshold": (6.0,), "min_attention": (1.0,)},
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Polls for 2024-W01 to W30, the 2021 election and Boundary data for 2024; a tiny grid."""
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    monkeypatch.setattr(calibration, "DEFAULT_GRID", TINY_GRID)
    return tmp_path


def calibrate_command(tmp_path, *extra):
    return main([
        "--data-dir", str(tmp_path / "data"), "calibrate", "--calibration-period", "2024-W09:2024-W22",
        "--horizon", "6", "--runs", "10", "--final-runs", "20", "--scale", "1000",
        "--results-dir", str(tmp_path / "results"), *extra,
    ])


def test_calibrate_command_saves_the_tuned_values_and_the_search(store, capsys):
    assert calibrate_command(store) == 0
    saved = list((store / "results").glob("calibration-*.json"))
    assert len(saved) == 1
    record = json.loads(saved[0].read_text(encoding="utf-8"))
    assert record["tuned_on"] == ["2024-02-26", "2024-05-27"]
    assert record["starts"] == ["2024-02-26", "2024-03-25", "2024-04-22"]  # W09, W13, W17
    assert len(record["search"]["reactions"]) == 4 and len(record["search"]["actors"]) == 2
    assert {"dawum", "gdelt-events", "wikipedia-pageviews", "eurostat-inflation"} <= set(record["data_versions"])
    out = capsys.readouterr().out
    assert "3 Backtests" in out and "Saved" in out
    assert "twin:calibrated" in out


def test_calibrate_command_skips_start_weeks_with_no_federal_election_before_them(store, monkeypatch, capsys):
    import dataclasses
    from whatif import cli
    from whatif.data import elections

    # As if the 2021 election, the only one registered, had been held on 2024-03-01.
    moved = tuple(dataclasses.replace(e, day=date(2024, 3, 1)) for e in elections.ELECTIONS if e.day.year == 2021)
    monkeypatch.setattr(elections, "ELECTIONS", moved)
    monkeypatch.setattr(cli, "ELECTIONS", moved)
    assert calibrate_command(store) == 0
    record = json.loads(next((store / "results").glob("calibration-*.json")).read_text(encoding="utf-8"))
    assert record["starts"] == ["2024-03-25", "2024-04-22"]  # W13, W17; W09 has no election before it
    out = capsys.readouterr().out
    assert "2 Backtests" in out and "skipped 1 start week before the first registered federal election" in out


def calibrated_file(tmp_path):
    assert calibrate_command(tmp_path) == 0
    return str(next((tmp_path / "results").glob("calibration-*.json")))


def test_a_backtest_after_the_period_runs_with_the_calibrated_values_and_records_them(store):
    path = calibrated_file(store)
    assert main([
        "--data-dir", str(store / "data"), "backtest", "--start", "2024-W23", "--horizon", "6", "--runs", "20",
        "--scale", "1000", "--results-dir", str(store / "out"), "--calibrated", path, "--party-actors", "stand-in",
    ]) == 0
    result = json.loads(next((store / "out").glob("backtest-*.json")).read_text(encoding="utf-8"))
    settings = result["settings"]["twin"]
    tuned = json.loads(open(path, encoding="utf-8").read())["tuned"]
    for key, value in tuned["twin"].items():
        assert settings[key] == value
    assert settings["stand_in"]["rebut_threshold"] == tuned["stand_in"]["rebut_threshold"]
    assert settings["calibrated"] == path
    assert settings["tuned_on"] == {"first": "2024-02-26", "last": "2024-05-27"}


def test_a_backtest_scoring_tuned_weeks_is_refused(store, capsys):
    path = calibrated_file(store)
    capsys.readouterr()
    assert main([
        "--data-dir", str(store / "data"), "backtest", "--start", "2024-W15", "--horizon", "6",
        "--scale", "1000", "--results-dir", str(store / "out"), "--calibrated", path,
    ]) == 1
    assert "tune on the Calibration Period only" in capsys.readouterr().err


def test_calibrated_values_and_first_guesses_are_not_mixed(store, capsys):
    path = calibrated_file(store)
    with pytest.raises(SystemExit):
        main(["--data-dir", str(store / "data"), "backtest", "--start", "2024-W23", "--calibrated", path,
              "--first-guess-reactions"])
    assert "not allowed with" in capsys.readouterr().err


def test_a_rolling_window_runs_with_calibrated_values_and_stand_ins(store, capsys):
    path = calibrated_file(store)
    capsys.readouterr()
    assert main([
        "--data-dir", str(store / "data"), "rolling", "--first", "2024-W23", "--horizon", "6", "--runs", "20",
        "--scale", "1000", "--results-dir", str(store / "out"), "--calibrated", path, "--party-actors", "stand-in",
        "--calibration-period", "2024-W09:2024-W22",
    ]) == 0
    summary = json.loads(next((store / "out").glob("rolling-*/summary.json")).read_text(encoding="utf-8"))
    assert summary["window"]["calibrated"] == path
    assert summary["window"]["party_actors"] == "stand-in"
    assert summary["summary"]["twin"]["backtests"] == 1
