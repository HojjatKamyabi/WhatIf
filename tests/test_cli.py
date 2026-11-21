import json
from datetime import date
from pathlib import Path

import pytest

from whatif.cli import main, parse_week
from whatif.data.polls import PARTIES

FIRST_MONDAY = date(2024, 1, 1)


def write_store(data_dir: Path, weeks: int = 30) -> None:
    """A DAWUM-shaped snapshot with one national poll per week."""
    surveys = {}
    for i in range(weeks):
        day = date.fromordinal(FIRST_MONDAY.toordinal() + 7 * i + 2).isoformat()
        surveys[str(i)] = {
            "Date": day, "Survey_Period": {"Date_Start": day, "Date_End": day},
            "Surveyed_Persons": "1000", "Parliament_ID": "0", "Institute_ID": "1",
            "Results": {"1": 30 - i * 0.1, "7": 18 + i * 0.1, "2": 16, "4": 12, "5": 6, "23": 5, "3": 5, "0": 8},
        }
    store = data_dir / "dawum"
    store.mkdir(parents=True)
    (store / "20260101T000000Z.json").write_text(json.dumps({"Database": {}, "Institutes": {}, "Surveys": surveys}))
    (store / "20260101T000000Z.meta.json").write_text(json.dumps({"licence": "ODC-ODbL"}))


def test_parse_week_accepts_iso_weeks_and_dates():
    assert parse_week("2025-W10") == date(2025, 3, 3)
    assert parse_week("2025-03-05") == date(2025, 3, 3)


def write_boundary_store(data_dir: Path, skip: str | None = None) -> None:
    """Economy, recorded Events and page views covering 2024, as the fetch commands would leave them."""
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    for name, level in (("eurostat-inflation", 2.5), ("eurostat-unemployment", 3.4)):
        if name == skip:
            continue
        series = {
            "updated": "2026-01-01",
            "dimension": {"time": {"category": {"index": {m: i for i, m in enumerate(months)}}}},
            "value": {str(i): level + 0.1 * i for i in range(len(months))},
        }
        store = data_dir / name
        store.mkdir(parents=True)
        (store / "20260101T000000Z.json").write_text(json.dumps(series))
        (store / "20260101T000000Z.meta.json").write_text(json.dumps({"licence": "recorded"}))
    ranged = {
        "gdelt-events": "day,party,topic,events,articles,tone\n2024-04-17,AfD,migration,1,2,-1.0\n",
        "wikipedia-pageviews": "day,article,views\n2024-04-17,Alternative_für_Deutschland,9000\n",
    }
    for name, payload in ranged.items():
        if name == skip:
            continue
        store = data_dir / name
        store.mkdir(parents=True)
        (store / "20260101T000000Z_20240101-20241231.csv").write_text(payload, encoding="utf-8")
        (store / "20260101T000000Z_20240101-20241231.meta.json").write_text(
            json.dumps({"licence": "recorded", "range_start": "2024-01-01", "range_end": "2024-12-31"})
        )


def run_backtest_command(tmp_path, *extra: str) -> dict:
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15",
        "--horizon", "6", "--runs", "50", "--scale", "1000", "--results-dir", str(tmp_path / "results"), *extra,
    ])
    assert code == 0
    saved = list((tmp_path / "results").glob("backtest-*.json"))
    assert len(saved) == 1
    return json.loads(saved[0].read_text())


def test_backtest_command_runs_the_twin_and_saves_a_self_contained_result(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)  # the last election before 2024-W15
    result = run_backtest_command(tmp_path)  # no Boundary data needed while reactions are off
    out = capsys.readouterr().out
    assert "naive" in out and "statistical" in out and "twin" in out
    assert result["data_versions"] == {
        "dawum": "20260101T000000Z",
        "election-results-2021": "20260101T000000Z",
        "constituency-structure-2021": "20260101T000000Z",
    }
    twin = result["settings"]["twin"]
    assert twin["influence"] == 0.1 and twin["poll_sample"] == 2000 and twin["degree"] == 10
    assert twin["scale"] == 1000 and twin["election_day"] == "2021-09-26"
    assert twin["inflation_reaction"] == twin["unemployment_reaction"] == twin["event_reaction"] == 0.0


def test_first_guess_reactions_feed_the_boundary_to_the_twin_and_record_its_sources(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    result = run_backtest_command(tmp_path, "--first-guess-reactions")
    assert result["data_versions"] == {
        "dawum": "20260101T000000Z",
        "election-results-2021": "20260101T000000Z",
        "constituency-structure-2021": "20260101T000000Z",
        "eurostat-inflation": "20260101T000000Z",
        "eurostat-unemployment": "20260101T000000Z",
        "gdelt-events": "20260101T000000Z_20240101-20241231",
        "wikipedia-pageviews": "20260101T000000Z_20240101-20241231",
    }
    twin = result["settings"]["twin"]
    assert twin["unemployment_reaction"] == 0.03 and twin["event_reaction"] == 0.002
    assert twin["topic_priorities"]["18-24"] == [1.0, 0.8, 1.6, 0.8, 0.8, 1.0]


@pytest.mark.parametrize(
    "missing, command",
    [
        ("eurostat-inflation", "fetch-economy"),
        ("eurostat-unemployment", "fetch-economy"),
        ("gdelt-events", "fetch-events"),
        ("wikipedia-pageviews", "fetch-attention"),
    ],
)
def test_backtest_without_boundary_data_says_what_to_fetch(tmp_path, capsys, missing, command):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data", skip=missing)
    assert main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--scale", "1000",
        "--first-guess-reactions",
    ]) == 1
    assert command in capsys.readouterr().err


def test_backtest_names_the_event_range_still_to_fetch(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    meta = next((tmp_path / "data" / "gdelt-events").glob("*.meta.json"))
    meta.write_text(json.dumps({"licence": "recorded", "range_start": "2024-03-01", "range_end": "2024-12-31"}))
    # 2024-W15 needs Events from eight weeks before it, from 2024-02-12, to set the usual level.
    assert main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--scale", "1000",
        "--first-guess-reactions",
    ]) == 1
    err = capsys.readouterr().err
    assert "fetch-events" in err and "2024-02-12" in err


def test_fetch_events_needs_a_date_range(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "fetch-events"])


def test_backtest_without_election_data_says_what_to_fetch(tmp_path, capsys):
    write_store(tmp_path / "data")
    assert main(["--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15"]) == 1
    assert "fetch-elections" in capsys.readouterr().err


def test_poll_average_command_shows_parties(tmp_path, capsys):
    write_store(tmp_path / "data")
    assert main(["--data-dir", str(tmp_path / "data"), "poll-average", "--last", "3"]) == 0
    out = capsys.readouterr().out
    assert all(party in out for party in PARTIES)


def test_missing_data_is_an_error_not_a_crash(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path / "empty"), "backtest", "--start", "2024-W15"]) == 1
    assert "fetch-polls" in capsys.readouterr().err


def write_election_store(data_dir: Path, year: int = 2025) -> None:
    """Saved results and structural data for one election, as a fetch would have left them."""
    fixtures = Path(__file__).parent / "fixtures" / "bundeswahlleiterin"
    for name, fixture in (
        (f"election-results-{year}", f"kerg2_{year}.csv"),
        (f"constituency-structure-{year}", f"strukturdaten_{year}.csv"),
    ):
        store = data_dir / name
        store.mkdir(parents=True)
        (store / "20260101T000000Z.csv").write_bytes((fixtures / fixture).read_bytes())
        (store / "20260101T000000Z.meta.json").write_text(json.dumps({"licence": "recorded"}))


def test_population_prints_the_fit_report_for_the_election_before_the_start_week(tmp_path, capsys):
    write_election_store(tmp_path)
    assert main(["--data-dir", str(tmp_path), "population", "--start", "2025-W40", "--scale", "1000"]) == 0
    out = capsys.readouterr().out
    assert "2025-02-23" in out
    assert "Flensburg – Schleswig" in out
    assert "party gap" in out


def test_population_without_election_data_says_what_to_fetch(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "population", "--start", "2025-W40"]) == 1
    assert "fetch-elections" in capsys.readouterr().err


def test_fetching_events_for_days_not_fully_published_is_refused(tmp_path, capsys):
    today = date.today().isoformat()
    assert main(["--data-dir", str(tmp_path), "fetch-events", "--start", "2025-01-01", "--end", today]) == 1
    assert "fully published" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


SCENARIO_FILE = """
[scenario.party-b-scandal]
description = "A scandal hits Party B in week 2"

[[scenario.party-b-scandal.event]]
week = 2
party = "AfD"
topic = "other"
shock = 30.0
"""


def scenario_setup(tmp_path: Path) -> Path:
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    path = tmp_path / "scenarios.toml"
    path.write_text(SCENARIO_FILE, encoding="utf-8")
    return path


def test_a_backtest_runs_under_a_named_scenario_and_records_it(tmp_path, capsys):
    path = scenario_setup(tmp_path)
    result = run_backtest_command(
        tmp_path, "--first-guess-reactions", "--scenario-file", str(path), "--scenario", "party-b-scandal"
    )
    scenario = result["settings"]["twin"]["scenario"]
    assert scenario["name"] == "party-b-scandal"
    assert scenario["events"] == [{"week": 2, "party": "AfD", "topic": "other", "shock": 30.0}]
    assert "party-b-scandal" in capsys.readouterr().out


def test_a_backtest_without_a_scenario_records_no_intervention(tmp_path):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    assert run_backtest_command(tmp_path)["settings"]["twin"]["scenario"]["name"] == "no-intervention"


def test_a_scenario_voters_cannot_react_to_is_refused(tmp_path, capsys):
    path = scenario_setup(tmp_path)
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--scale", "1000",
        "--scenario-file", str(path), "--scenario", "party-b-scandal",
    ])
    assert code == 1
    err = capsys.readouterr().err
    assert "event_reaction" in err and "--first-guess-reactions" in err


def test_an_unknown_scenario_name_lists_the_known_ones(tmp_path, capsys):
    path = scenario_setup(tmp_path)
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15",
        "--scenario-file", str(path), "--scenario", "nope",
    ])
    assert code == 1
    assert "party-b-scandal" in capsys.readouterr().err


def test_compare_shows_both_outcome_distributions_per_party_side_by_side(tmp_path, capsys):
    path = scenario_setup(tmp_path)
    code = main([
        "--data-dir", str(tmp_path / "data"), "compare", "--start", "2024-W15", "--horizon", "6",
        "--runs", "50", "--scale", "1000", "--first-guess-reactions", "--scenario-file", str(path),
        "--results-dir", str(tmp_path / "results"), "no-intervention", "party-b-scandal",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "no-intervention" in out and "party-b-scandal" in out
    assert all(party in out for party in PARTIES)
    saved = json.loads(next((tmp_path / "results").glob("compare-*.json")).read_text())
    assert set(saved["scores"]) == {"twin:no-intervention", "twin:party-b-scandal"}
    assert saved["settings"]["twin:party-b-scandal"]["scenario"]["name"] == "party-b-scandal"
    afd = PARTIES.index("AfD")
    usual = saved["scores"]["twin:no-intervention"]["quantiles"]["0.5"][-1][afd]
    scandal = saved["scores"]["twin:party-b-scandal"]["quantiles"]["0.5"][-1][afd]
    assert scandal < usual


def test_rolling_runs_a_window_and_writes_a_summary_plots_and_every_backtest(tmp_path, capsys):
    write_store(tmp_path / "data")  # weekly polls from 2024-W01 to 2024-W30
    write_election_store(tmp_path / "data", year=2021)
    code = main([
        "--data-dir", str(tmp_path / "data"), "rolling", "--first", "2024-W10", "--every", "4",
        "--horizon", "4", "--runs", "20", "--scale", "1000", "--results-dir", str(tmp_path / "results"),
        "--calibration-period", "2024-W10:2024-W13",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "naive" in out and "statistical" in out and "twin" in out
    assert "calibration period" in out.lower()

    run_dir = next((tmp_path / "results").glob("rolling-*"))
    backtests = sorted(run_dir.glob("backtest-*.json"))
    assert [json.loads(p.read_text())["config"]["start_week"] for p in backtests] == [
        "2024-03-04", "2024-04-01", "2024-04-29", "2024-05-27", "2024-06-24",
    ]
    assert len(list(run_dir.glob("backtest-*.png"))) == 5
    assert (run_dir / "summary.png").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["calibration_period"] == ["2024-03-04", "2024-03-25"]
    assert summary["summary"]["twin"]["backtests"] == 4  # the 2024-W10 start is in the Calibration Period
    assert summary["summary"]["naive"]["backtests"] == 5
    assert json.loads(backtests[0].read_text())["calibration"] is True


def test_a_single_backtest_also_saves_its_plot(tmp_path):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    run_backtest_command(tmp_path)
    assert len(list((tmp_path / "results").glob("backtest-*.png"))) == 1


def test_a_backtest_with_stand_in_party_actors_records_them_and_their_decisions(tmp_path):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    result = run_backtest_command(tmp_path, "--party-actors", "stand-in")
    twin = result["settings"]["twin"]
    assert twin["party_actors"] == "stand-in" and twin["decision_mode"] == "shared"
    assert twin["stand_in"]["tuned_on"] is None and twin["tuned_on"] is None  # untuned first guesses
    assert len(result["logs"]["twin"]) == 6 * 7  # every party, every week of the 6-week Horizon
    assert "gdelt-events" in result["data_versions"]  # Party Actors read the week's Events


def test_party_actors_need_the_recorded_events(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    code = main(["--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--party-actors", "stand-in"])
    assert code == 1
    assert "fetch-" in capsys.readouterr().err


def test_a_backtest_without_party_actors_records_none(tmp_path):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    result = run_backtest_command(tmp_path)
    assert result["settings"]["twin"]["party_actors"] == "none"
    assert result["logs"] == {}


MODELS_FILE = """
[model.fake]
base_url = "http://localhost:8000/v1"
model = "open-model-7b"
quantization = "4-bit AWQ"
training_cutoff = 2023-12-31
cutoff_source = "https://example.org/model-card"
input_price = 1.0
output_price = 4.0
"""


def fake_endpoint(monkeypatch, move="emphasise economy"):
    """Stand in for an OpenAI-compatible server: no request leaves the machine."""
    calls = []

    def post(url, body, headers):
        calls.append(json.loads(body))
        reply = {"move": move, "reason": "fake"}
        return json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}],
                           "usage": {"prompt_tokens": 500, "completion_tokens": 20}}).encode()

    monkeypatch.setattr("whatif.llm._post", post)
    return calls


def llm_backtest(tmp_path, *extra):
    models = tmp_path / "models.toml"
    models.write_text(MODELS_FILE, encoding="utf-8")
    return run_backtest_command(
        tmp_path, "--party-actors", "language-model", "--model", "fake", "--models-file", str(models), *extra
    )


def test_a_language_model_backtest_is_repeated_with_stand_ins_and_records_its_provenance(tmp_path, capsys, monkeypatch):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    calls = fake_endpoint(monkeypatch)
    result = llm_backtest(tmp_path)
    assert set(result["scores"]) == {"naive", "statistical", "twin:stand-in", "twin:language-model"}
    assert len(calls) == 6 * 7  # Shared Actor Decisions: Horizon × parties
    llm = result["settings"]["twin:language-model"]
    assert llm["model"]["model"] == "open-model-7b" and llm["model"]["quantization"] == "4-bit AWQ"
    assert llm["model"]["endpoint"] == "http://localhost:8000/v1"
    assert llm["model"]["training_cutoff"] == "2023-12-31"
    assert llm["cutoff_margin_months"] == 3
    assert llm["cutoff_label"] == "After"  # 2024-04-08 is more than 3 months past 2023-12-31
    assert llm["decision_mode"] == "shared" and result["settings"]["twin:stand-in"]["decision_mode"] == "shared"
    assert sorted(llm["masking"]["labels"].values()) == [f"Party {c}" for c in "ABCDEFG"]
    assert llm["usage"]["calls"] == 42 and llm["usage"]["cost_usd"] == pytest.approx(42 * (500 * 1 + 20 * 4) / 1e6)
    out = capsys.readouterr().out
    assert "twin:stand-in" in out and "twin:language-model" in out
    assert "Cutoff Label: After" in out and "cost" in out.lower()


def test_rerunning_a_language_model_backtest_makes_no_model_calls(tmp_path, monkeypatch):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    calls = fake_endpoint(monkeypatch)
    llm_backtest(tmp_path)
    for path in (tmp_path / "results").glob("*"):
        path.unlink()
    again = llm_backtest(tmp_path)
    assert len(calls) == 42  # all from the first run
    assert again["settings"]["twin:language-model"]["usage"]["cache_hits"] == 42


def test_per_run_language_model_backtests_default_to_20_runs(tmp_path, monkeypatch):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    calls = fake_endpoint(monkeypatch)
    models = tmp_path / "models.toml"
    models.write_text(MODELS_FILE, encoding="utf-8")
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--horizon", "2", "--scale", "1000",
        "--results-dir", str(tmp_path / "results"), "--party-actors", "language-model", "--model", "fake",
        "--models-file", str(models), "--decision-mode", "per-run",
    ])
    assert code == 0
    result = json.loads(next((tmp_path / "results").glob("backtest-*.json")).read_text())
    assert result["config"]["runs"] == 20
    assert len(calls) == 20 * 2 * 7
    assert result["settings"]["twin:stand-in"]["decision_mode"] == "per-run"


def test_a_missing_model_config_is_an_error_not_a_crash(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)
    write_boundary_store(tmp_path / "data")
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15", "--party-actors", "language-model",
        "--model", "fake", "--models-file", str(tmp_path / "nope.toml"),
    ])
    assert code == 1
    assert "models.example.toml" in capsys.readouterr().err
