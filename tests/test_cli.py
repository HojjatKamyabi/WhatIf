import json
from datetime import date
from pathlib import Path

from whatif.cli import main, parse_week
from whatif.polls import PARTIES

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


def test_backtest_command_runs_the_twin_and_saves_a_self_contained_result(tmp_path, capsys):
    write_store(tmp_path / "data")
    write_election_store(tmp_path / "data", year=2021)  # the last election before 2024-W15
    code = main([
        "--data-dir", str(tmp_path / "data"), "backtest", "--start", "2024-W15",
        "--horizon", "6", "--runs", "50", "--scale", "1000", "--results-dir", str(tmp_path / "results"),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "naive" in out and "statistical" in out and "twin" in out
    saved = list((tmp_path / "results").glob("backtest-*.json"))
    assert len(saved) == 1
    result = json.loads(saved[0].read_text())
    assert result["data_versions"] == {
        "dawum": "20260101T000000Z",
        "election-results-2021": "20260101T000000Z",
        "constituency-structure-2021": "20260101T000000Z",
    }
    twin = result["settings"]["twin"]
    assert twin["influence"] == 0.1 and twin["poll_sample"] == 2000 and twin["degree"] == 10
    assert twin["scale"] == 1000 and twin["election_day"] == "2021-09-26"


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
