import json
from datetime import date
from pathlib import Path

import pytest

from whatif.adapters import fetch, latest
from whatif.polls import DAWUM, PARTIES, parse_dawum
from whatif.store import MissingDataError

SAMPLE = Path(__file__).parent / "fixtures" / "dawum_sample.json"


def sample_database() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_only_national_polls_are_kept():
    polls = parse_dawum(sample_database())
    assert len(polls) == 2
    assert {p.institute for p in polls} == {"Forsa", "INSA"}


def test_shares_are_on_the_fixed_party_list_and_sum_to_100():
    for poll in parse_dawum(sample_database()):
        assert tuple(poll.shares) == PARTIES
        assert sum(poll.shares.values()) == pytest.approx(100)


def test_untracked_parties_count_as_others():
    forsa = next(p for p in parse_dawum(sample_database()) if p.institute == "Forsa")
    # Freie Wähler (3) plus Sonstige (5) out of a 100 total
    assert forsa.shares["Others"] == pytest.approx(8)
    assert forsa.shares["BSW"] == 0
    assert "BSW" not in forsa.reported
    assert "FDP" in forsa.reported


def test_field_end_date_and_missing_sample_size():
    forsa = next(p for p in parse_dawum(sample_database()) if p.institute == "Forsa")
    assert forsa.field_end == date(2026, 11, 2)
    assert forsa.sample_size is None


def test_fetch_records_provenance_and_is_read_back_offline(tmp_path):
    raw = SAMPLE.read_bytes()
    saved = fetch(DAWUM, tmp_path, download=lambda url: raw)

    loaded = latest(DAWUM, tmp_path)
    assert loaded.path == saved.path
    assert loaded.load() == sample_database()
    assert loaded.metadata["licence"] == "ODC-ODbL"
    assert "dawum.de" in loaded.metadata["attribution"]
    assert loaded.metadata["source_last_update"] == "2026-11-03T19:01:08+01:00"
    assert loaded.metadata["downloaded_at"]


def test_missing_store_is_reported(tmp_path):
    with pytest.raises(MissingDataError):
        latest(DAWUM, tmp_path)
