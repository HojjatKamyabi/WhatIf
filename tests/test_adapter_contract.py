"""The data adapter contract: rules every adapter must meet (spec stories 16, 67, 68)."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from whatif.adapters import Adapter, Licence, LicenceNotRecordedError, fetch, latest
from whatif.store import MissingDataError

FIXTURE = Path(__file__).parent / "fixtures" / "dawum_sample.json"


def unlicensed_adapter() -> Adapter:
    return Adapter(
        name="example",
        source="https://example.org/data",
        licence=None,
        record_type=dict,
        parse=lambda raw: [{"raw": raw}],
        fixture=FIXTURE,
    )


def test_an_adapter_without_a_recorded_licence_refuses_to_fetch(tmp_path):
    calls = []

    def download(url: str) -> bytes:
        calls.append(url)
        return b"{}"

    with pytest.raises(LicenceNotRecordedError):
        fetch(unlicensed_adapter(), tmp_path, download=download)
    assert calls == []
    assert not any(tmp_path.iterdir())


EXAMPLE_LICENCE = Licence(
    name="Example Open Licence",
    url="https://example.org/licence",
    checked_on=date(2026, 11, 5),
    attribution="Data from example.org",
)


def test_a_licensed_adapter_saves_a_snapshot_with_its_provenance(tmp_path):
    adapter = replace(unlicensed_adapter(), licence=EXAMPLE_LICENCE)
    saved = fetch(adapter, tmp_path, download=lambda url: b"payload")

    loaded = latest(adapter, tmp_path)
    assert loaded.path == saved.path
    assert loaded.raw() == b"payload"
    assert loaded.metadata["source"] == "https://example.org/data"
    assert loaded.metadata["licence"] == "Example Open Licence"
    assert loaded.metadata["licence_url"] == "https://example.org/licence"
    assert loaded.metadata["licence_checked_on"] == "2026-11-05"
    assert loaded.metadata["attribution"] == "Data from example.org"
    assert loaded.metadata["downloaded_at"]


def test_reading_a_source_that_was_never_fetched_is_reported(tmp_path):
    with pytest.raises(MissingDataError):
        latest(unlicensed_adapter(), tmp_path)


# The shared contract check. Every adapter registered in whatif.sources gets these tests for free.

from whatif.sources import ADAPTERS  # noqa: E402


def test_dawum_is_a_registered_adapter():
    assert "dawum" in {adapter.name for adapter in ADAPTERS}


def test_adapter_names_are_unique():
    names = [adapter.name for adapter in ADAPTERS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_adapter_meets_the_contract(adapter, tmp_path):
    assert adapter.source.startswith(("https://", "http://"))
    assert adapter.fixture.is_file(), "every adapter ships a saved sample fixture"

    records = adapter.parse(adapter.fixture.read_bytes())
    assert records, "the fixture must parse into at least one record"
    assert all(isinstance(r, adapter.record_type) for r in records), "records are typed"

    if adapter.licence is None:
        with pytest.raises(LicenceNotRecordedError):
            fetch(adapter, tmp_path, download=lambda url: pytest.fail("fetched without a licence"))
        return

    if adapter.collect is None:
        fetch(adapter, tmp_path, download=lambda url: adapter.fixture.read_bytes())
    else:  # fetched by date range: the fixture stands in for what collecting a range stores
        with pytest.raises(ValueError, match="date range"):
            fetch(adapter, tmp_path, download=lambda url: pytest.fail("fetched without a range"))
        stand_in = replace(adapter, collect=lambda download, start, end: adapter.fixture.read_bytes())
        fetch(stand_in, tmp_path, start=date(2025, 2, 1), end=date(2025, 2, 28))
    snapshot = latest(adapter, tmp_path)
    assert adapter.parse(snapshot.raw()) == records
    for key in ("source", "licence", "licence_url", "licence_checked_on", "downloaded_at"):
        assert snapshot.metadata[key], f"snapshot metadata records {key}"
    if adapter.collect is not None:
        assert (snapshot.metadata["range_start"], snapshot.metadata["range_end"]) == ("2025-02-01", "2025-02-28")
