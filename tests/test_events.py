"""Recorded Events (GDELT 2.0) and public attention (Wikipedia page views) as Boundary series."""

import io
import urllib.error
import zipfile
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from whatif.adapters import fetch
from whatif.boundary import attention, event_shock
from whatif.events import (
    GDELT,
    PAGEVIEWS,
    TOPICS,
    EventCount,
    PageViews,
    collect_gdelt,
    collect_pageviews,
    event_log,
    page_view_log,
)
from whatif.polls import PARTIES
from whatif.store import MissingDataError

FIXTURES = Path(__file__).parent / "fixtures"
SLOT = "http://data.gdeltproject.org/gdeltv2/20250210120000"


def zipped(name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


def not_found(url: str):
    raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def gdelt_download(url: str) -> bytes:
    """Only the 12:00 slot of 2025-02-10 exists; it holds the trimmed real export files."""
    if url == f"{SLOT}.export.CSV.zip":
        return zipped("export.CSV", (FIXTURES / "gdelt" / "export.CSV").read_bytes())
    if url == f"{SLOT}.translation.export.CSV.zip":
        return zipped("translation.export.CSV", (FIXTURES / "gdelt" / "translation.export.CSV").read_bytes())
    not_found(url)


def collected_events() -> list[EventCount]:
    return GDELT.parse(collect_gdelt(gdelt_download, date(2025, 2, 10), date(2025, 2, 10)))


def by_party(records: list[EventCount]) -> dict[str, EventCount]:
    return {r.party: r for r in records if r.party is not None}


def test_events_are_kept_only_when_they_involve_germany():
    records = collected_events()
    assert all(isinstance(r, EventCount) for r in records)
    assert sum(r.events for r in records) == 20  # 20 German rows out of 26 in the two files


def test_events_are_grouped_by_day_party_and_topic():
    parties = by_party(collected_events())
    assert set(parties) == {"CDU/CSU", "Greens", "AfD", "SPD"}

    merz = parties["CDU/CSU"]  # euronews: "chancellor hopeful merz open to reform of germanys debt brake"
    assert (merz.day, merz.topic, merz.events, merz.articles) == (date(2025, 2, 10), "economy", 3, 10)
    assert merz.tone == pytest.approx(-4.21940928270042)

    greens = parties["Greens"]  # yahoo: "greens climate change ignored german"
    assert (greens.topic, greens.events, greens.articles) == ("climate", 1, 4)
    assert greens.tone == pytest.approx(-5.85585585585586)


def test_translated_german_language_news_is_included():
    parties = by_party(collected_events())
    scholz = parties["SPD"]  # hna.de/wirtschaft/...-scholz-aussagen, from the translation stream
    assert (scholz.topic, scholz.events, scholz.articles) == ("economy", 3, 7)
    assert parties["AfD"].topic == "other"  # a Macedonian headline with no topic keyword


def test_missing_gdelt_files_are_counted_not_fatal():
    raw = collect_gdelt(gdelt_download, date(2025, 2, 10), date(2025, 2, 10))
    assert GDELT.describe(raw)["missing_files"] == 190  # 96 slots × 2 streams, 2 present


def test_gdelt_is_downloaded_by_date_range_only():
    requested = []

    def download(url: str) -> bytes:
        requested.append(url)
        not_found(url)

    collect_gdelt(download, date(2025, 2, 10), date(2025, 2, 11))
    assert len(requested) == 2 * 96 * 2
    assert all("/gdeltv2/2025021" in url for url in requested)
    assert not any("masterfilelist" in url for url in requested)


def test_the_event_log_covers_every_fetched_range(tmp_path):
    for start, end in ((date(2025, 1, 1), date(2025, 1, 31)), (date(2025, 2, 1), date(2025, 2, 28))):
        fetch(GDELT, tmp_path, download=gdelt_download, start=start, end=end)
    log = event_log(tmp_path)
    assert log.covers(date(2025, 1, 1), date(2025, 2, 28))
    assert not log.covers(date(2025, 1, 1), date(2025, 3, 1))
    assert sum(r.events for r in log.records) == 20


def test_an_event_log_with_nothing_fetched_says_what_to_fetch(tmp_path):
    with pytest.raises(MissingDataError, match="fetch-events"):
        event_log(tmp_path)


# --- Page views ------------------------------------------------------------------------------


def pageview_download(url: str) -> bytes:
    if "/Alternative_f%C3%BCr_Deutschland/" in url:
        return (FIXTURES / "wikimedia" / "afd.json").read_bytes()
    if "/Inflation/" in url:
        return (FIXTURES / "wikimedia" / "inflation.json").read_bytes()
    not_found(url)


def test_page_views_are_read_per_day_and_mapped_to_a_party_or_topic():
    records = PAGEVIEWS.parse(collect_pageviews(pageview_download, date(2025, 2, 3), date(2025, 2, 9)))
    assert all(isinstance(r, PageViews) for r in records)
    views = {(r.subject, r.day): r.views for r in records}
    assert views[("AfD", date(2025, 2, 3))] == 9681
    assert views[("economy", date(2025, 2, 9))] == 327
    assert {r.subject for r in records} == {"AfD", "economy"}  # articles without data are skipped


def test_page_view_log_without_a_fetch_says_what_to_fetch(tmp_path):
    with pytest.raises(MissingDataError, match="fetch-attention"):
        page_view_log(tmp_path)


# --- Weekly Boundary series built from them ---------------------------------------------------

START = date(2025, 3, 3)  # a Monday
AFD, MIGRATION = PARTIES.index("AfD"), TOPICS.index("migration")


def steady_afd_coverage(weeks: range, burst_week: int | None = None) -> list[EventCount]:
    """Ten articles a week about AfD and migration at tone -2, except one burst of 30 at tone -4."""
    records = []
    for w in weeks:
        day = START + timedelta(weeks=w)
        articles, tone = (30, -4.0) if w == burst_week else (10, -2.0)
        records.append(EventCount(day, "AfD", "migration", 1, articles, tone))
    return records


def test_event_shock_is_negative_coverage_above_the_party_s_usual_level():
    records = steady_afd_coverage(range(-8, 4), burst_week=2)
    log = event_log_from(records, START - timedelta(weeks=8), START + timedelta(weeks=4))
    shock = event_shock(log, START, horizon=4, baseline_weeks=8)
    assert shock.shape == (4, len(PARTIES), len(TOPICS))
    # Usual: 10 articles × tone 2 = 20 a week, on a volume of 10. Burst: 30 × 4 = 120, so (120 − 20) / 10.
    assert shock[2, AFD, MIGRATION] == pytest.approx(10.0)
    assert np.delete(shock.reshape(4, -1), AFD * len(TOPICS) + MIGRATION, axis=1)[2] == pytest.approx(0.0)
    assert shock[[0, 1, 3]] == pytest.approx(0.0)


def test_event_shock_refuses_days_outside_the_fetched_ranges():
    log = event_log_from([], START - timedelta(weeks=2), START + timedelta(weeks=4))
    with pytest.raises(MissingDataError, match="fetch-events"):
        event_shock(log, START, horizon=4, baseline_weeks=8)


def test_attention_is_page_views_relative_to_the_usual_level():
    records = [
        PageViews(START + timedelta(days=d), "Alternative_für_Deutschland", "AfD", 300 if d // 7 == 1 else 100)
        for d in range(-56, 28)
    ]
    log = page_view_log_from(records, START - timedelta(weeks=8), START + timedelta(weeks=4))
    party, topic = attention(log, START, horizon=4, baseline_weeks=8)
    assert party[:, AFD] == pytest.approx([1.0, 3.0, 1.0, 1.0])
    assert party[:, PARTIES.index("SPD")] == pytest.approx([1.0] * 4)  # no views at all: neutral
    assert topic == pytest.approx(np.ones((4, len(TOPICS))))


def test_attention_refuses_weeks_outside_the_fetched_range():
    log = page_view_log_from([], START - timedelta(weeks=8), START + timedelta(weeks=2))
    with pytest.raises(MissingDataError, match="fetch-attention"):
        attention(log, START, horizon=4, baseline_weeks=8)


def event_log_from(records, first: date, last: date):
    from whatif.events import EventLog

    return EventLog(records, ((first, last),))


def page_view_log_from(records, first: date, last: date):
    from whatif.events import PageViewLog

    return PageViewLog(records, (first, last))


def test_the_contract_fixtures_are_what_collecting_the_samples_stores():
    stored = collect_gdelt(gdelt_download, date(2025, 2, 10), date(2025, 2, 10))
    assert GDELT.parse(GDELT.fixture.read_bytes()) == GDELT.parse(stored)
    stored = collect_pageviews(pageview_download, date(2025, 2, 3), date(2025, 2, 9))
    assert PAGEVIEWS.parse(PAGEVIEWS.fixture.read_bytes()) == PAGEVIEWS.parse(stored)
