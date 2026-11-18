"""Recorded news Events for Germany from the GDELT 2.0 event database.

GDELT is terabytes in total, so it is only ever downloaded by date range: every 15-minute export
file in the range, in English and translated, filtered to Germany and aggregated by day, party
and topic before anything is stored. Which party and topic a record belongs to is not GDELT's
to say: the caller passes a `classify` function, so each project keeps its own mapping.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.parse
import zipfile
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

from whatif.data.adapters import Adapter, Licence, fetch_with_retries, snapshots

SOURCE = "http://data.gdeltproject.org/gdeltv2/"
LICENCE = Licence(
    name="GDELT Project terms of use: unlimited and unrestricted use, with a citation",
    url="https://www.gdeltproject.org/about.html#termsofuse",
    checked_on=date(2026, 11, 10),
    attribution="Event data: The GDELT Project (https://www.gdeltproject.org/)",
)
FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "gdelt" / "events_20250210.csv"

_STREAMS = ("export.CSV.zip", "translation.export.CSV.zip")  # English sources, translated sources
# Columns of a GDELT 2.0 event record (0-based).
_ACTOR1_NAME, _ACTOR1_COUNTRY, _ACTOR2_NAME, _ACTOR2_COUNTRY = 6, 7, 16, 17
_NUM_ARTICLES, _AVG_TONE, _ACTION_COUNTRY, _DATE_ADDED, _SOURCE_URL = 33, 34, 53, 59, 60
_COLUMNS = 61

# classify(actor-name words, URL-path words) -> (parties involved, topic). Words are lower case,
# with umlauts written out (ü -> ue); the two actors' names are separated by an empty word.
Classify = Callable[[list[str], list[str]], tuple[list[str], str]]


@dataclass(frozen=True)
class EventCount:
    """Recorded Events for Germany on one day, for one party (or none) and one topic."""

    day: date  # the day GDELT recorded them
    party: str | None  # the party `classify` named, or None when the Events involve no party
    topic: str  # the topic `classify` named
    events: int
    articles: int  # news articles reporting them
    tone: float  # article-weighted average tone (GDELT's AvgTone; negative is bad news)


def words(text: str) -> list[str]:
    """Lower-case words of a name or URL, with umlauts written out, for keyword matching."""
    text = urllib.parse.unquote(text).lower()
    for umlaut, written in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, written)
    return re.findall(r"[a-z]+", text)


def _is_germany(row: list[str]) -> bool:
    return row[_ACTOR1_COUNTRY] == "DEU" or row[_ACTOR2_COUNTRY] == "DEU" or row[_ACTION_COUNTRY] == "GM"


def _urls(start: date, end: date) -> list[str]:
    urls = []
    day = start
    while day <= end:
        for slot in range(96):
            stamp = datetime(day.year, day.month, day.day, slot // 4, 15 * (slot % 4))
            urls += [f"{SOURCE}{stamp:%Y%m%d%H%M%S}.{stream}" for stream in _STREAMS]
        day += timedelta(days=1)
    return urls


def collect(
    download: Callable[[str], bytes], start: date, end: date, classify: Classify, workers: int = 16
) -> bytes:
    """Every GDELT 2.0 event file from `start` to `end`, aggregated to EventCounts as CSV.

    Files GDELT never published (it has gaps) or that arrive damaged are counted, not fatal.
    """
    urls = _urls(start, end)
    totals: dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0.0])  # events, articles, weight, weight × tone
    missing = 0
    per_day = 96 * len(_STREAMS)  # downloaded a day at a time, so memory stays small on long ranges
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = (
            payload
            for first in range(0, len(urls), per_day)
            for payload in pool.map(lambda url: fetch_with_retries(download, url), urls[first : first + per_day])
        )
        for payload in payloads:
            if payload is None:
                missing += 1
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
            except zipfile.BadZipFile:
                missing += 1
                continue
            for line in text.splitlines():
                row = line.split("\t")
                if len(row) < _COLUMNS or not _is_germany(row):
                    continue
                day = date(int(row[_DATE_ADDED][:4]), int(row[_DATE_ADDED][4:6]), int(row[_DATE_ADDED][6:8]))
                path = words(urllib.parse.urlsplit(row[_SOURCE_URL]).path)
                names = words(row[_ACTOR1_NAME]) + [""] + words(row[_ACTOR2_NAME])
                articles, tone = int(row[_NUM_ARTICLES] or 0), float(row[_AVG_TONE] or 0.0)
                parties, topic = classify(names, path)
                for party in parties or [None]:
                    total = totals[(day, party, topic)]
                    total[0] += 1
                    total[1] += articles
                    total[2] += max(articles, 1)  # a record always counts, even if it names no article
                    total[3] += max(articles, 1) * tone

    out = io.StringIO()
    out.write(f"# files={len(urls) - missing} missing_files={missing}\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["day", "party", "topic", "events", "articles", "tone"])
    for (day, party, topic), (events, articles, weight, weighted) in sorted(
        totals.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2])
    ):
        writer.writerow([day.isoformat(), party or "", topic, events, articles, repr(weighted / weight)])
    return out.getvalue().encode("utf-8")


def parse(raw: bytes) -> list[EventCount]:
    lines = [line for line in raw.decode("utf-8").splitlines() if not line.startswith("#")]
    return [
        EventCount(
            day=date.fromisoformat(row["day"]),
            party=row["party"] or None,
            topic=row["topic"],
            events=int(row["events"]),
            articles=int(row["articles"]),
            tone=float(row["tone"]),
        )
        for row in csv.DictReader(lines)
    ]


def describe(raw: bytes) -> dict:
    header = raw.split(b"\n", 1)[0].decode("utf-8").lstrip("# ")
    return {key: int(value) for key, value in (item.split("=") for item in header.split())}


def gdelt_adapter(
    classify: Classify, name: str = "gdelt-events", fixture: Path = FIXTURE, fetch_command: str | None = None
) -> Adapter:
    """The GDELT adapter, aggregating by the parties and topics `classify` assigns."""
    return Adapter(
        name=name,
        source=SOURCE,
        licence=LICENCE,
        record_type=EventCount,
        parse=parse,
        fixture=fixture,
        suffix=".csv",
        describe=describe,
        fetch_command=fetch_command,
        collect=partial(collect, classify=classify),
    )


@dataclass(frozen=True)
class EventLog:
    """Every stored EventCount, and the day ranges that were fetched (a day with no records in a
    fetched range genuinely had no Events)."""

    records: list[EventCount]
    ranges: tuple[tuple[date, date], ...]

    def covers(self, first: date, last: date) -> bool:
        day = first
        while day <= last:
            if not any(start <= day <= end for start, end in self.ranges):
                return False
            day += timedelta(days=1)
        return True


def event_log(adapter: Adapter, data_dir: Path) -> EventLog:
    """Merge every stored range of `adapter`; where ranges overlap, the later download wins."""
    by_day: dict[date, list[EventCount]] = {}
    ranges = []
    for snapshot in snapshots(adapter, data_dir):
        start = date.fromisoformat(snapshot.metadata["range_start"])
        end = date.fromisoformat(snapshot.metadata["range_end"])
        ranges.append((start, end))
        day = start
        while day <= end:
            by_day[day] = []
            day += timedelta(days=1)
        for record in adapter.parse(snapshot.raw()):
            by_day.setdefault(record.day, []).append(record)
    return EventLog([r for day in sorted(by_day) for r in by_day[day]], tuple(ranges))
