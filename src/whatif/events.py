"""Recorded Events (GDELT 2.0) and public attention (Wikipedia page views): Boundary series.

GDELT is terabytes in total, so it is only ever downloaded by date range: every 15-minute
export file in the range, in English and translated, filtered to Germany and aggregated by
day, party and topic before anything is stored. Page views are daily counts for a fixed list
of German Wikipedia articles, one per party or topic.

The mapping from a GDELT record to a party and a topic, and the list of articles, are
modelling choices that belong to the researcher. They are the visible tables below.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.error
import urllib.parse
import zipfile
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from whatif.adapters import Adapter, Licence, snapshots
from whatif.polls import TRACKED_PARTIES

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

# Every Event is filed under one topic. "other" is anything the keywords below don't place.
TOPICS: tuple[str, ...] = ("economy", "migration", "climate", "security", "social", "other")

# --- The researcher's mapping from a GDELT record to a party and a topic ----------------------
#
# A record involves a party when one of these phrases appears, as whole words, in either actor's
# name or in the source article's URL path. Umlauts are written out (ü → ue), and a few are also
# given with the umlaut dropped (ü → u), because GDELT and URLs do both. A record can involve
# several parties; it counts once for each.
PARTY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CDU/CSU": (
        "cdu", "csu", "christian democrat", "christian democrats", "christian democratic union",
        "merz", "merkel", "laschet", "kramp karrenbauer", "soeder", "soder",
    ),
    "AfD": (
        "afd", "alternative for germany", "alternative fuer deutschland", "alternative fur deutschland",
        "weidel", "chrupalla", "hoecke", "hocke", "gauland", "meuthen",
    ),
    "SPD": (
        "spd", "social democrat", "social democrats", "social democratic party",
        "scholz", "esken", "klingbeil", "nahles",
    ),
    "Greens": ("greens", "green party", "gruene", "gruenen", "grune", "grunen", "habeck", "baerbock"),
    "Left": ("linke", "linken", "left party", "wissler", "reichinnek", "gysi"),
    "BSW": ("bsw", "wagenknecht"),
    "FDP": ("fdp", "free democrat", "free democrats", "free democratic party", "lindner"),
}

# A record's topic comes from its source URL path only (actor names are GDELT's own coding and
# say little about the policy area). The first topic, in TOPICS order, with a matching word wins.
# A word ending in "*" matches any word it starts; the others must match a whole word.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "economy": (
        "econom*", "inflation*", "wirtschaft*", "unemploy*", "arbeitslos*", "recession*", "rezession*",
        "steuer*", "budget*", "haushalt*", "schulden*", "wage", "wages", "lohn", "loehne", "job", "jobs",
        "tax", "taxes", "debt", "price", "prices", "preise",
    ),
    "migration": (
        "migra*", "immigra*", "refugee*", "fluecht*", "asyl*", "deport*", "abschieb*", "border", "borders",
        "grenze", "grenzen",
    ),
    "climate": (
        "climat*", "klima*", "energ*", "heizung*", "heating", "emission*", "coal", "kohle*", "renewabl*",
        "akw", "atomkraft*", "nuclear",
    ),
    "security": (
        "ukrain*", "russia*", "russland*", "militar*", "bundeswehr*", "defen*", "verteidigung*", "nato",
        "terror*", "anschlag*", "attack*", "police*", "polizei*", "crime*", "security", "sicherheit*",
        "war", "krieg*",
    ),
    "social": (
        "pension*", "rente*", "health*", "gesundheit*", "pflege*", "rent", "miete*", "housing", "wohnung*",
        "welfare", "buergergeld*",
    ),
}

# --- GDELT 2.0 event records --------------------------------------------------------------------

_GDELT = "http://data.gdeltproject.org/gdeltv2/"
_STREAMS = ("export.CSV.zip", "translation.export.CSV.zip")  # English sources, translated sources
# Columns of a GDELT 2.0 event record (0-based).
_ACTOR1_NAME, _ACTOR1_COUNTRY, _ACTOR2_NAME, _ACTOR2_COUNTRY = 6, 7, 16, 17
_NUM_ARTICLES, _AVG_TONE, _ACTION_COUNTRY, _DATE_ADDED, _SOURCE_URL = 33, 34, 53, 59, 60
_COLUMNS = 61


@dataclass(frozen=True)
class EventCount:
    """Recorded Events for Germany on one day, for one party (or none) and one topic."""

    day: date  # the day GDELT recorded them
    party: str | None  # a tracked party, or None when the Events involve no party
    topic: str  # one of TOPICS
    events: int
    articles: int  # news articles reporting them
    tone: float  # article-weighted average tone (GDELT's AvgTone; negative is bad news)


def _words(text: str) -> list[str]:
    text = urllib.parse.unquote(text).lower()
    for umlaut, written in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, written)
    return re.findall(r"[a-z]+", text)


def _parties(names: list[str], path: list[str]) -> list[str]:
    text = " " + " ".join(names + path) + " "
    return [p for p, phrases in PARTY_KEYWORDS.items() if any(f" {phrase} " in text for phrase in phrases)]


def _topic(path: list[str]) -> str:
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.endswith("*"):
                if any(word.startswith(keyword[:-1]) for word in path):
                    return topic
            elif keyword in path:
                return topic
    return "other"


def _is_germany(row: list[str]) -> bool:
    return row[_ACTOR1_COUNTRY] == "DEU" or row[_ACTOR2_COUNTRY] == "DEU" or row[_ACTION_COUNTRY] == "GM"


def _gdelt_urls(start: date, end: date) -> list[str]:
    urls = []
    day = start
    while day <= end:
        for slot in range(96):
            stamp = datetime(day.year, day.month, day.day, slot // 4, 15 * (slot % 4))
            urls += [f"{_GDELT}{stamp:%Y%m%d%H%M%S}.{stream}" for stream in _STREAMS]
        day += timedelta(days=1)
    return urls


def _fetch_with_retries(download: Callable[[str], bytes], url: str, attempts: int = 4) -> bytes | None:
    """The download, or None when the source says it doesn't exist (404). Retries anything else."""
    for attempt in range(attempts):
        try:
            return download(url)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    return None


def collect_gdelt(download: Callable[[str], bytes], start: date, end: date, workers: int = 16) -> bytes:
    """Every GDELT 2.0 event file from `start` to `end`, aggregated to EventCounts as CSV.

    Files GDELT never published (it has gaps) are counted, not fatal.
    """
    urls = _gdelt_urls(start, end)
    totals: dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0.0])  # events, articles, weight, weight × tone
    missing = 0
    per_day = 96 * len(_STREAMS)  # downloaded a day at a time, so memory stays small on long ranges
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = (
            payload
            for first in range(0, len(urls), per_day)
            for payload in pool.map(lambda url: _fetch_with_retries(download, url), urls[first : first + per_day])
        )
        for payload in payloads:
            if payload is None:
                missing += 1
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
            except zipfile.BadZipFile:  # a damaged file counts as missing rather than ending a long fetch
                missing += 1
                continue
            for line in text.splitlines():
                row = line.split("\t")
                if len(row) < _COLUMNS or not _is_germany(row):
                    continue
                day = date(int(row[_DATE_ADDED][:4]), int(row[_DATE_ADDED][4:6]), int(row[_DATE_ADDED][6:8]))
                path = _words(urllib.parse.urlsplit(row[_SOURCE_URL]).path)
                names = _words(row[_ACTOR1_NAME]) + [""] + _words(row[_ACTOR2_NAME])
                articles, tone = int(row[_NUM_ARTICLES] or 0), float(row[_AVG_TONE] or 0.0)
                topic = _topic(path)
                for party in _parties(names, path) or [None]:
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


def parse_events(raw: bytes) -> list[EventCount]:
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


def _describe_events(raw: bytes) -> dict:
    header = raw.split(b"\n", 1)[0].decode("utf-8").lstrip("# ")
    return {key: int(value) for key, value in (item.split("=") for item in header.split())}


GDELT = Adapter(
    name="gdelt-events",
    source=_GDELT,
    licence=Licence(
        name="GDELT Project terms of use: unlimited and unrestricted use, with a citation",
        url="https://www.gdeltproject.org/about.html#termsofuse",
        checked_on=date(2026, 11, 10),
        attribution="Event data: The GDELT Project (https://www.gdeltproject.org/)",
    ),
    record_type=EventCount,
    parse=parse_events,
    fixture=_FIXTURES / "gdelt" / "events_20250210.csv",
    suffix=".csv",
    describe=_describe_events,
    fetch_command="fetch-events",
    collect=collect_gdelt,
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


def event_log(data_dir: Path) -> EventLog:
    """Merge every stored GDELT range; where ranges overlap, the later download wins."""
    by_day: dict[date, list[EventCount]] = {}
    ranges = []
    for snapshot in snapshots(GDELT, data_dir):
        start = date.fromisoformat(snapshot.metadata["range_start"])
        end = date.fromisoformat(snapshot.metadata["range_end"])
        ranges.append((start, end))
        day = start
        while day <= end:
            by_day[day] = []
            day += timedelta(days=1)
        for record in GDELT.parse(snapshot.raw()):
            by_day.setdefault(record.day, []).append(record)
    return EventLog([r for day in sorted(by_day) for r in by_day[day]], tuple(ranges))


# --- Wikipedia page views -------------------------------------------------------------------------

# The researcher's list of German Wikipedia articles whose daily views measure attention to a
# party or a topic. CDU and CSU are two articles for one party.
ATTENTION_ARTICLES: dict[str, str] = {
    "Christlich_Demokratische_Union_Deutschlands": "CDU/CSU",
    "Christlich-Soziale_Union_in_Bayern": "CDU/CSU",
    "Alternative_für_Deutschland": "AfD",
    "Sozialdemokratische_Partei_Deutschlands": "SPD",
    "Bündnis_90/Die_Grünen": "Greens",
    "Die_Linke": "Left",
    "Bündnis_Sahra_Wagenknecht": "BSW",
    "Freie_Demokratische_Partei": "FDP",
    "Inflation": "economy",
    "Arbeitslosigkeit": "economy",
    "Migration": "migration",
    "Asylrecht_(Deutschland)": "migration",
    "Globale_Erwärmung": "climate",
    "Klimawandel": "climate",
    "Bundeswehr": "security",
    "Kriminalität": "security",
    "Rente": "social",
}
assert set(ATTENTION_ARTICLES.values()) <= set(TRACKED_PARTIES) | set(TOPICS)

_PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/de.wikipedia/all-access/user/"


@dataclass(frozen=True)
class PageViews:
    """Views of one German Wikipedia article on one day, by people (not bots)."""

    day: date
    article: str
    subject: str  # the party or topic the article stands for
    views: int


def collect_pageviews(download: Callable[[str], bytes], start: date, end: date) -> bytes:
    """Daily views of every article in ATTENTION_ARTICLES from `start` to `end`, as CSV.

    An article with no data for the range (for example a party founded later) is skipped.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["day", "article", "views"])
    for article in ATTENTION_ARTICLES:
        url = f"{_PAGEVIEWS}{urllib.parse.quote(article, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}"
        payload = _fetch_with_retries(download, url, attempts=6)
        if payload is None:
            continue
        for item in json.loads(payload)["items"]:
            stamp = item["timestamp"]
            writer.writerow([f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}", article, item["views"]])
    return out.getvalue().encode("utf-8")


def parse_pageviews(raw: bytes) -> list[PageViews]:
    """Read stored page views; articles no longer in ATTENTION_ARTICLES are left out on purpose."""
    return [
        PageViews(date.fromisoformat(row["day"]), row["article"], ATTENTION_ARTICLES[row["article"]], int(row["views"]))
        for row in csv.DictReader(raw.decode("utf-8").splitlines())
        if row["article"] in ATTENTION_ARTICLES
    ]


PAGEVIEWS = Adapter(
    name="wikipedia-pageviews",
    source=_PAGEVIEWS,
    licence=Licence(
        name="CC0 1.0 (Wikimedia analytics datasets)",
        url="https://dumps.wikimedia.org/legal.html",
        checked_on=date(2026, 11, 10),
        attribution="Page views: Wikimedia Foundation, de.wikipedia.org (CC0)",
    ),
    record_type=PageViews,
    parse=parse_pageviews,
    fixture=_FIXTURES / "wikimedia" / "pageviews_20250203-20250209.csv",
    suffix=".csv",
    fetch_command="fetch-attention",
    collect=collect_pageviews,
)


@dataclass(frozen=True)
class PageViewLog:
    records: list[PageViews]
    range: tuple[date, date]

    def covers(self, first: date, last: date) -> bool:
        return self.range[0] <= first and last <= self.range[1]


def page_view_log(data_dir: Path) -> PageViewLog:
    """The latest stored page views, with the range they were fetched for."""
    snapshot = snapshots(PAGEVIEWS, data_dir)[-1]
    span = (date.fromisoformat(snapshot.metadata["range_start"]), date.fromisoformat(snapshot.metadata["range_end"]))
    return PageViewLog(PAGEVIEWS.parse(snapshot.raw()), span)
