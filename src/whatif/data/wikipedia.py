"""Daily Wikipedia page views per article, from the Wikimedia page-view API.

Which articles to fetch, and what each one stands for, is the caller's choice: it passes a table
from article title to subject, so each project keeps its own list.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path

from whatif.data.adapters import Adapter, Licence, fetch_with_retries, snapshots

LICENCE = Licence(
    name="CC0 1.0 (Wikimedia analytics datasets)",
    url="https://dumps.wikimedia.org/legal.html",
    checked_on=date(2026, 11, 10),
    attribution="Page views: Wikimedia Foundation, de.wikipedia.org (CC0)",
)
FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "wikimedia" / "pageviews_20250203-20250209.csv"


def source(project: str = "de.wikipedia") -> str:
    return f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"


@dataclass(frozen=True)
class PageViews:
    """Views of one Wikipedia article on one day, by people (not bots)."""

    day: date
    article: str
    subject: str  # what the article stands for, from the caller's table
    views: int


def collect(
    download: Callable[[str], bytes], start: date, end: date, articles: Mapping[str, str], project: str = "de.wikipedia"
) -> bytes:
    """Daily views of every article in `articles` from `start` to `end`, as CSV.

    An article with no data for the range (for example one created later) is skipped. Wikimedia
    rate-limits fast requests, so each download is retried with backoff.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["day", "article", "views"])
    for article in articles:
        url = f"{source(project)}{urllib.parse.quote(article, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}"
        payload = fetch_with_retries(download, url, attempts=6)
        if payload is None:
            continue
        for item in json.loads(payload)["items"]:
            stamp = item["timestamp"]
            writer.writerow([f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}", article, item["views"]])
    return out.getvalue().encode("utf-8")


def parse(raw: bytes, articles: Mapping[str, str]) -> list[PageViews]:
    """Read stored page views; articles no longer in `articles` are left out on purpose."""
    return [
        PageViews(date.fromisoformat(row["day"]), row["article"], articles[row["article"]], int(row["views"]))
        for row in csv.DictReader(raw.decode("utf-8").splitlines())
        if row["article"] in articles
    ]


def pageviews_adapter(
    articles: Mapping[str, str],
    name: str = "wikipedia-pageviews",
    project: str = "de.wikipedia",
    fixture: Path = FIXTURE,
    fetch_command: str | None = None,
) -> Adapter:
    """The page-view adapter for `articles` (title -> subject) on one Wikipedia `project`."""
    return Adapter(
        name=name,
        source=source(project),
        licence=LICENCE,
        record_type=PageViews,
        parse=partial(parse, articles=articles),
        fixture=fixture,
        suffix=".csv",
        fetch_command=fetch_command,
        collect=partial(collect, articles=articles, project=project),
    )


@dataclass(frozen=True)
class PageViewLog:
    records: list[PageViews]
    range: tuple[date, date]

    def covers(self, first: date, last: date) -> bool:
        return self.range[0] <= first and last <= self.range[1]


def page_view_log(adapter: Adapter, data_dir: Path) -> PageViewLog:
    """The latest stored page views of `adapter`, with the range they were fetched for."""
    snapshot = snapshots(adapter, data_dir)[-1]
    span = (date.fromisoformat(snapshot.metadata["range_start"]), date.fromisoformat(snapshot.metadata["range_end"]))
    return PageViewLog(adapter.parse(snapshot.raw()), span)
