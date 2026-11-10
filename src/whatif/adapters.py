"""The data adapter contract: how every data source enters the local store.

See docs/adapter-contract.md for the written contract and the review checklist.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from whatif.store import MissingDataError, Snapshot


class LicenceNotRecordedError(RuntimeError):
    """Raised when an adapter is asked to fetch before its licence has been recorded."""


@dataclass(frozen=True)
class Licence:
    """The reuse terms of a source, as checked by a person on a given date."""

    name: str
    url: str
    checked_on: date
    attribution: str | None = None


@dataclass(frozen=True)
class Adapter:
    """One data source: where it comes from, under what licence, and how to read it."""

    name: str
    source: str
    licence: Licence | None
    record_type: type
    parse: Callable[[bytes], list]
    fixture: Path
    suffix: str = ".json"
    describe: Callable[[bytes], dict] | None = None  # extra metadata read from the download
    fetch_command: str | None = None  # the `whatif` command that fetches this source
    # For large sources fetched by date range: builds the stored bytes from many downloads
    # between two days (inclusive), using the `download` it is given.
    collect: Callable[[Callable[[str], bytes], date, date], bytes] | None = None


USER_AGENT = "WhatIf/0.1 (research prototype)"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch(
    adapter: Adapter,
    data_dir: Path,
    download: Callable[[str], bytes] = _download,
    start: date | None = None,
    end: date | None = None,
) -> Snapshot:
    """Download the source into the store, recording where, when and under what licence.

    An adapter with `collect` is fetched by date range only: `start` and `end` are required and
    recorded with the snapshot.
    """
    if adapter.licence is None:
        raise LicenceNotRecordedError(
            f"{adapter.name}: no licence recorded for {adapter.source}; check its reuse terms first."
        )
    if adapter.collect is not None:
        if start is None or end is None or end < start:
            raise ValueError(f"{adapter.name} is fetched by date range: give a start and an end day.")
        raw = adapter.collect(download, start, end)
    else:
        raw = download(adapter.source)
    downloaded_at = datetime.now(timezone.utc)

    source_dir = data_dir / adapter.name
    source_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{downloaded_at:%Y%m%dT%H%M%SZ}"
    if adapter.collect is not None:
        stamp += f"_{start:%Y%m%d}-{end:%Y%m%d}"
    path = source_dir / f"{stamp}{adapter.suffix}"
    path.write_bytes(raw)

    licence = adapter.licence
    metadata = {
        "source": adapter.source,
        "licence": licence.name,
        "licence_url": licence.url,
        "licence_checked_on": licence.checked_on.isoformat(),
        "attribution": licence.attribution,
        "downloaded_at": downloaded_at.isoformat(),
    }
    if adapter.collect is not None:
        metadata |= {"range_start": start.isoformat(), "range_end": end.isoformat()}
    if adapter.describe is not None:
        metadata |= adapter.describe(raw)
    _meta_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return Snapshot(path, metadata)


def latest(adapter: Adapter, data_dir: Path) -> Snapshot:
    """The most recent snapshot of the source in the store; never touches the network."""
    return snapshots(adapter, data_dir)[-1]


def snapshots(adapter: Adapter, data_dir: Path) -> list[Snapshot]:
    """Every snapshot of the source in the store, oldest first; never touches the network."""
    paths = sorted(
        p for p in (data_dir / adapter.name).glob(f"*{adapter.suffix}") if not p.name.endswith(".meta.json")
    )
    if not paths:
        how = f"Run `whatif {adapter.fetch_command}` first." if adapter.fetch_command else "Fetch it first."
        raise MissingDataError(f"No {adapter.name} snapshot in {data_dir}. {how}")
    return [Snapshot(p, json.loads(_meta_path(p).read_text(encoding="utf-8"))) for p in paths]


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(path.suffix) + ".meta.json")
