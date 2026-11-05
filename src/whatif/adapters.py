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


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch(adapter: Adapter, data_dir: Path, download: Callable[[str], bytes] = _download) -> Snapshot:
    """Download the source into the store, recording where, when and under what licence."""
    if adapter.licence is None:
        raise LicenceNotRecordedError(
            f"{adapter.name}: no licence recorded for {adapter.source}; check its reuse terms first."
        )
    raw = download(adapter.source)
    downloaded_at = datetime.now(timezone.utc)

    source_dir = data_dir / adapter.name
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{downloaded_at:%Y%m%dT%H%M%SZ}{adapter.suffix}"
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
    if adapter.describe is not None:
        metadata |= adapter.describe(raw)
    _meta_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return Snapshot(path, metadata)


def latest(adapter: Adapter, data_dir: Path) -> Snapshot:
    """The most recent snapshot of the source in the store; never touches the network."""
    snapshots = sorted(
        p for p in (data_dir / adapter.name).glob(f"*{adapter.suffix}") if not p.name.endswith(".meta.json")
    )
    if not snapshots:
        how = f"Run `whatif {adapter.fetch_command}` first." if adapter.fetch_command else "Fetch it first."
        raise MissingDataError(f"No {adapter.name} snapshot in {data_dir}. {how}")
    path = snapshots[-1]
    return Snapshot(path, json.loads(_meta_path(path).read_text(encoding="utf-8")))


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(path.suffix) + ".meta.json")
