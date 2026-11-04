"""The local data store: downloaded once, read offline by every Backtest."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DAWUM_URL = "https://api.dawum.de/"
DAWUM_LICENCE = "ODC-ODbL"
DAWUM_ATTRIBUTION = "Daten von dawum.de (Open Database License (ODbL))"


class MissingDataError(RuntimeError):
    """Raised when the data a Backtest needs is not in the local store."""


@dataclass(frozen=True)
class Snapshot:
    """One downloaded copy of a source, with where and when it came from."""

    path: Path
    metadata: dict

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return self.path.stem


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch_dawum(data_dir: Path, fetch: Callable[[str], bytes] = _download) -> Snapshot:
    """Download the DAWUM poll database into the store and record its provenance."""
    raw = fetch(DAWUM_URL)
    database = json.loads(raw.decode("utf-8"))
    downloaded_at = datetime.now(timezone.utc)

    source_dir = data_dir / "dawum"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{downloaded_at:%Y%m%dT%H%M%SZ}.json"
    path.write_bytes(raw)

    metadata = {
        "source": DAWUM_URL,
        "licence": DAWUM_LICENCE,
        "attribution": DAWUM_ATTRIBUTION,
        "downloaded_at": downloaded_at.isoformat(),
        "source_last_update": database.get("Database", {}).get("Last_Update"),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return Snapshot(path, metadata)


def latest_dawum(data_dir: Path) -> Snapshot:
    """The most recent DAWUM snapshot in the store."""
    snapshots = sorted(p for p in (data_dir / "dawum").glob("*.json") if not p.name.endswith(".meta.json"))
    if not snapshots:
        raise MissingDataError(f"No DAWUM poll snapshot in {data_dir}. Run `whatif fetch-polls` first.")
    path = snapshots[-1]
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return Snapshot(path, metadata)
