"""The local data store: downloaded once, read offline by every Backtest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class MissingDataError(RuntimeError):
    """Raised when the data a Backtest needs is not in the local store."""


@dataclass(frozen=True)
class Snapshot:
    """One downloaded copy of a source, with where and when it came from."""

    path: Path
    metadata: dict

    def raw(self) -> bytes:
        return self.path.read_bytes()

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return self.path.stem
