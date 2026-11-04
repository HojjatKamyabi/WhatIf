"""Saving Backtest results as self-contained files."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from whatif.backtest import BacktestResult
from whatif.polls import PARTIES


def save_result(result: BacktestResult, results_dir: Path) -> Path:
    """Write a Backtest result to its own JSON file and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    path = results_dir / (
        f"backtest-{result.config.start_week}-seed{result.config.seed}-{created:%Y%m%dT%H%M%SZ}.json"
    )
    payload = {"created_at": created.isoformat(), "parties": list(PARTIES), **asdict(result)}
    path.write_text(json.dumps(payload, indent=2, default=_encode), encoding="utf-8")
    return path


def _encode(value):
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}")
