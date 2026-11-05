"""National polls and the fixed party list."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from whatif.adapters import Adapter, Licence

# The fixed party list every Poll Average is expressed on. Any other party counts as Others.
PARTIES: tuple[str, ...] = ("CDU/CSU", "AfD", "SPD", "Greens", "Left", "BSW", "FDP", "Others")

# DAWUM party IDs for the parties we track by name. CDU (101) and CSU (102) only appear
# separately in some polls; nationally they count as one party.
_DAWUM_PARTY_IDS: dict[str, str] = {
    "1": "CDU/CSU",
    "101": "CDU/CSU",
    "102": "CDU/CSU",
    "7": "AfD",
    "2": "SPD",
    "4": "Greens",
    "5": "Left",
    "23": "BSW",
    "3": "FDP",
}

_DAWUM_BUNDESTAG_ID = "0"


TRACKED_PARTIES: tuple[str, ...] = PARTIES[:-1]


@dataclass(frozen=True)
class Poll:
    """One published national poll, with shares on the fixed party list summing to 100.

    `reported` holds the tracked parties the poll listed by name. A party it left out has
    share 0 here, but its real support is hidden inside Others, not absent.
    """

    institute: str
    published: date
    field_end: date
    sample_size: int | None
    shares: dict[str, float]
    reported: frozenset[str] = frozenset(TRACKED_PARTIES)


def parse_dawum(database: dict) -> list[Poll]:
    """Extract national (Bundestag) polls from a DAWUM database snapshot."""
    institutes = database.get("Institutes", {})
    polls = []
    for survey in database["Surveys"].values():
        if survey["Parliament_ID"] != _DAWUM_BUNDESTAG_ID:
            continue
        period = survey.get("Survey_Period") or {}
        published = date.fromisoformat(survey["Date"])
        field_end = date.fromisoformat(period["Date_End"]) if period.get("Date_End") else published
        institute = institutes.get(survey["Institute_ID"], {}).get("Name", survey["Institute_ID"])
        polls.append(
            Poll(
                institute=institute,
                published=published,
                field_end=field_end,
                sample_size=_parse_sample_size(survey.get("Surveyed_Persons")),
                shares=_map_shares(survey["Results"]),
                reported=frozenset(
                    _DAWUM_PARTY_IDS[i] for i in survey["Results"] if i in _DAWUM_PARTY_IDS
                ),
            )
        )
    return sorted(polls, key=lambda p: (p.field_end, p.institute))


def _parse_sample_size(raw) -> int | None:
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _map_shares(results: dict[str, float]) -> dict[str, float]:
    shares = dict.fromkeys(PARTIES, 0.0)
    for party_id, value in results.items():
        shares[_DAWUM_PARTY_IDS.get(party_id, "Others")] += float(value)
    total = sum(shares.values())
    if total <= 0:
        raise ValueError(f"Poll has no positive results: {results}")
    return {party: 100.0 * value / total for party, value in shares.items()}


def _describe_dawum(raw: bytes) -> dict:
    database = json.loads(raw.decode("utf-8"))
    return {"source_last_update": database.get("Database", {}).get("Last_Update")}


DAWUM = Adapter(
    name="dawum",
    source="https://api.dawum.de/",
    licence=Licence(
        name="ODC-ODbL",
        url="https://opendatacommons.org/licenses/odbl/1-0/",
        checked_on=date(2026, 11, 3),
        attribution="Daten von dawum.de (Open Database License (ODbL))",
    ),
    record_type=Poll,
    parse=lambda raw: parse_dawum(json.loads(raw.decode("utf-8"))),
    fixture=Path(__file__).parents[2] / "tests" / "fixtures" / "dawum_sample.json",
    describe=_describe_dawum,
    fetch_command="fetch-polls",
)
