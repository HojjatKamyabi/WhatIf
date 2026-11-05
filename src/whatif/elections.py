"""Federal election results and Constituency structural data (Federal Returning Officer open data)."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from whatif.adapters import Adapter, Licence
from whatif.polls import PARTIES
from whatif.store import MissingDataError

# Party names in the results files, mapped onto the fixed party list. Anything else is Others.
_PARTY_NAMES: dict[str, str] = {
    "CDU": "CDU/CSU",
    "CSU": "CDU/CSU",
    "AfD": "AfD",
    "SPD": "SPD",
    "GRÜNE": "Greens",
    "Die Linke": "Left",
    "DIE LINKE": "Left",
    "BSW": "BSW",
    "FDP": "FDP",
}

_PARTY_LIST_VOTE = "2"  # "Stimme" 2 is the party-list vote (Zweitstimme)


@dataclass(frozen=True)
class ConstituencyResult:
    """One Constituency's party-list vote in one federal election, on the fixed party list."""

    election_day: date
    constituency: int
    name: str
    eligible: int
    voted: int
    valid: int
    votes: dict[str, int]


def parse_results(raw: bytes) -> list[ConstituencyResult]:
    """Read a results-by-Constituency file ("kerg2.csv") into one result per Constituency."""
    lines = raw.decode("utf-8-sig").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Wahlart;"))
    rows = csv.DictReader(io.StringIO("\n".join(lines[header:])), delimiter=";")

    found: dict[int, dict] = {}
    for row in rows:
        if row["Gebietsart"] != "Wahlkreis":
            continue
        number = int(row["Gebietsnummer"])
        entry = found.setdefault(
            number,
            {
                "election_day": datetime.strptime(row["Wahltag"], "%d.%m.%Y").date(),
                "name": row["Gebietsname"],
                "votes": dict.fromkeys(PARTIES, 0),
            },
        )
        group, vote, count = row["Gruppenname"], row["Stimme"], _count(row["Anzahl"])
        if row["Gruppenart"] == "System-Gruppe":
            if group == "Wahlberechtigte":
                entry["eligible"] = count
            elif group in ("Wählende", "Wähler"):
                entry["voted"] = count
            elif group == "Gültige" and vote == _PARTY_LIST_VOTE:
                entry["valid"] = count
        elif vote == _PARTY_LIST_VOTE:
            entry["votes"][_PARTY_NAMES.get(group, "Others")] += count

    return [ConstituencyResult(constituency=number, **entry) for number, entry in sorted(found.items())]


def _count(text: str) -> int:
    return int(text) if text else 0


_AGE_BANDS: dict[str, str] = {
    "unter 18": "under 18",
    "18-24": "18-24",
    "25-34": "25-34",
    "35-59": "35-59",
    "60-74": "60-74",
    "75 und mehr": "75+",
}
_LAST_CONSTITUENCY = 299  # higher numbers are state (9xx) and national (999) totals


@dataclass(frozen=True)
class ConstituencyStructure:
    """Official structural statistics for one Constituency, as published for one federal election."""

    constituency: int
    name: str
    land: str
    population_density: float  # inhabitants per km²
    age_shares: dict[str, float]  # % of all inhabitants per age band
    disposable_income: float  # EUR per inhabitant per year
    unemployment_rate: float  # %


def parse_structure(raw: bytes) -> list[ConstituencyStructure]:
    """Read a structural-data file ("strukturdaten.csv"), matching columns by name, not position."""
    text = _decode(raw)
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Land;"))
    reader = csv.reader(io.StringIO("\n".join(lines[header:])), delimiter=";")
    columns = next(reader)

    def column(pattern: str) -> int:
        return next(i for i, name in enumerate(columns) if re.search(pattern, name))

    density = column(r"^Bevölkerungsdichte")
    income = column(r"^Verfügbares Einkommen der privaten Haushalte")
    unemployment = column(r"^Arbeitslosenquote .* - insgesamt")
    ages = {band: column(rf"^Alter von .* - {re.escape(label)} \(%\)") for label, band in _AGE_BANDS.items()}

    parsed = []
    for row in reader:
        if len(row) < len(columns) - 1 or not row[1].isdigit() or int(row[1]) > _LAST_CONSTITUENCY:
            continue
        parsed.append(
            ConstituencyStructure(
                constituency=int(row[1]),
                name=row[2],
                land=row[0],
                population_density=_number(row[density]),
                age_shares={band: _number(row[i]) for band, i in ages.items()},
                disposable_income=_number(row[income]),
                unemployment_rate=_number(row[unemployment]),
            )
        )
    return parsed


def _decode(raw: bytes) -> str:
    """The files are UTF-8 with BOM, except older ones in CP 1252."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _number(text: str) -> float:
    return float(text.replace(",", "."))


# --- The registered elections -------------------------------------------------------------

_BASE = "https://www.bundeswahlleiterin.de/dam/jcr"
_FIXTURES = Path(__file__).parents[2] / "tests" / "fixtures" / "bundeswahlleiterin"

# Licence checked 2026-11-05 and signed off by the researcher. Every file states Datenlizenz
# Deutschland – Namensnennung 2.0; the publisher's imprint (bundeswahlleiterin.de/info/impressum.html)
# allows reuse, including commercial, with "Die Bundeswahlleiterin" named as the source and any
# changes marked. In the structural data, columns 35–48 rest on raw data © Bundesagentur für Arbeit.
_DL_DE_BY_2 = dict(
    name="Datenlizenz Deutschland – Namensnennung – Version 2.0",
    url="https://www.govdata.de/dl-de/by-2-0",
    checked_on=date(2026, 11, 5),
)
_RESULTS_LICENCE = Licence(
    **_DL_DE_BY_2,
    attribution="Quelle: Die Bundeswahlleiterin, Wiesbaden; Daten als Berechnungsgrundlage verwendet",
)
_STRUCTURE_LICENCE = Licence(
    **_DL_DE_BY_2,
    attribution=(
        "Quelle: Die Bundeswahlleiterin, Wiesbaden (Strukturdaten der Wahlkreise; Rohdaten zu "
        "Beschäftigung und Arbeitslosigkeit © Bundesagentur für Arbeit); Daten als Berechnungsgrundlage verwendet"
    ),
)


@dataclass(frozen=True)
class FederalElection:
    """One federal election: its results and the structural data published for its Constituencies."""

    day: date
    results: Adapter
    structure: Adapter


def _election(day: date, results_url: str, structure_url: str) -> FederalElection:
    year = day.year
    return FederalElection(
        day=day,
        results=Adapter(
            name=f"election-results-{year}",
            source=results_url,
            licence=_RESULTS_LICENCE,
            record_type=ConstituencyResult,
            parse=parse_results,
            fixture=_FIXTURES / f"kerg2_{year}.csv",
            suffix=".csv",
            fetch_command="fetch-elections",
        ),
        structure=Adapter(
            name=f"constituency-structure-{year}",
            source=structure_url,
            licence=_STRUCTURE_LICENCE,
            record_type=ConstituencyStructure,
            parse=parse_structure,
            fixture=_FIXTURES / f"strukturdaten_{year}.csv",
            suffix=".csv",
            fetch_command="fetch-elections",
        ),
    )


ELECTIONS: tuple[FederalElection, ...] = (
    _election(
        date(2017, 9, 24),
        f"{_BASE}/0d1ea773-f3ca-40ea-b8ff-b031712707e1/btw17_kerg2.csv",
        f"{_BASE}/f7566722-a528-4b18-bea3-ea419371e300/btw2017_strukturdaten.csv",
    ),
    _election(
        date(2021, 9, 26),
        # Final results including the repeat election in parts of Berlin on 11 Feb 2024.
        f"{_BASE}/860495c9-83fb-4068-8a99-c1c985ffffd2/w-btw21_kerg2.csv",
        f"{_BASE}/b1d3fc4f-17eb-455f-a01c-a0bf32135c5d/btw21_strukturdaten.csv",
    ),
    _election(
        date(2025, 2, 23),
        "https://www.bundeswahlleiterin.de/bundestagswahlen/2025/ergebnisse/opendata/btw25/csv/kerg2.csv",
        f"{_BASE}/181f9e38-38db-4f64-991c-8141dfa0f2cb/btw2025_strukturdaten.csv",
    ),
)


def election_before(start_week: date) -> FederalElection:
    """The last registered federal election held before a Backtest's start week."""
    earlier = [e for e in ELECTIONS if e.day < start_week]
    if not earlier:
        raise MissingDataError(f"No federal election registered before {start_week}.")
    return earlier[-1]
