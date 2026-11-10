"""Monthly inflation and unemployment for Germany: the economy as a Boundary series.

Both come from Eurostat, which publishes the official German figures in harmonised form:
inflation is the HICP annual rate of change (%), unemployment the seasonally adjusted
unemployment rate (% of the labour force).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from whatif.adapters import Adapter, Licence
from whatif.store import MissingDataError

_EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eurostat"


@dataclass(frozen=True)
class MonthlyValue:
    """One month of an economic indicator for Germany."""

    month: date  # the first day of the month
    value: float


def parse_eurostat(raw: bytes) -> list[MonthlyValue]:
    """Read a single Eurostat series (JSON-stat 2.0, every dimension but time fixed) by month."""
    data = json.loads(raw)
    times = data["dimension"]["time"]["category"]["index"]
    by_position = {position: label for label, position in times.items()}
    values = []
    for position, value in data["value"].items():
        year, month = by_position[int(position)].split("-")
        values.append(MonthlyValue(date(int(year), int(month), 1), float(value)))
    return sorted(values, key=lambda v: v.month)


def _describe(raw: bytes) -> dict:
    return {"source_last_update": json.loads(raw).get("updated")}


def _eurostat_licence(dataset: str) -> Licence:
    return Licence(
        name="CC BY 4.0 (European Commission reuse policy, Decision 2011/833/EU)",
        url="https://eur-lex.europa.eu/eli/dec/2011/833/oj",
        checked_on=date(2026, 11, 10),
        attribution=f"Source: Eurostat ({dataset})",
    )


INFLATION = Adapter(
    name="eurostat-inflation",
    source=f"{_EUROSTAT}prc_hicp_minr?geo=DE&coicop18=TOTAL&unit=RCH_A&format=JSON&lang=EN",
    licence=_eurostat_licence("prc_hicp_minr"),
    record_type=MonthlyValue,
    parse=parse_eurostat,
    fixture=_FIXTURES / "inflation_de.json",
    describe=_describe,
    fetch_command="fetch-economy",
)

UNEMPLOYMENT = Adapter(
    name="eurostat-unemployment",
    source=f"{_EUROSTAT}une_rt_m?geo=DE&s_adj=SA&age=TOTAL&sex=T&unit=PC_ACT&format=JSON&lang=EN",
    licence=_eurostat_licence("une_rt_m"),
    record_type=MonthlyValue,
    parse=parse_eurostat,
    fixture=_FIXTURES / "unemployment_de.json",
    describe=_describe,
    fetch_command="fetch-economy",
)

INDICATORS: tuple[str, ...] = ("inflation", "unemployment")


@dataclass(frozen=True)
class Economy:
    """Germany's monthly inflation and unemployment, as recorded."""

    inflation: list[MonthlyValue]
    unemployment: list[MonthlyValue]

    def weekly(self, start_week: date, horizon: int) -> np.ndarray:
        """Each indicator for the week before `start_week` and every Horizon week: [week][indicator].

        A week takes the value of the month its Monday falls in. Refuses when a month is missing.
        """
        mondays = [start_week + timedelta(weeks=w) for w in range(-1, horizon)]
        series = [{v.month: v.value for v in self.inflation}, {v.month: v.value for v in self.unemployment}]
        table = np.empty((len(mondays), len(INDICATORS)))
        for i, monday in enumerate(mondays):
            month = monday.replace(day=1)
            for j, (name, values) in enumerate(zip(INDICATORS, series)):
                if month not in values:
                    raise MissingDataError(
                        f"No {name} figure for {month:%Y-%m}. Run `whatif fetch-economy` "
                        "(Eurostat publishes each month about a month later)."
                    )
                table[i, j] = values[month]
        return table
