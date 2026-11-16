"""Scenarios: named sets of dated interventions under which the Twin is run.

A Scenario changes the Boundary the Twin sees: it injects Events (as an Event Shock for a party
and topic in a given week) or overrides an economic indicator for a range of dates. Reaction
strengths are not part of a Scenario; they belong to the run, so a Scenario and "no
intervention" are always compared under the same Voter rules. Forced Party Actor moves arrive
with the Party Actors (issue 08).

Scenarios are written in a TOML file (default `scenarios.toml`):

    [scenario.party-b-scandal]
    description = "A scandal hits Party B in week 3"

    [[scenario.party-b-scandal.event]]
    week = 3            # Horizon week, 1 = the start week; or `date = 2024-06-05`
    party = "AfD"
    topic = "other"     # one of: economy, migration, climate, security, social, other
    shock = 20.0        # Event Shock: positive is bad news, negative good news

    [[scenario.party-b-scandal.indicator]]
    indicator = "unemployment"   # or "inflation"
    from = 2024-06-03
    until = 2024-06-23
    value = 6.0
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from whatif.boundary import WeeklyBoundary
from whatif.economy import INDICATORS
from whatif.events import TOPICS
from whatif.polls import TRACKED_PARTIES


class ScenarioError(ValueError):
    """A Scenario that is badly written, unknown, or can't apply to the Backtest asked for."""


@dataclass(frozen=True)
class InjectedEvent:
    party: str
    topic: str
    shock: float  # Event Shock: positive is bad news for the party, negative good news
    week: int | None = None  # Horizon week, 1 = the start week
    day: date | None = None  # or the day it happens

    def week_index(self, start_week: date) -> int:
        return self.week - 1 if self.week is not None else (self.day - start_week).days // 7

    def describe(self) -> dict:
        when = {"week": self.week} if self.week is not None else {"date": self.day.isoformat()}
        return when | {"party": self.party, "topic": self.topic, "shock": self.shock}


@dataclass(frozen=True)
class IndicatorOverride:
    indicator: str  # one of INDICATORS
    start: date
    end: date  # inclusive
    value: float

    def describe(self) -> dict:
        return {
            "indicator": self.indicator,
            "from": self.start.isoformat(),
            "until": self.end.isoformat(),
            "value": self.value,
        }


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    events: tuple[InjectedEvent, ...] = ()
    indicators: tuple[IndicatorOverride, ...] = ()

    @property
    def injects_events(self) -> bool:
        return bool(self.events)

    @property
    def overrides_indicators(self) -> bool:
        return bool(self.indicators)

    def apply(self, boundary: WeeklyBoundary, start_week: date) -> WeeklyBoundary:
        """The Boundary as this Scenario has it. Refuses interventions outside the Horizon."""
        if not self.events and not self.indicators:
            return boundary
        horizon = boundary.horizon
        last_day = start_week + timedelta(weeks=horizon) - timedelta(days=1)

        shock = boundary.event_shock.copy()
        for event in self.events:
            week = event.week_index(start_week)
            if not 0 <= week < horizon:
                raise ScenarioError(
                    f"Scenario {self.name!r} injects an Event outside the Horizon "
                    f"({start_week} to {last_day}): {event.describe()}"
                )
            shock[week, list(TRACKED_PARTIES).index(event.party), TOPICS.index(event.topic)] += event.shock

        economy = boundary.economy.copy()
        mondays = [start_week + timedelta(weeks=w) for w in range(horizon)]
        for override in self.indicators:
            weeks = [w for w, monday in enumerate(mondays) if override.start <= monday <= override.end]
            if not weeks:
                raise ScenarioError(
                    f"Scenario {self.name!r} overrides {override.indicator} outside the Horizon "
                    f"({start_week} to {last_day}): {override.describe()}"
                )
            economy[np.array(weeks) + 1, INDICATORS.index(override.indicator)] = override.value  # row 0: week before

        return replace(boundary, event_shock=shock, economy=economy)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "events": [e.describe() for e in self.events],
            "indicators": [i.describe() for i in self.indicators],
        }


NO_INTERVENTION = Scenario("no-intervention", "Nothing is changed: the Boundary as it was recorded.")

_EVENT_KEYS = {"week", "date", "party", "topic", "shock"}
_INDICATOR_KEYS = {"indicator", "from", "until", "value"}


def load_scenarios(path: Path) -> dict[str, Scenario]:
    """Every Scenario in the file, plus "no intervention". A missing file means only the default."""
    scenarios = {NO_INTERVENTION.name: NO_INTERVENTION}
    if not path.is_file():
        return scenarios
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ScenarioError(f"{path} is not valid TOML: {error}") from error
    for name, entry in table.get("scenario", {}).items():
        if name in scenarios:
            raise ScenarioError(f"{path}: {name!r} is built in and can't be redefined.")
        scenarios[name] = Scenario(
            name=name,
            description=str(entry.get("description", "")),
            events=tuple(_event(name, e) for e in entry.get("event", [])),
            indicators=tuple(_indicator(name, i) for i in entry.get("indicator", [])),
        )
    return scenarios


def scenario_named(path: Path, name: str) -> Scenario:
    scenarios = load_scenarios(path)
    if name not in scenarios:
        raise ScenarioError(f"No Scenario {name!r} in {path}. Known: {', '.join(scenarios)}.")
    return scenarios[name]


def _check_keys(scenario: str, kind: str, entry: dict, allowed: set[str]) -> None:
    unknown = set(entry) - allowed
    if unknown:
        raise ScenarioError(f"Scenario {scenario!r}: unknown {kind} field(s) {sorted(unknown)}.")


def _event(scenario: str, entry: dict) -> InjectedEvent:
    _check_keys(scenario, "event", entry, _EVENT_KEYS)
    party, topic = entry.get("party"), entry.get("topic")
    if party not in TRACKED_PARTIES:
        raise ScenarioError(f"Scenario {scenario!r}: party {party!r} is not one of {', '.join(TRACKED_PARTIES)}.")
    if topic not in TOPICS:
        raise ScenarioError(f"Scenario {scenario!r}: topic {topic!r} is not one of {', '.join(TOPICS)}.")
    week, day = entry.get("week"), entry.get("date")
    if (week is None) == (day is None):
        raise ScenarioError(f"Scenario {scenario!r}: an event needs either a week or a date, not both.")
    if week is not None and (not isinstance(week, int) or week < 1):
        raise ScenarioError(f"Scenario {scenario!r}: week must be a whole number from 1, got {week!r}.")
    if day is not None and not isinstance(day, date):
        raise ScenarioError(f"Scenario {scenario!r}: date must be written like 2024-06-05, got {day!r}.")
    if not isinstance(entry.get("shock"), int | float):
        raise ScenarioError(f"Scenario {scenario!r}: an event needs a numeric shock (positive is bad news).")
    return InjectedEvent(party=party, topic=topic, shock=float(entry["shock"]), week=week, day=day)


def _indicator(scenario: str, entry: dict) -> IndicatorOverride:
    _check_keys(scenario, "indicator", entry, _INDICATOR_KEYS)
    indicator = entry.get("indicator")
    if indicator not in INDICATORS:
        raise ScenarioError(f"Scenario {scenario!r}: indicator {indicator!r} is not one of {', '.join(INDICATORS)}.")
    start, end = entry.get("from"), entry.get("until")
    if not isinstance(start, date) or not isinstance(end, date) or end < start:
        raise ScenarioError(f"Scenario {scenario!r}: an indicator override needs dates `from` <= `until`.")
    if not isinstance(entry.get("value"), int | float):
        raise ScenarioError(f"Scenario {scenario!r}: an indicator override needs a numeric value.")
    return IndicatorOverride(indicator=indicator, start=start, end=end, value=float(entry["value"]))


def require_reactions(scenario: Scenario, parameters) -> None:
    """Refuse a Scenario whose interventions no Voter would react to: it would silently change nothing."""
    missing = []
    if scenario.injects_events and not parameters.event_reaction:
        missing.append("event_reaction")
    if scenario.overrides_indicators and not (parameters.inflation_reaction or parameters.unemployment_reaction):
        missing.append("inflation_reaction or unemployment_reaction")
    if missing:
        raise ScenarioError(
            f"Scenario {scenario.name!r} changes the Boundary, but Voters don't react to it: "
            f"{' and '.join(missing)} is 0 until calibrated. Run with --first-guess-reactions to explore it."
        )
