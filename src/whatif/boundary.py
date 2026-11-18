"""The Boundary as the Twin sees it: recorded series lined up week by week over a Horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from whatif.actors import EventSummary
from whatif.data.eurostat import Economy
from whatif.events import PARTY_FOUNDED, TOPICS, EventLog, PageViewLog
from whatif.data.polls import PARTIES
from whatif.data.store import MissingDataError

# Federal cabinets since 2013, by the day they took office: the parties Voters hold responsible
# for the economy. The FDP left the cabinet on 2024-11-07.
GOVERNMENTS: tuple[tuple[date, frozenset[str]], ...] = (
    (date(2013, 12, 17), frozenset({"CDU/CSU", "SPD"})),
    (date(2021, 12, 8), frozenset({"SPD", "Greens", "FDP"})),
    (date(2024, 11, 7), frozenset({"SPD", "Greens"})),
    (date(2025, 5, 6), frozenset({"CDU/CSU", "SPD"})),
)


def governing_parties(day: date) -> frozenset[str]:
    current = [parties for since, parties in GOVERNMENTS if since <= day]
    if not current:
        raise ValueError(f"No federal cabinet recorded for {day}.")
    return current[-1]


@dataclass(frozen=True)
class WeeklyBoundary:
    """Everything outside the Voters, for each week of a Horizon."""

    economy: np.ndarray  # [week][indicator]: the week before the start, then each Horizon week
    government: tuple[frozenset[str], ...]  # the governing parties in each Horizon week
    event_shock: np.ndarray  # [week][party][topic]: negative coverage above the party's usual level
    party_attention: np.ndarray  # [week][party]: page views relative to the usual level (1 = usual)
    topic_attention: np.ndarray  # [week][topic]: likewise, per topic
    # the week's Events as structured fields, for Party Actors; empty when no Events were recorded
    events: tuple[tuple[EventSummary, ...], ...] = field(default=())

    def events_in(self, week: int) -> tuple[EventSummary, ...]:
        return self.events[week] if week < len(self.events) else ()

    @property
    def horizon(self) -> int:
        return len(self.government)

    @classmethod
    def quiet(
        cls, horizon: int, government: frozenset[str], economy: tuple[float, float] = (2.0, 3.5)
    ) -> WeeklyBoundary:
        """A Horizon in which nothing happens: flat economy, usual coverage and attention."""
        return cls(
            economy=np.tile(np.asarray(economy, dtype=float), (horizon + 1, 1)),
            government=(government,) * horizon,
            event_shock=np.zeros((horizon, len(PARTIES), len(TOPICS))),
            party_attention=np.ones((horizon, len(PARTIES))),
            topic_attention=np.ones((horizon, len(TOPICS))),
        )


def build_boundary(
    start_week: date,
    horizon: int,
    economy: Economy,
    events: EventLog,
    views: PageViewLog,
    baseline_weeks: int,
) -> WeeklyBoundary:
    """Line up every Boundary series for a Backtest; refuses if any is missing for its period."""
    party_attention, topic_attention = attention(views, start_week, horizon, baseline_weeks)
    return WeeklyBoundary(
        economy=economy.weekly(start_week, horizon),
        government=tuple(governing_parties(start_week + timedelta(weeks=w)) for w in range(horizon)),
        event_shock=event_shock(events, start_week, horizon, baseline_weeks),
        party_attention=party_attention,
        topic_attention=topic_attention,
        events=event_summaries(events, start_week, horizon, baseline_weeks),
    )


def _window(start_week: date, horizon: int, baseline_weeks: int) -> tuple[date, date]:
    return start_week - timedelta(weeks=baseline_weeks), start_week + timedelta(weeks=horizon) - timedelta(days=1)


def _week_index(day: date, start_week: date) -> int:
    return (day - start_week).days // 7


def event_shock(log: EventLog, start_week: date, horizon: int, baseline_weeks: int) -> np.ndarray:
    """Each party's news coverage per topic and Horizon week, beyond its usual level.

    Bad news is negative coverage (articles × −tone, over records with negative tone) above the
    weekly average in the `baseline_weeks` before the start; good news is positive coverage above
    its usual level, and counts as a negative shock. A quiet week is neither: less bad news than
    usual is not good news. Coverage the Poll Average already reflects doesn't count again. The
    excess is divided by the party's usual weekly article count (at least 1), so it reads as tone
    points on the party's normal volume. Records about a party before it was founded are ignored.
    """
    first, last = _window(start_week, horizon, baseline_weeks)
    if not log.covers(first, last):
        raise MissingDataError(
            f"Recorded Events don't cover {first} to {last}. Run `whatif fetch-events` for that range."
        )
    weeks = baseline_weeks + horizon
    bad, good = np.zeros((weeks, len(PARTIES), len(TOPICS))), np.zeros((weeks, len(PARTIES), len(TOPICS)))
    volume = np.zeros((weeks, len(PARTIES)))
    for record in log.records:
        if record.party is None or not first <= record.day <= last:
            continue
        if record.day < PARTY_FOUNDED.get(record.party, date.min):
            continue
        week = _week_index(record.day, start_week) + baseline_weeks
        party, topic = PARTIES.index(record.party), TOPICS.index(record.topic)
        bad[week, party, topic] += record.articles * max(-record.tone, 0.0)
        good[week, party, topic] += record.articles * max(record.tone, 0.0)
        volume[week, party] += record.articles
    excess_bad = np.maximum(bad[baseline_weeks:] - bad[:baseline_weeks].mean(axis=0), 0.0)
    excess_good = np.maximum(good[baseline_weeks:] - good[:baseline_weeks].mean(axis=0), 0.0)
    usual_volume = np.maximum(volume[:baseline_weeks].mean(axis=0), 1.0)
    return (excess_bad - excess_good) / usual_volume[None, :, None]


def event_summaries(
    log: EventLog, start_week: date, horizon: int, baseline_weeks: int
) -> tuple[tuple[EventSummary, ...], ...]:
    """Each Horizon week's Events per party and topic, as the structured fields Party Actors see.

    Size is the week's articles relative to the party's usual weekly count over the
    `baseline_weeks` before the start (1 = a usual week); tone is the article-weighted average.
    Only records that involve a party count, and none before the party was founded. Nothing of a
    record's source (names, headlines, links) is kept.
    """
    first, last = _window(start_week, horizon, baseline_weeks)
    if not log.covers(first, last):
        raise MissingDataError(
            f"Recorded Events don't cover {first} to {last}. Run `whatif fetch-events` for that range."
        )
    usual = {p: 0.0 for p in PARTIES}
    weekly: dict[tuple[int, str, str], list[float]] = {}  # (week, party, topic) -> [articles, articles × tone]
    for record in log.records:
        if record.party is None or not first <= record.day <= last:
            continue
        if record.day < PARTY_FOUNDED.get(record.party, date.min):
            continue
        week = _week_index(record.day, start_week)
        if week < 0:
            usual[record.party] += record.articles / baseline_weeks
            continue
        totals = weekly.setdefault((week, record.party, record.topic), [0.0, 0.0])
        totals[0] += record.articles
        totals[1] += record.articles * record.tone
    return tuple(
        tuple(
            EventSummary(
                party=party,
                topic=topic,
                tone=round(weighted / articles, 3) if articles else 0.0,
                size=round(articles / max(usual[party], 1.0), 3),
            )
            for (w, party, topic), (articles, weighted) in sorted(weekly.items())
            if w == week
        )
        for week in range(horizon)
    )


def attention(log: PageViewLog, start_week: date, horizon: int, baseline_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """Page views per party and per topic in each Horizon week, relative to the usual weekly level.

    A party or topic with no views before the start (an article that didn't exist yet) is 1.
    """
    first, last = _window(start_week, horizon, baseline_weeks)
    if not log.covers(first, last):
        raise MissingDataError(
            f"Page views don't cover {first} to {last}. Run `whatif fetch-attention` for that range."
        )
    subjects = PARTIES + TOPICS
    weekly = np.zeros((baseline_weeks + horizon, len(subjects)))
    for record in log.records:
        if first <= record.day <= last:
            weekly[_week_index(record.day, start_week) + baseline_weeks, subjects.index(record.subject)] += record.views
    usual = weekly[:baseline_weeks].mean(axis=0)
    ratio = np.where(usual > 0, weekly[baseline_weeks:] / np.where(usual > 0, usual, 1.0), 1.0)
    return ratio[:, : len(PARTIES)], ratio[:, len(PARTIES) :]

