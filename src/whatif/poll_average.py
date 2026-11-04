"""The weekly Poll Average: WhatIf's Target Outcome."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from whatif.polls import PARTIES, TRACKED_PARTIES, Poll


def week_of(day: date) -> date:
    """The Monday that starts the calendar week containing `day`."""
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True)
class PollAverage:
    """Each party's weekly share of national vote intention.

    `shares[i, j]` is party `PARTIES[j]` in `weeks[i]`. Weeks with no poll repeat the
    previous week's shares and are marked in `carried`.
    """

    weeks: tuple[date, ...]
    shares: np.ndarray
    carried: np.ndarray

    def index_of(self, week: date) -> int:
        offset = (week - self.weeks[0]).days // 7
        if offset < 0 or offset >= len(self.weeks) or self.weeks[offset] != week:
            raise KeyError(week)
        return offset


def weekly_poll_average(polls: list[Poll], default_sample_size: int = 1000) -> PollAverage:
    """Average polls per calendar week of their field end date, weighted by sample size.

    Each tracked party is averaged only over the polls that reported it by name, so a poll
    that folds a party into Others doesn't drag that party down. If no poll in a week
    reported a party, it keeps the previous week's value. Others is the remainder to 100.
    Polls without a sample size count as `default_sample_size` respondents.
    """
    if not polls:
        raise ValueError("Cannot build a Poll Average from no polls")

    tracked = len(TRACKED_PARTIES)
    totals: dict[date, np.ndarray] = {}
    weights: dict[date, np.ndarray] = {}
    for poll in polls:
        week = week_of(poll.field_end)
        weight = float(poll.sample_size or default_sample_size)
        mask = np.array([party in poll.reported for party in TRACKED_PARTIES], dtype=float)
        vector = np.array([poll.shares[party] for party in TRACKED_PARTIES])
        totals[week] = totals.get(week, np.zeros(tracked)) + weight * mask * vector
        weights[week] = weights.get(week, np.zeros(tracked)) + weight * mask

    first, last = min(totals), max(totals)
    count = (last - first).days // 7 + 1
    weeks = tuple(first + timedelta(weeks=i) for i in range(count))
    shares = np.zeros((count, len(PARTIES)))
    carried = np.zeros(count, dtype=bool)
    previous = np.zeros(tracked)
    for i, week in enumerate(weeks):
        if week not in totals:
            shares[i] = shares[i - 1]
            carried[i] = True
            continue
        reported = weights[week] > 0
        parties = np.where(reported, totals[week] / np.where(reported, weights[week], 1.0), previous)
        others = max(0.0, 100.0 - parties.sum())
        row = np.append(parties, others)
        shares[i] = 100.0 * row / row.sum()
        previous = shares[i, :tracked]
    return PollAverage(weeks, shares, carried)
