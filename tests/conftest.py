from datetime import date, timedelta

import numpy as np
import pytest

from whatif.polls import PARTIES, TRACKED_PARTIES, Poll

FIRST_WEEK = date(2024, 1, 1)  # a Monday


def make_poll(
    week_index: int,
    shares: dict[str, float],
    sample_size: int | None = 1000,
    institute: str = "A",
    unreported: tuple[str, ...] = (),
) -> Poll:
    """A national poll whose fieldwork ends on the Wednesday of the given week.

    Parties in `unreported` were not listed by the poll; leave them out of `shares`.
    """
    full = dict.fromkeys(PARTIES, 0.0) | shares
    total = sum(full.values())
    field_end = FIRST_WEEK + timedelta(weeks=week_index, days=2)
    reported = frozenset(TRACKED_PARTIES) - set(unreported)
    return Poll(institute, field_end, field_end, sample_size, {p: 100 * v / total for p, v in full.items()}, reported)


@pytest.fixture
def poll_series() -> list[Poll]:
    """40 weeks of gently drifting, noisy polls: one per week."""
    rng = np.random.default_rng(42)
    base = np.array([30, 18, 16, 12, 6, 5, 5, 8], dtype=float)
    drift = np.array([-0.1, 0.15, 0.0, -0.05, 0.02, 0.0, -0.02, 0.0])
    polls = []
    for week in range(40):
        shares = np.clip(base + week * drift + rng.normal(0, 0.5, len(base)), 0.5, None)
        polls.append(make_poll(week, dict(zip(PARTIES, shares))))
    return polls
