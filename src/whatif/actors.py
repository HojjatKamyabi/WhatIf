"""Party Actors: each party except Others chooses one move per week from a fixed menu.

A Party Actor sees only a state summary: its own and its rivals' simulated support and the
week's Events as structured fields (party, topic, tone, size), never Voter data and never news
text. It returns a move and a reason. Implementations: the rule-based Stand-in here, and later
the language-model Party Actor, behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from whatif.backtest import CalibrationPeriod
from whatif.data.polls import TRACKED_PARTIES
from whatif.events import TOPICS

POLICY_TOPICS: tuple[str, ...] = tuple(t for t in TOPICS if t != "other")


@dataclass(frozen=True)
class Move:
    """One week's move: emphasise a policy topic, rebut the week's bad news, or hold still."""

    kind: str  # "emphasise", "rebut" or "hold"
    topic: str | None = None  # the topic, for "emphasise"

    def __str__(self) -> str:
        return f"emphasise {self.topic}" if self.kind == "emphasise" else self.kind

    @classmethod
    def parse(cls, text: str) -> Move:
        for move in MENU:
            if str(move) == text.strip().lower():
                return move
        raise ValueError(f"{text!r} is not a move; the menu is: {', '.join(map(str, MENU))}.")


MENU: tuple[Move, ...] = (*(Move("emphasise", t) for t in POLICY_TOPICS), Move("rebut"), Move("hold"))

# The topics each party is credible on (issue ownership): a first guess by the researcher, used by
# Voters (emphasis on an owned topic persuades more) and by the Stand-in (which topic to push).
ISSUE_OWNERSHIP: dict[str, tuple[str, ...]] = {
    "CDU/CSU": ("economy", "security"),
    "AfD": ("migration", "security"),
    "SPD": ("social", "economy"),
    "Greens": ("climate",),
    "Left": ("social",),
    "BSW": ("social", "security"),
    "FDP": ("economy",),
}
assert set(ISSUE_OWNERSHIP) == set(TRACKED_PARTIES)
assert all(set(topics) <= set(POLICY_TOPICS) for topics in ISSUE_OWNERSHIP.values())


@dataclass(frozen=True)
class EventSummary:
    """The week's recorded Events for one party and topic, as structured fields only."""

    party: str
    topic: str
    tone: float  # article-weighted average tone; negative is bad news
    size: float  # articles that week relative to the party's usual weekly count (1 = a usual week)


@dataclass(frozen=True)
class StateSummary:
    """Everything a Party Actor may see in one week. No Voter data, no names, no news text."""

    week: int  # Horizon week, 1 = the start week
    party: str  # the party deciding
    support: dict[str, float]  # simulated national support (%) of every party at the end of last week
    events: tuple[EventSummary, ...]  # the week's Events, all parties
    topic_attention: dict[str, float]  # public attention per policy topic relative to usual (1 = usual)


@dataclass(frozen=True)
class Decision:
    move: Move
    reason: str


class DecisionMaker(Protocol):
    """Anything that chooses a Party Actor's move from a state summary."""

    name: str

    def decide(self, state: StateSummary, menu: tuple[Move, ...]) -> Decision: ...


@dataclass(frozen=True)
class StandInParameters:
    """The Stand-in's rule parameters, tuned only on a Calibration Period, never on scored weeks."""

    rebut_threshold: float = 6.0  # bad news (size × −tone, summed) above which the party rebuts
    min_attention: float = 1.0  # an owned topic is worth emphasising from this attention up
    tuned_on: CalibrationPeriod | None = None  # None: untuned first guesses


class StandIn:
    """The rule-based Party Actor, with the same menu as the language-model one.

    1. If the week's bad news about the party is above the threshold, rebut it.
    2. Otherwise emphasise whichever of the party's own topics draws the most attention, if any
       draws at least the usual level.
    3. Otherwise hold.
    """

    name = "stand-in"

    def __init__(self, parameters: StandInParameters | None = None):
        self.parameters = parameters or StandInParameters()

    def decide(self, state: StateSummary, menu: tuple[Move, ...]) -> Decision:
        p = self.parameters
        bad_news = sum(e.size * max(-e.tone, 0.0) for e in state.events if e.party == state.party)
        if bad_news > p.rebut_threshold and Move("rebut") in menu:
            return Decision(Move("rebut"), f"bad news {bad_news:.1f} is above the threshold {p.rebut_threshold:g}")
        owned = [t for t in ISSUE_OWNERSHIP.get(state.party, ()) if Move("emphasise", t) in menu]
        if owned:
            topic = max(owned, key=lambda t: state.topic_attention.get(t, 0.0))
            attention = state.topic_attention.get(topic, 0.0)
            if attention >= p.min_attention:
                return Decision(Move("emphasise", topic), f"{topic} is its own topic with the most attention ({attention:.2f})")
        return Decision(Move("hold"), "no bad news to answer and none of its topics draws attention")
