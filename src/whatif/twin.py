"""The Twin: Voters from the Synthetic Population, stepped forward week by week."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from whatif.actors import ISSUE_OWNERSHIP, MENU, POLICY_TOPICS, DecisionMaker, Move, StateSummary
from whatif.backtest import CalibrationPeriod
from whatif.boundary import WeeklyBoundary
from whatif.events import TOPICS
from whatif.data.polls import PARTIES, TRACKED_PARTIES
from whatif.population import ADULT_AGE_BANDS, SyntheticPopulation

# How much each adult age band cares about each topic (economy, migration, climate, security,
# social, other), relative to 1. A first guess by the researcher, to be tuned only in the
# Calibration Period. The economy weight is further scaled by unemployment where the Voter lives.
DEFAULT_TOPIC_PRIORITIES: dict[str, tuple[float, ...]] = {
    "18-24": (1.0, 0.8, 1.6, 0.8, 0.8, 1.0),
    "25-34": (1.2, 0.9, 1.3, 0.8, 0.8, 1.0),
    "35-59": (1.3, 1.0, 1.0, 0.9, 0.8, 1.0),
    "60-74": (1.0, 1.2, 0.8, 1.1, 1.2, 1.0),
    "75+": (0.8, 1.2, 0.7, 1.2, 1.4, 1.0),
}

# The researcher's first guesses for how strongly Voters react to the Boundary. They made the
# Twin worse in four of five scored Backtests, so the defaults are zero until
# the Calibration Period sets them. These remain available for exploring, never for
# reported scores.
FIRST_GUESS_REACTIONS: dict[str, float] = {
    "inflation_reaction": 0.01,
    "unemployment_reaction": 0.03,
    "event_reaction": 0.002,
    "move_reaction": 0.01,
}


@dataclass(frozen=True)
class TwinParameters:
    """Every named, configurable parameter of the Twin's Voter rules."""

    degree: int = 10  # neighbours per Voter
    same_land_share: float = 0.2  # share of neighbours drawn from anywhere in the Voter's Land
    same_age_share: float = 0.5  # of the neighbours in the Voter's own Constituency, share in the same age band
    influence: float = 0.1  # how far each week a Voter's preference moves towards their neighbours' average
    poll_sample: int = 2000  # respondents in the weekly poll that measures the Twin
    # The economy: share of a Voter's support for the governing parties that moves to the others
    # per point rise (or back, per point fall). Inflation weighs more on Voters in poorer
    # Constituencies, unemployment on Voters where it is already high (both relative to the average).
    # Off (zero) until calibrated; see FIRST_GUESS_REACTIONS.
    inflation_reaction: float = 0.0
    unemployment_reaction: float = 0.0
    # Events: share of a Voter's support for a party that leaves it per unit of Event shock on a
    # topic the Voter gives priority 1, at usual attention. Good news draws support the other way.
    # Off (zero) until calibrated.
    event_reaction: float = 0.0
    event_baseline_weeks: int = 8  # weeks before the start that set each party's usual coverage
    max_attention: float = 5.0  # attention above this many times the usual level counts as this
    topic_priorities: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(DEFAULT_TOPIC_PRIORITIES))
    # Party Actor moves: share of a Voter's support that moves towards a party emphasising a topic
    # the Voter cares about more than average, per unit of extra priority, at usual attention.
    # Off (zero) until calibrated.
    move_reaction: float = 0.0
    owned_credibility: float = 1.0  # how much an emphasis persuades on a topic the party owns
    other_credibility: float = 0.3  # ... and on any other topic
    rebut_effect: float = 0.5  # share of the week's bad news about a party that a rebuttal takes away
    tuned_on: CalibrationPeriod | None = None  # the period these were tuned on; None: untuned first guesses

    @property
    def reacts_to_boundary(self) -> bool:
        """Whether any Voter rule uses the Boundary, so a Backtest needs its data."""
        return any((self.inflation_reaction, self.unemployment_reaction, self.event_reaction))

    def credibility(self, party: str, topic: str) -> float:
        return self.owned_credibility if topic in ISSUE_OWNERSHIP.get(party, ()) else self.other_credibility


@dataclass(frozen=True)
class TwinRun:
    """One run of the Twin from a starting Poll Average."""

    start: np.ndarray  # national shares (%) before the first step
    start_by_constituency: dict[int, dict[str, float]]  # shares (%) per Constituency before the first step
    end_by_constituency: dict[int, dict[str, float]]  # shares (%) per Constituency after the last step
    shares: np.ndarray  # national shares (%) after each weekly step: [week][party]
    disagreement: list[float]  # Neighbour Disagreement (points) at the start and after each week
    decisions: list[dict] = field(default_factory=list)  # Party Actor decisions, with their inputs


class Twin:
    """The simulated Focus Region: one Voter per group of eligible voters."""

    def __init__(
        self,
        population: SyntheticPopulation,
        parameters: TwinParameters | None = None,
        seed: int = 0,
        boundary: WeeklyBoundary | None = None,
        name: str = "twin",
        actors: Mapping[str, DecisionMaker] | None = None,
        forced_moves: Mapping[tuple[int, str], Move] | None = None,
        decision_mode: str = "shared",
    ):
        self.name = name  # how the Backtest result labels this Twin, e.g. "twin:<Scenario>"
        self.parameters = parameters or TwinParameters()
        self.boundary = boundary
        self.actors = dict(actors or {})  # party -> decision-maker; parties without one hold
        self.forced_moves = dict(forced_moves or {})  # (Horizon week index, party) -> move, from a Scenario
        if decision_mode not in ("shared", "per-run"):
            raise ValueError(f"decision_mode must be 'shared' or 'per-run', not {decision_mode!r}.")
        # shared: Party Actors decide once per week for all Runs; per-run: every Run decides anew
        self.decision_mode = decision_mode
        self.decision_log: list[dict] = []  # the decisions of the last forecast
        self._voters = population.voters()
        self._neighbours = _network(self._voters, self.parameters, np.random.default_rng(seed))
        voters = self._voters
        # Each Voter's own circumstances, relative to the average Voter.
        self._unemployment_weight = voters.unemployment_rate / voters.unemployment_rate.mean()
        self._inflation_weight = voters.disposable_income.mean() / voters.disposable_income
        priorities = np.array([self.parameters.topic_priorities[band] for band in ADULT_AGE_BANDS], dtype=float)
        self._topic_weight = priorities[voters.age]
        self._topic_weight[:, TOPICS.index("economy")] *= self._unemployment_weight

    def run(
        self,
        latest: np.ndarray,
        horizon: int,
        actors: Mapping[str, DecisionMaker] | None = None,
        run_index: int | None = None,
    ) -> TwinRun:
        """Start from the election map swung to `latest` (national %), then step `horizon` weeks.

        `actors` replaces the Twin's own Party Actors for this run (per-Run decisions); decisions
        are then logged with `run_index`.
        """
        if self.boundary is not None and horizon > self.boundary.horizon:
            raise ValueError(f"The Boundary covers {self.boundary.horizon} weeks, not the {horizon} asked for.")
        preferences = self._start(np.asarray(latest, dtype=float) / np.sum(latest))
        start = 100 * self._national(preferences)
        start_by_constituency = self._by_constituency(preferences)
        shares, disagreement, decisions = [], [self._disagreement(preferences)], []
        actors = self.actors if actors is None else dict(actors)
        for week in range(horizon):
            moves = self._decide(preferences, week, decisions, actors, run_index)
            preferences = self._step(preferences, week, moves)
            shares.append(100 * self._national(preferences))
            disagreement.append(self._disagreement(preferences))
        return TwinRun(
            start=start,
            start_by_constituency=start_by_constituency,
            end_by_constituency=self._by_constituency(preferences),
            shares=np.array(shares).reshape(horizon, len(PARTIES)),
            disagreement=disagreement,
            decisions=decisions,
        )

    # --- Party Actors -----------------------------------------------------------------------

    def _decide(
        self,
        preferences: np.ndarray,
        week: int,
        log: list[dict],
        actors: Mapping[str, DecisionMaker],
        run_index: int | None,
    ) -> dict[str, Move]:
        """Each party's move this week: a Scenario's forced move, else its Actor's, else hold.

        Actors see a state summary only: national support at the end of last week, the week's
        Events as structured fields, and attention per topic. Every decision is logged.
        """
        if not actors and not self.forced_moves:
            return {}
        support = {p: float(v) for p, v in zip(PARTIES, 100 * self._national(preferences))}
        events = self.boundary.events_in(week) if self.boundary is not None else ()
        attention = {
            t: float(self.boundary.topic_attention[week, TOPICS.index(t)]) if self.boundary is not None else 1.0
            for t in POLICY_TOPICS
        }
        moves = {}
        for party in TRACKED_PARTIES:
            actor = actors.get(party)
            proposed = None
            if actor is not None:
                summary = StateSummary(week + 1, party, dict(support), events, dict(attention))
                decision = actor.decide(summary, MENU)
                if decision.move not in MENU:
                    raise ValueError(f"{party}'s Party Actor chose {decision.move}, which is not on the menu.")
                proposed = decision
            forced = self.forced_moves.get((week, party))
            if forced is None and proposed is None:
                continue
            move = forced or proposed.move
            moves[party] = move
            entry = {} if run_index is None else {"run": run_index}
            log.append(entry | {
                "week": week + 1,
                "party": party,
                "move": str(move),
                "reason": "forced by the Scenario" if forced else proposed.reason,
                "forced": forced is not None,
                "proposed": str(proposed.move) if proposed else None,
                "state": {"support": support, "events": [asdict(e) for e in events], "topic_attention": attention},
            })
        return moves

    # --- Voter rules ------------------------------------------------------------------------

    def _step(self, preferences: np.ndarray, week: int, moves: Mapping[str, Move] | None = None) -> np.ndarray:
        """One week: social influence, then the economy, then the week's Events, then party moves."""
        moves = moves or {}
        influence = self.parameters.influence
        preferences = (1.0 - influence) * preferences + influence * preferences[self._neighbours].mean(axis=1)
        if self.boundary is not None:
            preferences = self._economy(preferences, week)
            rebutting = {p for p, m in moves.items() if m.kind == "rebut"}
            preferences = self._events(preferences, week, rebutting)
        return self._emphasis(preferences, week, moves)

    def _emphasis(self, preferences: np.ndarray, week: int, moves: Mapping[str, Move]) -> np.ndarray:
        """Voters who care more than average about a topic move towards a party that emphasises it,
        the more so on a topic the party owns and the more attention the topic gets."""
        reaction = self.parameters.move_reaction
        if not reaction:
            return preferences
        for party, move in moves.items():
            if move.kind != "emphasise":
                continue
            k = TOPICS.index(move.topic)
            weight = self._topic_weight[:, k]
            above_average = np.maximum(weight - weight.mean(), 0.0)
            attention = 1.0
            if self.boundary is not None:
                attention = float(np.clip(self.boundary.topic_attention[week, k], 0.0, self.parameters.max_attention))
            rate = -reaction * self.parameters.credibility(party, move.topic) * attention * above_average
            this_party = np.arange(len(PARTIES)) == PARTIES.index(party)
            preferences = _shift(preferences, this_party, rate, self._national(preferences))
        return preferences

    def _economy(self, preferences: np.ndarray, week: int) -> np.ndarray:
        """Hard times move support from the governing parties to the others; better times move it back."""
        change = self.boundary.economy[week + 1] - self.boundary.economy[week]  # inflation, unemployment
        pressure = (
            self.parameters.inflation_reaction * change[0] * self._inflation_weight
            + self.parameters.unemployment_reaction * change[1] * self._unemployment_weight
        )
        if not pressure.any():
            return preferences
        governing = np.isin(PARTIES, list(self.boundary.government[week]))
        return _shift(preferences, governing, pressure, self._national(preferences))

    def _events(self, preferences: np.ndarray, week: int, rebutting: set[str] = frozenset()) -> np.ndarray:
        """Bad news about a party moves Voters away from it, good news towards it.

        Each Voter weighs the news by how much they care about its topic, and by how much
        attention the party and the topic get that week. A party that rebuts takes away
        `rebut_effect` of that week's bad news about it.
        """
        cap = self.parameters.max_attention
        party_attention = np.clip(self.boundary.party_attention[week], 0.0, cap)
        topic_attention = np.clip(self.boundary.topic_attention[week], 0.0, cap)
        weighted_topics = self._topic_weight * topic_attention  # [voter][topic]
        for party in range(len(PARTIES) - 1):  # Others is no party: no Events, no attention
            shock = self.boundary.event_shock[week, party]
            if not shock.any():
                continue
            if PARTIES[party] in rebutting:
                shock = np.where(shock > 0, shock * (1.0 - self.parameters.rebut_effect), shock)
            reaction = self.parameters.event_reaction * party_attention[party] * (weighted_topics @ shock)
            this_party = np.arange(len(PARTIES)) == party
            preferences = _shift(preferences, this_party, reaction, self._national(preferences))
        return preferences

    def _disagreement(self, preferences: np.ndarray) -> float:
        """Neighbour Disagreement: how far, on average, a Voter's preference is from their neighbours'.

        Half the summed absolute difference, in percentage points: 0 means identical preferences,
        100 means no party in common.
        """
        gap = np.abs(preferences - preferences[self._neighbours].mean(axis=1)).sum(axis=1)
        return float(50 * gap.mean())

    def forecast(self, history: np.ndarray, horizon: int, runs: int, rng: np.random.Generator) -> np.ndarray:
        """The Twin as a Backtest forecaster: an Outcome Distribution shaped (runs, horizon, parties).

        Voters start from the last week of `history` and step forward. Each week, every Run
        measures them with its own poll: `poll_sample` respondents drawn from Voters in proportion
        to their chance of voting, each naming a party from their preference. That is a
        multinomial draw on the national shares, so Runs differ only by polling noise.
        """
        if self.decision_mode == "shared" or not self.actors:
            run = self.run(history[-1], horizon)
            self.decision_log = run.decisions  # Shared Actor Decisions: one set of moves for every Run
            shares = np.broadcast_to(run.shares / run.shares.sum(axis=1, keepdims=True), (runs, horizon, len(PARTIES)))
        else:  # per-Run decisions: each Run's Actors decide anew, with the Run's own seed
            per_run, self.decision_log = [], []
            for r in range(runs):
                actors = {p: (a.for_run(r) if hasattr(a, "for_run") else a) for p, a in self.actors.items()}
                run = self.run(history[-1], horizon, actors=actors, run_index=r)
                per_run.append(run.shares / run.shares.sum(axis=1, keepdims=True))
                self.decision_log += run.decisions
            shares = np.stack(per_run)
        sample = self.parameters.poll_sample
        counts = rng.multinomial(sample, shares)
        return 100.0 * counts / sample

    # --- starting state ---------------------------------------------------------------------

    def _start(self, target: np.ndarray) -> np.ndarray:
        """Each Voter's preference over the parties, swung uniformly so the nation matches `target`.

        Every Constituency gets the same additive swing per party (clipped at zero), found by
        iterating until the turnout-weighted national shares hit the target. Within a Constituency,
        each Voter's starting party is mixed with a common component, just enough that the
        Constituency average lands exactly on its swung shares.
        """
        voters = self._voters
        onehot = np.eye(len(PARTIES))[voters.party]
        numbers, index = np.unique(voters.constituency, return_inverse=True)
        election = np.stack([onehot[index == c].mean(axis=0) for c in range(len(numbers))])
        weight = np.bincount(index, weights=voters.turnout)
        weight /= weight.sum()

        swing = target - weight @ election
        for _ in range(200):
            swung = np.clip(election + swing, 0.0, None)
            swung /= swung.sum(axis=1, keepdims=True)
            miss = target - weight @ swung
            if np.abs(miss).max() < 1e-9:
                break
            swing += miss

        shrink = np.where(election > 0, 1.0 - swung / np.where(election > 0, election, 1.0), 0.0)
        mix = np.clip(shrink.max(axis=1), 0.0, 1.0)  # share of each Voter replaced by the common part
        common = np.where(
            mix[:, None] > 0,
            (swung - (1.0 - mix)[:, None] * election) / np.where(mix > 0, mix, 1.0)[:, None],
            0.0,
        )
        return (1.0 - mix[index])[:, None] * onehot + mix[index][:, None] * common[index]

    # --- measurement ------------------------------------------------------------------------

    def _national(self, preferences: np.ndarray) -> np.ndarray:
        weights = self._voters.turnout
        return weights @ preferences / weights.sum()

    def _by_constituency(self, preferences: np.ndarray) -> dict[int, dict[str, float]]:
        out = {}
        for number in np.unique(self._voters.constituency):
            mean = preferences[self._voters.constituency == number].mean(axis=0)
            out[int(number)] = {p: float(100 * mean[j]) for j, p in enumerate(PARTIES)}
        return out


def _network(voters, parameters: TwinParameters, rng: np.random.Generator) -> np.ndarray:
    """Each Voter's neighbours: nearby (own Constituency, often the same age band) and in their Land.

    Returns an array [voter][neighbour] of Voter indices. Neighbours are drawn with replacement.
    """
    degree = parameters.degree
    from_land = round(degree * parameters.same_land_share)
    same_age = round((degree - from_land) * parameters.same_age_share)
    anyone = degree - from_land - same_age

    neighbours = np.empty((len(voters.constituency), degree), dtype=np.int64)
    for number in np.unique(voters.constituency):
        members = np.flatnonzero(voters.constituency == number)
        neighbours[members, :anyone] = rng.choice(members, size=(len(members), anyone))
        for age in np.unique(voters.age[members]):
            peers = members[voters.age[members] == age]
            neighbours[peers, anyone : anyone + same_age] = rng.choice(peers, size=(len(peers), same_age))
    for land in np.unique(voters.land):
        members = np.flatnonzero(voters.land == land)
        neighbours[members, degree - from_land :] = rng.choice(members, size=(len(members), from_land))
    return neighbours


def _shift(preferences: np.ndarray, parties: np.ndarray, rate: np.ndarray, national: np.ndarray) -> np.ndarray:
    """Shrink each Voter's support for `parties` by the fraction `rate` (grow it, if negative).

    Good and bad news are symmetric: a party gains the same share of its own support that the
    same-sized bad news would cost it. Support leaving goes to the other parties in proportion
    to the Voter's own support for them (the national shares for a Voter who supports none of
    them); support arriving is taken from the other parties in proportion.
    """
    rate = np.clip(rate, -1.0, 1.0)
    if not rate.any():
        return preferences
    own = (preferences * parties).sum(axis=1)
    rest = 1.0 - own
    amount = np.maximum(rate * own, -rest)  # > 0 leaves the parties, < 0 joins them
    keep = np.where(own > 1e-12, 1.0 - amount / np.where(own > 1e-12, own, 1.0), 1.0)
    grow = np.where(rest > 1e-12, 1.0 + amount / np.where(rest > 1e-12, rest, 1.0), 1.0)
    shifted = preferences * np.where(parties, keep[:, None], grow[:, None])
    stranded = (rest <= 1e-12) & (amount > 0)  # Voters with no other party to move to
    if stranded.any():
        target = np.where(parties, 0.0, national)
        target = target / target.sum() if target.sum() > 0 else (~parties) / (~parties).sum()
        shifted[stranded] += amount[stranded, None] * target
    return shifted
