"""The Twin: Voters from the Synthetic Population, stepped forward week by week."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whatif.boundary import WeeklyBoundary
from whatif.events import TOPICS
from whatif.polls import PARTIES
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
    inflation_reaction: float = 0.01
    unemployment_reaction: float = 0.03
    # Events: share of a Voter's support for a party that leaves it per unit of Event shock on a
    # topic the Voter gives priority 1, at usual attention. Good news draws support the other way.
    event_reaction: float = 0.002
    event_baseline_weeks: int = 8  # weeks before the start that set each party's usual coverage
    max_attention: float = 5.0  # attention above this many times the usual level counts as this
    topic_priorities: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(DEFAULT_TOPIC_PRIORITIES))


@dataclass(frozen=True)
class TwinRun:
    """One run of the Twin from a starting Poll Average."""

    start: np.ndarray  # national shares (%) before the first step
    start_by_constituency: dict[int, dict[str, float]]  # shares (%) per Constituency before the first step
    end_by_constituency: dict[int, dict[str, float]]  # shares (%) per Constituency after the last step
    shares: np.ndarray  # national shares (%) after each weekly step: [week][party]
    disagreement: list[float]  # Neighbour Disagreement (points) at the start and after each week


class Twin:
    """The simulated Focus Region: one Voter per group of eligible voters."""

    name = "twin"

    def __init__(
        self,
        population: SyntheticPopulation,
        parameters: TwinParameters | None = None,
        seed: int = 0,
        boundary: WeeklyBoundary | None = None,
    ):
        self.parameters = parameters or TwinParameters()
        self.boundary = boundary
        self._voters = population.voters()
        self._neighbours = _network(self._voters, self.parameters, np.random.default_rng(seed))
        voters = self._voters
        # Each Voter's own circumstances, relative to the average Voter.
        self._unemployment_weight = voters.unemployment_rate / voters.unemployment_rate.mean()
        self._inflation_weight = voters.disposable_income.mean() / voters.disposable_income
        priorities = np.array([self.parameters.topic_priorities[band] for band in ADULT_AGE_BANDS], dtype=float)
        self._topic_weight = priorities[voters.age]
        self._topic_weight[:, TOPICS.index("economy")] *= self._unemployment_weight

    def run(self, latest: np.ndarray, horizon: int) -> TwinRun:
        """Start from the election map swung to `latest` (national %), then step `horizon` weeks."""
        if self.boundary is not None and horizon > self.boundary.horizon:
            raise ValueError(f"The Boundary covers {self.boundary.horizon} weeks, not the {horizon} asked for.")
        preferences = self._start(np.asarray(latest, dtype=float) / np.sum(latest))
        start = 100 * self._national(preferences)
        start_by_constituency = self._by_constituency(preferences)
        shares, disagreement = [], [self._disagreement(preferences)]
        for week in range(horizon):
            preferences = self._step(preferences, week)
            shares.append(100 * self._national(preferences))
            disagreement.append(self._disagreement(preferences))
        return TwinRun(
            start=start,
            start_by_constituency=start_by_constituency,
            end_by_constituency=self._by_constituency(preferences),
            shares=np.array(shares).reshape(horizon, len(PARTIES)),
            disagreement=disagreement,
        )

    # --- Voter rules ------------------------------------------------------------------------

    def _step(self, preferences: np.ndarray, week: int) -> np.ndarray:
        """One week: social influence, then the economy, then the week's Events."""
        influence = self.parameters.influence
        preferences = (1.0 - influence) * preferences + influence * preferences[self._neighbours].mean(axis=1)
        if self.boundary is not None:
            preferences = self._economy(preferences, week)
            preferences = self._events(preferences, week)
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
        national = self._national(preferences)
        preferences = _move(preferences, governing, np.clip(pressure, 0.0, 1.0), national)
        return _move(preferences, ~governing, np.clip(-pressure, 0.0, 1.0), national)

    def _events(self, preferences: np.ndarray, week: int) -> np.ndarray:
        """Bad news about a party moves Voters away from it, good news towards it.

        Each Voter weighs the news by how much they care about its topic, and by how much
        attention the party and the topic get that week.
        """
        cap = self.parameters.max_attention
        party_attention = np.clip(self.boundary.party_attention[week], 0.0, cap)
        topic_attention = np.clip(self.boundary.topic_attention[week], 0.0, cap)
        weighted_topics = self._topic_weight * topic_attention  # [voter][topic]
        for party in range(len(PARTIES) - 1):  # Others is no party: no Events, no attention
            shock = self.boundary.event_shock[week, party]
            if not shock.any():
                continue
            reaction = self.parameters.event_reaction * party_attention[party] * (weighted_topics @ shock)
            this_party = np.arange(len(PARTIES)) == party
            national = self._national(preferences)
            preferences = _move(preferences, this_party, np.clip(reaction, 0.0, 1.0), national)
            preferences = _move(preferences, ~this_party, np.clip(-reaction, 0.0, 1.0), national)
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
        run = self.run(history[-1], horizon)
        shares = run.shares / run.shares.sum(axis=1, keepdims=True)
        sample = self.parameters.poll_sample
        counts = rng.multinomial(sample, np.broadcast_to(shares, (runs, horizon, len(PARTIES))))
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


def _move(preferences: np.ndarray, losing: np.ndarray, rate: np.ndarray, national: np.ndarray) -> np.ndarray:
    """Move `rate` (per Voter) of each Voter's support for the `losing` parties to the other parties.

    It goes to the other parties in proportion to the Voter's own support for them; a Voter who
    supports none of them follows the national shares instead.
    """
    if not rate.any():
        return preferences
    moved = preferences * losing * rate[:, None]
    gaining = preferences * ~losing
    total = gaining.sum(axis=1, keepdims=True)
    fallback = np.where(~losing, national, 0.0)
    fallback = fallback / fallback.sum() if fallback.sum() > 0 else (~losing) / (~losing).sum()
    share = np.where(total > 1e-12, gaining / np.where(total > 1e-12, total, 1.0), fallback)
    return preferences - moved + moved.sum(axis=1, keepdims=True) * share
