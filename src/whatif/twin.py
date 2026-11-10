"""The Twin: Voters from the Synthetic Population, stepped forward week by week."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from whatif.polls import PARTIES
from whatif.population import SyntheticPopulation


@dataclass(frozen=True)
class TwinParameters:
    """Every named, configurable parameter of the Twin's Voter rules."""

    degree: int = 10  # neighbours per Voter
    same_land_share: float = 0.2  # share of neighbours drawn from anywhere in the Voter's Land
    same_age_share: float = 0.5  # of the neighbours in the Voter's own Constituency, share in the same age band
    influence: float = 0.1  # how far each week a Voter's preference moves towards their neighbours' average
    poll_sample: int = 2000  # respondents in the weekly poll that measures the Twin


@dataclass(frozen=True)
class TwinRun:
    """One run of the Twin from a starting Poll Average."""

    start: np.ndarray  # national shares (%) before the first step
    start_by_constituency: dict[int, dict[str, float]]  # shares (%) per Constituency before the first step
    shares: np.ndarray  # national shares (%) after each weekly step: [week][party]
    disagreement: list[float]  # Neighbour Disagreement (points) at the start and after each week


class Twin:
    """The simulated Focus Region: one Voter per group of eligible voters."""

    name = "twin"

    def __init__(self, population: SyntheticPopulation, parameters: TwinParameters | None = None, seed: int = 0):
        self.parameters = parameters or TwinParameters()
        self._voters = population.voters()
        self._neighbours = _network(self._voters, self.parameters, np.random.default_rng(seed))

    def run(self, latest: np.ndarray, horizon: int) -> TwinRun:
        """Start from the election map swung to `latest` (national %), then step `horizon` weeks."""
        preferences = self._start(np.asarray(latest, dtype=float) / np.sum(latest))
        start = 100 * self._national(preferences)
        start_by_constituency = self._by_constituency(preferences)
        shares, disagreement = [], [self._disagreement(preferences)]
        for _ in range(horizon):
            preferences = self._step(preferences)
            shares.append(100 * self._national(preferences))
            disagreement.append(self._disagreement(preferences))
        return TwinRun(
            start=start,
            start_by_constituency=start_by_constituency,
            shares=np.array(shares).reshape(horizon, len(PARTIES)),
            disagreement=disagreement,
        )

    # --- Voter rules ------------------------------------------------------------------------

    def _step(self, preferences: np.ndarray) -> np.ndarray:
        """One week: every Voter moves part of the way towards their neighbours' average."""
        influence = self.parameters.influence
        return (1.0 - influence) * preferences + influence * preferences[self._neighbours].mean(axis=1)

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
