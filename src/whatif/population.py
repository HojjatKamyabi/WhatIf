"""The Synthetic Population: made-up Voters matched per Constituency to official statistics.

No real person is modelled. Every Voter is drawn from aggregate figures for its Constituency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from whatif.elections import ConstituencyResult, ConstituencyStructure
from whatif.polls import PARTIES
from whatif.store import MissingDataError

ADULT_AGE_BANDS: tuple[str, ...] = ("18-24", "25-34", "35-59", "60-74", "75+")


@dataclass(frozen=True)
class ConstituencySummary:
    """What the Synthetic Population looks like in one Constituency."""

    voters: int
    preferences: dict[str, int]  # Voters whose starting preference is each party
    turnout: float  # every Voter's starting chance of voting
    ages: dict[str, int]  # Voters per adult age band
    land: str
    population_density: float  # inhabitants per km², standing in for urban versus rural
    disposable_income: float  # EUR per inhabitant per year
    unemployment_rate: float  # %


@dataclass(frozen=True)
class Voters:
    """The Synthetic Population as columns, one entry per Voter, for the Twin to run on."""

    constituency: np.ndarray  # Constituency number
    party: np.ndarray  # index into PARTIES of the starting preference
    age: np.ndarray  # index into ADULT_AGE_BANDS
    land: np.ndarray  # name of the Land
    turnout: np.ndarray  # starting chance of voting


class SyntheticPopulation:
    """Voters stored column-wise: one entry per Voter in each array.

    Every Voter has a Constituency, a starting party preference and an adult age band.
    Land, density, income, unemployment and turnout are the Constituency's own figures.
    """

    def __init__(
        self,
        constituency: np.ndarray,
        party: np.ndarray,
        age: np.ndarray,
        results: dict[int, ConstituencyResult],
        structure: dict[int, ConstituencyStructure],
    ):
        self._constituency = constituency
        self._party = party
        self._age = age
        self._results = results
        self._structure = structure

    def __len__(self) -> int:
        return len(self._constituency)

    def voters(self) -> Voters:
        turnout = {n: r.voted / r.eligible for n, r in self._results.items()}
        land = {n: s.land for n, s in self._structure.items()}
        return Voters(
            constituency=self._constituency.copy(),
            party=self._party.copy(),
            age=self._age.copy(),
            land=np.array([land[n] for n in self._constituency]),
            turnout=np.array([turnout[n] for n in self._constituency]),
        )

    def summary(self) -> dict[int, ConstituencySummary]:
        summaries = {}
        for number, result in self._results.items():
            mine = self._constituency == number
            parties = np.bincount(self._party[mine], minlength=len(PARTIES))
            ages = np.bincount(self._age[mine], minlength=len(ADULT_AGE_BANDS))
            facts = self._structure[number]
            summaries[number] = ConstituencySummary(
                voters=int(mine.sum()),
                preferences={p: int(parties[j]) for j, p in enumerate(PARTIES)},
                turnout=result.voted / result.eligible,
                ages={band: int(ages[j]) for j, band in enumerate(ADULT_AGE_BANDS)},
                land=facts.land,
                population_density=facts.population_density,
                disposable_income=facts.disposable_income,
                unemployment_rate=facts.unemployment_rate,
            )
        return summaries


def build_synthetic_population(
    results: Sequence[ConstituencyResult],
    structure: Sequence[ConstituencyStructure],
    scale: int,
    seed: int,
) -> SyntheticPopulation:
    """One Voter per `scale` eligible voters in every Constituency of the election.

    Starting preferences and adult age bands both reproduce the Constituency's figures by exact
    quota. Each is shuffled with the seed, so which Voters share a party and an age band is random:
    age is independent of party, because only aggregate figures exist (no survey data).
    """
    facts = {s.constituency: s for s in structure}
    missing = sorted(r.constituency for r in results if r.constituency not in facts)
    if missing:
        raise MissingDataError(f"No structural data for Constituencies {missing}.")

    rng = np.random.default_rng(seed)
    constituency, party, age = [], [], []
    for result in results:
        voters = max(1, round(result.eligible / scale))
        counts = _largest_remainder([result.votes[p] for p in PARTIES], voters)
        adult = np.array([facts[result.constituency].age_shares[band] for band in ADULT_AGE_BANDS])
        constituency.append(np.full(voters, result.constituency))
        party.append(rng.permutation(np.repeat(np.arange(len(PARTIES)), counts)))
        age.append(rng.permutation(np.repeat(np.arange(len(ADULT_AGE_BANDS)), _largest_remainder(adult, voters))))
    return SyntheticPopulation(
        np.concatenate(constituency),
        np.concatenate(party),
        np.concatenate(age),
        {r.constituency: r for r in results},
        facts,
    )


def _largest_remainder(votes: Sequence[int], seats: int) -> np.ndarray:
    """Split `seats` in proportion to `votes`: floors first, then the largest remainders."""
    quotas = np.asarray(votes, dtype=float) * seats / sum(votes)
    counts = np.floor(quotas).astype(int)
    leftover = seats - counts.sum()
    counts[np.argsort(-(quotas - counts), kind="stable")[:leftover]] += 1
    return counts


@dataclass(frozen=True)
class FitRow:
    """How closely one Constituency's Voters match the figures they were built from."""

    constituency: int
    name: str
    voters: int
    party_gap: float  # largest gap, in percentage points, between Voter preferences and the vote
    age_gap: float  # largest gap, in percentage points, between Voter ages and the adult age bands
    turnout: float

    @property
    def party_within_tolerance(self) -> bool:
        """Exact quota keeps every party within one Voter of its share of the vote."""
        return self.party_gap <= 100 / self.voters

    @property
    def age_within_tolerance(self) -> bool:
        """Exact quota keeps every age band within one Voter of its share of adults."""
        return self.age_gap <= 100 / self.voters


@dataclass(frozen=True)
class FitReport:
    rows: list[FitRow]

    @property
    def all_parties_within_tolerance(self) -> bool:
        return all(row.party_within_tolerance for row in self.rows)

    @property
    def all_ages_within_tolerance(self) -> bool:
        return all(row.age_within_tolerance for row in self.rows)

    def format(self) -> str:
        lines = [f"{'no.':>4}  {'Constituency':<40}{'Voters':>9}{'party gap':>11}{'age gap':>9}{'turnout':>9}"]
        for row in self.rows:
            lines.append(
                f"{row.constituency:>4}  {row.name[:39]:<40}{row.voters:>9}"
                f"{row.party_gap:>10.2f}p{row.age_gap:>8.2f}p{row.turnout:>9.1%}"
            )
        lines.append(
            f"Gaps in percentage points. Within one Voter everywhere: "
            f"parties {'yes' if self.all_parties_within_tolerance else 'NO'}, "
            f"age bands {'yes' if self.all_ages_within_tolerance else 'NO'}."
        )
        return "\n".join(lines)


def fit_report(population: SyntheticPopulation) -> FitReport:
    """Compare every Constituency's Voters with its election result and structural data."""
    rows = []
    for number, summary in population.summary().items():
        result = population._results[number]
        adult = np.array([population._structure[number].age_shares[b] for b in ADULT_AGE_BANDS])
        party_share = np.array([summary.preferences[p] for p in PARTIES]) / summary.voters
        vote_share = np.array([result.votes[p] for p in PARTIES]) / result.valid
        age_share = np.array([summary.ages[b] for b in ADULT_AGE_BANDS]) / summary.voters
        rows.append(
            FitRow(
                constituency=number,
                name=result.name,
                voters=summary.voters,
                party_gap=float(100 * np.abs(party_share - vote_share).max()),
                age_gap=float(100 * np.abs(age_share - adult / adult.sum()).max()),
                turnout=summary.turnout,
            )
        )
    return FitReport(rows)
