"""The simulation's memory: what species existed, where they came from, and when they ended.

Everything in this module is cold-path bookkeeping.  It never influences a tick -- no organism
behaves differently because of what is recorded here -- which is deliberate: a history that fed
back into the model would be a hidden selection pressure of exactly the kind the project
forbids.  Its only job is to make the question "why did this lineage become what it is?"
answerable after the fact.

Two decisions worth stating:

1. **A species identity is never reused.**  Extinction is permanent, so a resurrected id could
   only ever mean a bug or a lie in the lineage tree.  Ids are handed out by split order and
   the counter never rewinds.

2. **Extinction is declared once, on confirmation, and dated to the day the last member died.**
   ``sim.extinction_confirm_ticks`` guards against calling a transient dip an ending, but the
   ending itself happened when the population reached zero, not when the guard expired.  Both
   days are recorded, because the second explains why a run's report changed when it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config
from evosim.evolution.taxonomy import Split
from evosim.life.population import UNASSIGNED, Population

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

FOUNDER_SPECIES = 0


@dataclass(slots=True)
class SpeciesRecord:
    """One species: its parentage, its origin, its size, and its end if it has had one."""

    species_id: int
    parent: int
    origin_day: int
    origin_traits: tuple[float, ...]
    population: int
    peak_population: int
    peak_day: int
    last_seen_day: int
    extinct_day: int | None = None
    confirmed_day: int | None = None

    @property
    def extinct(self) -> bool:
        return self.extinct_day is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "parent": None if self.parent == UNASSIGNED else self.parent,
            "origin_day": self.origin_day,
            "origin_traits": list(self.origin_traits),
            "population": self.population,
            "peak_population": self.peak_population,
            "peak_day": self.peak_day,
            "last_seen_day": self.last_seen_day,
            "extinct_day": self.extinct_day,
            "confirmed_day": self.confirmed_day,
        }


@dataclass(frozen=True, slots=True)
class SpeciesSample:
    """One species at one sampled day."""

    species_id: int
    population: int
    mean_age: float
    mean_energy: float
    traits: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "population": self.population,
            "mean_age": self.mean_age,
            "mean_energy": self.mean_energy,
            "traits": list(self.traits),
        }


@dataclass(frozen=True, slots=True)
class Sample:
    """The whole biosphere at one sampled day, broken down by species."""

    day: int
    population: int
    species: tuple[SpeciesSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "population": self.population,
            "species": [entry.to_dict() for entry in self.species],
        }


@dataclass(slots=True)
class History:
    """Species records, the lineage tree they form, and the sampled time series."""

    extinction_confirm_ticks: int
    trait_names: tuple[str, ...]
    records: dict[int, SpeciesRecord] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    _next_species: int = 0

    @classmethod
    def from_config(cls, config: Config, trait_names: tuple[str, ...]) -> History:
        return cls(
            extinction_confirm_ticks=config.sim.extinction_confirm_ticks,
            trait_names=tuple(trait_names),
        )

    @property
    def next_species(self) -> int:
        """The identity the next split will be given."""

        return self._next_species

    @property
    def species_count(self) -> int:
        """How many species have ever existed, extinct ones included."""

        return len(self.records)

    def living(self) -> tuple[int, ...]:
        """Ids of species with living members, ascending."""

        return tuple(
            species for species, record in sorted(self.records.items())
            if record.population > 0
        )

    def extinct(self) -> tuple[int, ...]:
        """Ids of species whose extinction has been confirmed, ascending."""

        return tuple(
            species for species, record in sorted(self.records.items()) if record.extinct
        )

    # -- registration -----------------------------------------------------------------------

    def found(self, day: int, population: Population) -> SpeciesRecord:
        """Register the founder species and stamp it on every seeded organism."""

        if self.records:
            raise ValueError("the founder species has already been registered")
        population.species_id[population.active] = FOUNDER_SPECIES
        rows = np.arange(population.size, dtype=np.int64)
        return self._register(FOUNDER_SPECIES, UNASSIGNED, day, population, rows)

    def apply(
        self,
        day: int,
        population: Population,
        splits: list[Split],
    ) -> list[SpeciesRecord]:
        """Move each daughter group onto its own identity and record where it came from.

        Splits must be applied in the order the taxonomy found them: a later daughter group can
        be a subset of an earlier one, and applying them out of order would give an organism the
        identity of a species it is no longer in.
        """

        created: list[SpeciesRecord] = []
        for split in splits:
            if split.species != self._next_species:
                raise ValueError(
                    f"split claims species {split.species}, but the next identity to hand out "
                    f"is {self._next_species}"
                )
            population.species_id[population.active][split.rows] = split.species
            created.append(
                self._register(split.species, split.parent, day, population, split.rows)
            )
        return created

    def _register(
        self,
        species: int,
        parent: int,
        day: int,
        population: Population,
        rows: IntArray,
    ) -> SpeciesRecord:
        record = SpeciesRecord(
            species_id=species,
            parent=parent,
            origin_day=day,
            origin_traits=trait_means(population, rows),
            population=int(rows.size),
            peak_population=int(rows.size),
            peak_day=day,
            last_seen_day=day,
        )
        self.records[species] = record
        self._next_species = max(self._next_species, species + 1)
        return record

    # -- per-tick observation ---------------------------------------------------------------

    def observe(self, day: int, population: Population) -> int:
        """Update every record's population and confirm any extinctions come due.

        Returns the number of species declared extinct on this call.  This runs every tick
        rather than on the taxonomy cadence because an extinction date read off a 100-day grid
        would be an extinction date that is wrong by up to 100 days.
        """

        counts = np.bincount(
            population.species_id[population.active].astype(np.int64),
            minlength=self._next_species,
        )
        confirmed = 0
        for species, record in self.records.items():
            if record.extinct:
                continue
            count = int(counts[species]) if species < counts.size else 0
            record.population = count
            if count > 0:
                record.last_seen_day = day
                if count > record.peak_population:
                    record.peak_population = count
                    record.peak_day = day
            elif day - record.last_seen_day >= self.extinction_confirm_ticks:
                record.extinct_day = record.last_seen_day
                record.confirmed_day = day
                confirmed += 1
        return confirmed

    def sample(self, day: int, population: Population) -> Sample:
        """Record one point of the per-species time series."""

        active = population.active
        species = population.species_id[active].astype(np.int64)
        counts = np.bincount(species, minlength=max(self._next_species, 1))
        traits = population.phenotypes.traits[active].astype(np.float64)
        age = population.age[active].astype(np.float64)
        energy = population.energy[active].astype(np.float64)

        def totals(values: FloatArray) -> FloatArray:
            return np.bincount(species, weights=values, minlength=counts.size)

        # One bincount per locus rather than a scatter-add: the loop is over the 28 loci and
        # never over organisms, which is the rule the hot path is held to as well.
        trait_totals = np.stack(
            [totals(traits[:, locus]) for locus in range(traits.shape[1])], axis=1
        ) if traits.size else np.zeros((counts.size, 0))
        age_totals = totals(age)
        energy_totals = totals(energy)

        entries = [
            SpeciesSample(
                species_id=int(identity),
                population=int(counts[identity]),
                mean_age=float(age_totals[identity] / counts[identity]),
                mean_energy=float(energy_totals[identity] / counts[identity]),
                traits=tuple(
                    float(value) for value in trait_totals[identity] / counts[identity]
                ),
            )
            for identity in np.flatnonzero(counts)
        ]
        record = Sample(day=day, population=population.size, species=tuple(entries))
        self.samples.append(record)
        return record

    # -- reading it back --------------------------------------------------------------------

    def lineage(self, species: int) -> tuple[int, ...]:
        """The ancestral chain from the founder species down to ``species``."""

        if species not in self.records:
            raise KeyError(f"no such species: {species}")
        chain = [species]
        parent = self.records[species].parent
        while parent != UNASSIGNED:
            chain.append(parent)
            parent = self.records[parent].parent
        return tuple(reversed(chain))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole record for a run dump or an API."""

        return {
            "format": 1,
            "trait_names": list(self.trait_names),
            "extinction_confirm_ticks": self.extinction_confirm_ticks,
            "species": [
                self.records[species].to_dict() for species in sorted(self.records)
            ],
            "samples": [sample.to_dict() for sample in self.samples],
        }


def trait_means(population: Population, rows: IntArray) -> tuple[float, ...]:
    """Mean expressed trait vector over the given population rows."""

    traits = population.phenotypes.traits[population.active][rows].astype(np.float64)
    if traits.size == 0:
        return tuple(0.0 for _ in range(population.schema.n_loci))
    return tuple(float(value) for value in traits.mean(axis=0))


__all__ = [
    "FOUNDER_SPECIES",
    "History",
    "Sample",
    "SpeciesRecord",
    "SpeciesSample",
    "trait_means",
]
