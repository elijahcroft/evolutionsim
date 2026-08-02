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
from evosim.life.phenotype import DIET_NAMES, MORPHOLOGY_NAMES
from evosim.life.population import UNASSIGNED, Population
from evosim.world.biome import BIOME_NAMES

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

FOUNDER_SPECIES = 0

#: A biome has to hold at least this share of a species before the habitat line names it, and
#: the line names at most this many.  Both exist so the sentence stays a sentence: a species
#: spread thinly over nine biomes has no habitat worth naming, and saying so is more use than
#: reciting nine numbers.
HABITAT_FLOOR = 0.1
HABITAT_TERMS = 3


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
    #: Share of this species' members standing in each biome, in ``BIOME_NAMES`` order.
    habitat: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "population": self.population,
            "mean_age": self.mean_age,
            "mean_energy": self.mean_energy,
            "traits": list(self.traits),
            "habitat": list(self.habitat),
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
    biome_names: tuple[str, ...] = BIOME_NAMES
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

    def sample(self, day: int, population: Population, biome: IntArray) -> Sample:
        """Record one point of the per-species time series.

        ``biome`` is the day's classification flattened to one code per cell.  It is passed in
        rather than derived here because this module deliberately knows about arrays and not
        about the world layer -- and because the classification is a read of the world that the
        caller already has to have made.
        """

        active = population.active
        species = population.species_id[active].astype(np.int64)
        counts = np.bincount(species, minlength=max(self._next_species, 1))
        traits = population.phenotypes.traits[active].astype(np.float64)
        age = population.age[active].astype(np.float64)
        energy = population.energy[active].astype(np.float64)

        def totals(values: FloatArray) -> FloatArray:
            return np.bincount(species, weights=values, minlength=counts.size)

        # Where each organism is standing, counted per (species, biome) in one pass: the pair is
        # folded into a single index so this stays one bincount rather than one per species.
        n_biomes = len(self.biome_names)
        codes = np.asarray(biome).ravel()[population.cell[active].astype(np.intp)]
        occupancy = np.bincount(
            species * n_biomes + codes.astype(np.int64),
            minlength=counts.size * n_biomes,
        ).reshape(counts.size, n_biomes)

        # One bincount per locus rather than a scatter-add: the loop is over the loci and
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
                habitat=tuple(
                    float(value) for value in occupancy[identity] / counts[identity]
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

    def summary(self, population: Population) -> list[dict[str, Any]]:
        """Every species that has ever existed, newest first, with its lineage attached.

        Newest first because a run is usually read from its present backwards: the thing you
        want to open is the species that just appeared, not the founder you already know about.
        """

        return [
            {
                **self.records[species].to_dict(),
                "lineage": list(self.lineage(species)),
                "extinct": self.records[species].extinct,
            }
            for species in sorted(self.records, reverse=True)
        ]

    def report(self, species: int, population: Population) -> dict[str, Any]:
        """One species in full: its ancestry, its time series, and what has changed in it.

        ``drift`` is the point of the whole layer.  A trait mean on its own says what a species
        is; the *difference* from what it was at its origin, ranked by how far it moved relative
        to what that locus can do, is the beginning of an answer to why it became that.  It is
        ranked rather than merely listed because 28 unordered numbers are not an explanation.
        """

        if species not in self.records:
            raise KeyError(f"no such species: {species}")
        record = self.records[species]
        rows = np.flatnonzero(
            population.species_id[population.active].astype(np.int64) == species
        )
        current = trait_means(population, rows) if rows.size else None
        series = [
            {
                "day": sample.day,
                **entry.to_dict(),
            }
            for sample in self.samples
            for entry in sample.species
            if entry.species_id == species
        ]

        span = population.schema.span.astype(np.float64)
        reference = current if current is not None else (
            tuple(series[-1]["traits"]) if series else None
        )
        drift: list[dict[str, Any]] = []
        if reference is not None:
            drift = sorted(
                (
                    {
                        "name": name,
                        "origin": origin,
                        "now": now,
                        "change": now - origin,
                        # Divided by the locus span so loci measured in degrees and loci
                        # measured in offspring can be ranked against each other at all.
                        "span_fraction": (now - origin) / float(span[index])
                        if span[index] > 0.0
                        else 0.0,
                    }
                    for index, (name, origin, now) in enumerate(
                        zip(self.trait_names, record.origin_traits, reference)
                    )
                ),
                key=lambda entry: abs(entry["span_fraction"]),
                reverse=True,
            )

        return {
            **record.to_dict(),
            "extinct": record.extinct,
            "lineage": list(self.lineage(species)),
            "children": sorted(
                other for other, entry in self.records.items() if entry.parent == species
            ),
            "trait_names": list(self.trait_names),
            "current_traits": list(current) if current is not None else None,
            "series": series,
            "drift": drift,
            "morphology": self._morphology(population, rows, reference),
            "habitat": self._habitat(species),
        }

    def _habitat(self, species: int) -> dict[str, Any] | None:
        """Where this species was last seen living, and a sentence saying so.

        Read off the most recent sample that contains it rather than recounted live, for two
        reasons: it needs no ``World``, so ``report`` stays a pure read of what was recorded;
        and an extinct species keeps the habitat it had instead of losing it at the moment the
        question becomes historical.
        """

        for sample in reversed(self.samples):
            for entry in sample.species:
                if entry.species_id == species:
                    return {
                        "day": sample.day,
                        "names": list(self.biome_names),
                        "fractions": list(entry.habitat),
                        "line": habitat_line(entry.habitat, self.biome_names),
                    }
        return None

    def _morphology(
        self,
        population: Population,
        rows: IntArray,
        reference: tuple[float, ...] | None,
    ) -> dict[str, Any] | None:
        """This species' average body, in the form the creature viewer draws.

        A read of state that already exists: the same trait means the drift table is built from,
        re-keyed by gene name so the UI does not have to know column indices, plus the mean diet
        that decides the animal's colour.  The simulation is not told that a renderer exists --
        this is the API layer choosing what to expose, which is rule 4 of the roadmap.
        """

        if reference is None:
            return None
        genes = {
            name: reference[population.schema.index_of(name)]
            for name in MORPHOLOGY_NAMES
        }
        diet = population.phenotypes.diet[population.active][rows]
        body = {
            "volume": 0.0,
            "surface_area": 0.0,
            "mass": 0.0,
            "diet": dict.fromkeys(DIET_NAMES, 0.0),
        }
        if rows.size:
            phenotypes = population.phenotypes
            active = population.active
            body = {
                "volume": float(phenotypes.volume[active][rows].mean()),
                "surface_area": float(phenotypes.surface_area[active][rows].mean()),
                "mass": float(phenotypes.mass[active][rows].mean()),
                "diet": {
                    name: float(diet[:, index].mean())
                    for index, name in enumerate(DIET_NAMES)
                },
            }
        return {"genes": genes, **body}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole record for a run dump or an API."""

        return {
            "format": 1,
            "trait_names": list(self.trait_names),
            "biome_names": list(self.biome_names),
            "extinction_confirm_ticks": self.extinction_confirm_ticks,
            "species": [
                self.records[species].to_dict() for species in sorted(self.records)
            ],
            "samples": [sample.to_dict() for sample in self.samples],
        }


def habitat_line(fractions: tuple[float, ...], names: tuple[str, ...]) -> str:
    """Name the places a species actually lives, biggest share first.

    Generated from the occupancy numbers and never authored: the sentence contains nothing the
    distribution above it does not, which is the same rule the organism inspector's verdict is
    held to.  The floor and the term limit are what keep it a sentence -- a species spread over
    every biome is described by its largest three and the rest go unnamed rather than reciting
    ten numbers as prose.  Only a species with no members at all gets no line.
    """

    ranked = sorted(
        (
            (share, name)
            for share, name in zip(fractions, names)
            if share >= HABITAT_FLOOR
        ),
        reverse=True,
    )
    if not ranked:
        return ""
    return "; ".join(
        f"{name}, {share * 100:.0f}% of its range"
        if index == 0
        else f"{name}, {share * 100:.0f}%"
        for index, (share, name) in enumerate(ranked[:HABITAT_TERMS])
    )


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
    "habitat_line",
    "trait_means",
]
