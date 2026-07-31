"""Reproduction: the step that closes the loop from a cohort into a lineage.

Everything here is arithmetic on stored energy.  There is no fecundity bonus and no fitness
term: an organism has offspring because it accumulated enough energy to pay for them, and the
r/K tradeoff falls out of the fact that the same reserve buys either many cheap offspring or
few expensive ones.

Four decisions worth stating, because each one could reasonably have gone another way:

1. **Offspring are endowed from the parent's storage capacity, not their own.**  ``energy.yaml``
   describes the cost as ``parental_investment * offspring_storage_capacity``, but an offspring's
   capacity depends on a genome that does not exist until the parent has already committed to
   making it.  Estimating with the parent's capacity breaks that circularity, costs one mutation
   step of accuracy, and -- crucially -- keeps what the parent paid exactly equal to what the
   offspring received.  A ledger that balances is worth more than a second decimal place.

2. **A parent that cannot afford its whole brood has a smaller one**, rather than skipping the
   attempt.  All-or-nothing would make ``offspring_count`` a cliff: a lineage one unit of energy
   short of a brood of four would produce nothing at all rather than three.

3. **Reproduction happens after mortality**, so the dead do not breed and newborns are not aged,
   fed, or killed on the day they are born.

4. **The overhead is deposited as detritus, not deleted.**  ``reproduction.overhead`` above one
   models gametes and failed births; that material is real, and destroying it would put a third
   hole in the matter ledger that :mod:`evosim.sim` otherwise keeps closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config, ReproductionConfig
from evosim.life.genome import GenomeSchema
from evosim.life.movement import stochastic_round
from evosim.life.population import UNASSIGNED, Population
from evosim.rng import RngBundle
from evosim.world import World

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class MateSearchNotImplementedError(NotImplementedError):
    """Raised for a configured mate search radius the model cannot honour.

    Silently ignoring a configured value is precisely the failure this project's config
    discipline exists to prevent, so a radius beyond the parent's own cell fails loudly rather
    than quietly behaving as zero.
    """


@dataclass(frozen=True, slots=True)
class BirthStats:
    """What reproduction did this tick."""

    parents: int
    births: int
    sexual_births: int
    throttled: int
    energy_invested: float
    energy_overhead: float

    @property
    def energy_spent(self) -> float:
        """Total drawn from parental reserves, endowment plus overhead."""

        return self.energy_invested + self.energy_overhead


EMPTY = BirthStats(
    parents=0,
    births=0,
    sexual_births=0,
    throttled=0,
    energy_invested=0.0,
    energy_overhead=0.0,
)


@dataclass(frozen=True, slots=True)
class ReproductionModel:
    """Config-bound reproduction, vectorised over every parent at once."""

    reproduction: ReproductionConfig
    schema: GenomeSchema
    radiation: float
    neighbours: IntArray

    @classmethod
    def for_world(cls, config: Config, world: World, schema: GenomeSchema) -> ReproductionModel:
        if config.energy.reproduction.mate_search_radius > 0.0:
            raise MateSearchNotImplementedError(
                "energy.reproduction.mate_search_radius > 0 is not implemented; mates are "
                "found in the parent's own cell only. Set it to 0."
            )
        return cls(
            reproduction=config.energy.reproduction,
            schema=schema,
            radiation=config.planet.radiation,
            neighbours=world.grid.neighbour_indices().reshape(world.grid.n_cells, 4),
        )

    def reproduce(
        self,
        population: Population,
        world: World,
        rng: RngBundle,
    ) -> BirthStats:
        """Create this tick's offspring, charge their parents, and return what happened."""

        if population.size == 0:
            return EMPTY

        parents, brood = self._brood_sizes(population, rng)
        if parents.size == 0:
            return EMPTY

        parent_of = np.repeat(parents, brood)
        throttled = max(int(parent_of.size) - population.available, 0)
        if throttled:
            # Which offspring are dropped must be random. Taking the first that fit would drop
            # them in parent-row order, and rows are roughly age-ordered after compaction, so
            # the cap would quietly impose selection for older parents -- a hidden pressure of
            # exactly the kind this project forbids, and one that would contaminate every
            # selection result measured at the cap.
            keep = rng.repro.choice(parent_of.size, size=population.available, replace=False)
            keep.sort()
            parent_of = parent_of[keep]
        if parent_of.size == 0:
            return BirthStats(0, 0, 0, throttled, 0.0, 0.0)

        genomes, sexual, mates = self._offspring_genomes(population, parent_of, rng)
        cells = self._disperse(population.cell[parent_of], rng)
        endowment = (
            population.phenotypes.trait("parental_investment")[parent_of].astype(np.float64)
            * population.phenotypes.storage_capacity[parent_of].astype(np.float64)
        )

        overhead = (self.reproduction.overhead - 1.0) * endowment
        self._charge_parents(population, world, parent_of, endowment + overhead, overhead)

        population.add(
            genomes,
            cells,
            energy=endowment.astype(np.float32),
            parent_id=self._parentage(population, parent_of, mates),
            generation=population.generation[parent_of] + 1,
        )
        return BirthStats(
            parents=int(np.unique(parent_of).size),
            births=int(parent_of.size),
            sexual_births=int(np.count_nonzero(sexual)),
            throttled=throttled,
            energy_invested=float(endowment.sum()),
            energy_overhead=float(overhead.sum()),
        )

    # -- who breeds, and how many -------------------------------------------------------

    def _brood_sizes(
        self,
        population: Population,
        rng: RngBundle,
    ) -> tuple[IntArray, IntArray]:
        """Return the rows that reproduce and how many offspring each actually affords."""

        active = population.active
        phenotype = population.phenotypes.active
        capacity = phenotype.storage_capacity.astype(np.float64)
        energy = population.energy[active].astype(np.float64)

        ready = (
            (population.age[active] >= phenotype.trait("maturity_age"))
            & (energy >= phenotype.trait("repro_threshold").astype(np.float64) * capacity)
            & (capacity > 0.0)
        )
        candidates = np.flatnonzero(ready)
        if candidates.size == 0:
            return candidates.astype(np.int64), np.zeros(0, dtype=np.int64)

        cost_each = (
            self.reproduction.overhead
            * phenotype.trait("parental_investment")[candidates].astype(np.float64)
            * capacity[candidates]
        )
        wanted = stochastic_round(
            phenotype.trait("offspring_count")[candidates].astype(np.float64), rng.repro
        )
        affordable = np.floor(
            np.divide(
                energy[candidates],
                cost_each,
                out=np.zeros_like(cost_each),
                where=cost_each > 0.0,
            )
        ).astype(np.int64)

        brood = np.minimum(wanted, affordable)
        breeding = brood > 0
        return candidates[breeding].astype(np.int64), brood[breeding]

    # -- genomes -------------------------------------------------------------------------

    def _offspring_genomes(
        self,
        population: Population,
        parent_of: IntArray,
        rng: RngBundle,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], IntArray]:
        """Recombine or clone, then mutate.

        A sexual attempt that finds no compatible mate in the cell falls back to asexual rather
        than failing.  That is what makes ``sex_bias`` a smooth dial: a lineage can drift toward
        sex while it is still too sparse to find partners, without being punished for it.
        """

        mates, sexual = self._choose_mates(population, parent_of, rng)
        first = population.genomes[parent_of]
        second = population.genomes[np.maximum(mates, 0)]
        recombined = self.schema.recombine(first, second, rng.recombination)
        # An asexual offspring is a clone of its parent, so it takes the parent's genome
        # unchanged; only the sexual rows keep the recombined result.
        genomes = np.where(sexual[:, None, None], recombined, first)
        return self.schema.mutate(genomes, rng.mutation, self.radiation), sexual, mates

    def _choose_mates(
        self,
        population: Population,
        parent_of: IntArray,
        rng: RngBundle,
    ) -> tuple[IntArray, NDArray[np.bool_]]:
        """Pick a compatible partner in the parent's own cell, where one is wanted and exists.

        Mate compatibility is a genetic-distance threshold, which is what will let assortative
        mating reinforce a split once :mod:`evosim.evolution` exists: two diverging groups
        sharing a cell stop exchanging alleles before anything declares them separate species.
        """

        wants_sex = rng.repro.random(parent_of.size) < population.phenotypes.trait("sex_bias")[
            parent_of
        ].astype(np.float64)
        mates = np.full(parent_of.size, UNASSIGNED, dtype=np.int64)
        if not np.any(wants_sex):
            return mates, wants_sex

        # Mature organisms are the mating pool. Sorting them by cell turns "find somebody here"
        # into a pair of searchsorted bounds, which is the whole reason this is not a loop.
        phenotype = population.phenotypes.active
        pool = np.flatnonzero(
            population.age[population.active] >= phenotype.trait("maturity_age")
        )
        order = np.argsort(population.cell[pool], kind="stable")
        by_cell = pool[order]
        cells = population.cell[by_cell]

        seekers = np.flatnonzero(wants_sex)
        rows = parent_of[seekers]
        start = np.searchsorted(cells, population.cell[rows], side="left")
        stop = np.searchsorted(cells, population.cell[rows], side="right")
        occupants = stop - start

        # Draw from the block excluding the seeker itself, then step over its own slot.
        position = np.empty(pool.size, dtype=np.int64)
        position[order] = np.arange(pool.size)
        own_slot = position[np.searchsorted(pool, rows)]
        draw = start + (
            rng.repro.random(rows.size) * np.maximum(occupants - 1, 1)
        ).astype(np.int64)
        draw += draw >= own_slot
        chosen = by_cell[np.clip(draw, 0, by_cell.size - 1)]

        distance = self.schema.genetic_distance(
            population.genomes[rows], population.genomes[chosen]
        )
        compatible = (occupants >= 2) & (
            distance <= self.reproduction.mate_compatibility_distance
        )
        mates[seekers] = np.where(compatible, chosen, UNASSIGNED)
        wants_sex[seekers] = compatible
        return mates, wants_sex

    def _parentage(
        self,
        population: Population,
        parent_of: IntArray,
        mates: IntArray,
    ) -> IntArray:
        """Persistent IDs of both parents; the second is unset for a clone."""

        parents = np.full((parent_of.size, 2), UNASSIGNED, dtype=np.int64)
        parents[:, 0] = population.organism_id[parent_of]
        paired = mates >= 0
        parents[paired, 1] = population.organism_id[mates[paired]]
        return parents

    # -- placement and payment -------------------------------------------------------------

    def _disperse(self, origin: NDArray[np.integer], rng: RngBundle) -> NDArray[np.int32]:
        """Random-walk each offspring away from its parent's cell.

        Dispersal is undirected: a newborn has no information about where it is going, and
        giving it the parent's foraging judgement for free would be a hidden bonus.
        """

        cells = origin.astype(np.int64, copy=True)
        steps = stochastic_round(
            np.full(cells.size, self.reproduction.dispersal_radius), rng.repro
        )
        for step in range(int(steps.max()) if steps.size else 0):
            moving = np.flatnonzero(steps > step)
            if moving.size == 0:
                break
            options = self.neighbours[cells[moving]]
            reachable = options >= 0
            # Polar rows have three neighbours, not four. Weighting the draw by how many exist
            # keeps dispersal uniform over real directions rather than piling up on the pole.
            pick = (
                rng.repro.random(moving.size) * reachable.sum(axis=1)
            ).astype(np.int64)
            slot = np.argsort(~reachable, kind="stable")[np.arange(moving.size), pick]
            cells[moving] = options[np.arange(moving.size), slot]
        return cells.astype(np.int32)

    def _charge_parents(
        self,
        population: Population,
        world: World,
        parent_of: IntArray,
        spend: FloatArray,
        overhead: FloatArray,
    ) -> None:
        """Deduct the full cost from each parent and return the overhead to the world."""

        active = population.active
        charge = np.bincount(parent_of, weights=spend, minlength=population.size)
        population.energy[active] = (
            population.energy[active].astype(np.float64) - charge
        ).astype(np.float32)

        n_cells = world.grid.n_cells
        world.resources.add_detritus(
            np.bincount(
                population.cell[parent_of].astype(np.intp),
                weights=overhead,
                minlength=n_cells,
            ).reshape(world.grid.shape)
        )


__all__ = [
    "BirthStats",
    "MateSearchNotImplementedError",
    "ReproductionModel",
]
