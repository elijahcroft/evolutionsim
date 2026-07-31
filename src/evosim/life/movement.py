"""Movement over grid neighbours.

An organism compares its own cell with its four neighbours and picks one, weighted by how much
better each looks.  Three things are worth stating about how that comparison is built, because
each of them exists to avoid inventing a rule the energy model does not already imply:

1. **Cells are scored with the energy model itself**, via
   :meth:`~evosim.life.energy.EnergyModel.foraging_yield`.  There is no separate notion of
   "habitat quality", so preference cannot drift out of agreement with what actually feeds an
   organism.  It also means the movement layer introduces no coefficients of its own.

2. **The score is divided by basal cost** before becoming a choice weight.  Basal cost is the
   organism's own energy scale, so "worth moving for" means the same thing to a large animal
   and a small one, and no arbitrary temperature constant is needed to turn energy into a
   probability.

3. **``sense_range`` gates how much of that comparison an organism can see.**  Neighbours are
   one cell away, so an organism with a range below one cell weighs them only partially and one
   with no senses at all wanders blind.  Without this, ``sense_range`` would be a pure cost with
   nothing to earn and selection would drive it to zero on every planet.

The per-step work is vectorised over the organisms that still have steps left, so the cost of
this module scales with movement actually performed rather than with population size.  The only
Python loop is over step index, bounded by the fastest organism alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.life.census import CellCensus
from evosim.life.energy import EnergyModel
from evosim.life.population import Population
from evosim.life.predation import PredationModel
from evosim.life.sampling import stochastic_round
from evosim.world import World

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Candidate 0 is "stay"; candidates 1..4 are the north/east/south/west neighbours in the order
# Grid.neighbour_indices returns them, so candidate index minus one is a heading.
_STAY = 0


@dataclass(frozen=True, slots=True)
class MovementModel:
    """Neighbour lookup plus the cell-choice rule, bound to one world's geometry."""

    neighbours: IntArray

    @classmethod
    def for_world(cls, world: World) -> MovementModel:
        """Precompute the flat neighbour table once per world."""

        return cls(neighbours=world.grid.neighbour_indices().reshape(world.grid.n_cells, 4))

    def realized_speed(
        self,
        population: Population,
        energy: EnergyModel,
        drag: FloatArray,
        basal: FloatArray,
    ) -> FloatArray:
        """Genetic speed, clipped to what the aerobic ceiling can pay for.

        Returning the clipped value rather than silently failing to move matters: this is the
        speed the locomotion cost is charged on, so an organism on a low-oxygen planet is
        neither billed for nor credited with movement it could not perform.
        """

        active = population.active
        wanted = population.phenotypes.trait("move_speed", active).astype(np.float64)
        ceiling = energy.max_move_speed(
            population.phenotypes.mass[active].astype(np.float64),
            drag,
            basal,
        )
        return np.minimum(wanted, ceiling)

    def move(
        self,
        population: Population,
        world: World,
        energy: EnergyModel,
        predation: PredationModel,
        census: CellCensus,
        speed: FloatArray,
        basal: FloatArray,
        rng: np.random.Generator,
    ) -> IntArray:
        """Advance every organism and return how many cells each actually travelled.

        Cells are updated in place.  Headings are only rewritten by organisms that moved, so
        ``move_persistence`` remembers a direction across ticks spent standing still.
        """

        size = population.size
        moved = np.zeros(size, dtype=np.int64)
        if size == 0:
            return moved

        remaining = stochastic_round(np.maximum(speed, 0.0), rng)
        max_steps = int(remaining.max()) if remaining.size else 0
        for step in range(max_steps):
            movers = np.flatnonzero(remaining > step)
            if movers.size == 0:
                break
            self._take_one_step(
                population, world, energy, predation, census, basal, rng, movers
            )
            moved[movers] += 1
        return moved

    def _take_one_step(
        self,
        population: Population,
        world: World,
        energy: EnergyModel,
        predation: PredationModel,
        census: CellCensus,
        basal: FloatArray,
        rng: np.random.Generator,
        movers: IntArray,
    ) -> None:
        """Choose and apply one cell transition for the given rows."""

        origin = population.cell[movers].astype(np.intp)
        candidates = np.concatenate(
            (origin[:, None], self.neighbours[origin]), axis=1
        )
        # Polar rows have no north or south neighbour. Gather from a safe index and mask the
        # result, rather than branching, so the expression stays a single vectorised pass.
        reachable = candidates >= 0
        lookup = np.where(reachable, candidates, 0).astype(np.intp)

        phenotype = population.phenotypes.select(movers)
        score = energy.foraging_yield(
            phenotype,
            world.climate.insolation.ravel()[lookup],
            world.resources.nutrients.ravel()[lookup],
            world.climate.moisture.ravel()[lookup],
            world.resources.detritus.ravel()[lookup],
            world.climate.temperature_c.ravel()[lookup],
        )
        # Other organisms are food and danger as well as competition, and both sides of that
        # are priced by the predation model rather than by anything invented here. `fear`
        # weights the danger: at one an organism values its life at exactly what a predator
        # would gain from it, below one it discounts the risk, above one it is skittish.
        score = score + predation.expected_gain(
            phenotype,
            census.occupancy[lookup],
            census.mass[lookup],
            census.speed[lookup],
            census.autotroph[lookup],
            census.carcass_value[lookup],
        ) - phenotype.trait("fear").astype(np.float64)[:, None] * predation.expected_risk(
            phenotype,
            census.threat[lookup],
            census.occupancy[lookup],
            census.hunter_mass[lookup],
            census.speed[lookup],
            census.hunter_sense[lookup],
            census.hunter_aggression[lookup],
        )

        scale = np.maximum(basal[movers], np.finfo(np.float64).tiny)
        advantage = (score - score[:, _STAY][:, None]) / scale[:, None]
        knowledge = np.clip(
            phenotype.trait("sense_range").astype(np.float64), 0.0, 1.0
        )
        logits = knowledge[:, None] * advantage

        # Persistence is added directly to the previous heading's log-weight. The locus is
        # already in [0, 1], the natural scale for a log-odds nudge, so this needs no
        # coefficient of its own -- and it is free, as genome.yaml documents.
        heading = population.heading[movers]
        continuing = np.flatnonzero(heading >= 0)
        if continuing.size:
            logits[continuing, heading[continuing] + 1] += phenotype.trait(
                "move_persistence"
            ).astype(np.float64)[continuing]

        logits = np.where(reachable, logits, -np.inf)
        chosen = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)

        rows = np.arange(movers.size)
        destination = candidates[rows, chosen]
        stepped = chosen != _STAY
        population.cell[movers[stepped]] = destination[stepped].astype(np.int32)
        population.heading[movers[stepped]] = (chosen[stepped] - 1).astype(np.int8)


__all__ = ["MovementModel", "stochastic_round"]
