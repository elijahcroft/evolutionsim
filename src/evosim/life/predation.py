"""Predation and herbivory: the two intake channels that need another organism.

This is the first place where one organism's traits appear in another's outcome, and it is what
turns a collection of independent foragers into a food web.  Nothing here declares trophic
levels.  A "predator" is any organism whose diet logits give it herbivore or carnivore weight,
and "prey" is whoever it managed to catch; both roles are read off the same genome.

The contest is deliberately symmetric in its inputs.  Attacking costs the attacker its
``aggression``, ``move_speed`` and ``sense_range``, all of which it already pays for in
:mod:`evosim.life.energy`; escaping costs the prey its ``camouflage``, ``armor`` and speed,
which it also already pays for.  Neither side gets a term the other cannot buy, so an arms race
is a thing the coefficients permit rather than a thing the code stages.

Three mechanics worth stating:

1. **Encounters scale with local density**, so a crowded cell is dangerous and an empty one is
   not.  Predation therefore becomes viable only where prey is abundant, which is what stops a
   carnivore from evolving in an empty ocean.

2. **A prey can only be killed once.**  When several attackers succeed against the same
   individual in one tick, the first attack drawn takes it and the rest go hungry.  Without
   this, a crowded cell could produce more meat than it contained.

3. **What the killer cannot eat stays in the world.**  A carcass is worth its body plus the
   reserve its owner never spent; the attacker assimilates the fraction its digestion and diet
   match allow, and every remaining unit is deposited as detritus rather than deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config, IntakeConfig
from evosim.life.cells import CellNeighbourhood
from evosim.life.sampling import stochastic_round
from evosim.life.phenotype import PhenotypeBatch
from evosim.life.population import Population
from evosim.world import World

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class HuntStats:
    """What predation did this tick."""

    attacks: int
    kills: int
    hunters: int
    energy_gained: float
    carrion_returned: float

    @property
    def capture_rate(self) -> float:
        """Share of attacks that ended in a kill; the arms race in one number."""

        return self.kills / self.attacks if self.attacks else 0.0


EMPTY = HuntStats(attacks=0, kills=0, hunters=0, energy_gained=0.0, carrion_returned=0.0)


@dataclass(frozen=True, slots=True)
class PredationModel:
    """Config-bound encounter and capture equations, vectorised over every attack."""

    intake: IntakeConfig
    energy_density: float

    @classmethod
    def from_config(cls, config: Config) -> PredationModel:
        return cls(
            intake=config.energy.intake,
            energy_density=config.energy.energy_density,
        )

    def encounter_rate(
        self,
        aggression: FloatArray,
        move_speed: FloatArray,
        sense_range: FloatArray,
        prey_density: FloatArray,
    ) -> FloatArray:
        """Expected attacks per attacker per tick.

        Sensing enters squared because detection sweeps an area, which is what finally gives
        ``sense_range`` a second thing to earn: in M3 it bought better foraging ground, and here
        it buys prey.  The additive floors mean a sessile, unaggressive, blind organism still
        blunders into the occasional meal rather than being categorically unable to.
        """

        intake = self.intake
        return (
            intake.k_encounter
            * (0.5 + aggression)
            * (0.3 + move_speed)
            * (0.3 + sense_range) ** 2
            * prey_density
        )

    def capture_probability(
        self,
        attacker_mass: FloatArray,
        prey_mass: FloatArray,
        attacker_speed: FloatArray,
        prey_speed: FloatArray,
        attacker_sense: FloatArray,
        attacker_aggression: FloatArray,
        prey_camouflage: FloatArray,
        prey_armor: FloatArray,
    ) -> FloatArray:
        """Logistic on the weighted contest documented in ``energy.yaml``.

        Size enters as a log2 ratio, so what matters is how many times bigger the attacker is
        rather than by how much -- the difference between a whale and a shrimp, not between one
        gram and two.  Camouflage is divided by the attacker's sensing, which is the direct
        coupling that makes a predator's eyes and a prey's concealment an arms race rather than
        two independent traits.
        """

        intake = self.intake
        # Subtracting logarithms rather than dividing first: a near-zero prey mass makes the
        # ratio overflow, while the difference of logs stays finite and merely large.
        floor = np.finfo(np.float64).tiny
        log_ratio = np.log2(np.maximum(attacker_mass, floor)) - np.log2(
            np.maximum(prey_mass, floor)
        )
        contest = (
            intake.capture_bias
            + intake.capture_size_advantage * log_ratio
            + intake.capture_speed_advantage * (attacker_speed - prey_speed)
            + intake.capture_aggression * attacker_aggression
            - intake.capture_camouflage * prey_camouflage / (1.0 + attacker_sense)
            - intake.capture_armor * prey_armor
        )
        return 1.0 / (1.0 + np.exp(-np.clip(contest, -700.0, 700.0)))

    def diet_match(
        self,
        attacker_herbivore: FloatArray,
        attacker_carnivore: FloatArray,
        prey_autotroph: FloatArray,
    ) -> FloatArray:
        """How much of a carcass this attacker's gut is actually equipped to use.

        A herbivore's affinity pays off against autotrophic prey and a carnivore's against
        heterotrophic prey, so the softmaxed diet logits decide what a lineage can eat without
        anything assigning it a trophic level.  The floor keeps a specialist from starving beside
        edible food, which would make diet a trap rather than a tradeoff.
        """

        match = (
            attacker_herbivore * prey_autotroph
            + attacker_carnivore * (1.0 - prey_autotroph)
        )
        return np.maximum(match, self.intake.diet_match_floor)

    def expected_gain(
        self,
        phenotype: PhenotypeBatch,
        occupancy: FloatArray,
        prey_mass: FloatArray,
        prey_speed: FloatArray,
        prey_autotroph: FloatArray,
        carcass_value: FloatArray,
    ) -> FloatArray:
        """Energy a cell's occupants are worth to this organism per tick.

        This is the same encounter and capture arithmetic :meth:`hunt` resolves, evaluated
        against a cell's average occupant instead of a drawn individual.  Routing the movement
        score through the identical equations is what stops "where the prey is" from becoming a
        second, parallel set of rules that could disagree with what hunting actually pays.

        ``genome.yaml`` defines ``sense_range`` as habitat *and prey* detection; this is the
        second half of that.  Arguments broadcast against a trailing candidate-cell axis.
        """

        mass = phenotype.mass.astype(np.float64)[:, None]
        speed = phenotype.trait("move_speed").astype(np.float64)[:, None]
        sense = phenotype.trait("sense_range").astype(np.float64)[:, None]
        aggression = phenotype.trait("aggression").astype(np.float64)[:, None]

        rate = self.encounter_rate(
            aggression, speed, sense, np.maximum(occupancy - 1.0, 0.0)
        )
        capture = self.capture_probability(
            mass,
            prey_mass,
            speed,
            prey_speed,
            sense,
            aggression,
            np.zeros_like(prey_mass),
            np.zeros_like(prey_mass),
        )
        match = self.diet_match(
            phenotype.diet_component("herbivore").astype(np.float64)[:, None],
            phenotype.diet_component("carnivore").astype(np.float64)[:, None],
            prey_autotroph,
        )
        digestion = phenotype.trait("digestion_efficiency").astype(np.float64)[:, None]
        return rate * capture * carcass_value * match * digestion

    def expected_risk(
        self,
        phenotype: PhenotypeBatch,
        threat: FloatArray,
        occupancy: FloatArray,
        hunter_mass: FloatArray,
        hunter_speed: FloatArray,
        hunter_sense: FloatArray,
        hunter_aggression: FloatArray,
    ) -> FloatArray:
        """Energy this organism stands to lose to a cell's hunters per tick.

        Risk is priced in the only currency the model has: being eaten costs an organism its own
        body and everything it had saved.  Expressing danger that way lets ``fear`` weight it
        against food on the same scale, so avoiding predators needs no coefficient of its own --
        and a ``fear`` of one means valuing your life at exactly what it is worth.

        The threat a cell aims at any single occupant is its total hunting capability divided
        among the occupants, which is why a crowd is safer per head than a lone individual.
        """

        mass = phenotype.mass.astype(np.float64)[:, None]
        attacks = threat / np.maximum(occupancy, 1.0)
        capture = self.capture_probability(
            hunter_mass,
            mass,
            hunter_speed,
            phenotype.trait("move_speed").astype(np.float64)[:, None],
            hunter_sense,
            hunter_aggression,
            phenotype.trait("camouflage").astype(np.float64)[:, None],
            phenotype.trait("armor").astype(np.float64)[:, None],
        )
        worth = self.energy_density * mass + phenotype.storage_capacity.astype(
            np.float64
        )[:, None]
        return attacks * capture * worth

    def hunt(
        self,
        population: Population,
        world: World,
        rng: np.random.Generator,
        allowance: FloatArray,
    ) -> tuple[FloatArray, BoolArray, HuntStats]:
        """Resolve every attack this tick.

        Returns the energy each organism gained, which prey were killed, and the summary.  The
        caller owns applying the energy and the deaths, so that predation kills join hazard and
        starvation deaths in a single compaction rather than shuffling rows mid-tick.

        ``allowance`` is how much more each attacker can actually take up -- the smaller of the
        room left in its reserve and the metabolic ceiling its oxygen supply allows.  Meat beyond
        that is not deleted; it stays on the ground as carrion.
        """

        size = population.size
        gained = np.zeros(size, dtype=np.float64)
        killed = np.zeros(size, dtype=bool)
        if size < 2:
            return gained, killed, EMPTY

        phenotype = population.phenotypes.active
        attacker_rows, target_rows = self._draw_attacks(population, rng)
        if attacker_rows.size == 0:
            return gained, killed, EMPTY

        probability = self.capture_probability(
            phenotype.mass[attacker_rows].astype(np.float64),
            phenotype.mass[target_rows].astype(np.float64),
            phenotype.trait("move_speed")[attacker_rows].astype(np.float64),
            phenotype.trait("move_speed")[target_rows].astype(np.float64),
            phenotype.trait("sense_range")[attacker_rows].astype(np.float64),
            phenotype.trait("aggression")[attacker_rows].astype(np.float64),
            phenotype.trait("camouflage")[target_rows].astype(np.float64),
            phenotype.trait("armor")[target_rows].astype(np.float64),
        )
        success = rng.random(attacker_rows.size) < probability
        attacker_rows, target_rows = attacker_rows[success], target_rows[success]
        if attacker_rows.size == 0:
            return gained, killed, HuntStats(int(success.size), 0, 0, 0.0, 0.0)

        # One carcass, one killer. np.unique returns the first occurrence of each prey, and the
        # attack order is already the random order they were drawn in, so the winner is a fair
        # draw among whoever succeeded rather than whoever happens to sit in a lower row.
        target_rows, first = np.unique(target_rows, return_index=True)
        attacker_rows = attacker_rows[first]

        carcass = self.energy_density * phenotype.mass[target_rows].astype(
            np.float64
        ) + np.maximum(population.energy[target_rows].astype(np.float64), 0.0)
        edible = carcass * self.diet_match(
            phenotype.diet_component("herbivore")[attacker_rows].astype(np.float64),
            phenotype.diet_component("carnivore")[attacker_rows].astype(np.float64),
            phenotype.diet_component("autotroph")[target_rows].astype(np.float64),
        ) * phenotype.trait("digestion_efficiency")[attacker_rows].astype(np.float64)

        # The allowance is a budget across everything one attacker eats this tick, not a limit
        # per carcass. Spending it down in order also makes a sated hunter stop: once the budget
        # is gone the remaining kills yield nothing, and an attack that yields nothing is an
        # attack the prey survives. Without this a single predator would clear an entire cell in
        # a day and leave almost all of it uneaten.
        eaten = self._spend_allowance(attacker_rows, edible, allowance)
        fed = eaten > 0.0
        attacker_rows, target_rows = attacker_rows[fed], target_rows[fed]
        carcass, eaten = carcass[fed], eaten[fed]
        if target_rows.size == 0:
            return gained, killed, HuntStats(int(success.size), 0, 0, 0.0, 0.0)

        np.add.at(gained, attacker_rows, eaten)
        killed[target_rows] = True

        remains = carcass - eaten
        world.resources.add_detritus(
            np.bincount(
                population.cell[target_rows].astype(np.intp),
                weights=remains,
                minlength=world.grid.n_cells,
            ).reshape(world.grid.shape)
        )
        return (
            gained,
            killed,
            HuntStats(
                attacks=int(success.size),
                kills=int(target_rows.size),
                hunters=int(np.unique(attacker_rows).size),
                energy_gained=float(eaten.sum()),
                carrion_returned=float(remains.sum()),
            ),
        )

    @staticmethod
    def _spend_allowance(
        attacker_rows: IntArray,
        edible: FloatArray,
        allowance: FloatArray,
    ) -> FloatArray:
        """Draw down each attacker's budget across its own kills, in order.

        Grouping by attacker turns "how much was left when this carcass came up" into one sorted
        cumulative sum, so the budget is honoured exactly without iterating over predators.
        """

        order = np.argsort(attacker_rows, kind="stable")
        rows, portions = attacker_rows[order], edible[order]
        running = np.cumsum(portions)

        # Restart the running total at each attacker's first kill.
        first = np.searchsorted(rows, rows, side="left")
        running -= np.concatenate(([0.0], running))[first]
        capped = np.minimum(running, np.maximum(allowance[rows], 0.0))

        previous = np.concatenate(([0.0], capped[:-1]))
        previous[first == np.arange(rows.size)] = 0.0

        eaten = np.empty_like(edible)
        eaten[order] = capped - previous
        return eaten

    def _draw_attacks(
        self,
        population: Population,
        rng: np.random.Generator,
    ) -> tuple[IntArray, IntArray]:
        """Expand each attacker's encounter rate into concrete attacker/target pairs."""

        active = population.active
        phenotype = population.phenotypes.active
        cells = population.cell[active].astype(np.intp)

        # Density excludes the organism itself: a lone occupant meets nobody.
        occupancy = np.bincount(cells, minlength=population.n_cells)
        density = (occupancy[cells] - 1).astype(np.float64)

        rate = self.encounter_rate(
            phenotype.trait("aggression").astype(np.float64),
            phenotype.trait("move_speed").astype(np.float64),
            phenotype.trait("sense_range").astype(np.float64),
            density,
        )
        attempts = stochastic_round(np.maximum(rate, 0.0), rng)
        hunters = np.flatnonzero(attempts > 0)
        if hunters.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        attacker_rows = np.repeat(hunters, attempts[hunters])
        neighbourhood = CellNeighbourhood.build(
            np.arange(population.size, dtype=np.int64), population.cell
        )
        target_rows, found = neighbourhood.sample_other(
            attacker_rows, population.cell, rng
        )
        return attacker_rows[found], target_rows[found]


__all__ = ["HuntStats", "PredationModel"]
