"""The biological tick: the one place the population and the world touch.

``Simulation`` is pure and headless.  It never reads the wall clock, never imports from the
server or UI layers, and draws randomness only from its own :class:`~evosim.rng.RngBundle`, so
a run is reproducible from ``(config, seed)`` alone.

Tick order is a design decision, not an implementation detail, so it is stated once here and
followed exactly:

1. **Age.** Everyone alive at the start of the tick gets a day older.
2. **Sense and move.** Organisms compare their cell with its neighbours and travel, paying the
   locomotion cost of the medium they set out from.
3. **Pay.** Every cost is charged at the cell the organism ended the tick in.
4. **Eat.** Intake is capped first by oxygen, then by what the shared cell pool can actually
   supply, and the corresponding stock is removed from the world.
5. **Die.** Environmental hazards, senescence, and an empty ledger.  Corpses become detritus in
   the cell where they fell.
6. **Breed.** Survivors with enough reserve convert it into offspring.
7. **Regrow.** Climate advances and resource pools recover.
8. **Record.** Species are counted, extinctions confirmed, and -- on the configured cadences --
   the taxonomy is revisited and the time series sampled.

Moving before feeding is what makes movement worth its cost: an organism that finds a better
cell eats there the same day.  Breeding after mortality means the dead do not reproduce and
newborns are not aged, fed, or killed on the day they are born.  Regrowing last means organisms
consume the world as they found it that morning, so a cell cannot be harvested and replenished
within a single tick.  Recording happens after everything else and changes nothing: no organism
behaves differently for having been counted.

Energy is not conserved -- autotrophy creates it from light, which is the point -- but *matter*
is.  Every unit of nutrient or detritus an organism assimilates is removed from a pool; every
corpse is returned to one; and the energy lost to ``reproduction.overhead`` is deposited as
detritus rather than deleted.  :class:`TickStats` reports every side so the ledger can be
checked rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config
from evosim.evolution.taxonomy import TaxonomyModel
from evosim.history.records import History
from evosim.life.census import CellCensus
from evosim.life.energy import (
    Costs,
    EnergyModel,
    Environment,
    Intake,
    apply_resource_contention,
)
from evosim.life.mortality import MortalityModel, starved
from evosim.life.movement import MovementModel
from evosim.life.population import Population
from evosim.life.predation import HuntStats, PredationModel
from evosim.life.reproduction import BirthStats, ReproductionModel
from evosim.rng import RngBundle
from evosim.world import World

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TickStats:
    """What one tick did, in enough detail to explain it afterwards.

    Costs and deaths are broken out by cause because the project's stated goal is that outcomes
    be explainable after the fact; a single "500 died" number would make that impossible.
    """

    day: int
    population: int
    deaths: int
    deaths_starvation: int
    deaths_hazard: int
    deaths_predation: int
    attacks: int
    kills: int
    hunters: int
    intake_predation: float
    carrion_returned: float
    births: int
    sexual_births: int
    breeding_parents: int
    capacity_throttle: int
    species: int
    species_born: int
    species_extinct: int
    energy_to_offspring: float
    energy_reproduction_overhead: float
    energy_intake: float
    energy_cost: float
    cost_basal: float
    cost_support: float
    cost_locomotion: float
    cost_sensory: float
    cost_thermoregulation: float
    cost_armor: float
    intake_autotrophy: float
    intake_detritivory: float
    nutrients_drawn: float
    detritus_consumed: float
    detritus_deposited: float
    cells_moved: int
    mean_energy: float
    mean_energy_fullness: float

    @property
    def net_energy(self) -> float:
        """Population-wide surplus this tick; negative means the biosphere is running down."""

        return self.energy_intake - self.energy_cost

    @property
    def growth(self) -> int:
        """Net change in living organisms; the number a lineage's survival turns on."""

        return self.births - self.deaths


@dataclass(slots=True)
class Simulation:
    """A world, a population, and the tick that couples them."""

    config: Config
    rng: RngBundle
    world: World
    population: Population
    energy: EnergyModel
    mortality: MortalityModel
    movement: MovementModel
    predation: PredationModel
    reproduction: ReproductionModel
    taxonomy: TaxonomyModel
    history: History
    last_stats: TickStats | None = field(default=None)

    @classmethod
    def create(cls, config: Config) -> Simulation:
        """Build a complete simulation from configuration alone."""

        rng = RngBundle(config.sim.seed)
        world = World.create(config.planet, rng)
        population = Population.seed_founders(config, world, rng)
        history = History.from_config(config, population.phenotypes.trait_names)
        history.found(world.day, population)
        return cls(
            config=config,
            rng=rng,
            world=world,
            population=population,
            energy=EnergyModel.from_config(config),
            mortality=MortalityModel.from_config(config),
            movement=MovementModel.for_world(world),
            predation=PredationModel.from_config(config),
            reproduction=ReproductionModel.for_world(config, world, population.schema),
            taxonomy=TaxonomyModel.from_config(config, population.schema),
            history=history,
        )

    @property
    def day(self) -> int:
        return self.world.day

    def run(self, ticks: int) -> TickStats | None:
        """Advance ``ticks`` days and return the final tick's statistics."""

        if not isinstance(ticks, (int, np.integer)) or isinstance(ticks, bool):
            raise TypeError("ticks must be an integer")
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        stats = self.last_stats
        for _ in range(int(ticks)):
            stats = self.step()
        return stats

    def step(self) -> TickStats:
        """Advance exactly one simulated day."""

        population = self.population
        active = population.active
        population.age[active] += 1

        departure = Environment.sample(self.world, population.cell[active])
        phenotype = population.phenotypes.active
        drag = self.energy.medium_drag(departure.on_land)
        basal = self.energy.basal_cost(
            phenotype.mass.astype(np.float64),
            phenotype.trait("metabolic_rate").astype(np.float64),
            self.energy.upkeep_multiplier(phenotype),
            phenotype.trait("temp_optimum").astype(np.float64),
        )

        # Locomotion is billed on the continuous speed while the organism takes a whole number
        # of steps. That is the correct pairing: the cost is the expected work, and the steps
        # are one unbiased sample of it, so the two agree over a lifetime rather than per tick.
        speed = self.movement.realized_speed(
            population, self.energy, drag, basal, departure.on_land
        )
        census = CellCensus.build(
            population,
            self.config.energy.energy_density,
            self.config.energy.intake.k_encounter,
        )
        moved = self.movement.move(
            population,
            self.world,
            self.energy,
            self.predation,
            census,
            speed,
            basal,
            self.rng.move,
        )

        arrival = Environment.sample(self.world, population.cell[active])
        costs = self.energy.costs_for(phenotype, arrival, speed, drag)
        intake = self._feed(arrival, basal)

        # Storage capacity is a hard ceiling: an organism with nowhere to put a surplus simply
        # does not keep it. That surplus becomes offspring at the breeding stage below.
        capacity = phenotype.storage_capacity.astype(np.float64)
        updated = (
            population.energy[active].astype(np.float64) + intake.total - costs.total
        )
        settled = np.minimum(updated, capacity)

        # Hunting happens on settled ledgers, so a carcass is worth what its owner actually had
        # rather than what it held before paying for the day. What an attacker may take up is
        # bounded twice: by the room left in its reserve, and by the same aerobic ceiling that
        # limits passive intake -- energy.yaml caps *all* intake at a multiple of basal cost, and
        # a predator is not exempt from its own metabolism.
        allowance = np.minimum(
            capacity - settled,
            np.maximum(self.energy.aerobic_scope * basal - intake.total, 0.0),
        )
        gained, predated, hunt = self.predation.hunt(
            population, self.world, self.rng.encounter, allowance
        )
        population.energy[active] = (settled + gained).astype(np.float32)

        deposited, starvation, hazard, predation = self._reap(arrival, predated)
        births = self.reproduction.reproduce(population, self.world, self.rng)

        self.world.step()
        born, extinct = self._record()
        stats = self._summarise(
            costs=costs,
            intake=intake,
            deposited=deposited + births.energy_overhead + hunt.carrion_returned,
            starvation_deaths=starvation,
            hazard_deaths=hazard,
            predation_deaths=predation,
            cells_moved=int(moved.sum()),
            births=births,
            hunt=hunt,
            species_born=born,
            species_extinct=extinct,
        )
        self.last_stats = stats
        return stats

    # -- tick stages ---------------------------------------------------------------------

    def _feed(self, environment: Environment, basal: FloatArray) -> Intake:
        """Assimilate energy and remove the matching stock from the world.

        Contention is resolved per pool rather than jointly, because a cell's nutrients and its
        detritus are separate stocks: exhausting one says nothing about the other.
        """

        population = self.population
        active = population.active
        cells = population.cell[active].astype(np.intp)
        n_cells = self.world.grid.n_cells
        raw = self.energy.intake_for(population.phenotypes.active, environment, basal)

        nutrient_share = apply_resource_contention(
            raw.nutrient_draw,
            cells,
            self.world.resources.nutrients.reshape(-1),
            n_cells,
        )
        detritus_share = apply_resource_contention(
            raw.detritus_draw,
            cells,
            self.world.resources.detritus.reshape(-1),
            n_cells,
        )
        intake = Intake(
            autotrophy=raw.autotrophy * nutrient_share,
            detritivory=raw.detritivory * detritus_share,
            nutrient_draw=raw.nutrient_draw * nutrient_share,
            detritus_draw=raw.detritus_draw * detritus_share,
        )

        shape = self.world.grid.shape
        self.world.resources.nutrients -= np.bincount(
            cells, weights=intake.nutrient_draw, minlength=n_cells
        ).reshape(shape)
        self.world.resources.detritus -= np.bincount(
            cells, weights=intake.detritus_draw, minlength=n_cells
        ).reshape(shape)
        np.clip(
            self.world.resources.nutrients,
            0.0,
            None,
            out=self.world.resources.nutrients,
        )
        np.clip(
            self.world.resources.detritus, 0.0, None, out=self.world.resources.detritus
        )
        return intake

    def _reap(
        self,
        environment: Environment,
        predated: NDArray[np.bool_],
    ) -> tuple[float, int, int, int]:
        """Kill, return the dead to the detritus pool, and compact the population.

        A corpse carries both the body that was built and whatever the organism had not yet
        spent, so nothing an organism accumulated leaves the world when it dies.  Organisms
        killed by a predator are excluded from that deposit: the hunt already split their
        carcass between the killer and the ground, and depositing it again would let a food web
        manufacture matter.
        """

        population = self.population
        active = population.active
        phenotype = population.phenotypes.active

        hazards = self.mortality.hazards(
            phenotype, environment, population.age[active]
        )
        hazard_dead = self.rng.death.random(population.size) < hazards.combined
        starvation_dead = starved(population.energy[active])
        dead = hazard_dead | starvation_dead | predated
        if not np.any(dead):
            return 0.0, 0, 0, 0

        # Deaths are attributed in a fixed order so the causes always sum to the total: being
        # eaten beats an empty ledger, which beats whatever the hazard dice said.
        predation_deaths = int(np.count_nonzero(predated))
        starvation_deaths = int(np.count_nonzero(starvation_dead & ~predated))
        hazard_deaths = int(np.count_nonzero(dead)) - predation_deaths - starvation_deaths

        buried = dead & ~predated
        corpse = (
            self.config.energy.energy_density * phenotype.mass[buried].astype(np.float64)
            + np.maximum(population.energy[active][buried].astype(np.float64), 0.0)
        )
        cells = population.cell[active][buried].astype(np.intp)
        deposit = np.bincount(
            cells, weights=corpse, minlength=self.world.grid.n_cells
        ).reshape(self.world.grid.shape)
        self.world.resources.add_detritus(deposit)

        population.remove(dead)
        return float(corpse.sum()), starvation_deaths, hazard_deaths, predation_deaths

    def _record(self) -> tuple[int, int]:
        """Revisit the taxonomy on its cadence, then count and sample on theirs.

        The taxonomy runs first so that a species born today is counted today; ``observe`` runs
        every tick regardless, because an extinction date read off a 100-day grid would be an
        extinction date wrong by up to 100 days.
        """

        day = self.world.day
        sim = self.config.sim
        born = 0
        if day % sim.taxonomy_interval == 0:
            splits = self.taxonomy.split(
                self.population, self.rng.speciation, self.history.next_species
            )
            born = len(self.history.apply(day, self.population, splits))
        extinct = self.history.observe(day, self.population)
        if day % sim.sample_interval == 0:
            self.history.sample(day, self.population)
        return born, extinct

    def _summarise(
        self,
        *,
        costs: Costs,
        intake: Intake,
        deposited: float,
        starvation_deaths: int,
        hazard_deaths: int,
        predation_deaths: int,
        cells_moved: int,
        births: BirthStats,
        hunt: HuntStats,
        species_born: int,
        species_extinct: int,
    ) -> TickStats:
        """Reduce the tick's arrays to the scalars a run log can carry."""

        survivors = self.population.active
        energy = self.population.energy[survivors].astype(np.float64)
        capacity = self.population.phenotypes.storage_capacity[survivors].astype(
            np.float64
        )
        fullness = np.divide(
            energy, capacity, out=np.zeros_like(energy), where=capacity > 0.0
        )
        return TickStats(
            day=self.world.day,
            population=self.population.size,
            deaths=starvation_deaths + hazard_deaths + predation_deaths,
            deaths_starvation=starvation_deaths,
            deaths_hazard=hazard_deaths,
            deaths_predation=predation_deaths,
            attacks=hunt.attacks,
            kills=hunt.kills,
            hunters=hunt.hunters,
            intake_predation=hunt.energy_gained,
            carrion_returned=hunt.carrion_returned,
            births=births.births,
            sexual_births=births.sexual_births,
            breeding_parents=births.parents,
            capacity_throttle=births.throttled,
            species=len(self.history.living()),
            species_born=species_born,
            species_extinct=species_extinct,
            energy_to_offspring=births.energy_invested,
            energy_reproduction_overhead=births.energy_overhead,
            energy_intake=float(intake.total.sum()),
            energy_cost=float(costs.total.sum()),
            cost_basal=float(costs.basal.sum()),
            cost_support=float(costs.support.sum()),
            cost_locomotion=float(costs.locomotion.sum()),
            cost_sensory=float(costs.sensory.sum()),
            cost_thermoregulation=float(costs.thermoregulation.sum()),
            cost_armor=float(costs.armor.sum()),
            intake_autotrophy=float(intake.autotrophy.sum()),
            intake_detritivory=float(intake.detritivory.sum()),
            nutrients_drawn=float(intake.nutrient_draw.sum()),
            detritus_consumed=float(intake.detritus_draw.sum()),
            detritus_deposited=deposited,
            cells_moved=cells_moved,
            mean_energy=float(energy.mean()) if energy.size else 0.0,
            mean_energy_fullness=float(fullness.mean()) if fullness.size else 0.0,
        )


__all__ = ["Simulation", "TickStats"]
