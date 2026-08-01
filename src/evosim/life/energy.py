"""The energy-model seam: every cost and every intake an organism can experience.

Nothing in this project may make a trait advantageous or costly except through this module
and :mod:`evosim.life.mortality`.  A trait earns by appearing in an intake term here, and pays
by appearing in a cost term here.  There are no bonuses, multipliers, or special cases applied
elsewhere.

Every function is a broadcasting NumPy expression over explicit arrays rather than a method on
an organism.  That is what lets the movement layer evaluate the same intake formula at five
candidate cells at once, and what will let the morphology work after M5 replace the body terms
without the tick loop noticing.

Units.  One tick is one simulated day.  Energy is in the arbitrary unit that
``energy.energy_density`` converts body mass into, so an organism's stored energy, its costs,
its intake, and the corpse it leaves are all directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config, EnergyConfig, PlanetConfig
from evosim.life.phenotype import PhenotypeBatch
from evosim.world import World

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

_Q10_DEGREES_C = 10.0


def saturation(quantity: FloatArray, half_saturation: float) -> FloatArray:
    """Michaelis-Menten availability in ``[0, 1)``.

    Used for every consumable pool so that a resource is never a hard switch: doubling a scarce
    pool nearly doubles uptake, while doubling an abundant one barely matters.  That curvature
    is what makes depleted cells worth leaving.
    """

    return quantity / (quantity + half_saturation)


@dataclass(frozen=True, slots=True)
class Environment:
    """World fields sampled at each organism's cell.

    Sampling once per tick keeps the world's 2-D layout inside this class: every consumer below
    sees flat per-organism arrays and never indexes a grid.
    """

    temperature_c: FloatArray
    insolation: FloatArray
    light: FloatArray
    depth_km: FloatArray
    moisture: FloatArray
    nutrients: FloatArray
    detritus: FloatArray
    toxicity: FloatArray
    on_land: BoolArray

    @classmethod
    def sample(cls, world: World, cells: NDArray[np.integer]) -> Environment:
        """Gather every field an organism in ``cells`` experiences."""

        index = np.asarray(cells, dtype=np.intp)
        if index.size and (index.min() < 0 or index.max() >= world.grid.n_cells):
            raise IndexError(
                f"cell index outside [0, {world.grid.n_cells}) while sampling environment"
            )
        return cls(
            temperature_c=world.climate.temperature_c.ravel()[index],
            insolation=world.climate.insolation.ravel()[index],
            light=world.light.ravel()[index],
            depth_km=world.depth_km.ravel()[index],
            moisture=world.climate.moisture.ravel()[index],
            nutrients=world.resources.nutrients.ravel()[index],
            detritus=world.resources.detritus.ravel()[index],
            toxicity=world.toxicity.ravel()[index],
            on_land=world.terrain.land.ravel()[index],
        )


@dataclass(frozen=True, slots=True)
class Costs:
    """The six per-tick expenditures, kept separate so a run can be explained."""

    basal: FloatArray
    support: FloatArray
    locomotion: FloatArray
    sensory: FloatArray
    thermoregulation: FloatArray
    armor: FloatArray

    @property
    def total(self) -> FloatArray:
        return (
            self.basal
            + self.support
            + self.locomotion
            + self.sensory
            + self.thermoregulation
            + self.armor
        )


@dataclass(frozen=True, slots=True)
class Intake:
    """Assimilated energy, and the resource stock each channel consumed to produce it.

    ``nutrient_draw`` and ``detritus_draw`` are what must be removed from the world.  Keeping
    them beside the energy they bought is what makes the ledger checkable: no consumer can
    credit an organism without also naming the pool it came out of.
    """

    autotrophy: FloatArray
    detritivory: FloatArray
    nutrient_draw: FloatArray
    detritus_draw: FloatArray

    @property
    def total(self) -> FloatArray:
        return self.autotrophy + self.detritivory


@dataclass(frozen=True, slots=True)
class EnergyModel:
    """Config-bound evaluator for the cost and intake equations in ``energy.yaml``."""

    planet: PlanetConfig
    energy: EnergyConfig

    @classmethod
    def from_config(cls, config: Config) -> EnergyModel:
        return cls(planet=config.planet, energy=config.energy)

    # -- physiological ceiling -----------------------------------------------------------

    @property
    def aerobic_scope(self) -> float:
        """Multiple of basal cost that oxygen supply allows as intake plus activity.

        This is the only place planetary oxygen enters the simulation.  A thin-oxygen planet
        therefore limits body size and activity by making the metabolic ceiling too low to pay
        for them, not by any rule that mentions size.
        """

        intake = self.energy.intake
        ratio = self.planet.o2_fraction / self.planet.o2_reference
        return float(
            intake.aerobic_scope_max * ratio**intake.aerobic_scope_o2_exponent
        )

    def max_move_speed(
        self,
        phenotype: PhenotypeBatch,
        drag: FloatArray,
        basal: FloatArray,
        on_land: BoolArray,
    ) -> FloatArray:
        """Highest speed whose locomotion cost still fits inside the aerobic ceiling.

        Basal metabolism itself consumes one unit of scope, so ``scope - 1`` is what remains
        for activity.  Inverting the locomotion equation for speed is exact, so no organism can
        ever spend more on movement than its oxygen supply supports.

        Limbs then multiply the result, but only on land.  This is the one place morphology
        *earns* rather than costs, and it is deliberately narrow: a limb pushes against a
        substrate, so in water it is drag and nothing else.  A lineage that grows legs at sea
        pays for them every tick and gets nothing back, which is what makes limbs a decision.
        """

        costs = self.energy.costs
        budget = max(self.aerobic_scope - 1.0, 0.0) * basal
        denominator = (
            costs.k_move
            * phenotype.mass.astype(np.float64)
            * self.planet.gravity**costs.move_gravity_exponent
            * drag
            * self.frontal_drag(phenotype)
            * self.limb_drag(phenotype)
        )
        ceiling = np.sqrt(
            np.divide(
                budget,
                denominator,
                out=np.full_like(np.asarray(denominator, dtype=np.float64), np.inf),
                where=denominator > 0.0,
            )
        )
        return ceiling * np.where(on_land, self.limb_thrust(phenotype), 1.0)

    # -- morphology terms ----------------------------------------------------------------
    #
    # Three small factors, each written so that a founder-shaped animal evaluates to exactly
    # 1.0.  That is what let M8 add geometry to the cost equations without re-tuning a single
    # constant that M3, M7 and M7b had calibrated against the old scalar body.

    def frontal_drag(self, phenotype: PhenotypeBatch) -> FloatArray:
        """How blunt this body is, relative to a founder-shaped body of the same volume.

        Note what this is *not*: it is not frontal area. The locomotion equation is already
        proportional to mass, and mass and frontal area both grow with body size, so charging
        the raw area would bill a large animal twice for being large -- a body six times the
        founder's length would pay fifty times more to move, which is arithmetic, not biology.

        Dividing frontal area by ``volume**(2/3)`` removes the size and leaves the shape. The
        result is exactly 1 for any founder-shaped animal at any size, above 1 for a blunter one
        and below 1 for a more streamlined one, so streamlining becomes something selection can
        find while size keeps being paid for exactly once.
        """

        cross_section = phenotype.cross_section.astype(np.float64)
        volume = phenotype.volume.astype(np.float64)
        return (
            cross_section
            / np.cbrt(volume * volume)
            / self.energy.costs.drag_shape_reference
        )

    def limb_drag(self, phenotype: PhenotypeBatch) -> FloatArray:
        """What limbs cost to drag through any medium."""

        costs = self.energy.costs
        return 1.0 + costs.limb_drag_cost * phenotype.limb_count.astype(
            np.float64
        ) * phenotype.trait("limb_ratio").astype(np.float64)

    def limb_thrust(self, phenotype: PhenotypeBatch) -> FloatArray:
        """What limbs earn against a substrate.

        Splay is the difference between a leg and a paddle: limbs held under the body push
        against the ground, limbs held out to the side do not.
        """

        costs = self.energy.costs
        return 1.0 + costs.limb_thrust * phenotype.limb_count.astype(
            np.float64
        ) * phenotype.trait("limb_ratio").astype(np.float64) * phenotype.trait(
            "limb_splay"
        ).astype(np.float64)

    def support_morphology(self, phenotype: PhenotypeBatch) -> FloatArray:
        """The skeleton bill for a segmented or unbalanced body."""

        costs = self.energy.costs
        segments = phenotype.trait("segment_count").astype(np.float64)
        taper = phenotype.trait("taper").astype(np.float64)
        return (
            1.0
            + costs.support_segment_cost * np.maximum(segments - 1.0, 0.0)
            + costs.support_taper_cost * np.abs(taper)
        )

    # -- costs ---------------------------------------------------------------------------

    def basal_cost(
        self,
        mass: FloatArray,
        metabolic_rate: FloatArray,
        upkeep: FloatArray,
        body_temperature_c: FloatArray,
    ) -> FloatArray:
        """``k_basal * m^0.75 * metabolic_rate * Q10(T_body) * upkeep``.

        Body temperature is the organism's ``temp_optimum``, because that is the temperature it
        pays thermoregulation to hold.  Running hot is therefore not free: it doubles basal
        cost every ``basal_q10`` degrees, which is the counterweight to the wider thermal niche
        a high optimum opens up.
        """

        costs = self.energy.costs
        q10 = costs.basal_q10 ** (
            (body_temperature_c - costs.basal_reference_temp_c) / _Q10_DEGREES_C
        )
        return costs.k_basal * mass**costs.basal_mass_exponent * metabolic_rate * q10 * upkeep

    def upkeep_multiplier(self, phenotype: PhenotypeBatch) -> FloatArray:
        """Tissue that must be maintained whether or not it is used.

        Every term here is a generalist's bill: a wide thermal window, a good gut, radiation
        repair, and slow ageing are each paid for in basal metabolism every single tick.  This
        is what stops the genome from simply maximising all of them.
        """

        costs = self.energy.costs
        return (
            1.0
            + costs.upkeep_temp_tolerance * phenotype.trait("temp_tolerance")
            + costs.upkeep_digestion * phenotype.trait("digestion_efficiency")
            + costs.upkeep_radiation_tolerance * phenotype.trait("radiation_tolerance")
            + costs.upkeep_longevity * (1.0 - phenotype.trait("senescence_rate"))
        ).astype(np.float64)

    def medium_drag(self, on_land: BoolArray) -> FloatArray:
        """Resistance of the medium the organism is currently moving through."""

        costs = self.energy.costs
        return np.where(on_land, costs.drag_land, costs.drag_water).astype(np.float64)

    def relative_surface(self, surface_area: FloatArray) -> FloatArray:
        """Surface area as a multiple of the founder's, shared by heat loss and armor plating."""

        return surface_area / self.energy.costs.thermo_area_reference

    def thermoregulation_cost(
        self,
        surface_area: FloatArray,
        temp_optimum: FloatArray,
        temp_tolerance: FloatArray,
        temperature_c: FloatArray,
    ) -> FloatArray:
        """Cost of holding body temperature outside the free window, scaled by real area.

        Heat leaves across a surface, so this is charged on the integrated surface of the actual
        body.  Bergmann's rule still emerges -- a larger body has less surface per unit volume --
        but now so does the rest of the shape argument: a finned or flattened animal is expensive
        to keep warm in a cold cell and a compact one is not, and neither fact is written
        anywhere as a rule.
        """

        costs = self.energy.costs
        excess = thermal_excess(temperature_c, temp_optimum, temp_tolerance)
        return costs.k_thermo * self.relative_surface(surface_area) * excess

    def costs_for(
        self,
        phenotype: PhenotypeBatch,
        environment: Environment,
        speed: FloatArray,
        drag: FloatArray | None = None,
    ) -> Costs:
        """Evaluate every expenditure for one aligned batch of organisms.

        ``speed`` is passed in rather than read from the phenotype because the movement layer
        has already clipped it to what the aerobic ceiling allows; charging the genetic speed
        would bill an organism for movement its oxygen supply never let it perform.  ``drag``
        is separable for the same reason: an organism that swam ashore this tick paid the water
        it left, not the air it arrived in.
        """

        costs = self.energy.costs
        mass = phenotype.mass.astype(np.float64)
        surface_area = phenotype.surface_area.astype(np.float64)
        if drag is None:
            drag = self.medium_drag(environment.on_land)

        basal = self.basal_cost(
            mass,
            phenotype.trait("metabolic_rate").astype(np.float64),
            self.upkeep_multiplier(phenotype),
            phenotype.trait("temp_optimum").astype(np.float64),
        )

        # A dense atmosphere carries part of the body's weight, but only for organisms standing
        # on land; energy.yaml applies the buoyancy relief there and nowhere else.
        support = (
            costs.k_support
            * mass
            * self.planet.gravity
            * (1.0 + phenotype.trait("armor"))
            / phenotype.slenderness.astype(np.float64)
            * self.support_morphology(phenotype)
        )
        support = np.where(
            environment.on_land,
            support / (1.0 + costs.pressure_buoyancy * self.planet.pressure),
            support,
        )

        locomotion = (
            costs.k_move
            * mass
            * speed**2
            * self.planet.gravity**costs.move_gravity_exponent
            * drag
            * self.frontal_drag(phenotype)
            * self.limb_drag(phenotype)
        )
        sensory = (
            costs.k_sense
            * phenotype.trait("sense_range").astype(np.float64) ** costs.sense_range_exponent
            * mass**costs.sense_mass_exponent
        )
        thermoregulation = self.thermoregulation_cost(
            surface_area,
            phenotype.trait("temp_optimum").astype(np.float64),
            phenotype.trait("temp_tolerance").astype(np.float64),
            environment.temperature_c,
        )
        # Armor is plating over a surface, so it shares the surface term thermoregulation uses
        # rather than carrying a duplicate config key. energy.yaml records that choice.
        armor = (
            costs.k_armor
            * phenotype.trait("armor").astype(np.float64)
            * self.relative_surface(surface_area)
        )
        return Costs(
            basal=basal,
            support=np.asarray(support, dtype=np.float64),
            locomotion=np.asarray(locomotion, dtype=np.float64),
            sensory=np.asarray(sensory, dtype=np.float64),
            thermoregulation=np.asarray(thermoregulation, dtype=np.float64),
            armor=np.asarray(armor, dtype=np.float64),
        )

    # -- intake --------------------------------------------------------------------------

    def photosynthesis(
        self,
        mass: FloatArray,
        diet_autotroph: FloatArray,
        digestion_efficiency: FloatArray,
        light: FloatArray,
        nutrients: FloatArray,
        moisture: FloatArray,
    ) -> FloatArray:
        """Energy fixed from light, nutrients, and water.

        Mass enters at 2/3 while basal cost enters at 3/4, so an autotroph has a size at which
        its own upkeep overtakes what its surface can collect.  That ceiling, not any rule, is
        why an autotrophic lineage stays small until it finds a denser energy source.

        ``light`` is what reaches the organism, not what reaches the surface.  Below the photic
        depth it is effectively zero, so this term is too however autotrophic the genome is:
        the deep ocean forecloses one way of making a living rather than taxing it.
        """

        intake = self.energy.intake
        return (
            intake.k_photo
            * diet_autotroph
            * mass**intake.photo_mass_exponent
            * light
            * saturation(nutrients, intake.nutrient_half_saturation)
            * saturation(moisture, intake.moisture_half_saturation)
            * digestion_efficiency
        )

    def detritivory(
        self,
        mass: FloatArray,
        diet_detritus: FloatArray,
        digestion_efficiency: FloatArray,
        detritus: FloatArray,
    ) -> FloatArray:
        """Energy assimilated from the dead-biomass pool."""

        intake = self.energy.intake
        return (
            intake.k_detritus
            * diet_detritus
            * mass**intake.detritus_mass_exponent
            * saturation(detritus, intake.detritus_half_saturation)
            * digestion_efficiency
        )

    def foraging_yield(
        self,
        phenotype: PhenotypeBatch,
        light: FloatArray,
        nutrients: FloatArray,
        moisture: FloatArray,
        detritus: FloatArray,
        temperature_c: FloatArray,
    ) -> FloatArray:
        """Net cell-dependent energy: what an organism gains minus what the cell costs it.

        Only the terms that vary between cells appear, so this is exactly the quantity a cell
        choice can change.  The movement layer scores candidate cells with it, which means
        habitat preference is a consequence of the energy model rather than a second, parallel
        set of rules that could disagree with it.  Field arguments broadcast against a trailing
        candidate axis.
        """

        mass = phenotype.mass.astype(np.float64)[:, None]
        digestion = phenotype.trait("digestion_efficiency").astype(np.float64)[:, None]
        gain = self.photosynthesis(
            mass,
            phenotype.diet_component("autotroph").astype(np.float64)[:, None],
            digestion,
            light,
            nutrients,
            moisture,
        ) + self.detritivory(
            mass,
            phenotype.diet_component("detritus").astype(np.float64)[:, None],
            digestion,
            detritus,
        )
        return gain - self.thermoregulation_cost(
            phenotype.surface_area.astype(np.float64)[:, None],
            phenotype.trait("temp_optimum").astype(np.float64)[:, None],
            phenotype.trait("temp_tolerance").astype(np.float64)[:, None],
            temperature_c,
        )

    def intake_for(
        self,
        phenotype: PhenotypeBatch,
        environment: Environment,
        basal: FloatArray,
    ) -> Intake:
        """Evaluate assimilation and the resource draw that pays for it.

        Two limits are applied, in physiological then environmental order.  Oxygen caps what an
        organism *can* absorb; the shared cell pool then caps what it *may*, and that second
        step lives in :func:`apply_resource_contention` because it couples organisms together.

        The stock consumed equals the energy assimilated: an autotroph removes
        ``nutrient_draw_per_energy`` of nutrient per unit fixed, and a detritivore ingests
        ``energy / digestion_efficiency`` of detritus and voids the unassimilated remainder
        straight back into the same cell, so the pool falls by the assimilated amount only.
        """

        mass = phenotype.mass.astype(np.float64)
        digestion = phenotype.trait("digestion_efficiency").astype(np.float64)
        autotrophy = self.photosynthesis(
            mass,
            phenotype.diet_component("autotroph").astype(np.float64),
            digestion,
            environment.light,
            environment.nutrients,
            environment.moisture,
        )
        detritivory = self.detritivory(
            mass,
            phenotype.diet_component("detritus").astype(np.float64),
            digestion,
            environment.detritus,
        )

        gross = autotrophy + detritivory
        ceiling = self.aerobic_scope * basal
        scale = np.divide(
            ceiling,
            gross,
            out=np.ones_like(gross),
            where=gross > ceiling,
        )
        autotrophy = autotrophy * scale
        detritivory = detritivory * scale
        return Intake(
            autotrophy=autotrophy,
            detritivory=detritivory,
            nutrient_draw=autotrophy * self.energy.intake.nutrient_draw_per_energy,
            detritus_draw=detritivory,
        )


def thermal_excess(
    temperature_c: FloatArray,
    temp_optimum: FloatArray,
    temp_tolerance: FloatArray,
) -> FloatArray:
    """Degrees by which the environment falls outside the free thermal window.

    Shared by the thermoregulation cost and the thermal hazard so that a lineage can never be
    charged by one and spared by the other.
    """

    return np.maximum(np.abs(temperature_c - temp_optimum) - temp_tolerance, 0.0)


def apply_resource_contention(
    draw: FloatArray,
    cells: NDArray[np.integer],
    available: FloatArray,
    n_cells: int,
) -> FloatArray:
    """Return the per-organism share when a cell's pool cannot meet total demand.

    Everyone in an oversubscribed cell is scaled by the same factor, so a crowded cell starves
    its occupants in proportion to their appetite.  This is the whole of density dependence in
    M3: nothing anywhere counts neighbours, but a cell's pool is finite and shared.

    Two bin-counts and one gather; no per-organism iteration.
    """

    if draw.size == 0:
        return np.ones(0, dtype=np.float64)
    demand = np.bincount(cells, weights=draw, minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_scale = np.divide(
            available,
            demand,
            out=np.ones_like(demand),
            where=demand > available,
        )
    return np.clip(cell_scale[cells], 0.0, 1.0)


__all__ = [
    "Costs",
    "EnergyModel",
    "Environment",
    "Intake",
    "apply_resource_contention",
    "saturation",
    "thermal_excess",
]
