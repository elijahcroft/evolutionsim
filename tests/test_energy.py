"""Tests for the energy-model seam.

The project's central claim is that no trait is advantageous or costly except by earning or
spending energy.  These tests pin the shape of each equation, and -- more importantly -- pin
the *emergent* consequences the design documents promise, such as high gravity favouring small
bodies.  If one of those falls out of a rule rather than out of the arithmetic, it will not
survive a change to the coefficients, and these tests are what would catch that.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.energy import (
    EnergyModel,
    Environment,
    apply_resource_contention,
    saturation,
    thermal_excess,
)
from evosim.life.genome import GenomeSchema
from evosim.life.phenotype import PhenotypeBuffer
from evosim.rng import RngBundle
from evosim.world import World


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def model(config: Config) -> EnergyModel:
    return EnergyModel.from_config(config)


@pytest.fixture(scope="module")
def world(config: Config) -> World:
    return World.create(config.planet, RngBundle(config.sim.seed))


def phenotypes(config: Config, count: int = 1, **traits: float):
    """Build a batch of identical organisms, overriding named loci."""
    schema = GenomeSchema.from_config(config.genome)
    genomes = schema.founders(count)
    for name, value in traits.items():
        genomes[:, schema.index_of(name), :] = value
    return PhenotypeBuffer.from_genomes(genomes, schema).active


def uniform_environment(count: int, **fields: float) -> Environment:
    defaults = {
        "temperature_c": 15.0,
        "insolation": 1.0,
        "moisture": 1.0,
        "nutrients": 1.0,
        "detritus": 1.0,
        "toxicity": 0.0,
    }
    defaults.update(fields)
    return Environment(
        on_land=np.zeros(count, dtype=bool),
        **{name: np.full(count, value) for name, value in defaults.items()},
    )


# -- primitives ---------------------------------------------------------------------------


def test_saturation_is_half_at_the_half_saturation_constant():
    assert saturation(np.array([0.25]), 0.25) == pytest.approx(0.5)
    assert saturation(np.array([0.0]), 0.25) == pytest.approx(0.0)


def test_saturation_is_monotone_and_bounded():
    values = saturation(np.linspace(0.0, 1000.0, 500), 0.25)
    assert np.all(np.diff(values) > 0.0)
    assert np.all((values >= 0.0) & (values < 1.0))


def test_thermal_excess_is_zero_inside_the_tolerance_window():
    optimum = np.full(3, 15.0)
    tolerance = np.full(3, 8.0)
    excess = thermal_excess(np.array([15.0, 23.0, 30.0]), optimum, tolerance)
    assert excess == pytest.approx([0.0, 0.0, 7.0])


# -- environment sampling -----------------------------------------------------------------


def test_environment_sample_matches_the_world_fields(world: World):
    cells = np.array([0, 100, world.grid.n_cells - 1])
    sample = Environment.sample(world, cells)
    assert sample.temperature_c == pytest.approx(
        world.climate.temperature_c.ravel()[cells]
    )
    assert sample.nutrients == pytest.approx(world.resources.nutrients.ravel()[cells])
    assert np.array_equal(sample.on_land, world.terrain.land.ravel()[cells])


def test_environment_sample_rejects_out_of_range_cells(world: World):
    with pytest.raises(IndexError):
        Environment.sample(world, np.array([world.grid.n_cells]))


# -- costs --------------------------------------------------------------------------------


def test_basal_cost_follows_kleiber_scaling(model: EnergyModel):
    one = np.array([1.0])
    cost = model.basal_cost(one, one, one, np.array([15.0]))
    doubled = model.basal_cost(np.array([2.0]), one, one, np.array([15.0]))
    assert doubled / cost == pytest.approx(2.0**0.75)


def test_basal_cost_doubles_per_q10_step_of_body_temperature(config: Config, model):
    costs = config.energy.costs
    one = np.array([1.0])
    reference = model.basal_cost(one, one, one, np.array([costs.basal_reference_temp_c]))
    hotter = model.basal_cost(one, one, one, np.array([costs.basal_reference_temp_c + 10.0]))
    assert hotter / reference == pytest.approx(costs.basal_q10)


@pytest.mark.parametrize(
    "trait,value",
    [
        ("temp_tolerance", 40.0),
        ("digestion_efficiency", 0.9),
        ("radiation_tolerance", 4.0),
    ],
)
def test_every_upkeep_trait_raises_basal_cost(config: Config, model, trait, value):
    """Generalism and quality tissue must be paid for, or nothing constrains them."""
    baseline = model.upkeep_multiplier(phenotypes(config, **{trait: 0.05}))
    expensive = model.upkeep_multiplier(phenotypes(config, **{trait: value}))
    assert expensive > baseline


def test_slow_ageing_costs_upkeep(config: Config, model: EnergyModel):
    """`upkeep_longevity` charges for (1 - senescence_rate), so long life is not free."""
    short = model.upkeep_multiplier(phenotypes(config, senescence_rate=1.0))
    long = model.upkeep_multiplier(phenotypes(config, senescence_rate=0.0))
    assert long > short


def test_support_cost_is_linear_in_gravity_and_mass(config: Config):
    heavy = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, gravity=2.0))
    )
    normal = EnergyModel.from_config(config)
    environment = uniform_environment(1)
    speed = np.zeros(1)

    small = phenotypes(config, body_size=1.0)
    big = phenotypes(config, body_size=2.0)
    assert (
        heavy.costs_for(small, environment, speed).support
        / normal.costs_for(small, environment, speed).support
    ) == pytest.approx(2.0)
    assert (
        normal.costs_for(big, environment, speed).support
        / normal.costs_for(small, environment, speed).support
    ) == pytest.approx(8.0)  # mass = size^3


def test_slender_bodies_cost_less_to_support(config: Config, model: EnergyModel):
    environment = uniform_environment(1)
    speed = np.zeros(1)
    stocky = model.costs_for(phenotypes(config, body_slenderness=0.5), environment, speed)
    slender = model.costs_for(phenotypes(config, body_slenderness=2.5), environment, speed)
    assert slender.support < stocky.support


def test_atmospheric_buoyancy_relieves_support_on_land_only(config: Config):
    """energy.yaml grants the pressure relief to land-dwellers; water is left unchanged."""
    dense = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, pressure=10.0))
    )
    thin = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, pressure=1.0))
    )
    organism = phenotypes(config, body_size=1.0)
    speed = np.zeros(1)

    on_land = uniform_environment(1)
    on_land = replace(on_land, on_land=np.ones(1, dtype=bool))
    assert (
        dense.costs_for(organism, on_land, speed).support
        < thin.costs_for(organism, on_land, speed).support
    )

    at_sea = uniform_environment(1)
    assert dense.costs_for(organism, at_sea, speed).support == pytest.approx(
        thin.costs_for(organism, at_sea, speed).support
    )


def test_locomotion_cost_is_quadratic_in_speed(config: Config, model: EnergyModel):
    environment = uniform_environment(1)
    organism = phenotypes(config, body_size=1.0)
    slow = model.costs_for(organism, environment, np.array([1.0])).locomotion
    fast = model.costs_for(organism, environment, np.array([3.0])).locomotion
    assert fast / slow == pytest.approx(9.0)


def test_water_costs_more_to_move_through_than_land(config: Config, model: EnergyModel):
    organism = phenotypes(config, body_size=1.0)
    speed = np.array([1.0])
    at_sea = uniform_environment(1)
    on_land = replace(uniform_environment(1), on_land=np.ones(1, dtype=bool))
    assert (
        model.costs_for(organism, at_sea, speed).locomotion
        > model.costs_for(organism, on_land, speed).locomotion
    )


def test_thermoregulation_is_free_inside_tolerance_and_linear_outside(config, model):
    organism = phenotypes(config, temp_optimum=15.0, temp_tolerance=8.0)
    speed = np.zeros(1)
    inside = model.costs_for(organism, uniform_environment(1, temperature_c=20.0), speed)
    assert inside.thermoregulation == pytest.approx(0.0)

    near = model.costs_for(organism, uniform_environment(1, temperature_c=25.0), speed)
    far = model.costs_for(organism, uniform_environment(1, temperature_c=27.0), speed)
    assert far.thermoregulation / near.thermoregulation == pytest.approx(4.0 / 2.0)


def test_large_bodies_are_thermally_cheaper_per_unit_mass(config, model):
    """Bergmann's rule must emerge from the 2/3 surface exponent, not be asserted."""
    speed = np.zeros(1)
    cold = uniform_environment(1, temperature_c=-20.0)
    small = phenotypes(config, body_size=1.0)
    large = phenotypes(config, body_size=4.0)
    per_mass = [
        (model.costs_for(p, cold, speed).thermoregulation / p.mass).item()
        for p in (small, large)
    ]
    assert per_mass[1] < per_mass[0]


# -- intake -------------------------------------------------------------------------------


def test_aerobic_scope_is_one_at_the_reference_oxygen_fraction(config: Config):
    reference = replace(
        config.planet, o2_fraction=config.planet.o2_reference
    )
    model = EnergyModel.from_config(replace(config, planet=reference))
    assert model.aerobic_scope == pytest.approx(config.energy.intake.aerobic_scope_max)


def test_low_oxygen_lowers_the_metabolic_ceiling(config: Config):
    thin = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, o2_fraction=0.02))
    )
    assert thin.aerobic_scope < EnergyModel.from_config(config).aerobic_scope


def test_intake_is_capped_by_the_aerobic_scope(config: Config, model: EnergyModel):
    """A rich cell cannot be exploited faster than oxygen supply allows."""
    organism = phenotypes(config, body_size=1.0)
    rich = uniform_environment(1, nutrients=1e6, detritus=1e6, insolation=1e3)
    basal = np.array([1e-4])
    intake = model.intake_for(organism, rich, basal)
    assert intake.total == pytest.approx(model.aerobic_scope * basal)


def test_low_oxygen_limits_what_a_large_body_can_earn(config: Config):
    """The mechanism by which a thin-oxygen planet caps size, with no rule mentioning size."""
    rich = uniform_environment(1, nutrients=1e6, detritus=1e6, insolation=1e3)
    organism = phenotypes(config, body_size=4.0)
    basal = np.array([1e-3])
    thick = EnergyModel.from_config(config).intake_for(organism, rich, basal).total
    thin = (
        EnergyModel.from_config(
            replace(config, planet=replace(config.planet, o2_fraction=0.02))
        )
        .intake_for(organism, rich, basal)
        .total
    )
    assert thin < thick


def test_photosynthesis_rises_with_every_input(config: Config, model: EnergyModel):
    organism = phenotypes(config, body_size=1.0)
    basal = np.full(1, 1e3)  # high enough that the aerobic cap never binds
    poor = uniform_environment(1, insolation=0.1, nutrients=0.05, moisture=0.05)
    for field, value in (
        ("insolation", 1.0),
        ("nutrients", 5.0),
        ("moisture", 1.0),
    ):
        better = replace(poor, **{field: np.full(1, value)})
        assert (
            model.intake_for(organism, better, basal).autotrophy
            > model.intake_for(organism, poor, basal).autotrophy
        ), field


def test_specialisation_is_structural_not_a_bonus(config: Config, model: EnergyModel):
    """Diet logits are softmaxed, so gaining detritivory necessarily costs autotrophy."""
    basal = np.full(1, 1e3)
    environment = uniform_environment(1)
    autotroph = model.intake_for(phenotypes(config), environment, basal)
    detritivore = model.intake_for(
        phenotypes(config, aff_detritus=4.0), environment, basal
    )
    assert detritivore.detritivory > autotroph.detritivory
    assert detritivore.autotrophy < autotroph.autotrophy


def test_intake_draws_exactly_the_stock_it_credits(config: Config, model: EnergyModel):
    """The ledger: no organism may be credited without a pool being debited."""
    basal = np.full(4, 1e3)
    environment = uniform_environment(4)
    intake = model.intake_for(phenotypes(config, 4, aff_detritus=2.0), environment, basal)
    assert intake.nutrient_draw == pytest.approx(
        intake.autotrophy * config.energy.intake.nutrient_draw_per_energy
    )
    assert intake.detritus_draw == pytest.approx(intake.detritivory)


def test_autotroph_size_ceiling_exists_on_the_reference_planet(config, model):
    """Net energy must eventually turn negative with size, or bodies grow without limit."""
    environment = uniform_environment(1)
    net = []
    for size in (0.4, 1.0, 3.0, 6.0, 12.0):
        organism = phenotypes(config, body_size=size)
        basal = model.basal_cost(
            organism.mass.astype(np.float64),
            organism.trait("metabolic_rate").astype(np.float64),
            model.upkeep_multiplier(organism),
            organism.trait("temp_optimum").astype(np.float64),
        )
        intake = model.intake_for(organism, environment, basal).total
        cost = model.costs_for(organism, environment, np.zeros(1)).total
        net.append((intake - cost).item())
    assert net[0] > 0.0
    assert net[-1] < 0.0


def test_elaborate_senses_are_unaffordable_on_a_small_body(config: Config, model):
    """`sense_range` scales at 1.5 while its owner's intake scales at 2/3 of mass.

    A founder-sized organism with maximal senses therefore cannot pay for them, which is the
    counterweight to the foraging advantage sensing buys: perception is something a lineage has
    to grow into rather than something it can simply have.
    """
    environment = uniform_environment(1)
    basal = np.full(1, 1e3)
    intake = model.intake_for(phenotypes(config), environment, basal).total
    keen = model.costs_for(
        phenotypes(config, sense_range=8.0), environment, np.zeros(1)
    )
    assert keen.sensory > intake


def test_high_gravity_shrinks_the_largest_viable_body(config: Config):
    """The headline emergent claim: nothing anywhere says 'high gravity favours small'."""
    sizes = np.linspace(0.2, 12.0, 60)
    environment = uniform_environment(1)

    def largest_viable(gravity: float) -> float:
        model = EnergyModel.from_config(
            replace(config, planet=replace(config.planet, gravity=gravity))
        )
        viable = 0.0
        for size in sizes:
            organism = phenotypes(config, body_size=float(size))
            basal = model.basal_cost(
                organism.mass.astype(np.float64),
                organism.trait("metabolic_rate").astype(np.float64),
                model.upkeep_multiplier(organism),
                organism.trait("temp_optimum").astype(np.float64),
            )
            intake = model.intake_for(organism, environment, basal).total
            cost = model.costs_for(organism, environment, np.zeros(1)).total
            if intake > cost:
                viable = float(size)
        return viable

    assert largest_viable(4.0) < largest_viable(1.0)


# -- aerobic ceiling on activity ------------------------------------------------------------


def test_max_move_speed_exactly_exhausts_the_activity_budget(config: Config, model):
    mass = np.array([1.0])
    drag = np.array([config.energy.costs.drag_water])
    basal = np.array([0.01])
    speed = model.max_move_speed(mass, drag, basal)
    locomotion = model.costs_for(
        phenotypes(config, body_size=1.0), uniform_environment(1), speed
    ).locomotion
    assert locomotion == pytest.approx((model.aerobic_scope - 1.0) * basal)


def test_max_move_speed_is_unbounded_when_movement_is_free(config: Config):
    free = EnergyModel.from_config(
        replace(config, energy=replace(config.energy, costs=replace(config.energy.costs, k_move=0.0)))
    )
    assert np.isinf(free.max_move_speed(np.array([1.0]), np.array([1.0]), np.array([1.0])))


# -- contention -----------------------------------------------------------------------------


def test_contention_is_absent_when_the_pool_can_meet_demand():
    share = apply_resource_contention(
        np.array([0.1, 0.2]), np.array([0, 1]), np.array([1.0, 1.0]), 2
    )
    assert share == pytest.approx([1.0, 1.0])


def test_contention_shares_a_scarce_pool_in_proportion_to_appetite():
    """Density dependence with nothing counting neighbours: the pool is simply finite."""
    draw = np.array([3.0, 1.0])
    share = apply_resource_contention(draw, np.array([0, 0]), np.array([2.0, 9.0]), 2)
    assert share == pytest.approx([0.5, 0.5])
    assert float((draw * share).sum()) == pytest.approx(2.0)


def test_contention_leaves_unrelated_cells_alone():
    share = apply_resource_contention(
        np.array([10.0, 0.1]), np.array([0, 1]), np.array([1.0, 1.0]), 2
    )
    assert share[0] == pytest.approx(0.1)
    assert share[1] == pytest.approx(1.0)


def test_contention_handles_an_empty_population():
    share = apply_resource_contention(
        np.zeros(0), np.zeros(0, dtype=np.intp), np.array([1.0]), 1
    )
    assert share.shape == (0,)
