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
        # A surface organism by default: fully lit, no water column overhead. Tests that care
        # about the deep pass `light` and `depth_km` explicitly.
        "light": 1.0,
        "depth_km": 0.0,
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

    small = phenotypes(config, body_length=1.0)
    big = phenotypes(config, body_length=2.0)
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
    # Slenderness is 1/(2*radius_ratio): a wide body for its length is a stocky one. It pays
    # more support twice over now -- it is heavier, and it is less slender -- which is what a
    # derived slenderness buys over a locus that could have claimed either independently.
    stocky = model.costs_for(phenotypes(config, radius_ratio=0.6), environment, speed)
    slender = model.costs_for(phenotypes(config, radius_ratio=0.2), environment, speed)
    assert slender.support < stocky.support


def test_atmospheric_buoyancy_relieves_support_on_land_only(config: Config):
    """energy.yaml grants the pressure relief to land-dwellers; water is left unchanged."""
    dense = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, pressure=10.0))
    )
    thin = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, pressure=1.0))
    )
    organism = phenotypes(config, body_length=1.0)
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
    organism = phenotypes(config, body_length=1.0)
    slow = model.costs_for(organism, environment, np.array([1.0])).locomotion
    fast = model.costs_for(organism, environment, np.array([3.0])).locomotion
    assert fast / slow == pytest.approx(9.0)


def test_water_costs_more_to_move_through_than_land(config: Config, model: EnergyModel):
    organism = phenotypes(config, body_length=1.0)
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
    """Bergmann's rule must emerge from the body's real surface area, not be asserted."""
    speed = np.zeros(1)
    cold = uniform_environment(1, temperature_c=-20.0)
    small = phenotypes(config, body_length=1.0)
    large = phenotypes(config, body_length=4.0)
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
    organism = phenotypes(config, body_length=1.0)
    rich = uniform_environment(1, nutrients=1e6, detritus=1e6, insolation=1e3)
    basal = np.array([1e-4])
    intake = model.intake_for(organism, rich, basal)
    assert intake.total == pytest.approx(model.aerobic_scope * basal)


def test_low_oxygen_limits_what_a_large_body_can_earn(config: Config):
    """The mechanism by which a thin-oxygen planet caps size, with no rule mentioning size."""
    rich = uniform_environment(1, nutrients=1e6, detritus=1e6, insolation=1e3)
    organism = phenotypes(config, body_length=4.0)
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
    organism = phenotypes(config, body_length=1.0)
    basal = np.full(1, 1e3)  # high enough that the aerobic cap never binds
    poor = uniform_environment(1, light=0.1, nutrients=0.05, moisture=0.05)
    for field, value in (
        # `light`, not `insolation`: what reaches the organism is what it can fix.
        ("light", 1.0),
        ("nutrients", 5.0),
        ("moisture", 1.0),
    ):
        better = replace(poor, **{field: np.full(1, value)})
        assert (
            model.intake_for(organism, better, basal).autotrophy
            > model.intake_for(organism, poor, basal).autotrophy
        ), field


def test_the_deep_forecloses_autotrophy_however_bright_the_surface(
    config: Config, model: EnergyModel
):
    """Surface insolation cannot feed an organism the water column has cut off from it.

    This is the whole mechanism of the depth axis: below the photic depth, autotrophy is not
    merely a poor living, it is not a living at all, and no amount of sun on the waves changes
    that.  A lineage down there has to eat something that fell.
    """

    organism = phenotypes(config, body_length=1.0)
    basal = np.full(1, 1e3)
    lit = uniform_environment(1, insolation=1.0, light=1.0, depth_km=0.0)
    abyssal = uniform_environment(1, insolation=1.0, light=0.0, depth_km=5.0)

    assert model.intake_for(organism, lit, basal).autotrophy > 0.0
    assert model.intake_for(organism, abyssal, basal).autotrophy == 0.0


def test_each_diet_wins_its_own_zone(config: Config, model: EnergyModel):
    """The point of calibrating `k_detritus`: two livings, each best somewhere.

    Before M7b a pure detritivore was net-negative at every detritus level on every planet --
    intake is capped at `k_detritus`, and 0.06 was below the cost of simply existing.  No
    environment could make detritivory pay, so the dark ocean could not be a niche however much
    dead biomass fell into it.  What that constant buys is this table, and no more than this:
    sunlight still wins where there is sunlight.
    """

    def net(phenotype, environment):
        costs = model.costs_for(
            phenotype,
            environment,
            np.zeros(1),
            model.medium_drag(environment.on_land),
        )
        intake = model.intake_for(phenotype, environment, costs.basal)
        return float(intake.total.sum() - costs.total.sum())

    autotroph = phenotypes(
        config, aff_autotroph=6.0, aff_detritus=-2.0, aff_herbivore=-2.0, aff_carnivore=-2.0
    )
    detritivore = phenotypes(
        config, aff_autotroph=-2.0, aff_detritus=6.0, aff_herbivore=-2.0, aff_carnivore=-2.0
    )
    shelf = uniform_environment(1, light=1.0, nutrients=0.6, detritus=0.3, temperature_c=20.0)
    abyss = uniform_environment(
        1, light=0.0, nutrients=0.6, detritus=1.0, depth_km=4.0, temperature_c=20.0
    )

    # In the light, photosynthesis wins -- but detritivory is a living, not a death sentence.
    assert net(autotroph, shelf) > net(detritivore, shelf) > 0.0
    # In the dark, the ranking inverts, and it inverts because the light is gone.
    assert net(detritivore, abyss) > 0.0 > net(autotroph, abyss)


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
    # Out to the top of the body_length range. The founder sits at 1.0 and the range reaches
    # 30, which is the retired body_size range rescaled -- so this sweep covers the same span
    # of body masses it always did.
    for size in (0.4, 1.0, 3.0, 6.0, 12.0, 30.0):
        organism = phenotypes(config, body_length=size)
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
    sizes = np.linspace(0.5, 30.0, 60)
    environment = uniform_environment(1)

    def largest_viable(gravity: float) -> float:
        model = EnergyModel.from_config(
            replace(config, planet=replace(config.planet, gravity=gravity))
        )
        viable = 0.0
        for size in sizes:
            organism = phenotypes(config, body_length=float(size))
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
    """Inverting the locomotion equation must be exact, including its morphology factors."""
    organism = phenotypes(config, body_length=1.0)
    at_sea = np.zeros(1, dtype=bool)
    drag = np.array([config.energy.costs.drag_water])
    basal = np.array([0.01])
    speed = model.max_move_speed(organism, drag, basal, at_sea)
    locomotion = model.costs_for(
        organism, uniform_environment(1), speed
    ).locomotion
    assert locomotion == pytest.approx((model.aerobic_scope - 1.0) * basal)


def test_max_move_speed_is_unbounded_when_movement_is_free(config: Config):
    free = EnergyModel.from_config(
        replace(config, energy=replace(config.energy, costs=replace(config.energy.costs, k_move=0.0)))
    )
    assert np.isinf(
        free.max_move_speed(
            phenotypes(config), np.array([1.0]), np.array([1.0]), np.zeros(1, dtype=bool)
        )
    )


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


# -- morphology (milestone 8) ----------------------------------------------------------------


# What a founder organism cost and earned in a reference cell immediately *before* the
# morphology genome replaced the scalar body, captured by running the M7b code. Every intake
# and cost constant in energy.yaml was calibrated against these numbers across M3, M7 and M7b,
# so M8 was required to reproduce them rather than to re-tune around them.
_PRE_MORPHOLOGY_FOUNDER = {
    "mass": 0.06400000303983688,
    "storage_capacity": 0.12800000607967377,
    "water": {
        "basal": 0.003886098828619389,
        "support": 0.0003840000182390213,
        "locomotion": 1.638400126647952e-05,
        "sensory": 0.0003305878575493369,
        "thermoregulation": 0.0,
        "armor": 0.0,
        "max_move_speed": 8.149406028890926,
    },
    "land": {
        "basal": 0.003886098828619389,
        "support": 0.00036571430308478215,
        "locomotion": 1.02400007915497e-05,
        "sensory": 0.0003305878575493369,
        "thermoregulation": 0.0,
        "armor": 0.0,
        "max_move_speed": 10.308273851521312,
    },
    "intake": {
        "autotrophy": 0.016222200546801114,
        "detritivory": 0.0017532995899162084,
    },
}


@pytest.mark.parametrize("medium", ["water", "land"])
def test_the_founder_pays_exactly_what_it_paid_before_morphology(
    config: Config, model: EnergyModel, medium: str
):
    """The constraint the whole morphology milestone was built around.

    Replacing `mass = body_size**3` with an integrated body could have moved every number in
    energy.yaml out from under three milestones of calibration. It did not, and this is where
    that is asserted rather than assumed: `body_density` pins the founder's mass, and each new
    morphology factor is written so it evaluates to exactly 1.0 at founder proportions. A
    failure here means a coefficient in energy.yaml no longer means what its comment claims.
    """

    expected = _PRE_MORPHOLOGY_FOUNDER[medium]
    founder = phenotypes(config)
    environment = replace(
        uniform_environment(1), on_land=np.full(1, medium == "land", dtype=bool)
    )
    speed = founder.trait("move_speed").astype(np.float64)

    costs = model.costs_for(founder, environment, speed)
    assert float(founder.mass[0]) == pytest.approx(
        _PRE_MORPHOLOGY_FOUNDER["mass"], rel=1e-5
    )
    assert float(founder.storage_capacity[0]) == pytest.approx(
        _PRE_MORPHOLOGY_FOUNDER["storage_capacity"], rel=1e-5
    )
    for name in ("basal", "support", "locomotion", "sensory", "armor"):
        assert float(getattr(costs, name)[0]) == pytest.approx(
            expected[name], rel=1e-4
        ), name
    assert float(costs.thermoregulation[0]) == pytest.approx(
        expected["thermoregulation"], abs=1e-12
    )

    intake = model.intake_for(founder, environment, costs.basal)
    assert float(intake.autotrophy[0]) == pytest.approx(
        _PRE_MORPHOLOGY_FOUNDER["intake"]["autotrophy"], rel=1e-4
    )
    assert float(intake.detritivory[0]) == pytest.approx(
        _PRE_MORPHOLOGY_FOUNDER["intake"]["detritivory"], rel=1e-4
    )

    ceiling = model.max_move_speed(
        founder, model.medium_drag(environment.on_land), costs.basal, environment.on_land
    )
    assert float(ceiling[0]) == pytest.approx(expected["max_move_speed"], rel=1e-4)


def test_limbs_cost_drag_everywhere_and_earn_speed_only_on_land(
    config: Config, model: EnergyModel
):
    """The test that decides whether morphology is real or decorative.

    A limb has to appear on both sides of the ledger. Here it does: the limbed animal pays
    strictly more to move in either medium, and gets a strictly higher speed ceiling only where
    there is something to push against. Neither effect is allowed to be zero -- a free limb is a
    hidden bonus, and a purely costly one could never be selected for.
    """

    plain = phenotypes(config)
    limbed = phenotypes(config, limb_pairs=2.0, limb_ratio=0.5, limb_splay=1.0)
    assert float(limbed.limb_count[0]) == 4.0

    for on_land in (False, True):
        environment = replace(
            uniform_environment(1), on_land=np.full(1, on_land, dtype=bool)
        )
        drag = model.medium_drag(environment.on_land)
        speed = np.full(1, 1.0)

        plain_costs = model.costs_for(plain, environment, speed)
        limbed_costs = model.costs_for(limbed, environment, speed)
        # Same speed, same medium: the extra bill is the limbs and nothing else.
        assert float(limbed_costs.locomotion[0]) > float(plain_costs.locomotion[0])

        basal = np.full(1, 0.01)
        plain_ceiling = model.max_move_speed(
            plain, drag, basal, environment.on_land
        )
        limbed_ceiling = model.max_move_speed(
            limbed, drag, basal, environment.on_land
        )
        if on_land:
            assert float(limbed_ceiling[0]) > float(plain_ceiling[0])
        else:
            # At sea a limb is drag and nothing else, so it strictly lowers the ceiling.
            assert float(limbed_ceiling[0]) < float(plain_ceiling[0])


def test_a_segmented_or_lopsided_body_costs_more_to_hold_up(
    config: Config, model: EnergyModel
):
    """Support is where the skeleton is paid for, and both terms are zero for the founder."""

    environment = uniform_environment(1)
    speed = np.zeros(1)
    plain = model.costs_for(phenotypes(config), environment, speed)
    segmented = model.costs_for(
        phenotypes(config, segment_count=10.0), environment, speed
    )
    lopsided = model.costs_for(phenotypes(config, taper=0.9), environment, speed)

    assert float(segmented.support[0]) > float(plain.support[0])
    assert float(lopsided.support[0]) > float(plain.support[0])


def test_a_dorsal_fin_is_surface_without_volume(config: Config, model: EnergyModel):
    """A fin is cheap to grow and expensive to keep warm, which is what makes it a trade."""

    cold = uniform_environment(1, temperature_c=-20.0)
    speed = np.zeros(1)
    plain = phenotypes(config)
    finned = phenotypes(config, dorsal_fin=1.0)

    assert float(finned.mass[0]) == pytest.approx(float(plain.mass[0]), rel=1e-6)
    assert float(finned.surface_area[0]) > float(plain.surface_area[0])
    assert (
        model.costs_for(finned, cold, speed).thermoregulation[0]
        > model.costs_for(plain, cold, speed).thermoregulation[0]
    )
