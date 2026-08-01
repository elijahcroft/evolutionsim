"""Milestone 1 world invariants and acceptance tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evosim.config import Config
from evosim.rng import RngBundle
from evosim.world import World
from evosim.world.climate import Climate, daily_insolation
from evosim.world.grid import Grid
from evosim.world.resources import Resources
from evosim.world.terrain import Terrain, generate_terrain


@pytest.fixture
def config() -> Config:
    return Config.load()


def test_grid_latitudes_are_cell_centred_and_symmetric():
    grid = Grid(8, 4)
    assert grid.latitudes_deg == pytest.approx([67.5, 22.5, -22.5, -67.5])
    assert grid.latitudes_deg == pytest.approx(-grid.latitudes_deg[::-1])


def test_grid_wraps_east_west_but_not_across_poles():
    grid = Grid(8, 4)
    neighbours = grid.neighbour_indices()
    indices = grid.flat_indices()
    assert neighbours[1, 0, 3] == indices[1, -1]
    assert neighbours[1, -1, 1] == indices[1, 0]
    assert np.all(neighbours[0, :, 0] == -1)
    assert np.all(neighbours[-1, :, 2] == -1)


def test_cell_area_weights_shrink_toward_poles():
    weights = Grid(16, 8).cell_area_weights
    assert weights[0, 0] < weights[3, 0]
    assert weights[:, 0] == pytest.approx(weights[::-1, 0])


def test_terrain_hits_configured_land_fraction(config: Config):
    grid = Grid(config.planet.grid_width, config.planet.grid_height)
    terrain = generate_terrain(grid, config.planet.terrain, RngBundle(11).terrain)
    cell_tolerance = 1.0 / grid.n_cells
    assert terrain.land.mean() == pytest.approx(
        config.planet.terrain.land_fraction, abs=cell_tolerance
    )
    assert np.all(
        terrain.elevation_km[terrain.land] > config.planet.terrain.sea_level
    )
    assert np.all(
        terrain.elevation_km[terrain.water] <= config.planet.terrain.sea_level
    )
    assert terrain.elevation_km.max() <= (
        config.planet.terrain.sea_level + config.planet.terrain.max_elevation_km
    )
    assert terrain.elevation_km.min() >= (
        config.planet.terrain.sea_level - config.planet.terrain.max_elevation_km
    )


def test_terrain_is_deterministic_and_seed_sensitive(config: Config):
    grid = Grid(config.planet.grid_width, config.planet.grid_height)
    first = generate_terrain(grid, config.planet.terrain, RngBundle(9).terrain)
    repeat = generate_terrain(grid, config.planet.terrain, RngBundle(9).terrain)
    other = generate_terrain(grid, config.planet.terrain, RngBundle(10).terrain)
    assert np.array_equal(first.elevation_km, repeat.elevation_km)
    assert not np.array_equal(first.elevation_km, other.elevation_km)


def test_poles_are_colder_than_equator(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    polar = np.r_[world.climate.temperature_c[:4].ravel(),
                  world.climate.temperature_c[-4:].ravel()]
    equatorial = world.climate.temperature_c[30:34]
    assert polar.mean() < equatorial.mean()


def test_zero_tilt_has_no_seasonal_variation(config: Config):
    planet = replace(config.planet, axial_tilt=0.0)
    grid = Grid(planet.grid_width, planet.grid_height)
    start = daily_insolation(grid, planet, 0)
    quarter_year = daily_insolation(grid, planet, planet.year_length_days // 4)
    assert np.array_equal(start, quarter_year)

    world = World.create(planet, RngBundle(config.sim.seed))
    initial_temperature = world.climate.temperature_c.copy()
    world.step(planet.year_length_days)
    assert world.climate.temperature_c == pytest.approx(initial_temperature)


def test_high_tilt_continents_have_stronger_seasons_than_oceans(config: Config):
    planet = replace(config.planet, axial_tilt=60.0)
    world = World.create(planet, RngBundle(config.sim.seed))
    minimum = world.climate.temperature_c.copy()
    maximum = minimum.copy()
    for _ in range(planet.year_length_days):
        world.step()
        minimum = np.minimum(minimum, world.climate.temperature_c)
        maximum = np.maximum(maximum, world.climate.temperature_c)
    seasonal_range = maximum - minimum
    middle_latitudes = np.abs(
        np.broadcast_to(world.grid.latitudes_deg[:, None], world.grid.shape)
    ) < 60.0
    land = world.terrain.land & middle_latitudes
    water = world.terrain.water & middle_latitudes
    assert seasonal_range[land].mean() > seasonal_range[water].mean() * 1.5


def test_ocean_moisture_is_saturated(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    assert world.climate.moisture[world.terrain.water] == pytest.approx(
        config.planet.climate.moisture_ocean
    )
    assert np.all(world.climate.moisture >= config.planet.climate.moisture_min)


def test_resource_capacity_is_area_weighted(config: Config):
    grid = Grid(config.planet.grid_width, config.planet.grid_height)
    land = np.ones(grid.shape, dtype=np.bool_)
    terrain = Terrain(
        elevation_km=np.ones(grid.shape),
        land=land,
        ruggedness=np.zeros(grid.shape),
    )
    resources = Resources.initialize(grid, terrain, config.planet)
    equatorial = resources.nutrient_capacity[grid.height // 2, 0]
    polar = resources.nutrient_capacity[0, 0]
    assert polar < equatorial


def test_resources_never_go_negative(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    world.resources.nutrients.fill(-1.0)
    world.resources.detritus.fill(-1.0)
    world.step()
    assert np.all(world.resources.nutrients >= 0.0)
    assert np.all(world.resources.detritus >= 0.0)


# -- depth and light (milestone 7) --------------------------------------------------------


def test_depth_is_the_water_column_and_land_has_none(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    land = world.terrain.land
    assert np.all(world.depth_km[land] == 0.0)
    # The shallowest water cell sits exactly at sea level, so zero is a legitimate depth.
    assert np.all(world.depth_km[~land] >= 0.0)
    assert world.depth_km[~land].max() > 1.0
    expected = config.planet.terrain.sea_level - world.terrain.elevation_km[~land]
    assert world.depth_km[~land] == pytest.approx(expected)


def test_light_equals_insolation_on_land_and_decays_with_depth(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    world.step(30)
    land = world.terrain.land
    assert world.light[land] == pytest.approx(world.climate.insolation[land])
    # Attenuation only removes light; it can never add any.
    assert np.all(world.light <= world.climate.insolation + 1e-12)

    water = ~land
    order = np.argsort(world.depth_km[water])
    transmittance = world.transmittance[water][order]
    assert np.all(np.diff(transmittance) <= 1e-12), "deeper water must not be brighter"


def test_a_planet_in_clear_water_is_lit_all_the_way_down():
    """Attenuation is a property of the planet, not a hardcoded fact about oceans."""
    config = Config.load(overrides=["planet.climate.light_attenuation_per_km=0.0"])
    world = World.create(config.planet, RngBundle(config.sim.seed))
    assert world.light == pytest.approx(world.climate.insolation)
    assert np.all(world.transmittance == 1.0)


def test_deep_water_is_cold_regardless_of_the_sun_above_it(config: Config):
    """The thermocline: depth cools asymptotically, the way elevation cools linearly."""
    world = World.create(config.planet, RngBundle(config.sim.seed))
    world.step(200)
    water = world.terrain.water
    shallow = water & (world.depth_km < 0.25)
    deep = water & (world.depth_km > 3.0)

    assert world.climate.temperature_c[deep].mean() < (
        world.climate.temperature_c[shallow].mean() - 5.0
    )
    # Asymptotic, not runaway: nothing falls below the floor the planet declares.
    floor = config.planet.climate.deep_temperature_c
    assert world.climate.temperature_c[deep].min() > floor - 5.0

    # Equatorial and polar abyss converge on the same temperature; the sun above stops
    # mattering once the water column is deep enough.
    very_deep = world.depth_km > 4.0
    if very_deep.sum() > 20:
        assert world.climate.temperature_c[very_deep].std() < 3.0


def test_a_planet_without_a_thermocline_leaves_its_deep_water_alone():
    """Same seed, same terrain, same cells -- only the thermocline differs.

    Comparing deep cells against shallow ones inside a single world would not show this:
    basins and shelves sit at different latitudes, so they differ in temperature for reasons
    that have nothing to do with depth.
    """

    def deep(*overrides: str) -> tuple[float, float]:
        config = Config.load(overrides=list(overrides))
        world = World.create(config.planet, RngBundle(config.sim.seed))
        cells = world.terrain.water & (world.depth_km > 3.0)
        values = world.climate.temperature_c[cells]
        return float(values.mean()), float(values.std())

    on_mean, on_spread = deep()
    # A scale far larger than the deepest ocean makes the correction vanish.
    off_mean, off_spread = deep("planet.climate.thermocline_scale_km=1000.0")

    # With it, the abyss sits at the floor the planet declares and barely varies. Without it,
    # deep water is just whatever the sky above happens to be doing -- note the direction is
    # not "colder": the thermocline *warms* a polar basin toward 4 C as surely as it cools a
    # tropical one, which is what makes the deep one connected habitat rather than a rim.
    assert on_mean == pytest.approx(4.0, abs=1.0)
    assert on_spread < off_spread / 3.0


def test_detritus_sinks_downhill_without_creating_or_destroying_any(config: Config):
    """Marine snow relocates dead biomass; the pool it moves through is closed."""
    world = World.create(config.planet, RngBundle(config.sim.seed))
    resources = world.resources
    assert resources.sinks_anywhere

    rng = np.random.default_rng(0)
    resources.detritus[:] = rng.random(resources.detritus.shape)
    deep = world.depth_km > np.percentile(world.depth_km[world.terrain.water], 75)
    total_before = resources.detritus.sum()
    deep_before = resources.detritus[deep].sum()

    for _ in range(50):
        resources._sink(config.planet.resources.detritus_sink_fraction)

    assert resources.detritus.sum() == pytest.approx(total_before, rel=1e-12)
    assert resources.detritus[deep].sum() > deep_before
    assert np.all(resources.detritus >= 0.0)


def test_nothing_sinks_when_the_planet_says_it_does_not(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    world.resources.detritus[:] = 1.0
    world.resources._sink(0.0)
    assert np.all(world.resources.detritus == 1.0)


def test_world_is_reproducible(config: Config):
    first = World.create(config.planet, RngBundle(123))
    second = World.create(config.planet, RngBundle(123))
    first.step(20)
    second.step(20)
    for name in first.arrays():
        assert np.array_equal(first.arrays()[name], second.arrays()[name])


def test_world_rejects_negative_ticks(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    with pytest.raises(ValueError, match="non-negative"):
        world.step(-1)


def test_add_detritus_validates_shape_and_sign(config: Config):
    world = World.create(config.planet, RngBundle(config.sim.seed))
    with pytest.raises(ValueError, match="shape"):
        world.resources.add_detritus(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        world.resources.add_detritus(-np.ones(world.grid.shape))
