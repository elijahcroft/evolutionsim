"""Milestone 2 population storage, founder placement, and compaction tests."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from textwrap import dedent

import numpy as np
import pytest

from evosim.config import DEFAULT_CONFIG_DIR, Config
from evosim.life.population import (
    HabitatError,
    Population,
    PopulationCapacityError,
)
from evosim.rng import STREAM_NAMES, RngBundle
from evosim.world import World


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(DEFAULT_CONFIG_DIR)


def _world(config: Config, seed: int | None = None) -> tuple[World, RngBundle]:
    rng = RngBundle(config.sim.seed if seed is None else seed)
    return World.create(config.planet, rng), rng


def test_founders_fill_a_preallocated_dense_prefix(config: Config):
    world, rng = _world(config)
    population = Population.seed_founders(config, world, rng)
    count = config.sim.initial_population
    capacity = config.sim.max_population
    loci = config.genome.n_loci

    assert len(population) == count
    assert population.size == population.phenotypes.size == count
    assert population.capacity == capacity
    assert population.available == capacity - count
    assert population.genomes.shape == (capacity, loci, 2)
    assert population.genomes.dtype == np.float32
    assert population.phenotypes.traits.shape == (capacity, loci)
    assert population.cell.shape == population.energy.shape == (capacity,)
    assert np.array_equal(population.organism_id[:count], np.arange(count))
    assert np.all(population.organism_id[count:] == -1)


def test_founders_are_homozygous_at_configured_initial_values(config: Config):
    world, rng = _world(config)
    population = Population.seed_founders(config, world, rng)
    active_genomes = population.genomes[population.active]
    expected = config.genome.init_array()

    assert np.array_equal(active_genomes[:, :, 0], active_genomes[:, :, 1])
    assert np.allclose(active_genomes[:, :, 0], expected)
    assert np.allclose(population.phenotypes.traits[population.active], expected)


@pytest.mark.parametrize("habitat", ["water", "land", "any"])
def test_founder_cells_respect_configured_habitat(config: Config, habitat: str):
    configured = replace(config, sim=replace(config.sim, initial_habitat=habitat))
    world, rng = _world(configured)
    population = Population.seed_founders(configured, world, rng)
    cells = population.cell[population.active]

    assert np.all((cells >= 0) & (cells < world.grid.n_cells))
    if habitat == "water":
        assert np.all(world.terrain.water.ravel()[cells])
    elif habitat == "land":
        assert np.all(world.terrain.land.ravel()[cells])


def test_founder_sampling_probabilities_are_physical_cell_areas(config: Config):
    configured = replace(config, sim=replace(config.sim, initial_population=4))
    world, _ = _world(configured)

    class ChoiceRecorder:
        def __init__(self) -> None:
            self.options = None
            self.probabilities = None

        def choice(self, options, *, size, replace, p):
            assert replace is True
            self.options = np.asarray(options).copy()
            self.probabilities = np.asarray(p).copy()
            return np.resize(self.options, size)

    class InitOnlyBundle:
        def __init__(self) -> None:
            self.init = ChoiceRecorder()

    rng = InitOnlyBundle()
    Population.seed_founders(configured, world, rng)  # type: ignore[arg-type]
    valid = np.flatnonzero(world.terrain.water.ravel())
    expected = world.grid.cell_area_weights.ravel()[valid]
    expected /= expected.sum()

    assert np.array_equal(rng.init.options, valid)
    assert np.allclose(rng.init.probabilities, expected)


def test_requested_empty_habitat_fails_clearly(config: Config):
    no_land = replace(
        config.planet,
        terrain=replace(config.planet.terrain, land_fraction=0.0),
    )
    configured = replace(
        config,
        sim=replace(config.sim, initial_habitat="land"),
        planet=no_land,
    )
    world, rng = _world(configured)

    with pytest.raises(HabitatError, match="no land habitat"):
        Population.seed_founders(configured, world, rng)


def test_founder_state_is_seed_deterministic(config: Config):
    world_a, rng_a = _world(config, seed=123)
    world_b, rng_b = _world(config, seed=123)
    a = Population.seed_founders(config, world_a, rng_a)
    b = Population.seed_founders(config, world_b, rng_b)

    for name, array in a.active_arrays(copy=True).items():
        assert np.array_equal(array, b.active_arrays()[name]), name


def test_founder_seeding_only_consumes_the_init_stream(config: Config):
    used = RngBundle(77)
    reference = RngBundle(77)
    world_used = World.create(config.planet, used)
    World.create(config.planet, reference)
    Population.seed_founders(config, world_used, used)

    for name in set(STREAM_NAMES) - {"init"}:
        assert np.array_equal(used[name].random(16), reference[name].random(16)), name


def test_founder_energy_uses_newborn_energy_ledger(config: Config):
    world, rng = _world(config)
    population = Population.seed_founders(config, world, rng)
    active = population.active
    investment = population.phenotypes.trait("parental_investment", active)
    expected = investment * population.phenotypes.storage_capacity[active]

    assert np.all(population.energy[active] > 0.0)
    assert np.allclose(population.energy[active], expected)
    assert np.all(
        population.energy[active] <= population.phenotypes.storage_capacity[active]
    )


def test_add_and_remove_keep_every_array_aligned(config: Config):
    world, _ = _world(config)
    sim = replace(config.sim, max_population=5, initial_population=1)
    population = Population.empty(sim, config.genome, world)
    genomes = population.schema.founders(3)
    genomes[:, config.genome.index_of("body_size"), :] = np.array(
        [[0.2], [0.4], [0.8]], dtype=np.float32
    )

    ids = population.add(
        genomes,
        cells=np.array([10, 20, 30]),
        energy=np.array([1.0, 2.0, 3.0]),
        age=np.array([4, 5, 6]),
        generation=np.array([0, 1, 2]),
    )
    removed = population.remove(np.array([False, True, False]))

    assert np.array_equal(ids, [0, 1, 2])
    assert np.array_equal(removed, [1])
    assert len(population) == population.phenotypes.size == 2
    assert np.array_equal(population.organism_id[:2], [0, 2])
    assert np.array_equal(population.cell[:2], [10, 30])
    assert np.array_equal(population.energy[:2], [1.0, 3.0])
    assert np.array_equal(population.age[:2], [4, 6])
    assert np.array_equal(population.generation[:2], [0, 2])
    assert np.allclose(
        population.phenotypes.trait("body_size", population.active), [0.2, 0.8]
    )
    assert np.all(population.organism_id[2:3] == -1)

    new_id = population.add(
        population.schema.founders(1),
        cells=40,
        energy=1.0,
    )
    assert np.array_equal(new_id, [3]), "persistent IDs must never be recycled"


def test_capacity_overflow_is_atomic_and_explicit(config: Config):
    world, _ = _world(config)
    sim = replace(config.sim, max_population=2, initial_population=1)
    population = Population.empty(sim, config.genome, world)
    population.add(population.schema.founders(2), cells=[1, 2], energy=1.0)
    before = population.active_arrays(copy=True)

    with pytest.raises(PopulationCapacityError, match="0 of 2"):
        population.add(population.schema.founders(1), cells=3, energy=1.0)

    assert population.size == 2
    for name, array in before.items():
        assert np.array_equal(array, population.active_arrays()[name]), name


def test_derived_phenotype_failure_leaves_population_untouched(config: Config):
    world, _ = _world(config)
    body_index = config.genome.index_of("body_size")
    loci = list(config.genome.loci)
    loci[body_index] = replace(loci[body_index], high=1e20)
    oversized_body_config = replace(config.genome, loci=tuple(loci))
    sim = replace(config.sim, max_population=2, initial_population=1)
    population = Population.empty(sim, oversized_body_config, world)
    genome = population.schema.founders(1)
    genome[:, body_index, :] = 1e20

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="derived body geometry"):
            population.add(genome, cells=1, energy=1.0)

    assert population.size == population.phenotypes.size == 0
    assert population.organism_id[0] == -1
    assert population.cell[0] == -1
    assert population.energy[0] == 0.0
    assert np.all(population.genomes[0] == 0.0)


def test_remove_rejects_misaligned_masks(config: Config):
    world, _ = _world(config)
    sim = replace(config.sim, max_population=3, initial_population=1)
    population = Population.empty(sim, config.genome, world)
    population.add(population.schema.founders(2), cells=[1, 2], energy=1.0)

    with pytest.raises(ValueError, match="boolean"):
        population.remove(np.array([0, 1]))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        population.remove(np.array([True]))


def test_add_rejects_non_real_state(config: Config):
    world, _ = _world(config)
    sim = replace(config.sim, max_population=3, initial_population=1)
    population = Population.empty(sim, config.genome, world)
    complex_genome = population.schema.founders(1).astype(np.complex64)
    complex_genome += 1j

    with pytest.raises(TypeError, match="real numeric alleles"):
        population.add(complex_genome, cells=1, energy=1.0)
    with pytest.raises(TypeError, match="energy must contain numbers"):
        population.add(
            population.schema.founders(1),
            cells=1,
            energy=1.0 + 1.0j,
        )
    with pytest.raises(ValueError, match="does not fit in float32"):
        population.add(
            population.schema.founders(1),
            cells=1,
            energy=np.finfo(np.float64).max,
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        population.add(
            population.schema.founders(1),
            cells=1,
            energy=-1e-50,
        )
    with pytest.raises(ValueError, match="does not fit in int64"):
        population.add(
            population.schema.founders(1),
            cells=1,
            energy=1.0,
            parent_id=np.iinfo(np.uint64).max,
        )


def test_organism_serialization_is_a_renderer_friendly_dict(config: Config):
    world, rng = _world(config)
    population = Population.seed_founders(config, world, rng)
    organism = population.organism_dict(0)

    assert organism["id"] == 0
    assert organism["cell"] >= 0
    assert set(organism["genome"]) == {locus.name for locus in config.genome.loci}
    assert len(organism["genome"]["body_size"]) == 2
    assert isinstance(organism["phenotype"], dict)
    assert organism["phenotype"]["body"]["mass"] > 0.0
    assert organism["phenotype"]["diet"]["autotroph"] > 0.5


def test_population_hot_paths_have_no_python_iteration():
    """A coarse lint guard against accidentally adding per-organism Python loops."""
    for method in (
        Population.seed_founders,
        Population.add,
        Population.remove,
    ):
        tree = ast.parse(dedent(inspect.getsource(method)))
        forbidden = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp)
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree)), method
