"""Tests for the periodic split test.

The property this file exists to pin is agreement: the taxonomy must divide organisms on
exactly the metric, in exactly the units, that decides whether two of them can breed.  A
species concept that disagreed with the mating rule would let the simulation report two species
that are freely exchanging alleles, or one species whose halves have not interbred in
generations -- and every lineage tree afterwards would be fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from evosim.config import Config
from evosim.evolution.taxonomy import TaxonomyModel
from evosim.rng import RngBundle
from evosim.sim import Simulation

DIVERGING_LOCI = ("temp_optimum", "body_length", "move_speed")
OTHER_LOCI = ("aggression", "sense_range", "armor", "camouflage")


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def model(config: Config) -> TaxonomyModel:
    from evosim.life.genome import GenomeSchema

    return TaxonomyModel.from_config(config, GenomeSchema.from_config(config.genome))


def small(overrides: list[str] | None = None) -> Config:
    return Config.load(
        overrides=[
            "planet.grid_width=32",
            "planet.grid_height=16",
            "sim.initial_population=200",
            "sim.max_population=4000",
            *(overrides or []),
        ]
    )


def diverge(
    simulation: Simulation,
    rows: slice,
    factor: float = 1.0,
    loci: tuple[str, ...] = DIVERGING_LOCI,
) -> None:
    """Push a block of the population toward the far end of several loci."""
    population = simulation.population
    schema = population.schema
    for name in loci:
        index = schema.index_of(name)
        low = float(schema.low[index])
        high = float(schema.high[index])
        population.genomes[rows, index, :] = low + (high - low) * factor
    population.phenotypes.update(0, population.genomes[: population.size], schema)


# -- the metric ---------------------------------------------------------------------------------


def test_coordinate_distance_is_genetic_distance(model: TaxonomyModel, config: Config):
    """Euclidean distance in taxonomy space must equal the mating rule's distance exactly."""
    rng = np.random.default_rng(0)
    schema = model.schema
    left = schema.founders(16).copy()
    right = schema.founders(16).copy()
    span = schema.span[None, :, None].astype(np.float64)
    low = schema.low[None, :, None].astype(np.float64)
    right += rng.random(right.shape) * span * 0.5
    right = np.clip(right, low, low + span).astype(np.float32)

    coordinates = model.coordinates(left) - model.coordinates(right)
    euclidean = np.sqrt(np.sum(coordinates**2, axis=1))
    assert euclidean == pytest.approx(schema.genetic_distance(left, right), abs=1e-6)


def test_a_zero_weighted_locus_cannot_make_a_species(model: TaxonomyModel):
    """`distance_weights` decides what counts, and mutation_rate is configured not to."""
    assert model.scale[model.schema.index_of("mutation_rate")] == 0.0


# -- when a split happens -----------------------------------------------------------------------


def test_one_undivided_population_stays_one_species():
    simulation = Simulation.create(small())
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert splits == []


def test_two_separated_groups_become_two_species():
    simulation = Simulation.create(small())
    diverge(simulation, slice(0, 100))
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert len(splits) == 1
    assert splits[0].separation > simulation.taxonomy.split_distance
    assert splits[0].parent == 0
    assert splits[0].rows.size == 100


def test_a_split_that_stays_within_mating_distance_is_refused():
    """The test and the mating rule share one threshold, so this is the same statement twice."""
    simulation = Simulation.create(small())
    population = simulation.population
    schema = population.schema
    index = schema.index_of("temp_optimum")
    # A separation deliberately below the threshold: these organisms would still interbreed.
    offset = float(schema.span[index]) * simulation.taxonomy.split_distance * 0.5
    population.genomes[:100, index, :] += offset
    population.phenotypes.update(0, population.genomes[: population.size], schema)

    splits = simulation.taxonomy.split(
        population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert splits == []


def test_the_larger_group_keeps_the_ancestral_identity():
    simulation = Simulation.create(small())
    diverge(simulation, slice(0, 40))
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert len(splits) == 1
    assert splits[0].rows.size == 40


def test_splitting_recurses_until_nothing_more_splits():
    """One pass must recognise all the divergence present, not one species' worth of it."""
    simulation = Simulation.create(small())
    diverge(simulation, slice(0, 60), factor=1.0)
    diverge(simulation, slice(60, 120), loci=OTHER_LOCI)
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert len(splits) == 2
    assert {split.species for split in splits} == {1, 2}


def test_identical_organisms_never_split():
    simulation = Simulation.create(small())
    assert (
        simulation.taxonomy.split(simulation.population, RngBundle(9).speciation, 1) == []
    )


def test_an_empty_population_has_no_taxonomy():
    simulation = Simulation.create(small(["energy.mortality.background=1.0"]))
    simulation.step()
    assert simulation.population.size == 0
    assert simulation.taxonomy.split(simulation.population, RngBundle(1).speciation, 1) == []


def test_a_lone_divergent_organism_is_its_own_species():
    """No quorum constant exists, and the threshold is the whole definition."""
    simulation = Simulation.create(small())
    diverge(simulation, slice(0, 1))
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    assert len(splits) == 1
    assert splits[0].rows.tolist() == [0]


def test_the_split_is_reproducible_from_the_stream_alone():
    simulation = Simulation.create(small())
    diverge(simulation, slice(0, 100))
    first = simulation.taxonomy.split(simulation.population, RngBundle(3).speciation, 1)
    second = simulation.taxonomy.split(simulation.population, RngBundle(3).speciation, 1)
    assert [split.rows.tolist() for split in first] == [
        split.rows.tolist() for split in second
    ]
