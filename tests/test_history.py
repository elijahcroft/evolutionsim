"""Tests for the run's memory: lineage, extinction, and the sampled time series.

The point of this layer is that a finished run can be read rather than merely reported, so the
properties pinned here are the ones that would make a recorded history untrustworthy: an
extinction announced twice or reversed, a species id reused, a lineage that does not reach the
founder, or a record that differs between two runs of the same ``(config, seed)``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evosim.config import Config
from evosim.history import FOUNDER_SPECIES, History, habitat_line
from evosim.rng import RngBundle
from evosim.sim import Simulation
from evosim.world.biome import BIOME_NAMES


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


def biomes(simulation: Simulation) -> np.ndarray:
    """The day's biome codes, one per cell, in the shape ``History.sample`` wants them."""
    return simulation.world.biomes().ravel()


def diverged(simulation: Simulation, rows: slice) -> None:
    population = simulation.population
    schema = population.schema
    for name in ("temp_optimum", "body_length", "move_speed"):
        index = schema.index_of(name)
        population.genomes[rows, index, :] = schema.high[index]
    population.phenotypes.update(0, population.genomes[: population.size], schema)


def split_once(simulation: Simulation, day: int = 1) -> None:
    """Force one accepted split and register it, without waiting for the cadence."""
    diverged(simulation, slice(0, 80))
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, simulation.history.next_species
    )
    simulation.history.apply(day, simulation.population, splits)


# -- founding ------------------------------------------------------------------------------------


def test_every_founder_belongs_to_the_founder_species():
    simulation = Simulation.create(small())
    population = simulation.population
    assert np.all(population.species_id[population.active] == FOUNDER_SPECIES)
    assert simulation.history.living() == (FOUNDER_SPECIES,)
    assert simulation.history.lineage(FOUNDER_SPECIES) == (FOUNDER_SPECIES,)


def test_founding_twice_is_refused():
    simulation = Simulation.create(small())
    with pytest.raises(ValueError):
        simulation.history.found(0, simulation.population)


def test_an_offspring_inherits_its_parents_species():
    simulation = Simulation.create(small())
    simulation.run(60)
    population = simulation.population
    assert population.size > 0
    assert np.all(population.species_id[population.active] == FOUNDER_SPECIES)


# -- the lineage tree ------------------------------------------------------------------------------


def test_a_split_records_its_parent_and_its_origin_day():
    simulation = Simulation.create(small())
    split_once(simulation, day=7)
    record = simulation.history.records[1]
    assert record.parent == FOUNDER_SPECIES
    assert record.origin_day == 7
    assert record.population == 80
    assert simulation.history.lineage(1) == (FOUNDER_SPECIES, 1)


def test_the_daughter_organisms_carry_the_new_identity():
    simulation = Simulation.create(small())
    split_once(simulation)
    population = simulation.population
    assert np.count_nonzero(population.species_id[population.active] == 1) == 80


def test_origin_traits_describe_the_group_that_left():
    """A record must say what the species *was*, or the tree explains nothing."""
    simulation = Simulation.create(small())
    split_once(simulation)
    schema = simulation.population.schema
    index = schema.index_of("body_length")
    assert simulation.history.records[1].origin_traits[index] == pytest.approx(
        float(schema.high[index]), rel=1e-5
    )
    assert simulation.history.records[0].origin_traits[index] == pytest.approx(
        float(schema.init[index]), rel=1e-5
    )


def test_a_split_out_of_order_is_refused():
    simulation = Simulation.create(small())
    diverged(simulation, slice(0, 80))
    splits = simulation.taxonomy.split(
        simulation.population, RngBundle(1).speciation, 5
    )
    with pytest.raises(ValueError):
        simulation.history.apply(1, simulation.population, splits)


def test_an_unknown_species_has_no_lineage():
    simulation = Simulation.create(small())
    with pytest.raises(KeyError):
        simulation.history.lineage(42)


# -- extinction ------------------------------------------------------------------------------------


def test_extinction_waits_for_confirmation_and_is_declared_once():
    simulation = Simulation.create(small(["sim.extinction_confirm_ticks=5"]))
    history = simulation.history
    split_once(simulation, day=1)
    population = simulation.population

    # Every member of the daughter species dies on day 2, and nothing else changes.
    population.remove(population.species_id[population.active] == 1)
    assert history.observe(2, population) == 0
    assert history.records[1].population == 0
    assert not history.records[1].extinct

    assert history.observe(5, population) == 0  # four days since last seen: not yet
    assert history.observe(6, population) == 1  # five days: confirmed
    assert history.records[1].extinct_day == 1  # dated to when it actually ended
    assert history.records[1].confirmed_day == 6

    assert history.observe(7, population) == 0  # never announced twice
    assert history.extinct() == (1,)


def test_a_confirmed_species_is_not_revived_by_a_later_count():
    """Nothing can restore a species, so a stale count must not silently do it."""
    simulation = Simulation.create(small(["sim.extinction_confirm_ticks=1"]))
    history = simulation.history
    split_once(simulation, day=1)
    population = simulation.population
    population.species_id[population.active] = 1
    history.observe(2, population)
    assert history.records[FOUNDER_SPECIES].extinct

    history.observe(3, population)
    assert history.records[FOUNDER_SPECIES].extinct_day == 0
    assert history.records[FOUNDER_SPECIES].population == 0


def test_a_species_id_is_never_reused():
    simulation = Simulation.create(small(["sim.extinction_confirm_ticks=1"]))
    history = simulation.history
    split_once(simulation, day=1)
    population = simulation.population
    population.remove(population.species_id[population.active] == 1)
    history.observe(3, population)
    assert history.records[1].extinct
    assert history.next_species == 2


def test_peak_population_records_when_a_species_was_at_its_largest():
    simulation = Simulation.create(small())
    history = simulation.history
    population = simulation.population
    history.observe(1, population)
    peak = history.records[FOUNDER_SPECIES].peak_population
    population.remove(np.arange(population.size) < 50)
    history.observe(2, population)
    assert history.records[FOUNDER_SPECIES].peak_population == peak
    assert history.records[FOUNDER_SPECIES].peak_day == 0
    assert history.records[FOUNDER_SPECIES].population == peak - 50


# -- the time series ---------------------------------------------------------------------------------


def test_a_sample_breaks_the_population_down_by_species():
    simulation = Simulation.create(small())
    split_once(simulation)
    sample = simulation.history.sample(10, simulation.population, biomes(simulation))
    assert sample.day == 10
    assert sample.population == simulation.population.size
    assert [entry.species_id for entry in sample.species] == [0, 1]
    assert sum(entry.population for entry in sample.species) == sample.population


def test_sampled_trait_means_are_the_species_means():
    simulation = Simulation.create(small())
    split_once(simulation)
    population = simulation.population
    sample = simulation.history.sample(10, population, biomes(simulation))
    index = population.schema.index_of("body_length")
    for entry in sample.species:
        members = population.species_id[population.active] == entry.species_id
        expected = population.phenotypes.traits[population.active][members, index].mean()
        assert entry.traits[index] == pytest.approx(float(expected), rel=1e-5)


def test_samples_arrive_on_the_configured_interval():
    simulation = Simulation.create(small(["sim.sample_interval=10"]))
    simulation.run(35)
    assert [sample.day for sample in simulation.history.samples] == [10, 20, 30]


def test_an_extinct_biosphere_still_samples():
    simulation = Simulation.create(
        small(["energy.mortality.background=1.0", "sim.sample_interval=1"])
    )
    simulation.step()
    assert simulation.population.size == 0
    assert simulation.history.samples[-1].species == ()


# -- where they live (milestone 11) -------------------------------------------------------------------


def _cells_of(simulation: Simulation, biome: str) -> np.ndarray:
    codes = biomes(simulation)
    return np.flatnonzero(codes == BIOME_NAMES.index(biome))


def _house(simulation: Simulation, species: int, biome: str) -> None:
    """Move every member of a species into cells of one biome, and nowhere else."""
    population = simulation.population
    active = population.active
    rows = np.flatnonzero(population.species_id[active] == species)
    cells = _cells_of(simulation, biome)
    assert cells.size, f"the test planet has no {biome}"
    population.cell[active][rows] = cells[np.arange(rows.size) % cells.size]


def test_habitat_occupancy_is_a_distribution_over_the_biomes():
    simulation = Simulation.create(small())
    split_once(simulation)
    sample = simulation.history.sample(10, simulation.population, biomes(simulation))
    for entry in sample.species:
        assert len(entry.habitat) == len(BIOME_NAMES)
        assert sum(entry.habitat) == pytest.approx(1.0)
        assert all(share >= 0.0 for share in entry.habitat)


def test_sampled_occupancy_matches_a_hand_recount():
    simulation = Simulation.create(small())
    split_once(simulation)
    population = simulation.population
    codes = biomes(simulation)
    sample = simulation.history.sample(10, population, codes)
    where = codes[population.cell[population.active].astype(np.intp)]
    for entry in sample.species:
        members = population.species_id[population.active] == entry.species_id
        for index in range(len(BIOME_NAMES)):
            expected = (where[members] == index).mean()
            assert entry.habitat[index] == pytest.approx(float(expected))


def test_the_habitat_line_names_the_largest_shares_biggest_first():
    fractions = [0.0] * len(BIOME_NAMES)
    fractions[BIOME_NAMES.index("deep ocean")] = 0.3
    fractions[BIOME_NAMES.index("cold shelf")] = 0.62
    fractions[BIOME_NAMES.index("tundra")] = 0.08
    line = habitat_line(tuple(fractions), BIOME_NAMES)
    assert line == "cold shelf, 62% of its range; deep ocean, 30%"
    # The 8% is below the floor, so it is not named at all rather than named and ignored.
    assert "tundra" not in line


def test_the_habitat_line_stays_a_sentence_when_a_species_is_everywhere():
    even = tuple(1.0 / len(BIOME_NAMES) for _ in BIOME_NAMES)
    assert habitat_line(even, BIOME_NAMES).count(";") == 2
    # And a species with no members has nothing to say rather than a line of zeroes.
    assert habitat_line(tuple(0.0 for _ in BIOME_NAMES), BIOME_NAMES) == ""


def test_two_species_on_different_depths_get_different_habitat_lines():
    """The milestone 11 criterion: the text is derived from occupancy, never authored."""

    simulation = Simulation.create(small(["sim.sample_interval=1"]))
    split_once(simulation)
    _house(simulation, 0, "warm shelf")
    _house(simulation, 1, "deep ocean")
    simulation.history.sample(1, simulation.population, biomes(simulation))

    shallow = simulation.history.report(0, simulation.population)["habitat"]
    deep = simulation.history.report(1, simulation.population)["habitat"]
    assert shallow["line"] == "warm shelf, 100% of its range"
    assert deep["line"] == "deep ocean, 100% of its range"
    assert shallow["line"] != deep["line"]


def test_an_extinct_species_keeps_the_habitat_it_had():
    """``report`` reads the last sample rather than recounting, so history survives the end."""

    simulation = Simulation.create(small(["sim.sample_interval=1"]))
    split_once(simulation)
    _house(simulation, 1, "deep ocean")
    simulation.history.sample(1, simulation.population, biomes(simulation))

    population = simulation.population
    population.remove(population.species_id[population.active] == 1)
    report = simulation.history.report(1, simulation.population)
    assert report["current_traits"] is None
    assert report["habitat"]["line"] == "deep ocean, 100% of its range"


def test_a_species_never_sampled_reports_no_habitat():
    simulation = Simulation.create(small())
    assert simulation.history.report(0, simulation.population)["habitat"] is None


# -- serialization ------------------------------------------------------------------------------------


def test_the_record_serializes_to_json():
    simulation = Simulation.create(small(["sim.sample_interval=5"]))
    simulation.run(10)
    split_once(simulation, day=10)
    payload = json.loads(json.dumps(simulation.history.to_dict()))
    assert payload["trait_names"] == list(simulation.population.phenotypes.trait_names)
    assert [entry["species_id"] for entry in payload["species"]] == [0, 1]
    assert payload["species"][0]["parent"] is None
    assert payload["species"][1]["parent"] == 0
    assert [sample["day"] for sample in payload["samples"]] == [5, 10]


def test_two_runs_of_the_same_config_and_seed_record_the_same_history():
    """The milestone 5 criterion: the recorded history reproduces from (config, seed) alone."""
    left = Simulation.create(small(["sim.taxonomy_interval=25", "sim.sample_interval=25"]))
    right = Simulation.create(small(["sim.taxonomy_interval=25", "sim.sample_interval=25"]))
    left.run(120)
    right.run(120)
    assert left.history.to_dict() == right.history.to_dict()


def test_history_is_bookkeeping_and_never_read_by_the_tick():
    """No organism may behave differently for having been recorded."""
    baseline = Simulation.create(small())
    baseline.run(40)
    noisy = Simulation.create(small())
    for _ in range(40):
        noisy.step()
        noisy.history.sample(noisy.day, noisy.population, biomes(noisy))
    assert noisy.population.size == baseline.population.size
    assert np.array_equal(
        noisy.population.organism_id[noisy.population.active],
        baseline.population.organism_id[baseline.population.active],
    )


def test_history_from_config_carries_the_confirmation_window():
    config = small(["sim.extinction_confirm_ticks=17"])
    history = History.from_config(config, ("a", "b"))
    assert history.extinction_confirm_ticks == 17
    assert history.species_count == 0
