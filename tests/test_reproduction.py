"""Tests for reproduction.

The energy ledger is the centre of gravity again.  ``reproduction.overhead`` above one means
every birth destroys some of the parent's reserve, and that loss must land in the world as
detritus rather than vanishing -- otherwise a lineage could quietly launder matter out of the
simulation by breeding, and every carrying-capacity result afterwards would be wrong.

The other thing pinned here is that fecundity is bought, never granted: a parent that cannot
pay has a smaller brood, and one that cannot pay for a single offspring has none.
"""

from __future__ import annotations

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.reproduction import MateSearchNotImplementedError, ReproductionModel
from evosim.sim import Simulation

# Regrowth and decay off, so a pool moves only where an organism ate, died, or bred.
FROZEN = [
    "planet.resources.nutrient_regen_land=0.0",
    "planet.resources.nutrient_regen_water=0.0",
    "planet.resources.detritus_decay_rate=0.0",
]
# A small, well-fed, immortal population: reproduction is then the only thing happening.
FERTILE = [
    "planet.grid_width=32",
    "planet.grid_height=16",
    "sim.initial_population=200",
    "sim.max_population=20000",
    "energy.mortality.background=0.0",
    "energy.mortality.h_thermal_max=0.0",
    "genome.loci.senescence_rate.init=0.0",
    "genome.loci.maturity_age.init=1.0",
]


def sim(*overrides: str) -> Simulation:
    return Simulation.create(Config.load(overrides=[*FERTILE, *overrides]))


def crowded(*overrides: str) -> Simulation:
    """Everyone in one rich cell, going nowhere.

    Mate search is same-cell only, so a population scattered one-per-cell across an ocean is
    genuinely unable to breed sexually -- correct behaviour, but useless for testing that the
    sexual path works.  The cell is made rich because packing a population into one cell
    otherwise starves it through the very contention the ecology is supposed to have.
    """
    simulation = sim(
        "sim.initial_population=40",
        "genome.loci.move_speed.init=0.0",
        "energy.reproduction.dispersal_radius=0.0",
        "planet.resources.nutrient_regen_water=1.0",
        "planet.resources.nutrient_capacity_water=60.0",
        *overrides,
    )
    population = simulation.population
    population.cell[population.active] = population.cell[0]
    return simulation


def until_first_birth(simulation: Simulation, limit: int = 400):
    """Step until a tick produces offspring, and return that tick's statistics."""
    for _ in range(limit):
        stats = simulation.step()
        if stats.births:
            return stats
    raise AssertionError("no births occurred")


# -- eligibility ---------------------------------------------------------------------------


def test_a_lineage_reproduces_and_grows():
    simulation = sim()
    stats = until_first_birth(simulation)
    assert stats.births > 0
    assert stats.breeding_parents > 0
    simulation.run(120)
    assert simulation.population.size > 200


def test_nobody_breeds_before_maturity():
    """`maturity_age` must gate reproduction, or life history has no timescale."""
    simulation = sim("genome.loci.maturity_age.init=60.0")
    for _ in range(40):
        assert simulation.step().births == 0
    assert np.all(simulation.population.age[simulation.population.active] == 40)


def test_nobody_breeds_below_the_energy_threshold():
    """`repro_threshold` at its ceiling means a full tank is required."""
    simulation = sim("genome.loci.repro_threshold.init=1.0", "energy.intake.k_photo=0.0")
    for _ in range(30):
        assert simulation.step().births == 0


def test_a_parent_that_cannot_afford_a_brood_has_a_smaller_one():
    """Affordability must scale the brood rather than cancelling it.

    All-or-nothing would turn `offspring_count` into a cliff: a parent one unit short of four
    offspring would produce none at all rather than three.

    `offspring_count` is set high so that energy, not the locus, is the binding constraint --
    otherwise both parents simply have the brood they asked for and the test proves nothing.
    """
    crowded = "genome.loci.offspring_count.init=40.0"
    lavish = until_first_birth(sim(crowded, "genome.loci.parental_investment.init=0.50"))
    thrifty = until_first_birth(sim(crowded, "genome.loci.parental_investment.init=0.02"))
    assert thrifty.births > lavish.births


def test_offspring_count_sets_the_brood_size():
    few = until_first_birth(sim("genome.loci.offspring_count.init=1.0"))
    many = until_first_birth(sim("genome.loci.offspring_count.init=8.0"))
    assert many.births > few.births


# -- the energy ledger across a birth ---------------------------------------------------------


def test_a_birth_moves_exactly_the_energy_it_costs():
    """What the parents paid must equal what the offspring got, plus the overhead.

    Reproduction is driven directly rather than through `Simulation.step`, because a full tick
    also feeds and charges the parents and clips them at their storage ceiling.  Isolating the
    stage is the only way to assert the transfer exactly rather than to a tolerance that would
    hide a real leak.
    """
    simulation = sim(*FROZEN)
    simulation.run(5)  # clear the maturity gate
    population = simulation.population
    model = simulation.reproduction

    # Fill every tank directly rather than waiting for one to fill, so the reproduction stage
    # is the only thing that has touched these reserves.
    active = population.active
    population.energy[active] = population.phenotypes.storage_capacity[active]
    before = population.energy[active].astype(np.float64).sum()
    ids = population.organism_id[population.active].copy()
    births = model.reproduce(population, simulation.world, simulation.rng)
    assert births.births > 0

    active = population.active
    newborn = np.isin(population.organism_id[active], ids, invert=True)
    endowment = population.energy[active][newborn].astype(np.float64).sum()
    survivors = population.energy[active][~newborn].astype(np.float64).sum()

    assert births.energy_invested == pytest.approx(endowment, rel=1e-5)
    assert before - survivors == pytest.approx(births.energy_spent, rel=1e-5)
    assert births.energy_spent == pytest.approx(
        births.energy_invested * simulation.config.energy.reproduction.overhead, rel=1e-9
    )


def test_reproduction_overhead_becomes_detritus_rather_than_vanishing():
    """`overhead` models gametes and failed births; that material is real."""
    simulation = sim(*FROZEN, "energy.reproduction.overhead=2.0")
    while True:
        before = simulation.world.resources.detritus.sum()
        stats = simulation.step()
        if stats.births:
            break
    after = simulation.world.resources.detritus.sum()

    assert stats.energy_reproduction_overhead > 0.0
    assert after - before == pytest.approx(
        stats.detritus_deposited - stats.detritus_consumed, abs=1e-9
    )


def test_a_lossless_configuration_has_no_overhead_to_deposit():
    simulation = sim("energy.reproduction.overhead=1.0")
    stats = until_first_birth(simulation)
    assert stats.energy_reproduction_overhead == pytest.approx(0.0)


def test_no_parent_is_ever_charged_more_energy_than_it_has():
    simulation = sim("genome.loci.offspring_count.init=50.0")
    for _ in range(60):
        simulation.step()
        active = simulation.population.active
        assert np.all(simulation.population.energy[active] >= 0.0)


# -- offspring state ---------------------------------------------------------------------------


def test_offspring_record_their_parent_and_their_generation():
    simulation = sim()
    population = simulation.population
    founders = set(population.organism_id[population.active].tolist())
    until_first_birth(simulation)

    active = population.active
    newborn = np.isin(population.organism_id[active], list(founders), invert=True)
    assert newborn.any()
    assert np.all(population.generation[active][newborn] == 1)
    assert np.all(population.parent_id[active][newborn, 0] >= 0)
    # Asexual by default, so there is no second parent.
    assert np.all(population.parent_id[active][newborn, 1] == -1)
    assert np.all(population.age[active][newborn] == 0)


def test_offspring_are_born_in_their_parents_cell_without_dispersal():
    """Movement is disabled so the parents' cells at birth are the ones recorded up front."""
    simulation = sim(
        "energy.reproduction.dispersal_radius=0.0", "genome.loci.move_speed.init=0.0"
    )
    population = simulation.population
    founders = set(population.organism_id[population.active].tolist())
    parent_cells = set(population.cell[population.active].tolist())
    until_first_birth(simulation)

    active = population.active
    newborn = np.isin(population.organism_id[active], list(founders), invert=True)
    assert newborn.any()
    assert set(population.cell[active][newborn].tolist()) <= parent_cells


def _newborn_offsets(simulation: Simulation) -> np.ndarray:
    """Whether each newborn was placed off its own parent's cell."""
    population = simulation.population
    founders = population.organism_id[population.active].copy()
    until_first_birth(simulation)

    active = population.active
    newborn = np.isin(population.organism_id[active], founders, invert=True)
    # Movement happens before reproduction, so a parent's cell at the end of the tick is the
    # cell it bred in.
    parent_row = np.searchsorted(
        population.organism_id[active], population.parent_id[active][newborn, 0]
    )
    return population.cell[active][newborn] != population.cell[active][parent_row]


def test_dispersal_spreads_offspring_off_the_parent_cell():
    scattered = _newborn_offsets(sim("energy.reproduction.dispersal_radius=1.0"))
    assert scattered.any()


def test_zero_dispersal_leaves_every_offspring_on_its_parents_cell():
    stayed = _newborn_offsets(sim("energy.reproduction.dispersal_radius=0.0"))
    assert stayed.size > 0
    assert not stayed.any()


def test_offspring_never_land_outside_the_grid():
    simulation = sim("energy.reproduction.dispersal_radius=4.0")
    for _ in range(60):
        simulation.step()
        cells = simulation.population.cell[simulation.population.active]
        assert np.all((cells >= 0) & (cells < simulation.world.grid.n_cells))


def test_offspring_are_mutated_copies_rather_than_exact_clones():
    """Without heritable variation there is nothing for selection to act on."""
    simulation = sim("genome.loci.mutation_rate.init=0.5")
    until_first_birth(simulation)
    active = simulation.population.active
    spread = simulation.population.genomes[active].std(axis=0).sum()
    assert spread > 0.0


def test_the_mutation_rate_locus_controls_how_fast_variation_accumulates():
    """`mutation_rate` is itself evolvable, so it must actually do something."""

    def spread(rate: float) -> float:
        simulation = sim(f"genome.loci.mutation_rate.init={rate}")
        simulation.run(120)
        active = simulation.population.active
        return float(simulation.population.genomes[active].std(axis=0).sum())

    # The configured floor is 0.0001, so the slow lineage is very nearly clonal.
    assert spread(0.5) > spread(0.0001)


# -- sexual reproduction -------------------------------------------------------------------------


def test_reproduction_is_asexual_when_sex_bias_is_zero():
    stats = until_first_birth(sim())
    assert stats.sexual_births == 0


def test_a_full_sex_bias_produces_sexual_offspring_with_two_parents():
    simulation = crowded("genome.loci.sex_bias.init=1.0")
    population = simulation.population
    founders = set(population.organism_id[population.active].tolist())
    stats = until_first_birth(simulation)

    assert stats.sexual_births > 0
    active = population.active
    newborn = np.isin(population.organism_id[active], list(founders), invert=True)
    pairs = population.parent_id[active][newborn]
    sexual = pairs[:, 1] >= 0
    assert sexual.any()
    assert np.all(pairs[sexual, 0] != pairs[sexual, 1])


def test_an_incompatible_population_falls_back_to_asexual():
    """A mate too distant to breed with must not stop an organism reproducing at all.

    Half the cell is pushed to the far end of the `move_persistence` locus, which genome.yaml
    documents as free.  Using a costly locus instead would simply kill the divergent half and
    leave a homogeneous, mutually compatible population -- which is what makes this the
    mechanism that will let assortative mating reinforce a split in M5: two diverging groups
    sharing a cell stop exchanging alleles well before anything declares them separate species.
    """
    simulation = crowded(
        "genome.loci.sex_bias.init=1.0",
        "energy.reproduction.mate_compatibility_distance=0.02",
    )
    population = simulation.population
    locus = population.schema.index_of("move_persistence")
    half = population.size // 2
    population.genomes[:half, locus, :] = 1.0
    population.genomes[half:, locus, :] = 0.0
    population.phenotypes.update(0, population.genomes[: population.size], population.schema)

    stats = until_first_birth(simulation)
    assert stats.births > 0
    assert 0 < stats.sexual_births < stats.births


def test_a_solitary_organism_cannot_find_a_mate():
    simulation = crowded("genome.loci.sex_bias.init=1.0", "sim.initial_population=1")
    stats = until_first_birth(simulation)
    assert stats.births > 0
    assert stats.sexual_births == 0


def test_an_unimplemented_mate_search_radius_fails_loudly():
    """Silently treating a configured radius as zero is the bug the config discipline exists for."""
    with pytest.raises(MateSearchNotImplementedError):
        Simulation.create(Config.load(overrides=["energy.reproduction.mate_search_radius=2.0"]))


# -- capacity throttling ----------------------------------------------------------------------------


def test_hitting_the_population_cap_is_recorded_rather_than_silent():
    """sim.yaml requires a run that spent time at the cap to say so."""
    simulation = sim(
        "sim.max_population=260",
        "sim.initial_population=250",
        "genome.loci.offspring_count.init=20.0",
    )
    throttled = 0
    for _ in range(80):
        stats = simulation.step()
        throttled += stats.capacity_throttle
        assert simulation.population.size <= 260
    assert throttled > 0


def test_a_throttled_birth_is_not_charged_to_its_parent():
    """Only offspring that exist may cost anything, or the cap would silently tax breeders."""
    simulation = sim(
        "sim.max_population=205",
        "sim.initial_population=200",
        "genome.loci.offspring_count.init=30.0",
    )
    for _ in range(80):
        room = simulation.population.available
        stats = simulation.step()
        if stats.capacity_throttle:
            break
    assert stats.capacity_throttle > 0
    # Every birth that happened fitted in the room that existed; the rest were dropped, not
    # squeezed in and not billed.  `room` is read before the step, and the reap runs before the
    # breed, so this tick's dead have freed their slots by the time anything is born into them.
    assert stats.births <= room + stats.deaths
    assert simulation.population.size <= 205
    assert stats.energy_to_offspring > 0.0
    assert np.all(simulation.population.energy[simulation.population.active] >= 0.0)


# -- determinism ---------------------------------------------------------------------------------------


def test_reproduction_is_reproducible():
    left, right = sim(), sim()
    left.run(60)
    right.run(60)
    assert left.population.size == right.population.size
    assert np.array_equal(
        left.population.genomes[left.population.active],
        right.population.genomes[right.population.active],
    )
    assert left.rng.get_state() == right.rng.get_state()


def test_reproduction_consumes_only_its_own_streams():
    """Mortality and movement must not shift because a birth happened."""
    simulation = sim()
    population = simulation.population
    model = ReproductionModel.for_world(
        simulation.config, simulation.world, population.schema
    )
    simulation.run(30)  # let the population reach breeding condition

    before = simulation.rng.get_state()["streams"]
    model.reproduce(population, simulation.world, simulation.rng)
    after = simulation.rng.get_state()["streams"]
    changed = {name for name in before if before[name] != after[name]}
    assert changed <= {"repro", "mutation", "recombination"}
