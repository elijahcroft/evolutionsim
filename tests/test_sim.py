"""Tests for the biological tick.

The heart of this file is the ledger.  Energy is deliberately not conserved -- autotrophy
creates it from light -- but matter is, and the whole ecology rests on that: if organisms could
draw nutrients that were never removed from a cell, "carrying capacity" would be fiction and
every result about competition afterwards would be meaningless.  The ledger tests run with
regrowth and decay switched off, so the only thing that can move a pool is an organism.
"""

from __future__ import annotations

import ast
import inspect
from textwrap import dedent

import numpy as np
import pytest

from evosim.config import Config
from evosim.sim import Simulation

# Regrowth and decay off: the pools then change only where an organism ate or died.
FROZEN_RESOURCES = [
    "planet.resources.nutrient_regen_land=0.0",
    "planet.resources.nutrient_regen_water=0.0",
    "planet.resources.detritus_decay_rate=0.0",
]


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


def small(overrides: list[str] | None = None, **_: object) -> Config:
    """A cheap configuration: a small world and a small founder population."""
    return Config.load(
        overrides=[
            "planet.grid_width=32",
            "planet.grid_height=16",
            "sim.initial_population=200",
            "sim.max_population=2000",
            *(overrides or []),
        ]
    )


# -- construction ------------------------------------------------------------------------------


def test_create_seeds_the_configured_founder_population(config: Config):
    simulation = Simulation.create(config)
    assert simulation.population.size == config.sim.initial_population
    assert simulation.day == 0
    assert simulation.last_stats is None


def test_run_zero_ticks_changes_nothing(config: Config):
    simulation = Simulation.create(config)
    before = simulation.population.energy[simulation.population.active].copy()
    assert simulation.run(0) is None
    assert simulation.day == 0
    assert np.array_equal(simulation.population.energy[simulation.population.active], before)


@pytest.mark.parametrize("ticks", [-1, 1.5, True])
def test_run_rejects_a_bad_tick_count(config: Config, ticks):
    with pytest.raises((TypeError, ValueError)):
        Simulation.create(config).run(ticks)


# -- the tick happens ---------------------------------------------------------------------------


def test_a_tick_ages_every_survivor():
    simulation = Simulation.create(small())
    simulation.step()
    assert np.all(simulation.population.age[simulation.population.active] == 1)


def test_a_tick_advances_the_world_exactly_one_day():
    simulation = Simulation.create(small())
    simulation.run(5)
    assert simulation.day == 5
    assert simulation.last_stats is not None
    assert simulation.last_stats.day == 5


def test_organisms_feed_and_pay_on_the_reference_planet():
    """The founder lineage must be able to earn on the planet it was seeded into."""
    simulation = Simulation.create(small())
    stats = simulation.step()
    assert stats.intake_autotrophy > 0.0
    assert stats.cost_basal > 0.0
    assert stats.cost_support > 0.0
    assert stats.net_energy > 0.0


def test_reserves_build_up_while_the_ledger_is_positive():
    simulation = Simulation.create(small())
    first = simulation.step()
    later = simulation.run(20)
    assert later is not None
    assert later.mean_energy_fullness > first.mean_energy_fullness


def test_energy_never_exceeds_storage_capacity():
    simulation = Simulation.create(small())
    simulation.run(40)
    active = simulation.population.active
    assert np.all(
        simulation.population.energy[active]
        <= simulation.population.phenotypes.storage_capacity[active] + 1e-6
    )


# -- the matter ledger ----------------------------------------------------------------------------


def test_nutrients_fall_by_exactly_what_was_drawn():
    simulation = Simulation.create(small(FROZEN_RESOURCES))
    before = simulation.world.resources.nutrients.sum()
    stats = simulation.step()
    after = simulation.world.resources.nutrients.sum()
    assert stats.nutrients_drawn > 0.0
    assert before - after == pytest.approx(stats.nutrients_drawn, rel=1e-9)


def test_detritus_falls_by_what_was_eaten_and_rises_by_what_died():
    simulation = Simulation.create(
        small([*FROZEN_RESOURCES, "genome.loci.aff_detritus.init=2.0"])
    )
    for _ in range(6):
        before = simulation.world.resources.detritus.sum()
        stats = simulation.step()
        after = simulation.world.resources.detritus.sum()
        assert after - before == pytest.approx(
            stats.detritus_deposited - stats.detritus_consumed, abs=1e-9
        )
    assert stats.detritus_consumed > 0.0
    assert stats.detritus_deposited > 0.0


def test_a_corpse_returns_its_body_and_its_unspent_energy():
    """Nothing an organism accumulated may leave the world when it dies."""
    simulation = Simulation.create(
        small([*FROZEN_RESOURCES, "energy.mortality.background=1.0"])
    )
    population = simulation.population
    active = population.active
    expected = (
        simulation.config.energy.energy_density
        * population.phenotypes.mass[active].astype(np.float64).sum()
    )
    before = simulation.world.resources.detritus.sum()
    stats = simulation.step()

    assert stats.population == 0
    assert stats.deaths == 200
    # The corpses carry unspent energy too, so the deposit is at least the body mass.
    assert simulation.world.resources.detritus.sum() - before == pytest.approx(
        stats.detritus_deposited
    )
    assert stats.detritus_deposited > expected


def test_resource_pools_stay_within_their_physical_bounds():
    simulation = Simulation.create(small())
    for _ in range(30):
        simulation.step()
        resources = simulation.world.resources
        assert np.all(resources.nutrients >= 0.0)
        assert np.all(resources.nutrients <= resources.nutrient_capacity + 1e-9)
        assert np.all(resources.detritus >= 0.0)


def test_a_crowded_cell_cannot_be_drawn_past_empty():
    """Contention is what makes a cell's carrying capacity real rather than nominal."""
    simulation = Simulation.create(
        small([*FROZEN_RESOURCES, "sim.initial_population=1500"])
    )
    population = simulation.population
    population.cell[population.active] = 5  # everyone into one cell
    stats = simulation.step()
    assert stats.nutrients_drawn > 0.0
    assert np.all(simulation.world.resources.nutrients >= 0.0)


# -- mortality ------------------------------------------------------------------------------------


def test_starvation_is_recorded_separately_from_hazards():
    """A lineage that cannot pay its bills must die of that, and be seen to."""
    simulation = Simulation.create(
        small(["energy.intake.k_photo=0.0", "energy.intake.k_detritus=0.0"])
    )
    starved = 0
    for _ in range(30):
        stats = simulation.step()
        starved += stats.deaths_starvation
        if stats.population == 0:
            break
    assert starved > 0
    assert simulation.population.size == 0


def test_a_lethal_planet_kills_through_the_hazard_path():
    simulation = Simulation.create(small(["planet.base_toxicity=1000.0"]))
    stats = simulation.step()
    assert stats.deaths_hazard > 0


def test_extinction_is_stable():
    simulation = Simulation.create(small(["energy.mortality.background=1.0"]))
    simulation.step()
    assert simulation.population.size == 0
    stats = simulation.run(5)
    assert stats is not None
    assert stats.population == 0
    assert stats.energy_intake == 0.0
    assert simulation.day == 6


def test_a_cohort_that_cannot_reproduce_dies_of_old_age():
    """Senescence must still end a generation when nothing replaces it.

    This was M3's whole-run outcome. Reproduction is disabled here rather than deleting the
    test, because "the population survives" must remain a statement about births outpacing
    deaths and not an accident of organisms being unable to die.

    Breeding is blocked by making it unaffordable. `repro_threshold` at 1.0 does not work --
    energy is clipped at storage capacity, so that threshold is reached exactly rather than
    never -- and raising `maturity_age` would also switch off the senescence this is testing,
    because senescence is measured in units of maturity age.
    """
    simulation = Simulation.create(small(["energy.reproduction.overhead=1e9"]))
    simulation.run(200)
    assert simulation.population.size == 0


def test_a_founder_lineage_persists_beyond_its_first_generation():
    """The milestone 4 criterion: the lineage outlives the cohort that started it."""
    simulation = Simulation.create(small())
    simulation.run(300)
    population = simulation.population
    assert population.size > 0
    # Nobody alive is a founder, so this is descendants rather than long-lived originals.
    assert np.all(population.generation[population.active] > 0)


# -- species and history ----------------------------------------------------------------------------


# A life-history strategy at the opposite end of every locus that defines one: late-maturing,
# hoarding, and highly fecund.  These loci are used because they are survivable -- a body plan
# pushed this far starves within a tick, which would remove the very divergence under test --
# and because together they are 0.42 in mate-compatibility units, well past the threshold.
OPPOSITE_LIFE_HISTORY = (
    "maturity_age",
    "senescence_rate",
    "repro_threshold",
    "offspring_count",
    "parental_investment",
    "sex_bias",
    "move_persistence",
)


def diverge(simulation: Simulation, rows: np.ndarray) -> None:
    """Move a block of organisms to the far end of every life-history locus."""
    population = simulation.population
    schema = population.schema
    for name in OPPOSITE_LIFE_HISTORY:
        index = schema.index_of(name)
        population.genomes[rows, index, :] = schema.high[index]
    population.phenotypes.update(0, population.genomes[: population.size], schema)


def test_a_run_starts_as_one_species_and_stays_one_while_it_stays_one_kind():
    simulation = Simulation.create(small(["sim.taxonomy_interval=10"]))
    stats = simulation.run(40)
    assert stats.species == 1
    assert stats.species_born == 0
    assert stats.species_extinct == 0
    assert simulation.history.species_count == 1


def test_a_lineage_divided_between_two_habitats_becomes_two_species():
    """The milestone 5 criterion, with the divergence imposed rather than evolved.

    Nothing on the reference planet diverges this far on its own -- see DEVLOG -- so the two
    halves are seeded already divergent and put in separate habitats.  What is being tested is
    the whole path: that the ordinary tick notices, records two species with the right
    parentage, and leaves every organism with the group it belongs to.
    """
    simulation = Simulation.create(small(["sim.taxonomy_interval=3"]))
    population = simulation.population
    half = population.size // 2
    rows = np.arange(population.size)
    diverge(simulation, rows[:half])
    # Two habitats, one per hemisphere.  Both are the most livable water cell their hemisphere
    # has, because a habitat that kills its occupants would remove the divergence under test
    # rather than isolate it.
    world = simulation.world
    optimum = float(population.schema.init[population.schema.index_of("temp_optimum")])
    mismatch = np.where(
        world.terrain.water, np.abs(world.climate.temperature_c - optimum), np.inf
    )
    equator = world.grid.shape[0] // 2
    north = int(np.argmin(mismatch[:equator]))
    south = int(np.argmin(mismatch[equator:])) + equator * world.grid.shape[1]
    population.cell[rows[:half]] = north
    population.cell[rows[half:]] = south

    stats = simulation.run(3)
    assert stats.species == 2
    assert stats.species_born == 1
    assert simulation.history.lineage(1) == (0, 1)
    species = population.species_id[population.active]
    assert set(np.unique(species).tolist()) == {0, 1}
    # Each species is one coherent kind of organism, not a mixture of both.
    fecundity = population.phenotypes.trait("offspring_count")
    assert fecundity[species == 0].std() < 0.1
    assert fecundity[species == 1].std() < 0.1
    assert abs(fecundity[species == 0].mean() - fecundity[species == 1].mean()) > 1.0


def test_the_taxonomy_is_revisited_only_on_its_interval():
    simulation = Simulation.create(small(["sim.taxonomy_interval=3"]))
    population = simulation.population
    diverge(simulation, np.arange(60))

    born = [simulation.step().species_born for _ in range(6)]
    assert born[2] == 1  # day 3, and not before it
    assert sum(born) == 1


def test_an_extinct_biosphere_is_recorded_as_an_extinct_species():
    simulation = Simulation.create(
        small(["energy.mortality.background=1.0", "sim.extinction_confirm_ticks=3"])
    )
    stats = simulation.run(5)
    assert simulation.population.size == 0
    assert stats.species == 0
    assert simulation.history.extinct() == (0,)
    assert simulation.history.records[0].extinct_day == 0


# -- determinism ------------------------------------------------------------------------------------


def test_two_runs_of_the_same_config_and_seed_are_identical():
    left = Simulation.create(small())
    right = Simulation.create(small())
    left.run(25)
    right.run(25)

    assert left.population.size == right.population.size
    for name, array in left.population.active_arrays().items():
        assert np.array_equal(array, right.population.active_arrays()[name]), name
    assert np.array_equal(
        left.world.resources.nutrients, right.world.resources.nutrients
    )
    assert left.rng.get_state() == right.rng.get_state()


def test_a_different_seed_produces_a_different_run():
    left = Simulation.create(small())
    right = Simulation.create(small(["sim.seed=99"]))
    left.run(15)
    right.run(15)
    assert not np.array_equal(
        left.population.organism_id[left.population.active],
        right.population.organism_id[right.population.active],
    )


def test_the_tick_reads_no_state_outside_the_simulation():
    """Stepping one simulation must not perturb an independent one built the same way."""
    reference = Simulation.create(small())
    reference.run(10)

    disturbed = Simulation.create(small())
    noise = Simulation.create(small(["sim.seed=7"]))
    for _ in range(10):
        noise.step()
        disturbed.step()
    assert np.array_equal(
        reference.population.organism_id[reference.population.active],
        disturbed.population.organism_id[disturbed.population.active],
    )


# -- the organism inspector ------------------------------------------------------------------


def test_inspecting_an_organism_changes_nothing():
    """The inspector is a read: it re-evaluates the tick's equations and writes none of it back.

    This is the constraint the milestone was specified under, so it is asserted rather than
    assumed -- an inspector that quietly drew down a cell's nutrients would make looking at a
    run change the run.
    """

    simulation = Simulation.create(small())
    simulation.run(5)
    population = simulation.population
    before = {name: array.copy() for name, array in simulation.world.arrays().items()}
    energy = population.energy[population.active].copy()
    cells = population.cell[population.active].copy()

    simulation.inspect(int(population.organism_id[0]))

    for name, array in simulation.world.arrays().items():
        assert np.array_equal(array, before[name]), name
    assert np.array_equal(population.energy[population.active], energy)
    assert np.array_equal(population.cell[population.active], cells)


def test_inspecting_a_dead_organism_is_an_error():
    simulation = Simulation.create(small())
    simulation.run(1)
    with pytest.raises(KeyError):
        simulation.inspect(10**9)


def test_an_inspected_ledger_adds_up():
    """Every itemised number the panel shows must sum to the total shown beside it."""

    simulation = Simulation.create(small())
    simulation.run(5)
    panel = simulation.inspect(int(simulation.population.organism_id[0]))

    costs = panel["costs"]
    itemised = sum(value for name, value in costs.items() if name != "total")
    assert costs["total"] == pytest.approx(itemised)
    intake = panel["intake"]
    assert intake["total"] == pytest.approx(
        intake["autotrophy"] + intake["detritivory"] + intake["predation"]
    )
    assert panel["net_energy"] == pytest.approx(intake["total"] - costs["total"])
    assert 0.0 <= panel["hazards"]["combined"] <= 1.0


def test_a_starving_organism_explains_itself_from_its_own_panel():
    """The milestone's acceptance criterion.

    An organism that cannot photosynthesise is starving; its panel has to say so with numbers
    rather than leave it to be inferred -- a negative ledger, a named channel that is short,
    and a reserve with a countable number of days left in it.
    """

    simulation = Simulation.create(
        small(["energy.intake.k_photo=0.0", "energy.intake.k_detritus=0.0"])
    )
    simulation.run(3)
    panel = simulation.inspect(int(simulation.population.organism_id[0]))

    assert panel["intake"]["total"] == pytest.approx(0.0)
    assert panel["net_energy"] < 0.0
    # Which channel is emptying it, and how long that leaves: both readable off the panel.
    dominant = max(
        (name for name in panel["costs"] if name != "total"),
        key=lambda name: panel["costs"][name],
    )
    assert dominant == "basal"
    assert 0.0 < panel["energy"] / -panel["net_energy"] < float("inf")
    assert panel["energy"] < panel["thresholds"]["repro_energy"]


def test_the_inspector_reports_the_cell_the_organism_is_standing_in():
    simulation = Simulation.create(small())
    simulation.run(5)
    population = simulation.population
    panel = simulation.inspect(int(population.organism_id[0]))
    width = simulation.world.grid.width

    assert panel["cell"] == int(population.cell[0])
    assert panel["row"] * width + panel["column"] == panel["cell"]
    assert panel["environment"]["on_land"] == bool(
        simulation.world.terrain.land.ravel()[panel["cell"]]
    )
    assert panel["neighbours"] >= 1


# -- hot path ------------------------------------------------------------------------------------


def test_the_tick_has_no_per_organism_python_iteration():
    """The same coarse lint the population storage is held to, applied to the tick itself.

    ``Simulation.run``, ``MovementModel.move`` and ``ReproductionModel._disperse`` legitimately
    loop -- over ticks and over step index -- and are excluded; nothing here may loop over
    organisms.
    """
    from evosim.life.census import CellCensus
    from evosim.life.energy import EnergyModel
    from evosim.life.mortality import MortalityModel
    from evosim.life.predation import PredationModel
    from evosim.life.reproduction import ReproductionModel

    for method in (
        Simulation.step,
        Simulation._feed,
        Simulation._reap,
        Simulation._record,
        Simulation._summarise,
        EnergyModel.costs_for,
        EnergyModel.intake_for,
        EnergyModel.foraging_yield,
        MortalityModel.hazards,
        CellCensus.build,
        PredationModel.hunt,
        PredationModel._draw_attacks,
        PredationModel._spend_allowance,
        PredationModel.expected_gain,
        PredationModel.expected_risk,
        ReproductionModel.reproduce,
        ReproductionModel._brood_sizes,
        ReproductionModel._choose_mates,
    ):
        tree = ast.parse(dedent(inspect.getsource(method)))
        forbidden = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp)
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree)), method
