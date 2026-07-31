"""Tests for movement over grid neighbours.

Two properties matter more than the mechanics.  First, ``move_speed`` below one cell per tick
must still mean something, or the whole lower half of that locus becomes invisible to selection.
Second, ``sense_range`` must earn something -- an organism that can see a better neighbour must
actually reach it more often than a blind one -- because otherwise it is a pure cost and every
planet would drive it to zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.census import CellCensus
from evosim.life.energy import EnergyModel
from evosim.life.movement import MovementModel, stochastic_round
from evosim.life.population import Population
from evosim.life.predation import PredationModel
from evosim.rng import RngBundle
from evosim.world import World


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


def build(config: Config, seed: int = 0, **traits: float):
    """A world, a population with overridden loci, and the models that move it."""
    rng = RngBundle(seed)
    world = World.create(config.planet, rng)
    population = Population.seed_founders(config, world, rng)
    if traits:
        schema = population.schema
        active = population.active
        for name, value in traits.items():
            population.genomes[active, schema.index_of(name), :] = value
        population.phenotypes.update(0, population.genomes[active], schema)
    return world, population, EnergyModel.from_config(config), MovementModel.for_world(world), rng


def step(movement, population, world, energy, speed, basal, rng, config):
    """Drive one movement pass, building the per-cell census the way the tick does."""
    predation = PredationModel.from_config(config)
    census = CellCensus.build(
        population, config.energy.energy_density, config.energy.intake.k_encounter
    )
    return movement.move(
        population, world, energy, predation, census, speed, basal, rng
    )


def basal_of(population: Population, energy: EnergyModel) -> np.ndarray:
    phenotype = population.phenotypes.active
    return energy.basal_cost(
        phenotype.mass.astype(np.float64),
        phenotype.trait("metabolic_rate").astype(np.float64),
        energy.upkeep_multiplier(phenotype),
        phenotype.trait("temp_optimum").astype(np.float64),
    )


# -- sub-unit speeds --------------------------------------------------------------------------


def test_stochastic_round_only_produces_the_two_neighbouring_integers():
    rng = RngBundle(3).move
    result = stochastic_round(np.full(2000, 2.3), rng)
    assert set(np.unique(result)) <= {2, 3}


def test_stochastic_round_is_unbiased():
    """0.2 cells per tick must mean one step every fifth tick, not zero steps forever."""
    rng = RngBundle(4).move
    assert stochastic_round(np.full(200_000, 0.2), rng).mean() == pytest.approx(0.2, abs=0.01)


def test_stochastic_round_is_exact_on_whole_numbers():
    rng = RngBundle(5).move
    assert np.all(stochastic_round(np.full(1000, 3.0), rng) == 3)


def test_a_sub_unit_speed_still_moves_the_population(config: Config):
    world, population, energy, movement, rng = build(config, move_speed=0.2)
    moved = step(
        movement, population, world, energy, np.full(population.size, 0.2),
        basal_of(population, energy), rng.move, config,
    )
    assert 0 < moved.sum() < population.size


# -- the aerobic ceiling on speed ---------------------------------------------------------------


def test_realized_speed_never_exceeds_the_genetic_speed(config: Config):
    world, population, energy, movement, _ = build(config, move_speed=3.0)
    drag = energy.medium_drag(np.zeros(population.size, dtype=bool))
    speed = movement.realized_speed(population, energy, drag, basal_of(population, energy))
    assert np.all(speed <= 3.0 + 1e-6)


def test_thin_oxygen_slows_a_population_down(config: Config):
    from dataclasses import replace

    world, population, energy, movement, _ = build(config, move_speed=6.0)
    drag = energy.medium_drag(np.zeros(population.size, dtype=bool))
    basal = basal_of(population, energy)
    thin = EnergyModel.from_config(
        replace(config, planet=replace(config.planet, o2_fraction=0.005))
    )
    assert movement.realized_speed(population, thin, drag, basal).mean() < movement.realized_speed(
        population, energy, drag, basal
    ).mean()


# -- staying put ---------------------------------------------------------------------------------


def test_a_stationary_organism_never_changes_cell(config: Config):
    world, population, energy, movement, rng = build(config, move_speed=0.0)
    before = population.cell[population.active].copy()
    step(movement, population, world, energy, np.zeros(population.size), basal_of(population, energy), rng.move, config)
    assert np.array_equal(population.cell[population.active], before)


def test_headings_survive_a_tick_spent_standing_still(config: Config):
    """Persistence is only meaningful if a remembered direction outlives an idle tick."""
    world, population, energy, movement, rng = build(config, move_speed=1.0)
    basal = basal_of(population, energy)
    step(movement, population, world, energy, np.ones(population.size), basal, rng.move, config)
    headings = population.heading[population.active].copy()
    step(movement, population, world, energy, np.zeros(population.size), basal, rng.move, config)
    assert np.array_equal(population.heading[population.active], headings)


# -- geometry ------------------------------------------------------------------------------------


def test_movement_never_leaves_the_grid(config: Config):
    world, population, energy, movement, rng = build(config, move_speed=6.0, sense_range=8.0)
    basal = basal_of(population, energy)
    for _ in range(20):
        step(movement, population, world, energy, np.full(population.size, 6.0), basal, rng.move, config)
        cells = population.cell[population.active]
        assert np.all((cells >= 0) & (cells < world.grid.n_cells))


def test_movement_only_ever_visits_an_adjacent_cell_per_step(config: Config):
    world, population, energy, movement, rng = build(config, move_speed=1.0, sense_range=8.0)
    neighbours = world.grid.neighbour_indices().reshape(world.grid.n_cells, 4)
    before = population.cell[population.active].copy()
    step(movement, population, world, energy, np.ones(population.size), basal_of(population, energy), rng.move, config)
    after = population.cell[population.active]
    reachable = np.concatenate((before[:, None], neighbours[before]), axis=1)
    assert np.all((after[:, None] == reachable).any(axis=1))


def test_headings_point_at_the_cell_actually_entered(config: Config):
    world, population, energy, movement, rng = build(config, move_speed=1.0, sense_range=8.0)
    neighbours = world.grid.neighbour_indices().reshape(world.grid.n_cells, 4)
    before = population.cell[population.active].copy()
    step(movement, population, world, energy, np.ones(population.size), basal_of(population, energy), rng.move, config)
    after = population.cell[population.active]
    heading = population.heading[population.active]
    stepped = after != before
    assert stepped.any()
    assert np.array_equal(
        neighbours[before[stepped], heading[stepped]], after[stepped]
    )


# -- what sense_range earns --------------------------------------------------------------------


def test_sense_range_earns_better_foraging_ground(config: Config):
    """A seeing organism reaches the rich neighbour more often than a blind one does.

    Without this, `sense_range` would be a cost with nothing to buy and selection would drive
    it to zero on every planet, which would silently remove predation's precondition.

    Encounters are switched off for this one. The setup stacks the whole population into a
    single cell, which to a sharp-eyed organism is not a crowd but a larder -- and staying put
    to eat one's neighbours is the correct answer to that, just not the question being asked.
    Prey-seeking is tested on its own below.
    """
    quiet = Config.load(overrides=["energy.intake.k_encounter=0.0"])
    target = None
    reached = {}
    for sense in (0.0, 8.0):
        world, population, energy, movement, rng = build(
            quiet, seed=11, move_speed=1.0, sense_range=sense, move_persistence=0.0
        )
        # One cell is made far richer than everything around it, and the whole population is
        # placed on its western neighbour so the choice is a real one.
        centre = world.grid.n_cells // 2 + 40
        if target is None:
            target = centre
        west = world.grid.neighbour_indices().reshape(world.grid.n_cells, 4)[centre, 3]
        population.cell[population.active] = west
        world.resources.nutrients.reshape(-1)[centre] = 50.0
        world.climate.insolation.reshape(-1)[centre] = 5.0

        step(movement, population, world, energy, np.ones(population.size), basal_of(population, energy), rng.move, quiet)
        reached[sense] = int(np.count_nonzero(population.cell[population.active] == centre))

    assert reached[8.0] > reached[0.0]


def test_a_blind_population_spreads_over_every_direction(config: Config):
    """With nothing to see and nothing remembered, the choice must not be degenerate."""
    world, population, energy, movement, rng = build(
        config, seed=12, move_speed=1.0, sense_range=0.0, move_persistence=0.0
    )
    population.cell[population.active] = world.grid.n_cells // 2 + 17
    step(movement, population, world, energy, np.ones(population.size), basal_of(population, energy), rng.move, config)
    assert len(np.unique(population.cell[population.active])) == 5


def test_persistence_biases_a_blind_walk_toward_the_remembered_heading(config: Config):
    world, population, energy, movement, rng = build(
        config, seed=13, move_speed=1.0, sense_range=0.0, move_persistence=1.0
    )
    population.cell[population.active] = world.grid.n_cells // 2 + 17
    population.heading[population.active] = 1  # east
    basal = basal_of(population, energy)
    neighbours = world.grid.neighbour_indices().reshape(world.grid.n_cells, 4)
    east = neighbours[world.grid.n_cells // 2 + 17, 1]

    step(movement, population, world, energy, np.ones(population.size), basal, rng.move, config)
    went_east = np.count_nonzero(population.cell[population.active] == east)
    assert went_east > population.size / 5.0


def test_sensing_improves_the_ground_a_population_actually_occupies(config: Config):
    """The whole-run version of the previous test, on the reference planet.

    Mortality is switched off so the statistic can only move by migration; otherwise a drop
    would just be the badly-placed organisms dying, which proves nothing about movement.  The
    quantity measured is net foraging yield -- the same thing the choice rule optimises -- and
    not, say, temperature: an organism that trades a thermal penalty for far more light has
    chosen correctly, and a test that demanded cooler cells would be asserting a preference the
    energy model does not have.

    The comparison is between sighted and blind rather than against an absolute, because the
    absolute ratio also carries forty days of seasonal drift that has nothing to do with anyone
    choosing to go anywhere.  Only the difference between the two isolates movement.

    The sense values are kept low enough that the sensory cost does not starve anyone: at
    `sense_range` 2 on a founder-sized body the population collapses, and the survivors' ratio
    would then measure selective death rather than migration.
    """
    from evosim.sim import Simulation

    frozen = [
        "energy.mortality.background=0.0",
        "energy.mortality.h_thermal_max=0.0",
        "genome.loci.senescence_rate.init=0.0",
        # Reproduction is switched off as well as mortality: births in good cells would move
        # the statistic by demography rather than by anyone choosing to go anywhere.
        "genome.loci.maturity_age.init=3000.0",
        "genome.loci.move_speed.init=1.0",
        "sim.initial_population=400",
        "planet.grid_width=64",
        "planet.grid_height=32",
    ]

    def improvement(sense: float) -> float:
        simulation = Simulation.create(
            Config.load(overrides=[*frozen, f"genome.loci.sense_range.init={sense}"])
        )
        population = simulation.population

        def net_yield() -> float:
            phenotype = population.phenotypes.active
            from evosim.life.energy import Environment

            environment = Environment.sample(simulation.world, population.cell[population.active])
            fields = (
                environment.insolation,
                environment.nutrients,
                environment.moisture,
                environment.detritus,
                environment.temperature_c,
            )
            return float(
                simulation.energy.foraging_yield(
                    phenotype, *(field[:, None] for field in fields)
                ).mean()
            )

        before = net_yield()
        simulation.run(40)
        return net_yield() / before

    ratios = [improvement(sense) for sense in (0.0, 0.4, 1.0)]
    assert np.all(np.diff(ratios) > 0.0), ratios


def test_movement_consumes_only_the_move_stream(config: Config):
    """Movement must not disturb any other subsystem's position in its own stream."""
    world, population, energy, movement, rng = build(config, move_speed=2.0)
    before = rng.get_state()["streams"]
    step(
        movement, population, world, energy, np.full(population.size, 2.0),
        basal_of(population, energy), rng.move, config,
    )
    after = rng.get_state()["streams"]
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {"move"}


# -- prey seeking and fear -------------------------------------------------------------------


def crowd_next_door(config: Config, seed: int, decider: dict, crowd: dict | None = None):
    """One decider beside a cell packed with organisms of a separately chosen kind.

    The crowd's genome is set apart from the decider's on purpose. Giving them the same one
    would make "does a hunter approach prey" really ask "does a hunter approach other hunters",
    and the meal and the danger would then move together and cancel.
    """
    world, population, energy, movement, rng = build(config, seed=seed)
    schema = population.schema
    for name, value in (crowd or {}).items():
        population.genomes[:, schema.index_of(name), :] = value
    for name, value in decider.items():
        population.genomes[0, schema.index_of(name), :] = value
    population.phenotypes.update(0, population.genomes[: population.size], schema)

    neighbours = world.grid.neighbour_indices().reshape(world.grid.n_cells, 4)
    home = int(world.grid.n_cells // 2 + 37)
    east = int(neighbours[home, 1])
    population.cell[population.active] = east
    population.cell[0] = home
    population.heading[population.active] = -1
    return world, population, energy, movement, rng, home, east


def where_does_it_go(
    config: Config, trials: int, decider: dict, crowd: dict | None = None
) -> float:
    """Share of runs in which the lone decider walks into the crowded cell."""
    entered = 0
    for seed in range(trials):
        world, population, energy, movement, rng, home, east = crowd_next_door(
            config, seed, decider, crowd
        )
        speed = np.zeros(population.size)
        speed[0] = 1.0
        step(
            movement, population, world, energy, speed,
            basal_of(population, energy), rng.move, config,
        )
        entered += int(population.cell[0] == east)
    return entered / trials


HUNTER_TRAITS = {
    "sense_range": 4.0,
    "aggression": 5.0,
    "aff_herbivore": 4.0,
    "aff_autotroph": -2.0,
    "fear": 0.0,
    "move_persistence": 0.0,
}


def test_a_hunter_walks_toward_prey(config: Config):
    """genome.yaml defines `sense_range` as habitat *and prey* detection.

    Without this half, a predator wanders off its own food supply and predation cannot function
    in a spatial world at all -- it kills whatever shares its cell and then leaves.

    The control is the same hunter on a planet where encounters do not happen, which isolates
    prey-seeking from every other reason a cell might look good.
    """
    blind_to_prey = Config.load(overrides=["energy.intake.k_encounter=0.0"])
    assert where_does_it_go(config, 12, HUNTER_TRAITS) > where_does_it_go(
        blind_to_prey, 12, HUNTER_TRAITS
    )


def test_fear_drives_prey_away_from_a_dangerous_cell(config: Config):
    """`fear` weights being eaten against being fed, on the one scale the model has.

    The crowd here is made of hunters and the decider is ordinary prey, so the cell offers it
    almost nothing to eat and a great deal of danger. An organism that values its life at
    nothing still drifts in; one that overvalues it stays out. Without this the locus is dead
    config and prey have no behavioural defence at all.
    """
    dangerous = {"aggression": 5.0, "sense_range": 4.0}
    fearless = where_does_it_go(
        config, 12, {"fear": 0.0, "sense_range": 1.0, "move_persistence": 0.0}, dangerous
    )
    timid = where_does_it_go(
        config, 12, {"fear": 5.0, "sense_range": 1.0, "move_persistence": 0.0}, dangerous
    )
    assert timid < fearless
