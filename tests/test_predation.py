"""Tests for predation and herbivory.

This is the first interaction where one organism's genome decides another's fate, so the things
worth pinning are the ones that would let a food web cheat: a carcass eaten twice, meat created
that no body contained, or an attacker gaining more than its metabolism could process.

The arms-race structure is pinned too.  Every term that helps an attacker is a trait the
attacker pays for elsewhere, and every term that saves a prey is a trait the prey pays for --
if a future change gave either side something free, the coefficients in ``energy.yaml`` would
stop being the whole story about what selection favours.
"""

from __future__ import annotations

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.predation import PredationModel
from evosim.sim import Simulation


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def model(config: Config) -> PredationModel:
    return PredationModel.from_config(config)


def one(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float64)


HUNTER = {
    "aff_herbivore": 4.0,
    "aff_autotroph": -2.0,
    "aggression": 5.0,
    "move_speed": 2.0,
    "sense_range": 4.0,
    "body_size": 3.0,
}


def hunting_world(*overrides: str, hunters: int = 1, prey: int = 400) -> Simulation:
    """One cell holding a few large hunters and a crowd of founder-sized prey."""
    simulation = Simulation.create(
        Config.load(
            overrides=[
                "planet.grid_width=32",
                "planet.grid_height=16",
                f"sim.initial_population={prey + hunters}",
                "sim.max_population=20000",
                "genome.loci.move_speed.init=0.0",
                *overrides,
            ]
        )
    )
    population = simulation.population
    population.cell[population.active] = population.cell[0]
    schema = population.schema
    for name, value in HUNTER.items():
        population.genomes[:hunters, schema.index_of(name), :] = value
    population.phenotypes.update(0, population.genomes[: population.size], schema)
    population.energy[: population.size] = (
        population.phenotypes.storage_capacity[: population.size] * 0.5
    )
    return simulation


# -- the encounter equation ------------------------------------------------------------------


def test_an_empty_cell_offers_no_encounters(model: PredationModel):
    assert model.encounter_rate(one(5.0), one(5.0), one(5.0), one(0.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_every_encounter_factor_raises_the_rate(model: PredationModel, index: int):
    low = [one(0.5), one(0.5), one(0.5), one(10.0)]
    high = list(low)
    high[index] = high[index] * 2.0
    assert model.encounter_rate(*high) > model.encounter_rate(*low)


def test_sensing_enters_the_encounter_rate_quadratically(model: PredationModel):
    """Detection sweeps an area, which is the second thing `sense_range` earns."""
    near = model.encounter_rate(one(0.0), one(0.0), one(0.7), one(1.0))
    far = model.encounter_rate(one(0.0), one(0.0), one(1.7), one(1.0))
    assert far / near == pytest.approx((2.0 / 1.0) ** 2)


# -- the capture contest ----------------------------------------------------------------------


def base_capture(model: PredationModel, **changes: float) -> float:
    arguments = {
        "attacker_mass": 1.0,
        "prey_mass": 1.0,
        "attacker_speed": 1.0,
        "prey_speed": 1.0,
        "attacker_sense": 1.0,
        "attacker_aggression": 0.0,
        "prey_camouflage": 0.0,
        "prey_armor": 0.0,
    }
    arguments.update(changes)
    return float(model.capture_probability(**{k: one(v) for k, v in arguments.items()})[0])


def test_capture_probability_is_a_probability(model: PredationModel):
    for mass in (1e-6, 1.0, 1e6):
        value = base_capture(model, attacker_mass=mass)
        assert 0.0 <= value <= 1.0


def test_most_attacks_on_an_equal_opponent_fail(model: PredationModel):
    """`capture_bias` is negative so that hunting is hard by default."""
    assert base_capture(model) < 0.5


@pytest.mark.parametrize(
    "change,better",
    [
        ({"attacker_mass": 8.0}, True),
        ({"attacker_speed": 3.0}, True),
        ({"attacker_aggression": 4.0}, True),
        ({"prey_mass": 8.0}, False),
        ({"prey_speed": 3.0}, False),
        ({"prey_armor": 3.0}, False),
        ({"prey_camouflage": 3.0}, False),
    ],
)
def test_each_side_of_the_contest_pulls_its_own_way(model, change, better):
    baseline = base_capture(model)
    assert (base_capture(model, **change) > baseline) is better


def test_size_advantage_is_measured_in_doublings(model: PredationModel):
    """A log2 ratio means what matters is how many times bigger, not by how much."""
    logit = lambda p: np.log(p / (1.0 - p))
    step = logit(base_capture(model, attacker_mass=2.0)) - logit(base_capture(model))
    assert step == pytest.approx(
        model.intake.capture_size_advantage, rel=1e-6
    )


def test_sharper_senses_defeat_camouflage(model: PredationModel):
    """The direct coupling that makes eyes and concealment an arms race."""
    hidden = base_capture(model, prey_camouflage=4.0, attacker_sense=0.0)
    spotted = base_capture(model, prey_camouflage=4.0, attacker_sense=8.0)
    assert spotted > hidden


# -- diet match -------------------------------------------------------------------------------


def test_a_herbivore_profits_from_autotrophic_prey(model: PredationModel):
    grazing = model.diet_match(one(0.9), one(0.05), one(1.0))
    hunting = model.diet_match(one(0.9), one(0.05), one(0.0))
    assert grazing > hunting


def test_a_carnivore_profits_from_heterotrophic_prey(model: PredationModel):
    grazing = model.diet_match(one(0.05), one(0.9), one(1.0))
    hunting = model.diet_match(one(0.05), one(0.9), one(0.0))
    assert hunting > grazing


def test_the_wrong_food_still_yields_the_configured_floor(config, model):
    """A specialist beside inedible food must go hungry, not become unable to eat at all."""
    assert model.diet_match(one(0.0), one(0.0), one(1.0)) == pytest.approx(
        config.energy.intake.diet_match_floor
    )


# -- resolution in a real tick ------------------------------------------------------------------


def test_a_hunter_kills_and_is_fed_by_it():
    simulation = hunting_world()
    stats = simulation.step()
    assert stats.attacks > 0
    assert stats.kills > 0
    assert stats.deaths_predation == stats.kills
    assert stats.intake_predation > 0.0


def test_hunting_traits_are_what_make_a_hunter():
    """A crowd of sessile, unaggressive, barely-sighted organisms kills far less than a hunter.

    Every trait in the encounter equation is one the energy model charges for, so predation has
    to be bought rather than had.
    """
    idle = hunting_world(hunters=0)
    without = sum(idle.step().kills for _ in range(5))
    active = hunting_world(hunters=2)
    with_hunters = sum(active.step().kills for _ in range(5))
    assert with_hunters > without


def test_no_organism_attacks_itself():
    simulation = hunting_world(hunters=1, prey=0)
    stats = simulation.step()
    assert stats.attacks == 0
    assert stats.kills == 0


def test_a_carcass_feeds_exactly_one_killer():
    """Several successful attackers cannot share more meat than the prey contained."""
    simulation = hunting_world(hunters=40)
    for _ in range(5):
        stats = simulation.step()
        assert stats.kills <= stats.attacks
        assert stats.deaths_predation == stats.kills
        assert stats.population >= 0


def test_what_the_killer_cannot_eat_becomes_carrion():
    simulation = hunting_world(
        "planet.resources.detritus_decay_rate=0.0",
        "planet.resources.nutrient_regen_water=0.0",
        "planet.resources.nutrient_regen_land=0.0",
    )
    before = simulation.world.resources.detritus.sum()
    stats = simulation.step()
    after = simulation.world.resources.detritus.sum()

    assert stats.kills > 0
    assert stats.carrion_returned > 0.0
    assert after - before == pytest.approx(
        stats.detritus_deposited - stats.detritus_consumed, abs=1e-9
    )


def test_a_predated_organism_is_not_also_buried():
    """Depositing an eaten carcass again would let a food web manufacture matter."""
    simulation = hunting_world(
        "planet.resources.detritus_decay_rate=0.0",
        "planet.resources.nutrient_regen_water=0.0",
        "planet.resources.nutrient_regen_land=0.0",
        "energy.mortality.background=0.0",
    )
    population = simulation.population
    masses = population.phenotypes.mass[population.active].astype(np.float64)
    energies = population.energy[population.active].astype(np.float64)
    total = float(
        simulation.config.energy.energy_density * masses.sum() + energies.sum()
    )

    before = simulation.world.resources.detritus.sum()
    simulation.step()
    added = simulation.world.resources.detritus.sum() - before
    # Everything deposited came out of bodies that existed, so the deposit cannot exceed the
    # whole population's worth of matter.
    assert 0.0 < added < total


def test_predation_intake_respects_the_aerobic_ceiling():
    """energy.yaml caps *all* intake at a multiple of basal cost; a predator is not exempt."""
    generous = hunting_world("energy.intake.aerobic_scope_max=40.0").step()
    stingy = hunting_world("energy.intake.aerobic_scope_max=1.0").step()
    assert stingy.intake_predation < generous.intake_predation


def test_predation_deaths_are_reported_separately():
    simulation = hunting_world("energy.mortality.background=0.0")
    stats = simulation.step()
    assert stats.deaths_predation > 0
    assert stats.deaths == stats.deaths_predation + stats.deaths_hazard + stats.deaths_starvation


def test_predation_consumes_only_the_encounter_stream():
    simulation = hunting_world()
    population = simulation.population
    capacity = population.phenotypes.storage_capacity[population.active].astype(np.float64)
    before = simulation.rng.get_state()["streams"]
    simulation.predation.hunt(
        population, simulation.world, simulation.rng.encounter, capacity
    )
    after = simulation.rng.get_state()["streams"]
    assert {name for name in before if before[name] != after[name]} == {"encounter"}


# -- is the niche reachable at all? ----------------------------------------------------------------


def test_a_hunting_genome_can_pay_for_itself():
    """Predation must be a living somebody could make, or the whole channel is decoration.

    This is deliberately a statement about one hand-built genome rather than about evolution
    finding it. M4 records that the reverse is not true on the reference planet: from the
    autotroph founder the local gradient points away from hunting, because the traits that make
    predation pay only pay once several of them are large at the same time.
    """
    simulation = hunting_world(hunters=1, prey=600)
    population = simulation.population
    hunter = population.organism_id[:1].copy()
    reserve = float(population.energy[0])

    for _ in range(4):
        simulation.step()
        alive = np.isin(population.organism_id[population.active], hunter)
        assert alive.any(), "the hunter died before it could show a profit"

    # Its reserve grew, so hunting more than covered a large body, fast movement and sharp
    # senses -- the three things that make it a hunter and the three it pays most for.
    assert float(population.energy[population.active][alive][0]) > reserve
