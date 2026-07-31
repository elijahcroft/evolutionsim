"""Directional selection: the tests milestone 4 exists to make possible.

Everything before this milestone could be checked one tick at a time.  These cannot: they assert
that over hundreds of ticks and dozens of generations the *distribution of genomes* moves in the
direction the energy model implies, with nothing anywhere naming a fitness function.

Two properties are worth stating about how they are written:

* **Each is a comparison, not an absolute.**  A trait mean after 400 ticks depends on mutation,
  drift, and which cells happened to be lucky.  Asserting "temp_optimum ends near 21.8" would be
  fitting the test to one run.  Asserting "a lineage on a cold planet ends colder than the same
  lineage on a warm one" is a claim about selection that survives a change of seed.

* **A locus already at its optimum is the control.**  If a matched trait wandered as far as a
  mismatched one climbed, the climb would be drift rather than selection, and every result in
  this file would be noise.

These runs are the slowest tests in the suite by design: selection is not observable quickly.
"""

from __future__ import annotations

import numpy as np
import pytest

from evosim.config import Config
from evosim.sim import Simulation

# A small world that reaches many generations quickly. The raised mutation rate and lowered
# maturity age buy generations per tick; neither changes the direction selection pushes, only
# how fast the answer arrives.
EVOLVING = [
    "planet.grid_width=32",
    "planet.grid_height=16",
    "sim.initial_population=400",
    "sim.max_population=3000",
    "genome.loci.mutation_rate.init=0.1",
    "genome.loci.maturity_age.init=10.0",
]
TICKS = 400


def evolve(*overrides: str, ticks: int = TICKS) -> Simulation:
    simulation = Simulation.create(Config.load(overrides=[*EVOLVING, *overrides]))
    simulation.run(ticks)
    assert simulation.population.size > 0, "lineage went extinct before selection was measurable"
    return simulation


def mean_trait(simulation: Simulation, name: str) -> float:
    population = simulation.population
    return float(population.phenotypes.trait(name, population.active).mean())


# -- the lineage survives at all -----------------------------------------------------------


def test_a_lineage_outlives_many_generations():
    """M3's cohort died at day 45; M4's lineage must still be here after hundreds of days."""
    simulation = evolve()
    population = simulation.population
    assert population.size > 0
    assert int(population.generation[population.active].max()) > 10


# -- thermal adaptation ---------------------------------------------------------------------


def test_a_mismatched_thermal_optimum_climbs_toward_its_habitat():
    """The prediction milestone 3 recorded but could not test.

    The founder is seeded two degrees below the water it lives in. Nothing tells it to warm up;
    a colder-than-optimal organism simply pays thermoregulation every tick and carries a thermal
    hazard, so its descendants are outbred by whichever mutants sit closer to the water.
    """
    assert mean_trait(evolve("genome.loci.temp_optimum.init=20.0"), "temp_optimum") > 20.3


def test_a_matched_thermal_optimum_stays_put():
    """The control: without a mismatch to correct, the same locus must not wander."""
    assert mean_trait(evolve(), "temp_optimum") == pytest.approx(22.0, abs=0.5)


def test_a_colder_planet_selects_a_colder_lineage():
    """The strongest form: one founder, two planets, opposite answers.

    If the direction of selection tracks the planet rather than the genome, then nothing in the
    code is choosing the outcome -- which is the entire claim the project rests on.
    """
    cold = mean_trait(
        evolve("planet.climate.base_temperature_c=-33.0"), "temp_optimum"
    )
    warm = mean_trait(evolve("planet.climate.base_temperature_c=-18.0"), "temp_optimum")
    assert cold < warm


# -- the digestion prediction ------------------------------------------------------------------


def test_digestion_efficiency_climbs_as_milestone_3_predicted():
    """M3 worked out that `upkeep_digestion` is too cheap and predicted this; here it is.

    Raising `digestion_efficiency` from 0.40 to 0.95 multiplies intake by 2.4 while adding only
    about 26% to basal cost, so the locus should climb toward its upper bound. The test records
    the prediction rather than blessing the balance: it is evidence that `upkeep_digestion`
    wants raising, not evidence that 0.9 is the right value.
    """
    assert mean_trait(evolve(), "digestion_efficiency") > 0.42


# -- the population cap must not select ------------------------------------------------------------


def test_capacity_throttling_does_not_select_for_older_parents():
    """A regression guard on a real bug: the cap once dropped births in parent-row order.

    Rows are roughly age-ordered after compaction, so taking "the first that fit" quietly
    selected for the offspring of older parents. It was invisible in every single-tick test and
    it reversed the direction of thermal selection above. The cap is a memory budget; it must
    have no opinion about who breeds.
    """
    simulation = Simulation.create(
        Config.load(
            overrides=[
                *EVOLVING,
                "sim.max_population=1200",
                "sim.initial_population=1000",
                "genome.loci.offspring_count.init=6.0",
            ]
        )
    )
    population = simulation.population
    ages, weighted = [], []
    for _ in range(200):
        eligible = population.age[population.active].copy()
        ids = population.organism_id[population.active].copy()
        stats = simulation.step()
        if not stats.capacity_throttle:
            continue
        active = population.active
        newborn = np.isin(population.organism_id[active], ids, invert=True)
        parents = population.parent_id[active][newborn, 0]
        rows = np.searchsorted(ids, parents)
        ages.append(eligible.mean())
        weighted.append(eligible[rows].mean())

    assert len(ages) > 5, "the cap never bound, so nothing was tested"
    # Parents of surviving offspring must not be systematically older than the population they
    # were drawn from. A generous tolerance still catches the original bug, which biased this
    # by far more than a single tick of ageing.
    assert np.mean(weighted) - np.mean(ages) < 5.0
