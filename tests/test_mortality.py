"""Tests for environmental, age-related, and starvation mortality.

Mortality is the second and last way a trait may be disadvantageous, so the same discipline
applies here as in the energy tests: every hazard must be zero when its mismatch is zero, must
be reducible by the tolerance that is supposed to counter it, and must never combine into a
certainty just because several are present at once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.mortality import Hazards, MortalityModel, saturating_hazard, starved
from tests.test_energy import phenotypes, uniform_environment


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def model(config: Config) -> MortalityModel:
    return MortalityModel.from_config(config)


def ages(count: int, value: int = 0) -> np.ndarray:
    return np.full(count, value, dtype=np.int64)


# -- hazard shape ---------------------------------------------------------------------------


def test_saturating_hazard_is_zero_at_no_mismatch_and_half_at_one():
    assert saturating_hazard(0.4, np.array([0.0])) == pytest.approx(0.0)
    assert saturating_hazard(0.4, np.array([1.0])) == pytest.approx(0.2)


def test_saturating_hazard_never_reaches_its_maximum():
    values = saturating_hazard(0.4, np.linspace(0.0, 1e6, 500))
    assert np.all(values < 0.4)
    assert np.all(np.diff(values) > 0.0)


def test_hazards_combine_as_independent_probabilities():
    """Three coin flips must leave a survivor, so hazards multiply rather than add."""
    half = np.full(1, 0.5)
    zero = np.zeros(1)
    hazards = Hazards(
        background=half,
        thermal=half,
        radiation=half,
        toxicity=zero,
        pressure=zero,
        senescence=zero,
    )
    assert hazards.combined == pytest.approx(1.0 - 0.5**3)


def test_combined_hazard_is_a_probability_even_when_every_hazard_is_certain():
    one = np.ones(1)
    hazards = Hazards(
        background=one,
        thermal=one,
        radiation=one,
        toxicity=one,
        pressure=one,
        senescence=one,
    )
    assert hazards.combined == pytest.approx(1.0)


# -- individual hazards -----------------------------------------------------------------------


def test_a_comfortable_organism_faces_only_background_mortality(config, model):
    """On a benign planet inside its thermal window, nothing but bad luck kills."""
    organism = phenotypes(config, temp_optimum=15.0, temp_tolerance=8.0)
    hazards = model.hazards(organism, uniform_environment(1, temperature_c=15.0), ages(1))
    assert hazards.thermal == pytest.approx(0.0)
    assert hazards.radiation == pytest.approx(0.0)
    assert hazards.toxicity == pytest.approx(0.0)
    assert hazards.pressure == pytest.approx(0.0)
    assert hazards.senescence == pytest.approx(0.0)
    assert hazards.combined == pytest.approx(config.energy.mortality.background)


def test_thermal_hazard_reaches_half_its_maximum_at_the_lethal_margin(config, model):
    mortality = config.energy.mortality
    organism = phenotypes(config, temp_optimum=15.0, temp_tolerance=0.0)
    environment = uniform_environment(
        1, temperature_c=15.0 + mortality.thermal_lethal_margin_c
    )
    hazards = model.hazards(organism, environment, ages(1))
    assert hazards.thermal == pytest.approx(mortality.h_thermal_max / 2.0)


def test_thermal_tolerance_buys_survival(config: Config, model: MortalityModel):
    environment = uniform_environment(1, temperature_c=40.0)
    narrow = model.hazards(phenotypes(config, temp_tolerance=0.0), environment, ages(1))
    wide = model.hazards(phenotypes(config, temp_tolerance=30.0), environment, ages(1))
    assert wide.thermal < narrow.thermal


def test_radiation_hazard_needs_a_radioactive_planet(config: Config):
    calm = MortalityModel.from_config(config)
    hot = MortalityModel.from_config(
        replace(config, planet=replace(config.planet, radiation=5.0))
    )
    environment = uniform_environment(1)
    assert calm.hazards(phenotypes(config), environment, ages(1)).radiation == pytest.approx(0.0)
    assert hot.hazards(phenotypes(config), environment, ages(1)).radiation > 0.0


def test_radiation_tolerance_has_diminishing_returns(config: Config):
    """Tolerance divides rather than subtracts, so it cannot be maxed out cheaply."""
    model = MortalityModel.from_config(
        replace(config, planet=replace(config.planet, radiation=5.0))
    )
    environment = uniform_environment(1)
    hazard = [
        model.hazards(
            phenotypes(config, radiation_tolerance=t), environment, ages(1)
        ).radiation.item()
        for t in (0.0, 1.0, 2.0, 3.0)
    ]
    gains = -np.diff(hazard)
    assert np.all(gains > 0.0)
    assert np.all(np.diff(gains) < 0.0)


def test_toxicity_hazard_tracks_the_local_field(config: Config, model: MortalityModel):
    clean = model.hazards(phenotypes(config), uniform_environment(1), ages(1))
    poisoned = model.hazards(
        phenotypes(config), uniform_environment(1, toxicity=3.0), ages(1)
    )
    assert clean.toxicity == pytest.approx(0.0)
    assert poisoned.toxicity > 0.0


def test_pressure_hazard_measures_mismatch_from_one_atmosphere(config: Config):
    environment = uniform_environment(1)
    earthlike = MortalityModel.from_config(config)
    crushing = MortalityModel.from_config(
        replace(config, planet=replace(config.planet, pressure=50.0))
    )
    assert earthlike.hazards(phenotypes(config), environment, ages(1)).pressure == pytest.approx(0.0)

    exposed = crushing.hazards(phenotypes(config, pressure_tolerance=0.0), environment, ages(1))
    adapted = crushing.hazards(phenotypes(config, pressure_tolerance=5.0), environment, ages(1))
    assert exposed.pressure > adapted.pressure > 0.0


def test_depth_is_what_makes_pressure_tolerance_a_live_locus(config: Config):
    """Before M7 this locus could not be selected on at all.

    Pressure was a planetary constant, and on a 1 atm planet the mismatch term was identically
    zero for every organism everywhere -- so `pressure_tolerance` drifted and nothing else.
    With the water column counted, the deep costs something specific to occupy and paying for
    tolerance buys something specific back.
    """

    model = MortalityModel.from_config(config)
    assert config.planet.pressure == pytest.approx(1.0)

    surface = uniform_environment(1, depth_km=0.0)
    abyss = uniform_environment(1, depth_km=5.0)
    organism = phenotypes(config, pressure_tolerance=0.0)

    assert model.hazards(organism, surface, ages(1)).pressure == pytest.approx(0.0)
    assert model.hazards(organism, abyss, ages(1)).pressure > 0.0

    tolerant = phenotypes(config, pressure_tolerance=5.0)
    assert (
        model.hazards(tolerant, abyss, ages(1)).pressure
        < model.hazards(organism, abyss, ages(1)).pressure
    )


def test_senescence_rises_with_age_and_saturates_at_certainty(config, model):
    organism = phenotypes(config, maturity_age=20.0, senescence_rate=0.05)
    environment = uniform_environment(1)
    hazard = [
        model.hazards(organism, environment, ages(1, age)).senescence.item()
        for age in (0, 10, 20, 40, 10_000)
    ]
    assert hazard[0] == pytest.approx(0.0)
    assert np.all(np.diff(hazard) > 0.0)
    assert hazard[-1] == pytest.approx(1.0)


def test_senescence_is_measured_in_units_of_maturity_age(config, model):
    """A slow life history postpones ageing rather than merely enduring it."""
    environment = uniform_environment(1)
    quick = model.hazards(phenotypes(config, maturity_age=10.0), environment, ages(1, 30))
    slow = model.hazards(phenotypes(config, maturity_age=1000.0), environment, ages(1, 30))
    assert slow.senescence < quick.senescence


def test_a_high_senescence_rate_shortens_life(config: Config, model: MortalityModel):
    environment = uniform_environment(1)
    durable = model.hazards(phenotypes(config, senescence_rate=0.0), environment, ages(1, 40))
    frail = model.hazards(phenotypes(config, senescence_rate=0.9), environment, ages(1, 40))
    assert durable.senescence == pytest.approx(0.0)
    assert frail.senescence > 0.0


# -- starvation -------------------------------------------------------------------------------


def test_starvation_is_deterministic_at_or_below_zero():
    assert np.array_equal(
        starved(np.array([1e-6, 0.0, -1.0])), np.array([False, True, True])
    )


def test_hazards_are_aligned_with_the_batch(config: Config, model: MortalityModel):
    hazards = model.hazards(phenotypes(config, 5), uniform_environment(5), ages(5))
    for name in ("background", "thermal", "radiation", "toxicity", "pressure", "senescence"):
        assert getattr(hazards, name).shape == (5,), name
