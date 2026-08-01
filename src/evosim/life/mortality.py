"""Environmental, age-related, and starvation mortality.

Together with :mod:`evosim.life.energy` this is the complete list of ways a trait can be
disadvantageous.  Each environmental hazard is an independent per-tick probability, so they
combine as ``1 - prod(1 - h)`` rather than by addition: three hazards of 0.5 leave a real
survival chance, and no combination can exceed certainty.

Every hazard shares one saturating shape, ``h_max * x / (x + 1)`` on a scaled mismatch.  The
consequence is deliberate: a mild excursion is survivable and a large one is merely very
likely to kill, so a population pushed past its tolerance bleeds rather than vanishing at a
threshold.  A cliff would make evolution look like a step function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config, MortalityConfig, PlanetConfig
from evosim.life.energy import Environment, thermal_excess
from evosim.life.phenotype import PhenotypeBatch

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def saturating_hazard(maximum: float, mismatch: FloatArray) -> FloatArray:
    """Probability rising from zero toward ``maximum`` as ``mismatch`` grows past one."""

    return maximum * mismatch / (mismatch + 1.0)


@dataclass(frozen=True, slots=True)
class Hazards:
    """The per-tick death probability from each named cause, kept separate.

    A run is only explainable if "the lineage died" can be resolved into which hazard did it,
    so these are reported rather than pre-summed.
    """

    background: FloatArray
    thermal: FloatArray
    radiation: FloatArray
    toxicity: FloatArray
    pressure: FloatArray
    senescence: FloatArray

    @property
    def combined(self) -> FloatArray:
        """Probability of dying from at least one independent hazard this tick."""

        survival = (
            (1.0 - self.background)
            * (1.0 - self.thermal)
            * (1.0 - self.radiation)
            * (1.0 - self.toxicity)
            * (1.0 - self.pressure)
            * (1.0 - self.senescence)
        )
        return np.clip(1.0 - survival, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class MortalityModel:
    """Config-bound evaluator for the hazard equations in ``energy.yaml``."""

    planet: PlanetConfig
    mortality: MortalityConfig

    @classmethod
    def from_config(cls, config: Config) -> MortalityModel:
        return cls(planet=config.planet, mortality=config.energy.mortality)

    def hazards(
        self,
        phenotype: PhenotypeBatch,
        environment: Environment,
        age: NDArray[np.integer],
    ) -> Hazards:
        """Evaluate every hazard for one aligned batch of organisms."""

        config = self.mortality
        count = phenotype.size

        excess = thermal_excess(
            environment.temperature_c,
            phenotype.trait("temp_optimum").astype(np.float64),
            phenotype.trait("temp_tolerance").astype(np.float64),
        )
        thermal = saturating_hazard(
            config.h_thermal_max,
            excess / config.thermal_lethal_margin_c,
        )

        # Both radiation and pressure tolerance work by division, so tolerance has diminishing
        # returns: the first unit buys far more survival than the fifth. That curvature is what
        # stops a hostile planet from simply selecting all tolerances to their upper bound.
        radiation = saturating_hazard(
            config.h_radiation_max,
            config.radiation_scale
            * self.planet.radiation
            / (1.0 + phenotype.trait("radiation_tolerance").astype(np.float64)),
        )
        toxicity = saturating_hazard(
            config.h_toxicity_max,
            config.toxicity_scale * environment.toxicity,
        )
        # Pressure is the water column overhead, not a planetary constant: without depth the
        # mismatch term is identically zero on a 1 atm world and the locus cannot be selected
        # on at all. With it, the deep is a place that costs something specific to occupy.
        local_pressure = (
            self.planet.pressure
            + self.planet.pressure_per_km_depth * environment.depth_km
        )
        pressure = saturating_hazard(
            config.h_pressure_max,
            config.pressure_mismatch_scale
            * np.abs(local_pressure - 1.0)
            / (1.0 + phenotype.trait("pressure_tolerance").astype(np.float64)),
        )

        # Senescence is expressed in units of the organism's own maturity age, so a slow life
        # history postpones ageing rather than merely surviving it. The cubic exponent makes
        # the onset abrupt enough that maturity_age is under real selection.
        maturity = np.maximum(phenotype.trait("maturity_age").astype(np.float64), 1.0)
        relative_age = np.asarray(age, dtype=np.float64) / maturity
        senescence = np.clip(
            phenotype.trait("senescence_rate").astype(np.float64)
            * relative_age**config.senescence_exponent,
            0.0,
            1.0,
        )

        return Hazards(
            background=np.full(count, config.background, dtype=np.float64),
            thermal=np.asarray(thermal, dtype=np.float64),
            radiation=np.broadcast_to(
                np.asarray(radiation, dtype=np.float64), (count,)
            ).copy(),
            toxicity=np.asarray(toxicity, dtype=np.float64),
            pressure=np.broadcast_to(
                np.asarray(pressure, dtype=np.float64), (count,)
            ).copy(),
            senescence=np.asarray(senescence, dtype=np.float64),
        )


def starved(energy: NDArray[np.floating]) -> BoolArray:
    """Organisms whose reserves have run out.

    Starvation is deterministic rather than probabilistic: an empty ledger is not a risk, it is
    an outcome, and making it a hazard roll would let organisms survive on negative energy.
    """

    return np.asarray(energy) <= 0.0


__all__ = [
    "Hazards",
    "MortalityModel",
    "saturating_hazard",
    "starved",
]
