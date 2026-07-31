"""Vectorised biological state and the ecology that acts on it.

Milestone 2 established representation and deterministic founder seeding; milestone 3 adds the
energy, movement, and mortality models that make organisms ecological.  Reproduction, selection,
and predation are intentionally added by later layers.
"""

from evosim.life.census import CellCensus
from evosim.life.energy import (
    Costs,
    EnergyModel,
    Environment,
    Intake,
    apply_resource_contention,
    saturation,
    thermal_excess,
)
from evosim.life.genome import GenomeArray, GenomeSchema, TraitArray
from evosim.life.mortality import Hazards, MortalityModel, saturating_hazard, starved
from evosim.life.movement import MovementModel
from evosim.life.phenotype import (
    DIET_NAMES,
    PhenotypeBatch,
    PhenotypeBuffer,
    diet_softmax,
)
from evosim.life.predation import HuntStats, PredationModel
from evosim.life.reproduction import BirthStats, ReproductionModel
from evosim.life.sampling import stochastic_round
from evosim.life.population import (
    HabitatError,
    Population,
    PopulationCapacityError,
)

__all__ = [
    "DIET_NAMES",
    "Costs",
    "EnergyModel",
    "Environment",
    "GenomeArray",
    "GenomeSchema",
    "HabitatError",
    "Hazards",
    "Intake",
    "BirthStats",
    "CellCensus",
    "HuntStats",
    "MortalityModel",
    "MovementModel",
    "PhenotypeBatch",
    "PhenotypeBuffer",
    "Population",
    "PopulationCapacityError",
    "PredationModel",
    "ReproductionModel",
    "TraitArray",
    "apply_resource_contention",
    "diet_softmax",
    "saturating_hazard",
    "saturation",
    "starved",
    "stochastic_round",
    "thermal_excess",
]
