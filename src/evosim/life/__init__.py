"""Vectorised biological state for genomes, phenotypes, and populations.

Milestone 2 establishes representation and deterministic founder seeding only.  Ecological
energy flow, movement, mortality, and reproduction are intentionally added by later layers.
"""

from evosim.life.genome import GenomeArray, GenomeSchema, TraitArray
from evosim.life.phenotype import (
    DIET_NAMES,
    PhenotypeBatch,
    PhenotypeBuffer,
    diet_softmax,
)
from evosim.life.population import (
    HabitatError,
    Population,
    PopulationCapacityError,
)

__all__ = [
    "DIET_NAMES",
    "GenomeArray",
    "GenomeSchema",
    "HabitatError",
    "PhenotypeBatch",
    "PhenotypeBuffer",
    "Population",
    "PopulationCapacityError",
    "TraitArray",
    "diet_softmax",
]
