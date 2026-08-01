"""What the run remembers about itself.

Pure bookkeeping over the species the :mod:`evosim.evolution` layer identifies: the lineage
tree, extinction records, and the sampled time series.  Nothing here is read back by the
simulation, so a recorded history can never become a hidden selection pressure.
"""

from evosim.history.records import (
    FOUNDER_SPECIES,
    History,
    Sample,
    SpeciesRecord,
    SpeciesSample,
    trait_means,
)

__all__ = [
    "FOUNDER_SPECIES",
    "History",
    "Sample",
    "SpeciesRecord",
    "SpeciesSample",
    "trait_means",
]
