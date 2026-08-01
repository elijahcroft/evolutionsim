"""What acts on lineages rather than on individuals.

The :mod:`evosim.life` layer is about organisms: what they cost, what they eat, what kills
them.  This layer is about the groups those organisms fall into over time.  It depends on
``life`` and on nothing above it.
"""

from evosim.evolution.taxonomy import MAX_ITERATIONS, Split, TaxonomyModel

__all__ = ["MAX_ITERATIONS", "Split", "TaxonomyModel"]
