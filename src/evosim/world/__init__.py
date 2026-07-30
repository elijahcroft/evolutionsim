"""Deterministic planetary environment fields.

The world package owns all grid geometry and environmental state.  Later simulation layers
consume its arrays, but must not reproduce assumptions about the equirectangular grid.
"""

from evosim.world.world import World

__all__ = ["World"]
