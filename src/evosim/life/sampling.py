"""Small stochastic helpers shared by the layers that act on organisms.

These live apart from any one model because movement, reproduction, and predation all need
them, and routing them through whichever module happened to define one first would couple those
layers to each other for no reason.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def stochastic_round(values: FloatArray, rng: np.random.Generator) -> IntArray:
    """Round to whole units without bias.

    A speed of 0.2 cells per tick must mean one step every fifth tick, not zero steps forever
    and not one step every tick.  Rounding down would make every sub-unit value identical to
    zero and delete the entire lower half of such a locus from selection -- the same argument
    applies to a brood of 2.3 offspring and to a dispersal radius of 0.5 cells.
    """

    floor = np.floor(values)
    return (floor + (rng.random(values.shape) < (values - floor))).astype(np.int64)


__all__ = ["stochastic_round"]
