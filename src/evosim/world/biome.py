"""What kind of place each cell is.

A biome here is a *reading* of the world, not a layer of it.  Nothing is generated, stored or
stepped: the classification is a pure function of fields that already exist -- the land mask,
the light that reaches the floor, the temperature and the surface moisture -- and it is computed
on demand every time somebody asks.  That is deliberate.  A stored biome would be a second
description of the planet capable of disagreeing with the first, and an organism could then be
sitting in "desert" while the moisture it is actually charged for says otherwise.

Two consequences worth stating plainly:

1. **It moves.**  ``temperature_c`` relaxes toward its equilibrium every tick and ``moisture``
   and ``light`` are recomputed from the day's insolation, so a high-latitude cell genuinely
   passes from ``polar ice`` to ``cold shelf`` and back over a year.  That is the honest answer
   on a planet with an axial tilt, and it is what makes a *sampled distribution* of occupancy
   the right thing to record rather than a single static lookup.

2. **The deep ocean is cut on the water column, not on the day's sun.**  Photosynthesis reads
   ``light = insolation x transmittance``; a place is dark because of the second factor and a
   *day* is dark because of the first.  Classifying on ``light`` would have made every polar
   cell "deep ocean" for half the year -- the first version did, and a test caught it -- so the
   cut is on ``transmittance``, which is the same attenuation physics with the season taken out.
   It still means "where autotrophy is impossible", rather than a second definition in
   kilometres free to drift away from the one the energy model uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to the type checker
    from evosim.config import BiomeConfig
    from evosim.world.climate import Climate
    from evosim.world.terrain import Terrain

#: Biome name by code.  The index *is* the code, and the order is the order the UI legend and
#: the habitat line read in, so entries are appended rather than reordered.
BIOME_NAMES: tuple[str, ...] = (
    "deep ocean",
    "polar ice",
    "cold shelf",
    "warm shelf",
    "ice cap",
    "tundra",
    "desert",
    "grassland",
    "forest",
    "tropical forest",
)


def classify(
    terrain: Terrain,
    transmittance: NDArray[np.float64],
    climate: Climate,
    config: BiomeConfig,
) -> NDArray[np.int8]:
    """Which biome each cell is today, as a code into :data:`BIOME_NAMES`.

    One ``np.select`` over the whole grid: the loop is over the ten classes and never over the
    cells, which is the rule the tick loop is held to as well.  Conditions are evaluated in
    order and the first match wins, so each one only has to say what distinguishes it from the
    classes *below* it -- water is split by darkness, then by ice, then by temperature; land by
    ice, then by cold, then by how wet and finally how warm it is.

    Darkness is tested first because it is the one property of a water cell that does not move:
    the column above it is as deep in winter as in summer.
    """

    land = terrain.land
    water = ~land
    temperature = climate.temperature_c
    moisture = climate.moisture

    frozen = temperature <= config.freeze_c
    cold = temperature <= config.temperate_c

    conditions = [
        water & (transmittance < config.photic_transmittance),   # deep ocean
        water & frozen,                               # polar ice
        water & cold,                                 # cold shelf
        water,                                        # warm shelf
        land & frozen,                                # ice cap
        land & cold,                                  # tundra
        land & (moisture < config.arid_moisture),     # desert
        land & (moisture < config.humid_moisture),    # grassland
        land & (temperature <= config.tropical_c),    # forest
    ]
    # The final class is the default rather than a tenth condition, so every cell is guaranteed
    # a code and there is no "unclassified" value for a caller to have to handle.
    tropical_forest = len(BIOME_NAMES) - 1
    codes = np.select(
        conditions,
        list(range(len(conditions))),
        default=tropical_forest,
    )
    return codes.astype(np.int8)


__all__ = ["BIOME_NAMES", "classify"]
