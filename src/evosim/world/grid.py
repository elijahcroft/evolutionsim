"""Geometry for the equirectangular surface grid.

Rows are ordered north to south and columns west to east.  Longitude wraps; latitude does not.
Keeping those details here makes a future grid replacement a world-layer change rather than a
rewrite of the biological simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Grid:
    """Immutable geometry for a cell-centred latitude/longitude grid."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("grid dimensions must both be at least 2")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def latitudes_deg(self) -> FloatArray:
        """Latitude of each row centre, from just below +90 to just above -90."""
        row = np.arange(self.height, dtype=np.float64)
        return 90.0 - (row + 0.5) * (180.0 / self.height)

    @property
    def longitudes_deg(self) -> FloatArray:
        """Longitude of each column centre in [-180, 180)."""
        column = np.arange(self.width, dtype=np.float64)
        return -180.0 + (column + 0.5) * (360.0 / self.width)

    @property
    def latitude_radians(self) -> FloatArray:
        return np.deg2rad(self.latitudes_deg)

    @property
    def cell_area_weights(self) -> FloatArray:
        """Relative cell areas, broadcast to the grid shape.

        An equirectangular cell's physical area is proportional to cos(latitude).  Equatorial
        cells therefore have weight near one while polar cells carry much less resource.
        """
        row_weights = np.cos(self.latitude_radians)
        return np.broadcast_to(row_weights[:, None], self.shape).copy()

    def flat_indices(self) -> IntArray:
        return np.arange(self.n_cells, dtype=np.int64).reshape(self.shape)

    def neighbour_indices(self) -> IntArray:
        """Flat indices of north, east, south, west neighbours.

        Missing north/south neighbours at the polar edges are ``-1``.  East/west neighbours
        always wrap across the longitude seam.
        """
        indices = self.flat_indices()
        neighbours = np.full((*self.shape, 4), -1, dtype=np.int64)
        neighbours[1:, :, 0] = indices[:-1, :]
        neighbours[:, :, 1] = np.roll(indices, -1, axis=1)
        neighbours[:-1, :, 2] = indices[1:, :]
        neighbours[:, :, 3] = np.roll(indices, 1, axis=1)
        return neighbours
