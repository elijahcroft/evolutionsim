"""Procedural, deterministic surface elevation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evosim.config import TerrainConfig
from evosim.world.grid import Grid

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Terrain:
    elevation_km: FloatArray
    land: BoolArray
    ruggedness: FloatArray

    @property
    def water(self) -> BoolArray:
        return ~self.land


def generate_terrain(
    grid: Grid,
    config: TerrainConfig,
    rng: np.random.Generator,
) -> Terrain:
    """Generate layered periodic value noise and enforce the requested land fraction."""
    field = np.zeros(grid.shape, dtype=np.float64)
    amplitude = 1.0
    amplitude_total = 0.0
    frequency = config.base_frequency

    for _ in range(config.octaves):
        horizontal_points = max(2, int(round(frequency)))
        vertical_points = max(2, int(round(frequency * grid.height / grid.width)))
        coarse = rng.normal(size=(vertical_points, horizontal_points))
        field += amplitude * _resample_periodic_longitude(coarse, grid)
        amplitude_total += amplitude
        amplitude *= config.persistence
        frequency *= config.lacunarity

    if amplitude_total:
        field /= amplitude_total

    land = _highest_cells(field, config.land_fraction)
    elevation = _scale_around_sea_level(
        field, land, config.sea_level, config.max_elevation_km
    )
    ruggedness = _ruggedness(elevation)
    return Terrain(elevation_km=elevation, land=land, ruggedness=ruggedness)


def _resample_periodic_longitude(coarse: FloatArray, grid: Grid) -> FloatArray:
    """Bilinearly interpolate a coarse field, wrapping only the longitude dimension."""
    coarse_height, coarse_width = coarse.shape

    x = np.arange(grid.width, dtype=np.float64) * coarse_width / grid.width
    x0 = np.floor(x).astype(np.int64) % coarse_width
    x1 = (x0 + 1) % coarse_width
    tx = x - np.floor(x)

    if coarse_height == 1:
        y = np.zeros(grid.height, dtype=np.float64)
    else:
        y = np.linspace(0.0, coarse_height - 1, grid.height)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.minimum(y0 + 1, coarse_height - 1)
    ty = y - y0

    top = coarse[y0[:, None], x0[None, :]] * (1.0 - tx)[None, :]
    top += coarse[y0[:, None], x1[None, :]] * tx[None, :]
    bottom = coarse[y1[:, None], x0[None, :]] * (1.0 - tx)[None, :]
    bottom += coarse[y1[:, None], x1[None, :]] * tx[None, :]
    return top * (1.0 - ty)[:, None] + bottom * ty[:, None]


def _highest_cells(field: FloatArray, fraction: float) -> BoolArray:
    count = int(round(field.size * fraction))
    land = np.zeros(field.size, dtype=np.bool_)
    if count:
        # Stable sorting makes the result deterministic even in the unlikely event of ties.
        order = np.argsort(field, axis=None, kind="stable")
        land[order[-count:]] = True
    return land.reshape(field.shape)


def _scale_around_sea_level(
    field: FloatArray,
    land: BoolArray,
    sea_level: float,
    maximum_relief_km: float,
) -> FloatArray:
    """Map land above and water below the configured sea-level threshold."""
    elevation = np.full(field.shape, sea_level, dtype=np.float64)
    if maximum_relief_km == 0.0:
        return elevation

    if land.any():
        values = field[land]
        low = values.min()
        span = max(values.max() - low, np.finfo(np.float64).eps)
        scaled = np.maximum((values - low) / span, np.finfo(np.float64).eps)
        elevation[land] = sea_level + maximum_relief_km * scaled
    water = ~land
    if water.any():
        values = field[water]
        high = values.max()
        span = max(high - values.min(), np.finfo(np.float64).eps)
        elevation[water] = sea_level - maximum_relief_km * ((high - values) / span)
    return elevation


def _ruggedness(elevation: FloatArray) -> FloatArray:
    east = np.roll(elevation, -1, axis=1)
    west = np.roll(elevation, 1, axis=1)
    north = np.vstack((elevation[:1, :], elevation[:-1, :]))
    south = np.vstack((elevation[1:, :], elevation[-1:, :]))
    return np.maximum.reduce(
        (
            np.abs(elevation - east),
            np.abs(elevation - west),
            np.abs(elevation - north),
            np.abs(elevation - south),
        )
    )
