"""Seasonal insolation, temperature, and surface moisture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evosim.config import PlanetConfig
from evosim.world.grid import Grid
from evosim.world.terrain import Terrain

FloatArray = NDArray[np.float64]
_EARTH_DAY_HOURS = 24.0


@dataclass(slots=True)
class Climate:
    insolation: FloatArray
    temperature_c: FloatArray
    moisture: FloatArray
    _distance_from_water: FloatArray

    @classmethod
    def initialize(
        cls,
        grid: Grid,
        terrain: Terrain,
        config: PlanetConfig,
        day: int = 0,
    ) -> Climate:
        insolation = daily_insolation(grid, config, day)
        equilibrium = equilibrium_temperature(terrain, insolation, config)
        distance = distance_from_water(terrain.water)
        moisture = surface_moisture(terrain, distance, insolation, config)
        return cls(insolation, equilibrium, moisture, distance)

    def step(
        self,
        grid: Grid,
        terrain: Terrain,
        config: PlanetConfig,
        day: int,
    ) -> None:
        self.insolation = daily_insolation(grid, config, day)
        target = equilibrium_temperature(terrain, self.insolation, config)
        inertia = np.where(
            terrain.land, config.climate.inertia_land, config.climate.inertia_water
        )
        self.temperature_c += inertia * (target - self.temperature_c)
        self.moisture = surface_moisture(
            terrain, self._distance_from_water, self.insolation, config
        )


def daily_insolation(grid: Grid, config: PlanetConfig, day: int) -> FloatArray:
    """Daily-mean top-of-atmosphere insolation with polar day/night.

    The expression is the standard sunrise-hour-angle integral.  It is normalised so an
    equatorial equinox receives one unit before the configured solar and day-length scales.
    """
    phase = 2.0 * np.pi * (day % config.year_length_days) / config.year_length_days
    declination = np.deg2rad(config.axial_tilt) * np.sin(phase)
    latitude = grid.latitude_radians

    argument = -np.tan(latitude) * np.tan(declination)
    sunset_angle = np.arccos(np.clip(argument, -1.0, 1.0))
    sunset_angle = np.where(argument <= -1.0, np.pi, sunset_angle)
    sunset_angle = np.where(argument >= 1.0, 0.0, sunset_angle)

    daily_mean = (
        sunset_angle * np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.sin(sunset_angle)
    )
    equatorial_equinox_mean = 1.0
    relative = np.maximum(daily_mean / equatorial_equinox_mean, 0.0)
    relative *= config.solar_constant * config.day_length_hours / _EARTH_DAY_HOURS
    return np.broadcast_to(relative[:, None], grid.shape).copy()


def equilibrium_temperature(
    terrain: Terrain,
    insolation: FloatArray,
    config: PlanetConfig,
) -> FloatArray:
    """Surface equilibrium, cooled upward by elevation and downward by depth.

    The two corrections are deliberately symmetric.  Elevation cools linearly through a lapse
    rate; depth cools asymptotically toward ``deep_temperature_c``, because a water column does
    not keep getting colder without limit -- below the thermocline it is simply cold.  Land has
    zero depth and deep water has zero elevation, so each term only ever touches its own medium.
    """

    climate = config.climate
    above_sea_level = np.maximum(terrain.elevation_km - config.terrain.sea_level, 0.0)
    surface = (
        climate.base_temperature_c
        + climate.insolation_amplitude_c * insolation
        - climate.lapse_rate_c_per_km * above_sea_level
    )
    depth_km = np.maximum(config.terrain.sea_level - terrain.elevation_km, 0.0)
    descent = 1.0 - np.exp(-depth_km / climate.thermocline_scale_km)
    return surface - (surface - climate.deep_temperature_c) * descent


def distance_from_water(water: NDArray[np.bool_]) -> FloatArray:
    """Manhattan distance to ocean, with longitude wrapping and polar edges closed."""
    if water.all():
        return np.zeros(water.shape, dtype=np.float64)
    if not water.any():
        return np.full(water.shape, max(water.shape), dtype=np.float64)

    distance = np.where(water, 0.0, np.inf)
    for _ in range(water.size):
        previous = distance
        north = np.vstack((distance[:1, :], distance[:-1, :]))
        south = np.vstack((distance[1:, :], distance[-1:, :]))
        candidate = np.minimum.reduce(
            (
                distance,
                np.roll(distance, 1, axis=1) + 1.0,
                np.roll(distance, -1, axis=1) + 1.0,
                north + 1.0,
                south + 1.0,
            )
        )
        distance = candidate
        if np.array_equal(distance, previous):
            break
    return distance


def surface_moisture(
    terrain: Terrain,
    distance: FloatArray,
    insolation: FloatArray,
    config: PlanetConfig,
) -> FloatArray:
    climate = config.climate
    proximity = np.exp(-climate.moisture_decay_per_cell * distance)
    solar_scale = config.solar_constant * config.day_length_hours / _EARTH_DAY_HOURS
    if solar_scale > 0.0:
        evaporation = np.clip(insolation / solar_scale, 0.0, 1.0)
    else:
        evaporation = np.zeros(insolation.shape, dtype=np.float64)
    land_moisture = climate.moisture_min + (
        climate.moisture_ocean - climate.moisture_min
    ) * proximity * evaporation
    return np.where(terrain.land, land_moisture, climate.moisture_ocean)
