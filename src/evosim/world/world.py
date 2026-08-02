"""Container and update boundary for all planetary fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evosim.config import PlanetConfig
from evosim.rng import RngBundle
from evosim.world.biome import classify
from evosim.world.climate import Climate
from evosim.world.grid import Grid
from evosim.world.resources import Resources
from evosim.world.terrain import Terrain, generate_terrain

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class World:
    """Complete mutable environment at one simulated day.

    ``insolation`` is what arrives at the surface; ``light`` is what reaches the sea floor an
    organism actually occupies, after the water column above it has absorbed the rest.  On land
    the two are equal.  Photosynthesis reads ``light`` and nothing else does, which is what
    makes the deep ocean a place where autotrophy is impossible rather than merely poor.
    """

    config: PlanetConfig
    grid: Grid
    terrain: Terrain
    climate: Climate
    resources: Resources
    toxicity: FloatArray
    depth_km: FloatArray
    transmittance: FloatArray
    light: FloatArray
    day: int = 0

    @classmethod
    def create(cls, config: PlanetConfig, rng: RngBundle) -> World:
        grid = Grid(config.grid_width, config.grid_height)
        terrain = generate_terrain(grid, config.terrain, rng.terrain)
        climate = Climate.initialize(grid, terrain, config)
        resources = Resources.initialize(grid, terrain, config)
        relative_highland = np.maximum(
            terrain.elevation_km - config.terrain.sea_level, 0.0
        )
        if config.terrain.max_elevation_km > 0.0:
            relative_highland /= config.terrain.max_elevation_km
        toxicity = config.base_toxicity + (
            config.toxicity_elevation_coupling * relative_highland
        )
        depth_km = np.maximum(config.terrain.sea_level - terrain.elevation_km, 0.0)
        # Transmittance is a property of the water column, so it is computed once rather than
        # every tick; only the insolation arriving at the surface changes with the season.
        transmittance = np.exp(-config.climate.light_attenuation_per_km * depth_km)
        world = cls(
            config,
            grid,
            terrain,
            climate,
            resources,
            toxicity,
            depth_km,
            transmittance,
            climate.insolation * transmittance,
        )
        return world

    def step(self, ticks: int = 1) -> None:
        if not isinstance(ticks, (int, np.integer)) or isinstance(ticks, bool):
            raise TypeError("ticks must be an integer")
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        for _ in range(int(ticks)):
            self.day += 1
            self.climate.step(self.grid, self.terrain, self.config, self.day)
            self.light = self.climate.insolation * self.transmittance
            self.resources.step(self.terrain, self.climate, self.config)

    def biomes(self) -> NDArray[np.int8]:
        """What kind of place each cell is today, as a code into ``BIOME_NAMES``.

        Derived on demand and never stored, for the reason given in ``biome.py``: a kept copy
        would be a second description of the planet, free to disagree with the first.  It moves
        with the season, because the fields it reads do.
        """

        return classify(self.terrain, self.transmittance, self.climate, self.config.biome)

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Named arrays for diagnostics and stable, headless output."""
        return {
            "elevation_km": self.terrain.elevation_km,
            "land": self.terrain.land,
            "depth_km": self.depth_km,
            "insolation": self.climate.insolation,
            "light": self.light,
            "temperature_c": self.climate.temperature_c,
            "moisture": self.climate.moisture,
            "nutrients": self.resources.nutrients,
            "detritus": self.resources.detritus,
            "toxicity": self.toxicity,
            "biome": self.biomes(),
        }
