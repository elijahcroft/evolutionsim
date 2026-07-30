"""Container and update boundary for all planetary fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evosim.config import PlanetConfig
from evosim.rng import RngBundle
from evosim.world.climate import Climate
from evosim.world.grid import Grid
from evosim.world.resources import Resources
from evosim.world.terrain import Terrain, generate_terrain

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class World:
    """Complete mutable environment at one simulated day."""

    config: PlanetConfig
    grid: Grid
    terrain: Terrain
    climate: Climate
    resources: Resources
    toxicity: FloatArray
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
        return cls(config, grid, terrain, climate, resources, toxicity)

    def step(self, ticks: int = 1) -> None:
        if not isinstance(ticks, (int, np.integer)) or isinstance(ticks, bool):
            raise TypeError("ticks must be an integer")
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        for _ in range(int(ticks)):
            self.day += 1
            self.climate.step(self.grid, self.terrain, self.config, self.day)
            self.resources.step(self.terrain, self.climate, self.config)

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """Named arrays for diagnostics and stable, headless output."""
        return {
            "elevation_km": self.terrain.elevation_km,
            "land": self.terrain.land,
            "insolation": self.climate.insolation,
            "temperature_c": self.climate.temperature_c,
            "moisture": self.climate.moisture,
            "nutrients": self.resources.nutrients,
            "detritus": self.resources.detritus,
            "toxicity": self.toxicity,
        }
