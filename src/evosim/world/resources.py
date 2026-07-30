"""Renewable nutrient and detritus pools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from evosim.config import PlanetConfig
from evosim.world.climate import Climate
from evosim.world.grid import Grid
from evosim.world.terrain import Terrain

FloatArray = NDArray[np.float64]
_Q10_DEGREES_C = 10.0


@dataclass(slots=True)
class Resources:
    nutrients: FloatArray
    detritus: FloatArray
    nutrient_capacity: FloatArray

    @classmethod
    def initialize(
        cls,
        grid: Grid,
        terrain: Terrain,
        config: PlanetConfig,
    ) -> Resources:
        base_capacity = np.where(
            terrain.land,
            config.resources.nutrient_capacity_land,
            config.resources.nutrient_capacity_water,
        )
        capacity = base_capacity * grid.cell_area_weights
        return cls(
            nutrients=capacity.copy(),
            detritus=np.zeros(grid.shape, dtype=np.float64),
            nutrient_capacity=capacity,
        )

    def step(
        self,
        terrain: Terrain,
        climate: Climate,
        config: PlanetConfig,
    ) -> None:
        resource = config.resources
        moisture_scale = np.divide(
            climate.moisture,
            config.climate.moisture_ocean,
            out=np.zeros_like(climate.moisture),
            where=config.climate.moisture_ocean > 0.0,
        )
        max_ruggedness = terrain.ruggedness.max()
        if max_ruggedness > 0.0:
            upwelling = terrain.ruggedness / max_ruggedness
        else:
            upwelling = np.zeros_like(terrain.ruggedness)
        regeneration_rate = np.where(
            terrain.land,
            resource.nutrient_regen_land * moisture_scale,
            resource.nutrient_regen_water * (1.0 + resource.upwelling_bonus * upwelling),
        )
        regeneration_rate = np.clip(regeneration_rate, 0.0, 1.0)
        self.nutrients += regeneration_rate * (self.nutrient_capacity - self.nutrients)

        decay_rate = resource.detritus_decay_rate * np.power(
            resource.detritus_decay_q10,
            (climate.temperature_c - resource.detritus_reference_temp_c) / _Q10_DEGREES_C,
        )
        decayed = self.detritus * np.clip(decay_rate, 0.0, 1.0)
        self.detritus -= decayed
        self.nutrients += decayed

        np.clip(self.detritus, 0.0, None, out=self.detritus)
        np.clip(self.nutrients, 0.0, self.nutrient_capacity, out=self.nutrients)

    def add_detritus(self, amount: FloatArray) -> None:
        if amount.shape != self.detritus.shape:
            raise ValueError(
                f"detritus shape {amount.shape} does not match world shape {self.detritus.shape}"
            )
        if np.any(amount < 0.0):
            raise ValueError("detritus additions must be non-negative")
        self.detritus += amount
