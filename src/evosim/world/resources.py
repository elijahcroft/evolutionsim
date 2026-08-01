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
    sink_target: NDArray[np.intp]

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
            sink_target=_sink_targets(grid, terrain, config),
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

        self._sink(resource.detritus_sink_fraction)

    def _sink(self, fraction: float) -> None:
        """Move a fraction of each water cell's detritus to its deepest water neighbour.

        Written as a subtraction and a ``bincount`` of the same quantities so the pool is
        conserved to the last float: sinking relocates dead biomass, it never creates or
        destroys any.  Cells with no deeper neighbour route to themselves and so move nothing,
        which is what makes the abyssal floors of the world accumulate rather than leak.
        """

        if fraction <= 0.0:
            return
        flat = self.detritus.reshape(-1)
        moving = self.sink_target != np.arange(flat.size)
        departing = np.where(moving, flat * fraction, 0.0)
        flat -= departing
        flat += np.bincount(self.sink_target, weights=departing, minlength=flat.size)

    @property
    def sinks_anywhere(self) -> bool:
        """True when at least one cell has a deeper neighbour to sink into."""
        return bool(np.any(self.sink_target != np.arange(self.sink_target.size)))

    def add_detritus(self, amount: FloatArray) -> None:
        if amount.shape != self.detritus.shape:
            raise ValueError(
                f"detritus shape {amount.shape} does not match world shape {self.detritus.shape}"
            )
        if np.any(amount < 0.0):
            raise ValueError("detritus additions must be non-negative")
        self.detritus += amount


def _sink_targets(
    grid: Grid,
    terrain: Terrain,
    config: PlanetConfig,
) -> NDArray[np.intp]:
    """Route each water cell to its deepest water neighbour, or to itself if it is a basin.

    Bathymetry does not change, so this is resolved once at world creation and the per-tick
    transfer becomes a single gather.  Land is excluded on both ends: detritus on a hillside
    does not slide into the sea in this model, and nothing sinks *out* of the ocean.
    """

    depth = np.maximum(config.terrain.sea_level - terrain.elevation_km, 0.0).reshape(-1)
    water = terrain.water.reshape(-1)
    neighbours = grid.neighbour_indices().reshape(grid.n_cells, 4)
    identity = np.arange(grid.n_cells)

    exists = neighbours >= 0
    lookup = np.where(exists, neighbours, 0)
    usable = exists & water[lookup] & water[:, None]
    # -inf for anything unusable, so argmax can never select it.
    candidate_depth = np.where(usable, depth[lookup], -np.inf)

    best = np.argmax(candidate_depth, axis=1)
    target = neighbours[identity, best]
    downhill = usable[identity, best] & (candidate_depth[identity, best] > depth)
    return np.where(downhill, target, identity).astype(np.intp)
