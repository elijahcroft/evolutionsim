"""Per-cell summaries of who is standing where.

Movement has to answer two questions that no single organism's own state can answer: "is there
food here?" and "is there something here that eats me?".  Both are properties of a cell's
occupants, and both are needed for five candidate cells per organism per step, so computing
them per organism would be quadratic in a crowded cell.

Reducing the whole population to one row per cell once per tick makes both a gather.  Every
field below is a plain :func:`numpy.bincount`, so the census costs one pass over the population
regardless of how many organisms end up consulting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.life.population import Population

FloatArray: TypeAlias = NDArray[np.float64]


def _mean_per_cell(
    cells: NDArray[np.intp],
    values: FloatArray,
    counts: FloatArray,
    n_cells: int,
) -> FloatArray:
    # An empty population makes bincount return an integer array, which cannot receive a
    # float division in place; asking for float64 explicitly keeps the extinction case working.
    total = np.asarray(
        np.bincount(cells, weights=values, minlength=n_cells), dtype=np.float64
    )
    return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0.0)


@dataclass(frozen=True, slots=True)
class CellCensus:
    """What an average occupant of each cell looks like.

    Means rather than distributions: an organism deciding where to walk is making a rough
    judgement, and carrying the full distribution of every neighbour's traits into a five-cell
    comparison would cost far more than the decision is worth.  The approximation is stated
    here rather than hidden, because it is the reason a cell holding one huge predator reads the
    same as one holding many small ones of equal total threat.
    """

    occupancy: FloatArray
    """Organisms per cell."""

    carcass_value: FloatArray
    """Mean energy an occupant's body plus reserve is worth to whoever eats it."""

    mass: FloatArray
    """Mean body mass of an occupant."""

    autotroph: FloatArray
    """Mean autotroph diet fraction, which decides who profits from eating here."""

    speed: FloatArray
    """Mean movement speed, for both escaping and chasing."""

    threat: FloatArray
    """Summed encounter capability of the occupants: how much hunting happens here."""

    hunter_mass: FloatArray
    hunter_sense: FloatArray
    hunter_aggression: FloatArray

    @classmethod
    def build(
        cls,
        population: Population,
        energy_density: float,
        k_encounter: float,
    ) -> CellCensus:
        """Reduce the living population to one row per cell."""

        active = population.active
        phenotype = population.phenotypes.active
        n_cells = population.n_cells
        cells = population.cell[active].astype(np.intp)

        counts = np.bincount(cells, minlength=n_cells).astype(np.float64)
        mass = phenotype.mass.astype(np.float64)
        carcass = energy_density * mass + np.maximum(
            population.energy[active].astype(np.float64), 0.0
        )
        aggression = phenotype.trait("aggression").astype(np.float64)
        speed = phenotype.trait("move_speed").astype(np.float64)
        sense = phenotype.trait("sense_range").astype(np.float64)

        # The same product that drives the encounter rate, summed rather than averaged: two
        # half-hearted hunters really are as dangerous as one committed one.
        threat = np.asarray(
            np.bincount(
                cells,
                weights=k_encounter * (0.5 + aggression) * (0.3 + speed) * (0.3 + sense) ** 2,
                minlength=n_cells,
            ),
            dtype=np.float64,
        )

        return cls(
            occupancy=counts,
            carcass_value=_mean_per_cell(cells, carcass, counts, n_cells),
            mass=_mean_per_cell(cells, mass, counts, n_cells),
            autotroph=_mean_per_cell(
                cells, phenotype.diet_component("autotroph").astype(np.float64), counts, n_cells
            ),
            speed=_mean_per_cell(cells, speed, counts, n_cells),
            threat=threat,
            hunter_mass=_mean_per_cell(cells, mass, counts, n_cells),
            hunter_sense=_mean_per_cell(cells, sense, counts, n_cells),
            hunter_aggression=_mean_per_cell(cells, aggression, counts, n_cells),
        )


__all__ = ["CellCensus"]
