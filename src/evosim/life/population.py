"""Vectorised, capacity-bounded storage for living organisms.

The active population is always a dense prefix of every array.  Births append to that prefix
and deaths compact it with NumPy indexing, so later hot-path systems can operate on
``array[:population.size]`` without a Python loop or an indirection table.

This module deliberately stores body-derived state only through :mod:`evosim.life.phenotype`.
It never derives mass, storage, or geometry from a locus itself; that seam is what allows the
scalar body model to be replaced by real morphology after milestone 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config, GenomeConfig, SimConfig
from evosim.life.genome import GenomeSchema
from evosim.life.phenotype import PhenotypeBuffer
from evosim.rng import RngBundle
from evosim.world import World

Float32Array = NDArray[np.float32]
Int8Array = NDArray[np.int8]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]

UNASSIGNED = -1


class PopulationCapacityError(RuntimeError):
    """Raised when an append would exceed the configured population budget."""


class HabitatError(RuntimeError):
    """Raised when the configured founder habitat has no cells on the generated world."""


@dataclass(slots=True)
class Population:
    """Struct-of-arrays population with a dense active prefix.

    Arrays are allocated once to ``capacity``.  Only rows ``[:size]`` are alive; rows after
    that are spare slots.  Persistent organism IDs are independent of row numbers because
    compaction changes rows whenever organisms die.
    """

    schema: GenomeSchema
    capacity: int
    n_cells: int
    n_directions: int
    genomes: Float32Array
    phenotypes: PhenotypeBuffer
    cell: Int32Array
    energy: Float32Array
    age: Int64Array
    heading: Int8Array
    organism_id: Int64Array
    parent_id: Int64Array
    species_id: Int32Array
    generation: Int32Array
    size: int = 0
    _next_id: int = 0

    @classmethod
    def empty(
        cls,
        sim: SimConfig,
        genome: GenomeConfig,
        world: World,
    ) -> Population:
        """Allocate all population storage without creating an organism."""
        schema = GenomeSchema.from_config(genome)
        capacity = sim.max_population
        neighbour_count = world.grid.neighbour_indices().shape[-1]
        return cls(
            schema=schema,
            capacity=capacity,
            n_cells=world.grid.n_cells,
            n_directions=neighbour_count,
            genomes=np.zeros((capacity, schema.n_loci, 2), dtype=np.float32),
            phenotypes=PhenotypeBuffer.allocate(capacity, schema),
            cell=np.full(capacity, UNASSIGNED, dtype=np.int32),
            energy=np.zeros(capacity, dtype=np.float32),
            age=np.zeros(capacity, dtype=np.int64),
            heading=np.full(capacity, UNASSIGNED, dtype=np.int8),
            organism_id=np.full(capacity, UNASSIGNED, dtype=np.int64),
            parent_id=np.full((capacity, 2), UNASSIGNED, dtype=np.int64),
            species_id=np.full(capacity, UNASSIGNED, dtype=np.int32),
            generation=np.zeros(capacity, dtype=np.int32),
        )

    @classmethod
    def seed_founders(
        cls,
        config: Config,
        world: World,
        rng: RngBundle,
    ) -> Population:
        """Allocate a population and seed its configured founder lineage.

        Founder cells are sampled with replacement from the requested habitat, weighted by
        physical cell area.  Area weighting avoids reintroducing an equirectangular projection
        bias in which the tiny polar cells would be as likely as equatorial cells.

        Founders use the same starting-energy rule as future newborns: their expressed
        ``parental_investment`` fraction of adult storage capacity.  This introduces no extra
        founder-only tuning constant and follows the energy ledger documented in energy.yaml.
        """
        population = cls.empty(config.sim, config.genome, world)
        habitat = _habitat_mask(config.sim.initial_habitat, world)
        valid_cells = np.flatnonzero(habitat.ravel())
        if valid_cells.size == 0:
            raise HabitatError(
                f"cannot seed {config.sim.initial_population} founders: "
                f"world has no {config.sim.initial_habitat} habitat cells"
            )

        weights = world.grid.cell_area_weights.ravel()[valid_cells].astype(
            np.float64, copy=True
        )
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise HabitatError(
                f"cannot seed founders: {config.sim.initial_habitat} habitat has no "
                "positive-area cells"
            )
        weights /= weight_sum
        cells = rng.init.choice(
            valid_cells,
            size=config.sim.initial_population,
            replace=True,
            p=weights,
        ).astype(np.int32, copy=False)

        founders = population.schema.founders(config.sim.initial_population)
        population.add(founders, cells, energy=0.0)
        active = population.active
        investment = population.phenotypes.trait("parental_investment", active)
        initial_energy = investment * population.phenotypes.storage_capacity[active]
        population.energy[active] = initial_energy.astype(np.float32, copy=False)
        return population

    @property
    def active(self) -> slice:
        """Slice selecting every living row."""
        return slice(0, self.size)

    @property
    def available(self) -> int:
        return self.capacity - self.size

    @property
    def phenotype(self) -> PhenotypeBuffer:
        """Compatibility alias for the singular conceptual phenotype cache."""
        return self.phenotypes

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        genomes: NDArray[np.floating[Any]],
        cells: Any,
        energy: Any,
        *,
        age: Any = 0,
        heading: Any = UNASSIGNED,
        parent_id: Any = UNASSIGNED,
        species_id: Any = UNASSIGNED,
        generation: Any = 0,
    ) -> Int64Array:
        """Append a vectorised batch and return its persistent organism IDs.

        The operation validates the complete batch before changing ``size``.  Capacity
        overflow is always explicit so a later reproduction system can record a
        ``capacity_throttle`` event instead of silently losing births.
        """
        raw_batch = np.asarray(genomes)
        expected_tail = (self.schema.n_loci, 2)
        if raw_batch.ndim != 3 or raw_batch.shape[1:] != expected_tail:
            raise ValueError(
                f"genomes must have shape (count, {self.schema.n_loci}, 2), "
                f"got {raw_batch.shape}"
            )
        if (
            np.issubdtype(raw_batch.dtype, np.bool_)
            or np.issubdtype(raw_batch.dtype, np.complexfloating)
            or not np.issubdtype(raw_batch.dtype, np.number)
        ):
            raise TypeError("genomes must contain real numeric alleles")
        count = raw_batch.shape[0]
        if count > self.available:
            raise PopulationCapacityError(
                f"cannot add {count} organisms: {self.available} of "
                f"{self.capacity} population slots are available"
            )
        if not np.all(np.isfinite(raw_batch)):
            raise ValueError("genomes must contain only finite alleles")
        low = self.schema.low[None, :, None]
        high = self.schema.high[None, :, None]
        if np.any((raw_batch < low) | (raw_batch > high)):
            raise ValueError("genome allele lies outside its configured locus bounds")
        batch = raw_batch.astype(np.float32, copy=False)

        cell_values = _integer_vector("cells", cells, count, np.int32)
        if np.any((cell_values < 0) | (cell_values >= self.n_cells)):
            raise ValueError(f"cells must lie in [0, {self.n_cells})")
        energy_values = _float_vector("energy", energy, count, low=0.0)
        age_values = _integer_vector("age", age, count, np.int64)
        if np.any(age_values < 0):
            raise ValueError("age must be non-negative")
        heading_values = _integer_vector("heading", heading, count, np.int8)
        if np.any(
            (heading_values < UNASSIGNED) | (heading_values >= self.n_directions)
        ):
            raise ValueError(
                f"heading must be {UNASSIGNED} or lie in [0, {self.n_directions})"
            )
        parent_values = _parent_matrix(parent_id, count)
        if np.any(parent_values < UNASSIGNED):
            raise ValueError(f"parent IDs must be {UNASSIGNED} or non-negative")
        species_values = _integer_vector("species_id", species_id, count, np.int32)
        if np.any(species_values < UNASSIGNED):
            raise ValueError(f"species IDs must be {UNASSIGNED} or non-negative")
        generation_values = _integer_vector("generation", generation, count, np.int32)
        if np.any(generation_values < 0):
            raise ValueError("generation must be non-negative")

        if count == 0:
            return np.empty(0, dtype=np.int64)

        start = self.size
        stop = start + count
        target = slice(start, stop)
        ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)

        # Phenotype.update derives and validates all cached geometry before writing its
        # buffer.  Do it first so a derived overflow cannot leave partially written
        # population rows behind.
        self.phenotypes.update(start, batch, self.schema)
        self.genomes[target] = batch
        self.cell[target] = cell_values
        self.energy[target] = energy_values
        self.age[target] = age_values
        self.heading[target] = heading_values
        self.organism_id[target] = ids
        self.parent_id[target] = parent_values
        self.species_id[target] = species_values
        self.generation[target] = generation_values

        self.size = stop
        self._next_id += count
        return ids

    def remove(self, dead: NDArray[np.bool_]) -> Int64Array:
        """Remove a boolean-selected batch with stable, vectorised compaction."""
        mask = np.asarray(dead)
        if mask.dtype != np.bool_ or mask.shape != (self.size,):
            raise ValueError(
                f"dead mask must be boolean with shape ({self.size},), got "
                f"{mask.dtype} {mask.shape}"
            )
        if not np.any(mask):
            return np.empty(0, dtype=np.int64)

        old_size = self.size
        removed_ids = self.organism_id[:old_size][mask].copy()
        survivors = np.flatnonzero(~mask)
        new_size = survivors.size

        self.genomes[:new_size] = self.genomes[survivors]
        self.cell[:new_size] = self.cell[survivors]
        self.energy[:new_size] = self.energy[survivors]
        self.age[:new_size] = self.age[survivors]
        self.heading[:new_size] = self.heading[survivors]
        self.organism_id[:new_size] = self.organism_id[survivors]
        self.parent_id[:new_size] = self.parent_id[survivors]
        self.species_id[:new_size] = self.species_id[survivors]
        self.generation[:new_size] = self.generation[survivors]
        self.phenotypes.compact(survivors, old_size)

        cleared = slice(new_size, old_size)
        self.genomes[cleared] = 0.0
        self.cell[cleared] = UNASSIGNED
        self.energy[cleared] = 0.0
        self.age[cleared] = 0
        self.heading[cleared] = UNASSIGNED
        self.organism_id[cleared] = UNASSIGNED
        self.parent_id[cleared] = UNASSIGNED
        self.species_id[cleared] = UNASSIGNED
        self.generation[cleared] = 0
        self.size = int(new_size)
        return removed_ids

    def active_arrays(
        self,
        *,
        copy: bool = False,
    ) -> dict[str, NDArray[np.generic]]:
        """Return aligned active state for diagnostics (not a resume snapshot)."""
        active = self.active
        arrays: dict[str, NDArray[np.generic]] = {
            "organism_id": self.organism_id[active],
            "parent_id": self.parent_id[active],
            "species_id": self.species_id[active],
            "generation": self.generation[active],
            "cell": self.cell[active],
            "age": self.age[active],
            "heading": self.heading[active],
            "energy": self.energy[active],
            "genome": self.genomes[active],
            "traits": self.phenotypes.traits[active],
            "diet": self.phenotypes.diet[active],
            "mass": self.phenotypes.mass[active],
            "storage_capacity": self.phenotypes.storage_capacity[active],
        }
        if copy:
            return {name: value.copy() for name, value in arrays.items()}
        return arrays

    def organism_dict(self, row: int) -> dict[str, Any]:
        """Serialize one organism for a renderer/API on the cold path."""
        if not isinstance(row, (int, np.integer)) or isinstance(row, bool):
            raise TypeError("organism row must be an integer")
        row = int(row)
        if row < 0 or row >= self.size:
            raise IndexError(f"organism row {row} is outside active range [0, {self.size})")

        parents = self.parent_id[row]
        genome = {
            name: [float(self.genomes[row, locus, 0]), float(self.genomes[row, locus, 1])]
            for locus, name in enumerate(self.schema.names)
        }
        return {
            "id": int(self.organism_id[row]),
            "parents": [None if value < 0 else int(value) for value in parents],
            "species_id": (
                None if self.species_id[row] < 0 else int(self.species_id[row])
            ),
            "generation": int(self.generation[row]),
            "cell": int(self.cell[row]),
            "age": int(self.age[row]),
            "heading": None if self.heading[row] < 0 else int(self.heading[row]),
            "energy": float(self.energy[row]),
            "genome": genome,
            "phenotype": self.phenotypes.organism_dict(row, self.schema),
        }

    @property
    def memory_bytes(self) -> int:
        """Bytes reserved by population-owned NumPy arrays."""
        return (
            self.genomes.nbytes
            + self.cell.nbytes
            + self.energy.nbytes
            + self.age.nbytes
            + self.heading.nbytes
            + self.organism_id.nbytes
            + self.parent_id.nbytes
            + self.species_id.nbytes
            + self.generation.nbytes
            + self.phenotypes.memory_bytes
        )


def _habitat_mask(habitat: str, world: World) -> NDArray[np.bool_]:
    if habitat == "water":
        return world.terrain.water
    if habitat == "land":
        return world.terrain.land
    if habitat == "any":
        return np.ones(world.grid.shape, dtype=np.bool_)
    # SimConfig validates this, but keep this boundary safe for direct callers/tests.
    raise ValueError(f"unknown founder habitat {habitat!r}")


def _integer_vector(
    name: str,
    value: Any,
    count: int,
    dtype: type[np.signedinteger[Any]],
) -> NDArray[np.signedinteger[Any]]:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    try:
        broadcast = np.broadcast_to(raw, (count,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or have shape ({count},)") from exc
    info = np.iinfo(dtype)
    if np.any((broadcast < info.min) | (broadcast > info.max)):
        raise ValueError(f"{name} value does not fit in {np.dtype(dtype).name}")
    return broadcast.astype(dtype, copy=False)


def _float_vector(
    name: str,
    value: Any,
    count: int,
    *,
    low: float | None = None,
) -> Float32Array:
    raw = np.asarray(value)
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise TypeError(f"{name} must contain numbers")
    try:
        broadcast = np.broadcast_to(raw, (count,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or have shape ({count},)") from exc
    if not np.all(np.isfinite(broadcast)):
        raise ValueError(f"{name} must contain only finite values")
    if low is not None and np.any(broadcast < low):
        raise ValueError(f"{name} must be >= {low}")
    limit = float(np.finfo(np.float32).max)
    if np.any((broadcast < -limit) | (broadcast > limit)):
        raise ValueError(f"{name} value does not fit in float32")
    converted = broadcast.astype(np.float32, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} value does not fit in float32")
    return converted


def _parent_matrix(value: Any, count: int) -> Int64Array:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("parent_id must contain integers")
    if raw.ndim == 0:
        raw = np.broadcast_to(raw, (count, 2))
    elif raw.shape != (count, 2):
        raise ValueError(f"parent_id must be scalar or have shape ({count}, 2)")
    info = np.iinfo(np.int64)
    if np.any((raw < info.min) | (raw > info.max)):
        raise ValueError("parent_id value does not fit in int64")
    return raw.astype(np.int64, copy=False)
