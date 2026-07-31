"""Vectorised expression and scalar-body geometry.

This module is the sole boundary between expressed scalar loci and body-derived
quantities.  Milestone 2 deliberately uses the following simple interpretation:

``mass = body_size ** 3``

``adult storage_capacity = mass * energy_storage / body_slenderness``

Keeping both derivations here lets a later morphology model replace the scalar
body without teaching population storage or the energy tick about its geometry.
The continuously expressed ``offspring_count`` trait is not rounded here;
stochastic rounding belongs to an individual reproduction event.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float32]

DIET_NAMES: tuple[str, ...] = (
    "autotroph",
    "detritus",
    "herbivore",
    "carnivore",
)
_DIET_LOCUS_NAMES: tuple[str, ...] = tuple(f"aff_{name}" for name in DIET_NAMES)
_BODY_SIZE = "body_size"
_ENERGY_STORAGE = "energy_storage"
_BODY_SLENDERNESS = "body_slenderness"


class GenomeSchemaLike(Protocol):
    """The small genome interface needed by phenotype expression."""

    names: tuple[str, ...]
    n_loci: int

    def index_of(self, name: str) -> int: ...

    def express(self, genomes: ArrayLike) -> FloatArray: ...


def diet_softmax(logits: ArrayLike) -> FloatArray:
    """Return stable four-way diet fractions as ``float32``.

    Subtracting the row maximum before exponentiation prevents overflow even for
    extreme finite logits.  At least one shifted value is zero, so every row has
    a positive, finite denominator.
    """

    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(DIET_NAMES):
        raise ValueError(
            f"diet logits must have shape (batch, {len(DIET_NAMES)}); "
            f"got {values.shape}"
        )
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        raise TypeError(f"diet logits must contain real numbers; got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError("diet logits must contain only finite values")

    with np.errstate(over="ignore", under="ignore"):
        working = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(working)):
        raise ValueError("diet logits must be representable as finite float64 values")
    # A finite minimum subtracted from a finite maximum can intentionally overflow to
    # negative infinity. exp(-inf) is exactly the desired zero-probability limit.
    with np.errstate(over="ignore", under="ignore"):
        shifted = working - np.max(working, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
    fractions = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
    return fractions.astype(np.float32, copy=False)


@dataclass(slots=True)
class PhenotypeBatch:
    """Aligned views of a contiguous batch of expressed organisms.

    Arrays may be live views into a :class:`PhenotypeBuffer`; mutating them
    therefore mutates that buffer.  This is intentional for hot-path consumers
    that need views rather than per-tick copies.
    """

    traits: FloatArray
    diet: FloatArray
    mass: FloatArray
    storage_capacity: FloatArray
    trait_names: tuple[str, ...]
    _indices: dict[str, int]

    @property
    def size(self) -> int:
        return int(self.traits.shape[0])

    def __len__(self) -> int:
        return self.size

    def trait(self, name: str) -> FloatArray:
        """Return the batch column for a named expressed trait."""

        try:
            index = self._indices[name]
        except KeyError:
            raise KeyError(f"no such expressed trait: {name!r}") from None
        return self.traits[:, index]

    def diet_component(self, name: str) -> FloatArray:
        """Return one named diet-fraction column."""

        try:
            index = DIET_NAMES.index(name)
        except ValueError:
            raise KeyError(f"no such diet component: {name!r}") from None
        return self.diet[:, index]

    def organism_dict(self, row: int) -> dict[str, Any]:
        """Serialize one batch row for renderer/API use on the cold path."""

        row = _row_index(row, self.size)
        # These small comprehensions are deliberately confined to serialization,
        # never used by the population hot path.
        traits = {
            name: float(self.traits[row, index])
            for name, index in self._indices.items()
        }
        diet = {
            name: float(self.diet[row, index])
            for index, name in enumerate(DIET_NAMES)
        }
        return {
            "traits": traits,
            "diet": diet,
            "body": {
                "mass": float(self.mass[row]),
                "storage_capacity": float(self.storage_capacity[row]),
            },
        }


@dataclass(slots=True)
class PhenotypeBuffer:
    """Preallocated phenotype cache with a dense active prefix."""

    capacity: int
    trait_names: tuple[str, ...]
    traits: FloatArray
    diet: FloatArray
    mass: FloatArray
    storage_capacity: FloatArray
    size: int
    _indices: dict[str, int]
    _schema: GenomeSchemaLike
    _diet_indices: NDArray[np.intp]
    _body_size_index: int
    _energy_storage_index: int
    _body_slenderness_index: int

    @classmethod
    def allocate(
        cls,
        capacity: int,
        schema: GenomeSchemaLike,
    ) -> PhenotypeBuffer:
        """Allocate zero-filled storage for ``capacity`` organisms."""

        capacity = _non_negative_integer("capacity", capacity)
        names, indices = _schema_metadata(schema)
        diet_indices = np.asarray(
            [indices[name] for name in _DIET_LOCUS_NAMES],
            dtype=np.intp,
        )
        return cls(
            capacity=capacity,
            trait_names=names,
            traits=np.zeros((capacity, len(names)), dtype=np.float32),
            diet=np.zeros((capacity, len(DIET_NAMES)), dtype=np.float32),
            mass=np.zeros(capacity, dtype=np.float32),
            storage_capacity=np.zeros(capacity, dtype=np.float32),
            size=0,
            _indices=indices,
            _schema=schema,
            _diet_indices=diet_indices,
            _body_size_index=indices[_BODY_SIZE],
            _energy_storage_index=indices[_ENERGY_STORAGE],
            _body_slenderness_index=indices[_BODY_SLENDERNESS],
        )

    @classmethod
    def from_genomes(
        cls,
        genomes: ArrayLike,
        schema: GenomeSchemaLike,
        *,
        capacity: int | None = None,
    ) -> PhenotypeBuffer:
        """Allocate a buffer and express an initial genome batch into it."""

        batch = np.asarray(genomes)
        if batch.ndim != 3:
            raise ValueError(
                "genomes must have shape (batch, loci, 2); "
                f"got {batch.shape}"
            )
        requested_capacity = batch.shape[0] if capacity is None else capacity
        buffer = cls.allocate(requested_capacity, schema)
        buffer.update(0, batch, schema)
        return buffer

    @property
    def active(self) -> PhenotypeBatch:
        """Return live views over the dense active prefix."""

        return self.batch(0, self.size)

    @property
    def memory_bytes(self) -> int:
        """Bytes reserved by phenotype-owned NumPy arrays."""

        return (
            self.traits.nbytes
            + self.diet.nbytes
            + self.mass.nbytes
            + self.storage_capacity.nbytes
        )

    def __len__(self) -> int:
        return self.size

    def update(
        self,
        start: int,
        genomes: ArrayLike,
        schema: GenomeSchemaLike,
    ) -> PhenotypeBatch:
        """Express ``genomes`` into a contiguous buffer range.

        Existing rows may be overwritten.  New rows must extend the active
        prefix contiguously; gaps are rejected.  The complete result is
        validated before any buffer array is changed.
        """

        start = _non_negative_integer("start", start)
        if start > self.size:
            raise ValueError(
                f"update would leave an inactive gap: start {start} exceeds size {self.size}"
            )
        self._require_compatible_schema(schema)
        batch = np.asarray(genomes)
        expected_tail = (len(self.trait_names), 2)
        if batch.ndim != 3 or batch.shape[1:] != expected_tail:
            raise ValueError(
                f"genomes must have shape (batch, {len(self.trait_names)}, 2); "
                f"got {batch.shape}"
            )

        count = int(batch.shape[0])
        stop = start + count
        if stop > self.capacity:
            raise ValueError(
                f"phenotype update [{start}:{stop}] exceeds capacity {self.capacity}"
            )

        expressed = np.asarray(schema.express(batch), dtype=np.float32)
        if expressed.shape != (count, len(self.trait_names)):
            raise ValueError(
                "schema.express returned shape "
                f"{expressed.shape}; expected {(count, len(self.trait_names))}"
            )
        if not np.all(np.isfinite(expressed)):
            raise ValueError("expressed traits must contain only finite values")

        body_size = expressed[:, self._body_size_index]
        energy_storage = expressed[:, self._energy_storage_index]
        body_slenderness = expressed[:, self._body_slenderness_index]
        if np.any(body_size < 0.0):
            raise ValueError("expressed body_size must be non-negative")
        if np.any(energy_storage < 0.0):
            raise ValueError("expressed energy_storage must be non-negative")
        if np.any(body_slenderness <= 0.0):
            raise ValueError("expressed body_slenderness must be positive")

        diet = diet_softmax(expressed[:, self._diet_indices])
        mass = (body_size * body_size * body_size).astype(np.float32, copy=False)
        storage_capacity = (
            mass * energy_storage / body_slenderness
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(storage_capacity)):
            raise ValueError("derived body geometry must contain only finite values")

        target = slice(start, stop)
        self.traits[target] = expressed
        self.diet[target] = diet
        self.mass[target] = mass
        self.storage_capacity[target] = storage_capacity
        self.size = max(self.size, stop)
        return self.batch(start, stop)

    def compact(
        self,
        survivor_indices: ArrayLike,
        old_size: int,
    ) -> PhenotypeBatch:
        """Stably compact selected old rows into a new dense prefix."""

        old_size = _non_negative_integer("old_size", old_size)
        if old_size != self.size:
            raise ValueError(
                f"old_size {old_size} does not match phenotype size {self.size}"
            )
        survivors = np.asarray(survivor_indices)
        if survivors.ndim != 1 or not np.issubdtype(survivors.dtype, np.integer):
            raise TypeError("survivor_indices must be a one-dimensional integer array")
        if np.issubdtype(survivors.dtype, np.bool_):
            raise TypeError("survivor_indices must contain integer row indices, not booleans")
        if np.any((survivors < 0) | (survivors >= old_size)):
            raise IndexError(f"survivor index is outside old active range [0, {old_size})")
        if survivors.size and np.unique(survivors).size != survivors.size:
            raise ValueError("survivor_indices must not contain duplicates")

        new_size = int(survivors.size)
        # Advanced indexing materialises the right-hand side, so overlapping
        # source and destination rows remain safe during in-place compaction.
        self.traits[:new_size] = self.traits[survivors]
        self.diet[:new_size] = self.diet[survivors]
        self.mass[:new_size] = self.mass[survivors]
        self.storage_capacity[:new_size] = self.storage_capacity[survivors]

        cleared = slice(new_size, old_size)
        self.traits[cleared] = 0.0
        self.diet[cleared] = 0.0
        self.mass[cleared] = 0.0
        self.storage_capacity[cleared] = 0.0
        self.size = new_size
        return self.active

    def batch(self, start: int = 0, stop: int | None = None) -> PhenotypeBatch:
        """Return live views over a contiguous part of the active prefix."""

        start = _non_negative_integer("start", start)
        stop = self.size if stop is None else _non_negative_integer("stop", stop)
        if start > stop:
            raise ValueError(f"batch start {start} exceeds stop {stop}")
        if stop > self.size:
            raise IndexError(
                f"batch stop {stop} exceeds active phenotype size {self.size}"
            )
        rows = slice(start, stop)
        return PhenotypeBatch(
            traits=self.traits[rows],
            diet=self.diet[rows],
            mass=self.mass[rows],
            storage_capacity=self.storage_capacity[rows],
            trait_names=self.trait_names,
            _indices=self._indices,
        )

    def trait(
        self,
        name: str,
        rows: slice | NDArray[np.integer[Any]] | None = None,
    ) -> FloatArray:
        """Return a named trait view/selection from the buffer."""

        try:
            index = self._indices[name]
        except KeyError:
            raise KeyError(f"no such expressed trait: {name!r}") from None
        selection: Any = slice(0, self.size) if rows is None else rows
        return self.traits[selection, index]

    def diet_component(
        self,
        name: str,
        rows: slice | NDArray[np.integer[Any]] | None = None,
    ) -> FloatArray:
        """Return a named diet-fraction view/selection from the buffer."""

        try:
            index = DIET_NAMES.index(name)
        except ValueError:
            raise KeyError(f"no such diet component: {name!r}") from None
        selection: Any = slice(0, self.size) if rows is None else rows
        return self.diet[selection, index]

    def organism_dict(
        self,
        row: int,
        schema: GenomeSchemaLike | None = None,
    ) -> dict[str, Any]:
        """Serialize one active row for renderer/API use on the cold path."""

        if schema is not None:
            self._require_compatible_schema(schema)
        row = _row_index(row, self.size)
        return self.batch(row, row + 1).organism_dict(0)

    def _require_compatible_schema(self, schema: GenomeSchemaLike) -> None:
        if schema is not self._schema:
            raise ValueError(
                "genome schema does not match this phenotype buffer; use the exact schema "
                "the buffer was allocated with"
            )


def _schema_metadata(
    schema: GenomeSchemaLike,
) -> tuple[tuple[str, ...], dict[str, int]]:
    try:
        n_loci = schema.n_loci
        raw_names = schema.names
        index_of = schema.index_of
        express = schema.express
    except AttributeError as exc:
        raise TypeError(
            "schema must expose names, n_loci, index_of(name), and express(genomes)"
        ) from exc
    if not callable(index_of) or not callable(express):
        raise TypeError("schema index_of and express attributes must be callable")
    if isinstance(n_loci, (bool, np.bool_)) or not isinstance(
        n_loci, (int, np.integer)
    ):
        raise TypeError("schema.n_loci must be an integer")

    names = tuple(raw_names)
    if len(names) != int(n_loci):
        raise ValueError(
            f"schema exposes {len(names)} names for n_loci={int(n_loci)}"
        )
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("schema locus names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("schema locus names must be unique")

    indices = {name: int(index_of(name)) for name in names}
    if set(indices.values()) != set(range(len(names))):
        raise ValueError("schema.index_of must map names onto every locus column exactly once")
    missing = [
        name
        for name in (*_DIET_LOCUS_NAMES, _BODY_SIZE, _ENERGY_STORAGE, _BODY_SLENDERNESS)
        if name not in indices
    ]
    if missing:
        raise ValueError(
            "phenotype schema is missing required locus name(s): "
            + ", ".join(missing)
        )
    return names, indices


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _row_index(row: int, size: int) -> int:
    row = _non_negative_integer("row", row)
    if row >= size:
        raise IndexError(f"phenotype row {row} is outside active range [0, {size})")
    return row


__all__ = [
    "DIET_NAMES",
    "PhenotypeBatch",
    "PhenotypeBuffer",
    "diet_softmax",
]
