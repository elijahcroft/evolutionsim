"""Vectorised expression and body geometry.

This module is the sole boundary between expressed loci and body-derived quantities.
Milestone 8 replaced the scalar body (``mass = body_size ** 3``) with a real body plan: thirteen
morphology loci describe a shape, and volume, surface area, frontal area, limb count and
slenderness are *integrated out of that shape* rather than asserted.  Nothing outside this module
knows what an organism looks like; the energy model sees only the derived quantities, exactly as
it previously saw only ``mass``.

The body is a tube swept along a straight spine, plus an optional head, tail, limbs and dorsal
fin.  Its radius profile is defined by ``genome.morphology`` in ``config/genome.yaml``, and the
renderer in ``src/evosim/ui/index.html`` reads the same constants and draws the same profile --
so the animal on screen is the animal that pays the costs, which is the whole point of making
morphology enter the energy equations at all.

Volume and surface area are sums over conical frustums between consecutive profile samples,
which is precisely the geometry of the triangle mesh the renderer builds.  They are therefore
the volume and area of the drawn animal, not of an idealised solid it approximates.

``mass = volume * body_density``, and ``adult storage_capacity = mass * energy_storage /
slenderness``, where ``slenderness = 1 / (2 * radius_ratio)`` -- what the retired
``body_slenderness`` locus used to assert, now a consequence of the body's proportions.

The continuously expressed ``offspring_count`` trait is not rounded here; stochastic rounding
belongs to an individual reproduction event.  ``segment_count``, ``radial_symmetry`` and
``limb_pairs`` are likewise continuous loci, rounded only where geometry needs a whole number.
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
_ENERGY_STORAGE = "energy_storage"

#: The thirteen loci that describe a body plan, in the order the renderer expects them.
MORPHOLOGY_NAMES: tuple[str, ...] = (
    "body_length",
    "radius_ratio",
    "fullness",
    "taper",
    "segment_count",
    "segment_depth",
    "radial_symmetry",
    "limb_pairs",
    "limb_ratio",
    "limb_splay",
    "head_size",
    "tail_ratio",
    "dorsal_fin",
)

# Where the tail's base radius is sampled from, as a position along the body profile. The tail
# grows out of the body just inboard of the aft tip, where the profile still has some width.
_TAIL_ATTACH_T = 0.06
# The dorsal fin runs over this fraction of the body, and a half-sine has this mean height.
_FIN_LENGTH_FRACTION = 0.62
_FIN_MEAN_HEIGHT_FACTOR = 2.0 / np.pi


class MorphologyShape(Protocol):
    """The geometry constants ``body_geometry`` needs, as ``MorphologyConfig`` supplies them."""

    profile_samples: int
    body_density: float
    min_radius_ratio: float
    taper_gain: float
    segment_relief: float
    head_radius_gain: float
    head_squash: float
    tail_base_gain: float
    tail_base_offset_ratio: float
    limb_thickness_ratio: float
    fin_span: float


class GenomeSchemaLike(Protocol):
    """The small genome interface needed by phenotype expression."""

    names: tuple[str, ...]
    n_loci: int
    morphology: MorphologyShape

    def index_of(self, name: str) -> int: ...

    def express(self, genomes: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class BodyGeometry:
    """Everything the rest of the simulation is allowed to know about a body's shape.

    Each field is a ``float64`` array with one entry per organism.  The energy model consumes
    these and never sees a morphology locus, which is what keeps the body plan replaceable.
    """

    volume: NDArray[np.float64]
    surface_area: NDArray[np.float64]
    cross_section: NDArray[np.float64]
    limb_count: NDArray[np.float64]
    slenderness: NDArray[np.float64]


def radius_profile(
    body_radius: NDArray[np.float64],
    fullness: NDArray[np.float64],
    taper: NDArray[np.float64],
    segment_count: NDArray[np.float64],
    segment_depth: NDArray[np.float64],
    t: NDArray[np.float64],
    shape: MorphologyShape,
) -> NDArray[np.float64]:
    """Body radius at positions ``t`` along the spine, 0 at the tail end and 1 at the head end.

    This is the single statement of what a body's outline is.  ``buildCreature`` in the UI is a
    transcription of it, and the volume and surface-area sums below are integrals of it, so all
    three agree by construction rather than by maintenance.

    The three factors, in order: ``(4t(1-t))**fullness`` is a spindle that vanishes at both ends
    and peaks mid-body, with a low exponent giving a near-cylindrical worm and a high one a
    lens; ``taper`` slides the bulk fore or aft; and the cosine ripples the surface into visible
    somites.  Arrays broadcast against a trailing sample axis.
    """

    spindle = np.maximum(4.0 * t * (1.0 - t), 1e-4) ** fullness
    radius = body_radius * spindle
    radius = radius * (1.0 + taper * (t - 0.5) * shape.taper_gain)
    ripple = np.cos(2.0 * np.pi * t * segment_count)
    radius = radius * (
        1.0 - segment_depth * shape.segment_relief * (1.0 - ripple) * 0.5
    )
    # The floor is a fraction of this body's own radius, not an absolute length. An absolute
    # floor would be most of a small animal and nothing to a large one, which would break
    # geometric similarity: doubling every gene would stop doubling the animal.
    return np.maximum(radius, shape.min_radius_ratio * body_radius)


def body_geometry(
    expressed: FloatArray,
    indices: dict[str, int],
    shape: MorphologyShape,
) -> BodyGeometry:
    """Integrate expressed morphology loci into the quantities costs are charged on.

    Vectorised over the whole batch: the only loop-shaped thing here is the trailing sample
    axis, which NumPy walks, so this respects the no-per-organism-Python rule even though it is
    doing a numerical integral per animal.
    """

    def locus(name: str) -> NDArray[np.float64]:
        return expressed[:, indices[name]].astype(np.float64)[:, None]

    length = locus("body_length")
    radius_ratio = locus("radius_ratio")
    fullness = locus("fullness")
    taper = locus("taper")
    segment_count = locus("segment_count")
    segment_depth = locus("segment_depth")
    body_radius = radius_ratio * length

    samples = np.linspace(0.0, 1.0, shape.profile_samples, dtype=np.float64)[None, :]
    profile = radius_profile(
        body_radius, fullness, taper, segment_count, segment_depth, samples, shape
    )

    # Conical frustums between consecutive samples: exactly the mesh the renderer builds.
    # End caps are omitted because the profile has already closed to the radius floor at both
    # tips, so a cap contributes a part in ten thousand of the body's own cross-section.
    step = length / float(shape.profile_samples - 1)
    near, far = profile[:, :-1], profile[:, 1:]
    volume = (np.pi / 3.0) * step * np.sum(
        near * near + near * far + far * far, axis=1, keepdims=True
    )
    slant = np.hypot(step, far - near)
    surface_area = np.pi * np.sum((near + far) * slant, axis=1, keepdims=True)

    # Tail: a cone continuing aft from where the body has almost closed.
    tail_length = locus("tail_ratio") * length
    tail_radius = (
        radius_profile(
            body_radius,
            fullness,
            taper,
            segment_count,
            segment_depth,
            np.full((1, 1), _TAIL_ATTACH_T),
            shape,
        )
        * shape.tail_base_gain
        + shape.tail_base_offset_ratio * body_radius
    )
    volume = volume + (np.pi / 3.0) * tail_radius * tail_radius * tail_length
    surface_area = surface_area + np.pi * tail_radius * np.hypot(tail_radius, tail_length)

    # Head: a sphere flattened slightly in the vertical.
    head_radius = locus("head_size") * body_radius * shape.head_radius_gain
    volume = volume + (4.0 / 3.0) * np.pi * head_radius**3 * shape.head_squash
    surface_area = surface_area + 4.0 * np.pi * head_radius**2 * shape.head_squash

    # Limbs: cylinders. radial_symmetry 1 means bilateral -- a limb either side per pair --
    # while higher symmetry arrays that many limbs around the body axis instead.
    pairs = np.round(locus("limb_pairs"))
    symmetry = np.round(locus("radial_symmetry"))
    limb_count = pairs * np.where(symmetry <= 1.0, 2.0, symmetry)
    limb_radius = shape.limb_thickness_ratio * body_radius
    limb_length = locus("limb_ratio") * length
    volume = volume + limb_count * np.pi * limb_radius**2 * limb_length
    surface_area = surface_area + limb_count * 2.0 * np.pi * limb_radius * limb_length

    # Dorsal fin: a sheet. Two sides of area, negligible volume -- cheap to grow and expensive
    # to keep warm, which is the trade that makes it interesting rather than decorative.
    fin_height = locus("dorsal_fin") * body_radius * shape.fin_span * _FIN_MEAN_HEIGHT_FACTOR
    surface_area = surface_area + 2.0 * fin_height * _FIN_LENGTH_FRACTION * length

    # Frontal area seen by the medium. The head sits in front of the body rather than beside
    # it, so the two do not add; a limb held out from the axis presents its full side profile.
    widest = np.maximum(np.max(profile, axis=1, keepdims=True), head_radius)
    cross_section = np.pi * widest * widest + limb_count * 2.0 * limb_radius * limb_length

    slenderness = 1.0 / (2.0 * radius_ratio)
    return BodyGeometry(
        volume=volume[:, 0],
        surface_area=surface_area[:, 0],
        cross_section=cross_section[:, 0],
        limb_count=limb_count[:, 0],
        slenderness=slenderness[:, 0],
    )


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
    volume: FloatArray
    surface_area: FloatArray
    cross_section: FloatArray
    limb_count: FloatArray
    slenderness: FloatArray
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
                "volume": float(self.volume[row]),
                "surface_area": float(self.surface_area[row]),
                "cross_section": float(self.cross_section[row]),
                "limb_count": float(self.limb_count[row]),
                "slenderness": float(self.slenderness[row]),
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
    volume: FloatArray
    surface_area: FloatArray
    cross_section: FloatArray
    limb_count: FloatArray
    slenderness: FloatArray
    size: int
    _indices: dict[str, int]
    _schema: GenomeSchemaLike
    _shape: MorphologyShape
    _diet_indices: NDArray[np.intp]
    _energy_storage_index: int

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
            volume=np.zeros(capacity, dtype=np.float32),
            surface_area=np.zeros(capacity, dtype=np.float32),
            cross_section=np.zeros(capacity, dtype=np.float32),
            limb_count=np.zeros(capacity, dtype=np.float32),
            slenderness=np.zeros(capacity, dtype=np.float32),
            size=0,
            _indices=indices,
            _schema=schema,
            _shape=schema.morphology,
            _diet_indices=diet_indices,
            _energy_storage_index=indices[_ENERGY_STORAGE],
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
            + self.volume.nbytes
            + self.surface_area.nbytes
            + self.cross_section.nbytes
            + self.limb_count.nbytes
            + self.slenderness.nbytes
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

        energy_storage = expressed[:, self._energy_storage_index]
        if np.any(energy_storage < 0.0):
            raise ValueError("expressed energy_storage must be non-negative")

        diet = diet_softmax(expressed[:, self._diet_indices])
        geometry = body_geometry(expressed, self._indices, self._shape)
        if np.any(geometry.volume <= 0.0):
            raise ValueError("derived body volume must be positive")
        mass = (geometry.volume * self._shape.body_density).astype(
            np.float32, copy=False
        )
        storage_capacity = (
            mass * energy_storage / geometry.slenderness
        ).astype(np.float32, copy=False)
        finite = (
            np.all(np.isfinite(mass))
            and np.all(np.isfinite(storage_capacity))
            and np.all(np.isfinite(geometry.surface_area))
            and np.all(np.isfinite(geometry.cross_section))
        )
        if not finite:
            raise ValueError("derived body geometry must contain only finite values")

        target = slice(start, stop)
        self.traits[target] = expressed
        self.diet[target] = diet
        self.mass[target] = mass
        self.storage_capacity[target] = storage_capacity
        self.volume[target] = geometry.volume
        self.surface_area[target] = geometry.surface_area
        self.cross_section[target] = geometry.cross_section
        self.limb_count[target] = geometry.limb_count
        self.slenderness[target] = geometry.slenderness
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
        # Every array is written out by name rather than walked in a loop. That is not style:
        # tests/test_phenotype.py parses this method and rejects any Python iteration in it, and
        # the rule is worth more than the brevity a loop over seven attributes would buy.
        self.traits[:new_size] = self.traits[survivors]
        self.diet[:new_size] = self.diet[survivors]
        self.mass[:new_size] = self.mass[survivors]
        self.storage_capacity[:new_size] = self.storage_capacity[survivors]
        self.volume[:new_size] = self.volume[survivors]
        self.surface_area[:new_size] = self.surface_area[survivors]
        self.cross_section[:new_size] = self.cross_section[survivors]
        self.limb_count[:new_size] = self.limb_count[survivors]
        self.slenderness[:new_size] = self.slenderness[survivors]

        cleared = slice(new_size, old_size)
        self.traits[cleared] = 0.0
        self.diet[cleared] = 0.0
        self.mass[cleared] = 0.0
        self.storage_capacity[cleared] = 0.0
        self.volume[cleared] = 0.0
        self.surface_area[cleared] = 0.0
        self.cross_section[cleared] = 0.0
        self.limb_count[cleared] = 0.0
        self.slenderness[cleared] = 0.0
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
        return self._gather(rows)

    def select(self, rows: NDArray[np.integer[Any]]) -> PhenotypeBatch:
        """Return a batch gathered from arbitrary active rows.

        Unlike :meth:`batch`, the arrays are copies rather than views, because fancy indexing
        cannot alias.  This exists for read-only consumers that work on a scattered subset --
        the movement layer scoring only the organisms that are actually moving this tick --
        so they need not evaluate the whole population to reach a handful of rows.
        """

        selection = np.asarray(rows)
        if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
            raise TypeError("rows must be a one-dimensional integer array")
        if selection.size and (
            selection.min() < 0 or selection.max() >= self.size
        ):
            raise IndexError(f"row index is outside active range [0, {self.size})")
        return self._gather(selection)

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

    def _gather(self, rows: Any) -> PhenotypeBatch:
        """Build a batch over ``rows``, which may be a slice (views) or indices (copies).

        The single place batch construction happens, so :meth:`batch` and :meth:`select` cannot
        drift apart as derived quantities are added.
        """

        return PhenotypeBatch(
            traits=self.traits[rows],
            diet=self.diet[rows],
            mass=self.mass[rows],
            storage_capacity=self.storage_capacity[rows],
            volume=self.volume[rows],
            surface_area=self.surface_area[rows],
            cross_section=self.cross_section[rows],
            limb_count=self.limb_count[rows],
            slenderness=self.slenderness[rows],
            trait_names=self.trait_names,
            _indices=self._indices,
        )

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
        for name in (*_DIET_LOCUS_NAMES, *MORPHOLOGY_NAMES, _ENERGY_STORAGE)
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
    "MORPHOLOGY_NAMES",
    "BodyGeometry",
    "PhenotypeBatch",
    "PhenotypeBuffer",
    "body_geometry",
    "diet_softmax",
    "radius_profile",
]
