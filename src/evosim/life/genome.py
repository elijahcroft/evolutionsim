"""Vectorised diploid genome operations.

A genome batch always has shape ``(organisms, loci, 2)``.  The final axis is
the pair of homologous alleles.  Public operations return ``float32`` arrays
and never modify their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evosim.config import GenomeConfig, MorphologyConfig, MutationConfig


GenomeArray: TypeAlias = NDArray[np.float32]
TraitArray: TypeAlias = NDArray[np.float32]


def _readonly_float32(values: ArrayLike) -> NDArray[np.float32]:
    """Return a genuinely immutable float32 array backed by immutable bytes."""

    materialised = np.asarray(values, dtype=np.float32)
    frozen = np.frombuffer(materialised.tobytes(order="C"), dtype=np.float32)
    return frozen.reshape(materialised.shape)


def _require_generator(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an explicit numpy.random.Generator; "
            f"got {type(rng).__name__}"
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class GenomeSchema:
    """Immutable, array-oriented view of a :class:`GenomeConfig`.

    The config is converted once so hot genome and phenotype operations do not
    iterate over ``LocusSpec`` objects.  All cached arrays and the name mapping
    are read-only.
    """

    names: tuple[str, ...]
    indices: Mapping[str, int]
    bounds: NDArray[np.float32]
    low: NDArray[np.float32]
    high: NDArray[np.float32]
    init: NDArray[np.float32]
    sigma: NDArray[np.float32]
    span: NDArray[np.float32]
    distance_weights: NDArray[np.float32]
    mutation: MutationConfig
    morphology: MorphologyConfig
    mutation_rate_index: int | None
    radiation_tolerance_index: int | None
    _distance_weight_sum: float

    def __init__(self, config: GenomeConfig) -> None:
        if not isinstance(config, GenomeConfig):
            raise TypeError(
                "config must be an evosim.config.GenomeConfig; "
                f"got {type(config).__name__}"
            )

        names = tuple(locus.name for locus in config.loci)
        indices = MappingProxyType({name: i for i, name in enumerate(names)})
        low_values = config.low_array()
        high_values = config.high_array()
        bounds = _readonly_float32(np.column_stack((low_values, high_values)))

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "low", bounds[:, 0])
        object.__setattr__(self, "high", bounds[:, 1])
        object.__setattr__(self, "init", _readonly_float32(config.init_array()))
        object.__setattr__(self, "sigma", _readonly_float32(config.sigma_array()))
        object.__setattr__(
            self,
            "span",
            _readonly_float32(
                bounds[:, 1].astype(np.float64) - bounds[:, 0].astype(np.float64)
            ),
        )
        object.__setattr__(
            self,
            "distance_weights",
            _readonly_float32(config.distance_weight_array()),
        )
        object.__setattr__(self, "mutation", config.mutation)
        object.__setattr__(self, "morphology", config.morphology)
        object.__setattr__(
            self,
            "mutation_rate_index",
            indices.get("mutation_rate"),
        )
        object.__setattr__(
            self,
            "radiation_tolerance_index",
            indices.get("radiation_tolerance"),
        )
        object.__setattr__(
            self,
            "_distance_weight_sum",
            float(np.sum(self.distance_weights, dtype=np.float64)),
        )

    @classmethod
    def from_config(cls, config: GenomeConfig) -> GenomeSchema:
        """Build a schema from the validated genome configuration."""

        return cls(config)

    @property
    def n_loci(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        """Return the stable column index for ``name``."""

        try:
            return self.indices[name]
        except KeyError:
            raise KeyError(f"no such locus: {name!r}") from None

    def founders(self, count: int) -> GenomeArray:
        """Construct ``count`` homozygous copies of the configured founder."""

        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError(f"count must be an integer, got {type(count).__name__}")
        count = int(count)
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")

        shape = (count, self.n_loci, 2)
        return np.broadcast_to(self.init[None, :, None], shape).copy()

    def express(self, genomes: ArrayLike) -> TraitArray:
        """Express diploid genomes as clipped allele means.

        Returns an array with shape ``(organisms, loci)``.
        """

        alleles = self._genomes(genomes, name="genomes")
        # Accumulate the two homologs in float64 so two individually valid large float32
        # alleles cannot overflow before their mean is taken.
        expressed = np.mean(alleles, axis=2, dtype=np.float64).astype(np.float32)
        return np.clip(expressed, self.low, self.high)

    def genetic_distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
    ) -> NDArray[np.float32]:
        """Return weighted RMS distance for corresponding genome rows.

        Locus differences are measured between expressed traits and divided by
        that locus's configured span.  The two batches must contain the same
        number of organisms; the result has shape ``(organisms,)``.
        """

        left_genomes = self._genomes(left, name="left")
        right_genomes = self._genomes(right, name="right")
        self._require_same_count(left_genomes, right_genomes, "left", "right")

        delta = (self.express(left_genomes) - self.express(right_genomes)) / self.span
        mean_square = np.sum(
            delta * delta * self.distance_weights,
            axis=1,
            dtype=np.float64,
        ) / self._distance_weight_sum
        return np.sqrt(np.maximum(mean_square, np.float32(0.0))).astype(
            np.float32,
            copy=False,
        )

    def distance(
        self,
        left: ArrayLike,
        right: ArrayLike,
    ) -> NDArray[np.float32]:
        """Alias for :meth:`genetic_distance`."""

        return self.genetic_distance(left, right)

    def recombine(
        self,
        parent_a: ArrayLike,
        parent_b: ArrayLike,
        rng: np.random.Generator,
    ) -> GenomeArray:
        """Create one offspring per corresponding parent pair.

        Each offspring independently receives one allele per locus from
        ``parent_a`` and one from ``parent_b``.
        """

        _require_generator(rng)
        first = self._genomes(parent_a, name="parent_a")
        second = self._genomes(parent_b, name="parent_b")
        self._require_same_count(first, second, "parent_a", "parent_b")

        choice_shape = first.shape[:2]
        first_choice = rng.integers(0, 2, size=choice_shape, dtype=np.int8)
        second_choice = rng.integers(0, 2, size=choice_shape, dtype=np.int8)

        offspring = np.empty(first.shape, dtype=np.float32)
        offspring[:, :, 0] = np.take_along_axis(
            first,
            first_choice[:, :, None],
            axis=2,
        )[:, :, 0]
        offspring[:, :, 1] = np.take_along_axis(
            second,
            second_choice[:, :, None],
            axis=2,
        )[:, :, 0]
        return offspring

    def mutate(
        self,
        genomes: ArrayLike,
        rng: np.random.Generator,
        radiation: float = 0.0,
    ) -> GenomeArray:
        """Return a mutated copy of ``genomes``.

        Every allele mutates independently.  Its probability is the organism's
        expressed ``mutation_rate`` multiplied by the radiation excess
        ``1 + radiation_sensitivity * radiation / (1 + radiation_tolerance)``
        and clipped to one.
        Mutational effects are Gaussian with the locus sigma; configured
        large-effect events multiply that sigma by ``large_effect_scale``.
        """

        _require_generator(rng)
        alleles = self._genomes(genomes, name="genomes")
        radiation_value = self._radiation(radiation)
        if self.mutation_rate_index is None:
            raise ValueError(
                "genome schema has no 'mutation_rate' locus required for mutation"
            )

        rate_index = self.mutation_rate_index
        base_rate = np.mean(
            alleles[:, rate_index, :],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
        base_rate = np.clip(
            base_rate,
            self.low[rate_index],
            self.high[rate_index],
        )
        if self.radiation_tolerance_index is None:
            radiation_tolerance = np.zeros(alleles.shape[0], dtype=np.float32)
        else:
            tolerance_index = self.radiation_tolerance_index
            radiation_tolerance = np.mean(
                alleles[:, tolerance_index, :],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
            radiation_tolerance = np.clip(
                radiation_tolerance,
                self.low[tolerance_index],
                self.high[tolerance_index],
            )
            radiation_tolerance = np.maximum(radiation_tolerance, 0.0)
        radiation_factor = 1.0 + (
            self.mutation.radiation_sensitivity
            * radiation_value
            / (1.0 + radiation_tolerance)
        )
        effective_rate = np.clip(base_rate * radiation_factor, 0.0, 1.0)

        mutation_mask = rng.random(alleles.shape, dtype=np.float32) < effective_rate[
            :, None, None
        ]
        mutated = alleles.copy()
        flat_indices = np.flatnonzero(mutation_mask)
        mutation_count = flat_indices.size

        if mutation_count:
            locus_indices = (flat_indices // 2) % self.n_loci
            large_effect = (
                rng.random(mutation_count, dtype=np.float32)
                < self.mutation.p_large_effect
            )
            effect_sigma = self.sigma[locus_indices].astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                effect_sigma[large_effect] *= self.mutation.large_effect_scale
                effects = rng.standard_normal(mutation_count) * effect_sigma
            # Zero normal deviate times an overflowing scale has the limiting effect zero,
            # not NaN. Infinite non-zero effects are safe: clipping below maps them to the
            # corresponding configured bound.
            effects[np.isnan(effects)] = 0.0
            flat = mutated.reshape(-1)
            updated = flat[flat_indices].astype(np.float64) + effects
            flat[flat_indices] = np.clip(
                updated,
                self.low[locus_indices],
                self.high[locus_indices],
            ).astype(np.float32)

        np.clip(
            mutated,
            self.low[None, :, None],
            self.high[None, :, None],
            out=mutated,
        )
        return mutated

    def _genomes(self, genomes: ArrayLike, *, name: str) -> GenomeArray:
        array = np.asarray(genomes)
        expected = f"(batch, {self.n_loci}, 2)"
        if array.ndim != 3 or array.shape[1:] != (self.n_loci, 2):
            raise ValueError(
                f"{name} must have shape {expected}; got {array.shape}"
            )
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(
                f"{name} must contain real numeric alleles; got dtype {array.dtype}"
            )
        converted = array.astype(np.float32, copy=False)
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"{name} must contain only finite alleles")
        return converted

    @staticmethod
    def _require_same_count(
        left: GenomeArray,
        right: GenomeArray,
        left_name: str,
        right_name: str,
    ) -> None:
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"{left_name} and {right_name} must contain the same number of genomes; "
                f"got {left.shape[0]} and {right.shape[0]}"
            )

    @staticmethod
    def _radiation(radiation: float) -> float:
        if isinstance(radiation, (bool, np.bool_)) or not isinstance(
            radiation,
            Real,
        ):
            raise TypeError(
                f"radiation must be a non-negative finite number, got "
                f"{type(radiation).__name__}"
            )
        value = float(radiation)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"radiation must be a non-negative finite number, got {radiation!r}"
            )
        return value


__all__ = ["GenomeArray", "GenomeSchema", "TraitArray"]
