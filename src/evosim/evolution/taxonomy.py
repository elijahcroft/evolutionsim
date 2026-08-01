"""Deciding, periodically, how many kinds of organism the population contains.

A species here is not a label anything applies from outside; it is the answer to a question the
reproduction layer already asks every time an organism looks for a mate.  Two organisms
interbreed when their standardised genetic distance is below
``energy.reproduction.mate_compatibility_distance``.  This module therefore splits a species in
exactly one circumstance: when its members fall into two groups whose *centres* are further
apart than that same threshold, so that a typical member of one group could not breed with a
typical member of the other.  Assortative mating and the split test agree by construction, and
no coefficient enters the model that was not already in ``energy.yaml``.

The split test is 2-means run in the same space :meth:`GenomeSchema.genetic_distance` measures
in: each expressed trait divided by its locus span and scaled by the square root of its
distance weight, which makes plain Euclidean distance between two coordinate vectors *equal* to
the genetic distance between the genomes.  Anything else would let the taxonomy drift out of
agreement with the mating rule.

Three decisions worth stating:

1. **The larger cluster keeps the ancestral identity.**  A split is a bifurcation, and calling
   both halves new would throw away the continuity the lineage tree exists to record.  The
   smaller group is the daughter; ties go to the group containing the lower population row, so
   the outcome is a function of the state and not of dictionary order.

2. **Splitting recurses until nothing more splits.**  ``taxonomy_interval`` sets how often
   taxonomy is *revisited*, not how much divergence one pass may recognise; a population left
   alone for many intervals must not be capped at one new species per pass.

3. **There is no minimum species size.**  A single organism far enough from everything else is
   reproductively isolated by the very threshold above, which is the whole definition in use
   here.  If it fails to leave descendants its species is recorded and then declared extinct --
   which is the honest outcome, and cheaper than inventing a quorum constant that
   ``config/`` does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from evosim.config import Config
from evosim.life.genome import GenomeSchema
from evosim.life.population import Population

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

#: Convergence bound for the 2-means iteration.  This is a numerical safeguard, not a
#: biological tunable: assignments stabilise in a handful of passes, and a cap merely stops a
#: pathological cycle from running forever.
MAX_ITERATIONS = 32


@dataclass(frozen=True, slots=True)
class Split:
    """One accepted bifurcation: which rows leave, and how far apart the halves are."""

    species: int
    """Identity given to the daughter group."""

    parent: int
    """The species that bifurcated; it keeps its identity and its larger half."""

    rows: IntArray
    """Population rows that move to the daughter species."""

    separation: float
    """Genetic distance between the two cluster centres, in mate-compatibility units."""


@dataclass(frozen=True, slots=True)
class TaxonomyModel:
    """The periodic split test, bound to a genome schema and the mating threshold."""

    schema: GenomeSchema
    split_distance: float
    scale: FloatArray
    """Per-locus coordinate scale making Euclidean distance equal genetic distance."""

    @classmethod
    def from_config(cls, config: Config, schema: GenomeSchema) -> TaxonomyModel:
        weights = schema.distance_weights.astype(np.float64)
        total = float(weights.sum())
        span = schema.span.astype(np.float64)
        return cls(
            schema=schema,
            split_distance=config.energy.reproduction.mate_compatibility_distance,
            scale=np.sqrt(weights / total) / span,
        )

    def coordinates(self, genomes: NDArray[np.floating]) -> FloatArray:
        """Place genomes in the space the mating threshold is measured in."""

        return self.schema.express(genomes).astype(np.float64) * self.scale

    def split(
        self,
        population: Population,
        rng: np.random.Generator,
        next_species: int,
    ) -> list[Split]:
        """Return every bifurcation the current population supports.

        Splits are returned in the order they were found and must be applied in that order:
        a later split's rows are a subset of an earlier daughter group.  Nothing here writes to
        the population, because who allocates a species identity is the history layer's
        business, not the taxonomy's.
        """

        if population.size == 0:
            return []

        active = population.active
        coordinates = self.coordinates(population.genomes[active])
        labels = population.species_id[active].astype(np.int64, copy=True)

        pending = [int(value) for value in np.unique(labels)]
        splits: list[Split] = []
        while pending:
            species = pending.pop()
            rows = np.flatnonzero(labels == species)
            outcome = _two_means(coordinates[rows], rng)
            if outcome is None:
                continue
            daughter_mask, separation = outcome
            if separation <= self.split_distance:
                continue

            # The smaller half becomes the daughter; _two_means already orders its label so
            # that group 1 is the smaller (ties broken by first row), so no choice is made here.
            daughter_rows = rows[daughter_mask]
            labels[daughter_rows] = next_species
            splits.append(
                Split(
                    species=next_species,
                    parent=species,
                    rows=daughter_rows,
                    separation=separation,
                )
            )
            # Either half may still contain more than one kind of organism.
            pending.extend((species, next_species))
            next_species += 1
        return splits


def _two_means(
    points: FloatArray,
    rng: np.random.Generator,
) -> tuple[NDArray[np.bool_], float] | None:
    """Cluster ``points`` in two and report the distance between the centres.

    Returns ``None`` when no two-way split exists at all -- fewer than two points, points that
    are all identical, or an iteration that empties a cluster.  The returned mask selects the
    smaller cluster, with ties resolved toward the cluster *not* containing the first point, so
    the caller never has to break a tie of its own.
    """

    count = points.shape[0]
    if count < 2:
        return None

    # k-means++ initialisation: the second centre is drawn in proportion to squared distance
    # from the first, which is what makes a genuinely bimodal population find both modes on the
    # first try rather than depending on how many restarts anyone budgeted for.
    first = int(rng.integers(count))
    spread = np.sum((points - points[first]) ** 2, axis=1)
    total = float(spread.sum())
    if total <= 0.0:
        return None
    second = int(np.searchsorted(np.cumsum(spread), rng.random() * total, side="right"))
    centres = np.stack((points[first], points[min(second, count - 1)]))

    labels = np.zeros(count, dtype=np.bool_)
    for iteration in range(MAX_ITERATIONS):
        updated = np.sum((points - centres[1]) ** 2, axis=1) < np.sum(
            (points - centres[0]) ** 2, axis=1
        )
        if iteration and np.array_equal(updated, labels):
            break
        labels = updated
        upper = int(np.count_nonzero(labels))
        if upper == 0 or upper == count:
            return None
        centres = np.stack((points[~labels].mean(axis=0), points[labels].mean(axis=0)))

    separation = float(np.sqrt(np.sum((centres[0] - centres[1]) ** 2)))
    upper = int(np.count_nonzero(labels))
    if upper * 2 > count:
        labels = ~labels
    elif upper * 2 == count:
        # An exact halving: keep the group without the first row, so the ancestral identity
        # stays with the half the arbitrary initialisation started from.
        labels = labels if not labels[0] else ~labels
    return labels, separation


__all__ = ["MAX_ITERATIONS", "Split", "TaxonomyModel"]
