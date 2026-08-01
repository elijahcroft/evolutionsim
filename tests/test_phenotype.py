"""Vectorised phenotype expression and body-geometry tests.

The scalar-body derivations these started as (M2) were replaced in M8 by integrals over a real
morphology genome; the alignment, aliasing and validation tests around them are unchanged,
because the buffer contract they pin is exactly what let the body be replaced.
"""

from __future__ import annotations

import ast
import inspect
from textwrap import dedent

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.genome import GenomeSchema
from evosim.life.phenotype import (
    DIET_NAMES,
    MORPHOLOGY_NAMES,
    PhenotypeBuffer,
    diet_softmax,
)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def schema(config: Config) -> GenomeSchema:
    return GenomeSchema.from_config(config.genome)


def test_buffer_preallocates_aligned_float32_arrays(schema: GenomeSchema):
    buffer = PhenotypeBuffer.allocate(7, schema)

    assert buffer.capacity == 7
    assert buffer.size == 0
    assert buffer.traits.shape == (7, schema.n_loci)
    assert buffer.diet.shape == (7, 4)
    derived = (
        buffer.mass,
        buffer.storage_capacity,
        buffer.volume,
        buffer.surface_area,
        buffer.cross_section,
        buffer.limb_count,
        buffer.slenderness,
    )
    assert all(array.shape == (7,) for array in derived)
    assert all(array.dtype == np.float32 for array in derived)
    assert buffer.traits.dtype == np.float32
    assert buffer.diet.dtype == np.float32
    assert buffer.memory_bytes == sum(
        array.nbytes for array in (buffer.traits, buffer.diet, *derived)
    )
    assert len(buffer.active) == 0


@pytest.mark.parametrize("capacity", [-1, 1.5, True])
def test_buffer_rejects_invalid_capacity(schema: GenomeSchema, capacity):
    with pytest.raises((TypeError, ValueError)):
        PhenotypeBuffer.allocate(capacity, schema)


def test_founder_traits_and_diet_are_expressed_without_input_mutation(
    schema: GenomeSchema,
):
    genomes = schema.founders(3)
    before = genomes.copy()
    buffer = PhenotypeBuffer.allocate(5, schema)

    batch = buffer.update(0, genomes, schema)

    assert np.array_equal(genomes, before)
    assert buffer.size == batch.size == 3
    assert np.allclose(batch.traits, schema.init)
    assert np.allclose(batch.diet.sum(axis=1), 1.0)
    expected_logits = schema.init[
        [schema.index_of(f"aff_{name}") for name in DIET_NAMES]
    ]
    expected = np.exp(expected_logits - expected_logits.max())
    expected /= expected.sum()
    assert np.allclose(batch.diet, expected)
    assert np.all(batch.diet_component("autotroph") > 0.5)


def test_diet_softmax_is_stable_for_extreme_finite_logits():
    maximum = np.finfo(np.float64).max
    logits = np.array(
        [
            [1.0e30, 0.0, -1.0e30, 1.0e30],
            [-1.0e30, -1.0e30, -1.0e30, -1.0e30],
            [maximum, -maximum, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    before = logits.copy()

    with np.errstate(over="raise"):
        fractions = diet_softmax(logits)

    assert np.array_equal(logits, before)
    assert fractions.dtype == np.float32
    assert np.all(np.isfinite(fractions))
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert fractions[0] == pytest.approx([0.5, 0.0, 0.0, 0.5])
    assert fractions[1] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert fractions[2] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_diet_softmax_validates_shape_type_and_finiteness():
    with pytest.raises(ValueError, match="shape"):
        diet_softmax(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="real numbers"):
        diet_softmax(np.full((2, 4), "omnivore"))
    with pytest.raises(ValueError, match="finite"):
        diet_softmax(np.array([[0.0, 1.0, np.nan, 2.0]]))


def test_mass_and_storage_follow_from_the_integrated_body(schema: GenomeSchema):
    """Mass is volume times density, and storage divides by *derived* slenderness.

    Slenderness is no longer a locus that can disagree with the body's proportions: it is
    1/(2*radius_ratio), so a body twice as wide for its length is half as slender by
    construction rather than by a second gene saying so.
    """

    genomes = schema.founders(3)
    radius_ratio = np.array([0.50, 0.25, 0.10], dtype=np.float32)
    energy_storage = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    genomes[:, schema.index_of("radius_ratio"), :] = radius_ratio[:, None]
    genomes[:, schema.index_of("energy_storage"), :] = energy_storage[:, None]

    phenotype = PhenotypeBuffer.from_genomes(genomes, schema)
    density = schema.morphology.body_density
    expected_slenderness = 1.0 / (2.0 * radius_ratio)

    assert phenotype.slenderness[:3] == pytest.approx(expected_slenderness, rel=1e-6)
    assert phenotype.mass[:3] == pytest.approx(
        phenotype.volume[:3] * density, rel=1e-6
    )
    assert phenotype.storage_capacity[:3] == pytest.approx(
        phenotype.mass[:3] * energy_storage / expected_slenderness, rel=1e-6
    )


def test_the_founder_body_is_pinned_to_its_pre_morphology_mass(schema: GenomeSchema):
    """The anchor the whole of M8 rests on.

    Every intake and cost constant in energy.yaml was calibrated in M3, M7 and M7b against a
    founder of mass 0.064. Replacing the scalar body with a real one had to leave that number
    where it was, or the morphology genome would have silently invalidated three milestones of
    tuning. body_density is chosen to make this exact; if this test fails, something moved in
    the founder's geometry and energy.yaml no longer means what its comments say.
    """

    founder = PhenotypeBuffer.from_genomes(schema.founders(1), schema).active

    assert float(founder.mass[0]) == pytest.approx(0.064, rel=1e-5)
    assert float(founder.storage_capacity[0]) == pytest.approx(0.128, rel=1e-5)
    assert float(founder.slenderness[0]) == pytest.approx(1.0, rel=1e-6)
    assert float(founder.limb_count[0]) == 0.0


def test_geometry_scales_as_a_solid_and_shape_is_size_independent(schema: GenomeSchema):
    """Doubling every length multiplies volume by eight and area by four.

    This is what makes the derived quantities safe to use in cost equations: they are genuine
    geometric quantities with the dimensions they claim, not indices that happen to grow.
    """

    genomes = schema.founders(2)
    genomes[:, schema.index_of("body_length"), :] = np.array(
        [[1.0], [2.0]], dtype=np.float32
    )
    buffer = PhenotypeBuffer.from_genomes(genomes, schema)

    # rel=1e-4 rather than tighter: the derived arrays are stored as float32, so a few parts in
    # 100000 of slack is the storage format, not the geometry.
    assert buffer.volume[1] == pytest.approx(buffer.volume[0] * 8.0, rel=1e-4)
    assert buffer.surface_area[1] == pytest.approx(
        buffer.surface_area[0] * 4.0, rel=1e-4
    )
    assert buffer.cross_section[1] == pytest.approx(
        buffer.cross_section[0] * 4.0, rel=1e-4
    )
    # Surface area per unit volume falls with size: Bergmann's rule, as geometry rather than
    # as a rule written down anywhere.
    assert (
        buffer.surface_area[1] / buffer.volume[1]
        < buffer.surface_area[0] / buffer.volume[0]
    )


def test_a_slender_body_has_more_surface_than_a_fat_one_of_equal_volume(
    schema: GenomeSchema,
):
    """Shape, not just size, decides how much surface an animal has to keep warm."""

    genomes = schema.founders(2)
    # Volume goes as body_length**3 * radius_ratio**2, so four times the length at an eighth
    # of the radius ratio is the same animal's worth of flesh drawn out into a worm.
    genomes[0, schema.index_of("body_length"), :] = 1.0
    genomes[0, schema.index_of("radius_ratio"), :] = 0.5
    genomes[1, schema.index_of("body_length"), :] = 4.0
    genomes[1, schema.index_of("radius_ratio"), :] = 0.0625
    buffer = PhenotypeBuffer.from_genomes(genomes, schema)

    assert buffer.volume[1] == pytest.approx(buffer.volume[0], rel=1e-4)
    assert buffer.surface_area[1] > buffer.surface_area[0]
    assert buffer.slenderness[1] > buffer.slenderness[0]


def test_limb_count_rounds_pairs_and_symmetry(schema: GenomeSchema):
    """Counts are continuous loci; only the geometry rounds them."""

    genomes = schema.founders(4)
    pairs = np.array([0.4, 0.6, 2.0, 2.0], dtype=np.float32)
    symmetry = np.array([1.0, 1.0, 1.0, 4.0], dtype=np.float32)
    genomes[:, schema.index_of("limb_pairs"), :] = pairs[:, None]
    genomes[:, schema.index_of("radial_symmetry"), :] = symmetry[:, None]
    buffer = PhenotypeBuffer.from_genomes(genomes, schema)

    # 0.4 pairs rounds to none; 0.6 rounds to one pair, which is two limbs when bilateral;
    # two pairs at fourfold symmetry is eight limbs arrayed around the axis.
    assert buffer.limb_count[:4] == pytest.approx([0.0, 2.0, 4.0, 8.0])
    assert buffer.volume[1] > buffer.volume[0]


def test_contiguous_updates_can_overwrite_or_extend_but_not_leave_gaps(
    schema: GenomeSchema,
):
    buffer = PhenotypeBuffer.allocate(4, schema)
    first = schema.founders(2)
    first[:, schema.index_of("body_length"), :] = np.array(
        [[0.5], [1.0]], dtype=np.float32
    )
    buffer.update(0, first, schema)

    replacement = schema.founders(2)
    replacement[:, schema.index_of("body_length"), :] = np.array(
        [[2.0], [3.0]], dtype=np.float32
    )
    updated = buffer.update(1, replacement, schema)

    assert buffer.size == 3
    assert updated.size == 2
    assert buffer.trait("body_length") == pytest.approx([0.5, 2.0, 3.0])
    assert np.shares_memory(buffer.active.traits, buffer.traits)
    assert np.shares_memory(buffer.active.trait("body_length"), buffer.traits)

    with pytest.raises(ValueError, match="inactive gap"):
        buffer.update(4, schema.founders(0), schema)
    with pytest.raises(ValueError, match="capacity"):
        buffer.update(3, schema.founders(2), schema)
    with pytest.raises(ValueError, match="shape"):
        buffer.update(0, np.zeros((1, schema.n_loci)), schema)


def test_buffer_rejects_a_distinct_schema_even_with_the_same_locus_order(
    config: Config,
    schema: GenomeSchema,
):
    buffer = PhenotypeBuffer.allocate(1, schema)
    other_schema = GenomeSchema(config.genome)

    with pytest.raises(ValueError, match="exact schema"):
        buffer.update(0, other_schema.founders(1), other_schema)


def test_compaction_preserves_alignment_and_clears_released_rows(
    schema: GenomeSchema,
):
    genomes = schema.founders(4)
    values = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    genomes[:, schema.index_of("body_length"), :] = values[:, None]
    buffer = PhenotypeBuffer.from_genomes(genomes, schema, capacity=6)
    old_diet = buffer.diet[:4].copy()
    old_mass = buffer.mass[:4].copy()

    active = buffer.compact(np.array([0, 2, 3]), old_size=4)

    assert buffer.size == active.size == 3
    assert buffer.trait("body_length") == pytest.approx(values[[0, 2, 3]])
    assert np.array_equal(buffer.diet[:3], old_diet[[0, 2, 3]])
    assert np.array_equal(buffer.mass[:3], old_mass[[0, 2, 3]])
    assert np.all(buffer.traits[3:4] == 0.0)
    assert np.all(buffer.diet[3:4] == 0.0)
    assert np.all(buffer.mass[3:4] == 0.0)
    assert np.all(buffer.storage_capacity[3:4] == 0.0)


def test_compaction_rejects_invalid_indices_and_size(schema: GenomeSchema):
    buffer = PhenotypeBuffer.from_genomes(schema.founders(3), schema)
    with pytest.raises(ValueError, match="does not match"):
        buffer.compact(np.array([0, 1]), old_size=2)
    with pytest.raises(TypeError, match="integer"):
        buffer.compact(np.array([True, False, True]), old_size=3)
    with pytest.raises(IndexError, match="outside"):
        buffer.compact(np.array([0, 3]), old_size=3)
    with pytest.raises(ValueError, match="duplicates"):
        buffer.compact(np.array([0, 0]), old_size=3)


def test_named_views_and_batch_bounds(schema: GenomeSchema):
    buffer = PhenotypeBuffer.from_genomes(schema.founders(3), schema)

    assert np.shares_memory(buffer.trait("body_length"), buffer.traits)
    assert np.shares_memory(buffer.diet_component("autotroph"), buffer.diet)
    assert len(buffer.batch(1, 3)) == 2
    with pytest.raises(KeyError, match="no such expressed trait"):
        buffer.trait("wings")
    with pytest.raises(KeyError, match="no such diet"):
        buffer.diet_component("omnivore")
    with pytest.raises(ValueError, match="exceeds stop"):
        buffer.batch(2, 1)
    with pytest.raises(IndexError, match="exceeds active"):
        buffer.batch(0, 4)


def test_serialization_contains_traits_diet_and_body_geometry(
    schema: GenomeSchema,
):
    genomes = schema.founders(1)
    genomes[:, schema.index_of("offspring_count"), :] = np.float32(2.75)
    buffer = PhenotypeBuffer.from_genomes(genomes, schema)

    serialized = buffer.organism_dict(0, schema)

    assert set(serialized) == {"traits", "diet", "body"}
    assert set(serialized["traits"]) == set(schema.names)
    assert set(serialized["diet"]) == set(DIET_NAMES)
    assert serialized["traits"]["offspring_count"] == pytest.approx(2.75)
    assert isinstance(serialized["traits"]["offspring_count"], float)
    assert serialized["body"]["mass"] > 0.0
    assert serialized["body"]["storage_capacity"] > 0.0
    with pytest.raises(IndexError, match="outside active"):
        buffer.organism_dict(1, schema)


def test_hot_path_update_and_compaction_have_no_python_iteration():
    for method in (PhenotypeBuffer.update, PhenotypeBuffer.compact):
        tree = ast.parse(dedent(inspect.getsource(method)))
        forbidden = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp)
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree)), method


def test_offspring_bodies_resemble_their_parent_more_than_the_population(
    schema: GenomeSchema,
):
    """Morphology is heritable, which is the property that makes it evolvable at all.

    The morphology spike answered this by eye -- offspring visibly looked like their parent --
    and that was the finding that justified building the genome. Here it is asserted: with a
    varied population and each organism mutated once, an offspring's body-shape vector stays
    closer to the body it came from than to the population's average body. If this ever fails,
    mutation has become a random reset and selection has nothing to accumulate.
    """

    rng = np.random.default_rng(20260801)
    count = 400
    parents = schema.founders(count)
    # Spread the population out first, so "closer to your parent than to the mean" is a real
    # claim about inheritance rather than an artefact of everyone starting identical.
    for name in ("body_length", "radius_ratio", "fullness", "taper", "head_size"):
        index = schema.index_of(name)
        low, high = float(schema.low[index]), float(schema.high[index])
        spread = rng.uniform(low, min(high, low + 0.4 * (high - low)), size=count)
        parents[:, index, :] = spread[:, None].astype(np.float32)
    offspring = schema.mutate(parents, rng)

    def shape_vectors(genomes: np.ndarray) -> np.ndarray:
        buffer = PhenotypeBuffer.from_genomes(genomes, schema)
        columns = [schema.index_of(name) for name in MORPHOLOGY_NAMES]
        # Standardised by locus span, so a gene measured in body lengths and one measured in
        # limb pairs contribute comparably rather than the widest locus deciding everything.
        return buffer.traits[:count, columns] / schema.span[columns]

    parent_shapes = shape_vectors(parents)
    child_shapes = shape_vectors(offspring)
    population_mean = parent_shapes.mean(axis=0)

    to_parent = np.linalg.norm(child_shapes - parent_shapes, axis=1)
    to_mean = np.linalg.norm(child_shapes - population_mean, axis=1)

    assert to_parent.mean() < to_mean.mean()
    # Not merely closer on average: closer for the overwhelming majority of individuals.
    assert (to_parent < to_mean).mean() > 0.95
