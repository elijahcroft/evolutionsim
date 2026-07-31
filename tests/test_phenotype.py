"""Milestone 2 vectorised phenotype and scalar-body geometry tests."""

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
    assert buffer.mass.shape == buffer.storage_capacity.shape == (7,)
    assert buffer.traits.dtype == np.float32
    assert buffer.diet.dtype == np.float32
    assert buffer.mass.dtype == np.float32
    assert buffer.storage_capacity.dtype == np.float32
    assert buffer.memory_bytes == sum(
        array.nbytes
        for array in (
            buffer.traits,
            buffer.diet,
            buffer.mass,
            buffer.storage_capacity,
        )
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


def test_body_geometry_uses_the_documented_m2_interpretation(
    schema: GenomeSchema,
):
    genomes = schema.founders(3)
    body_size = np.array([0.2, 0.4, 0.8], dtype=np.float32)
    energy_storage = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    slenderness = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    genomes[:, schema.index_of("body_size"), :] = body_size[:, None]
    genomes[:, schema.index_of("energy_storage"), :] = energy_storage[:, None]
    genomes[:, schema.index_of("body_slenderness"), :] = slenderness[:, None]

    phenotype = PhenotypeBuffer.from_genomes(genomes, schema)
    expected_mass = body_size**3
    expected_storage = expected_mass * energy_storage / slenderness

    assert phenotype.mass[:3] == pytest.approx(expected_mass)
    assert phenotype.storage_capacity[:3] == pytest.approx(expected_storage)


def test_contiguous_updates_can_overwrite_or_extend_but_not_leave_gaps(
    schema: GenomeSchema,
):
    buffer = PhenotypeBuffer.allocate(4, schema)
    first = schema.founders(2)
    first[:, schema.index_of("body_size"), :] = np.array(
        [[0.2], [0.4]], dtype=np.float32
    )
    buffer.update(0, first, schema)

    replacement = schema.founders(2)
    replacement[:, schema.index_of("body_size"), :] = np.array(
        [[0.8], [1.0]], dtype=np.float32
    )
    updated = buffer.update(1, replacement, schema)

    assert buffer.size == 3
    assert updated.size == 2
    assert buffer.trait("body_size") == pytest.approx([0.2, 0.8, 1.0])
    assert np.shares_memory(buffer.active.traits, buffer.traits)
    assert np.shares_memory(buffer.active.trait("body_size"), buffer.traits)

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
    values = np.array([0.2, 0.4, 0.8, 1.6], dtype=np.float32)
    genomes[:, schema.index_of("body_size"), :] = values[:, None]
    buffer = PhenotypeBuffer.from_genomes(genomes, schema, capacity=6)
    old_diet = buffer.diet[:4].copy()
    old_mass = buffer.mass[:4].copy()

    active = buffer.compact(np.array([0, 2, 3]), old_size=4)

    assert buffer.size == active.size == 3
    assert buffer.trait("body_size") == pytest.approx(values[[0, 2, 3]])
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

    assert np.shares_memory(buffer.trait("body_size"), buffer.traits)
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
