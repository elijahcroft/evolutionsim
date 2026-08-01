"""Milestone 2 vectorised genome invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from evosim.config import Config
from evosim.life.genome import GenomeSchema
from evosim.rng import RngBundle


@pytest.fixture
def schema() -> GenomeSchema:
    return GenomeSchema(Config.load().genome)


def test_schema_caches_an_immutable_config_view(schema: GenomeSchema):
    config = Config.load().genome

    assert schema.names == tuple(locus.name for locus in config.loci)
    assert schema.n_loci == config.n_loci
    assert schema.index_of("body_length") == config.index_of("body_length")
    assert schema.bounds.shape == (config.n_loci, 2)
    assert schema.low.dtype == np.float32
    assert np.array_equal(schema.init, config.init_array())
    assert np.array_equal(schema.sigma, config.sigma_array())
    assert np.array_equal(schema.span, config.span_array())
    assert np.array_equal(schema.distance_weights, config.distance_weight_array())

    for cached in (
        schema.bounds,
        schema.low,
        schema.high,
        schema.init,
        schema.sigma,
        schema.span,
        schema.distance_weights,
    ):
        assert not cached.flags.writeable
        with pytest.raises(ValueError):
            cached.setflags(write=True)

    with pytest.raises(TypeError):
        schema.indices["new_locus"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        schema.names = ()  # type: ignore[misc]
    with pytest.raises(KeyError, match="no such locus"):
        schema.index_of("not_a_locus")


def test_founders_are_homozygous_float32_copies(schema: GenomeSchema):
    founders = schema.founders(7)

    assert founders.shape == (7, schema.n_loci, 2)
    assert founders.dtype == np.float32
    assert np.array_equal(founders[:, :, 0], founders[:, :, 1])
    assert np.array_equal(founders[0, :, 0], schema.init)

    founders[0, 0, 0] = schema.low[0]
    assert founders[1, 0, 0] == schema.init[0]
    assert schema.init[0] != founders[0, 0, 0]


@pytest.mark.parametrize("bad_count", [-1, 1.5, "2", True])
def test_founders_reject_invalid_counts(schema: GenomeSchema, bad_count):
    with pytest.raises((TypeError, ValueError), match="count"):
        schema.founders(bad_count)


def test_express_is_a_clipped_allele_mean_and_preserves_input(
    schema: GenomeSchema,
):
    genomes = schema.founders(3)
    genomes[0, :, :] = (schema.low - schema.span)[:, None]
    genomes[1, :, :] = (schema.high + schema.span)[:, None]
    genomes[2, :, 0] = schema.low
    genomes[2, :, 1] = schema.high
    original = genomes.copy()

    expressed = schema.express(genomes)

    assert expressed.shape == (3, schema.n_loci)
    assert expressed.dtype == np.float32
    assert np.array_equal(expressed[0], schema.low)
    assert np.array_equal(expressed[1], schema.high)
    assert expressed[2] == pytest.approx((schema.low + schema.high) / 2.0)
    assert np.array_equal(genomes, original)


def test_expression_does_not_overflow_when_large_alleles_have_a_finite_mean():
    config = Config.load().genome
    temperature = config.index_of("temp_optimum")
    loci = list(config.loci)
    loci[temperature] = replace(
        loci[temperature],
        low=2e38,
        high=3e38,
        init=2e38,
        sigma=1e36,
    )
    schema = GenomeSchema(replace(config, loci=tuple(loci)))
    genome = schema.founders(1)
    genome[:, temperature, 0] = schema.low[temperature]
    genome[:, temperature, 1] = schema.high[temperature]

    expressed = schema.express(genome)

    assert np.isfinite(expressed[0, temperature])
    assert expressed[0, temperature] == pytest.approx(2.5e38, rel=1e-6)


def test_genetic_distance_is_weighted_standardised_rms(
    schema: GenomeSchema,
):
    left = schema.founders(2)
    right = schema.founders(2)
    locus = schema.index_of("body_length")
    left[0, locus, :] = schema.low[locus]
    right[0, locus, :] = schema.high[locus]

    distance = schema.genetic_distance(left, right)
    expected = np.sqrt(
        schema.distance_weights[locus] / schema.distance_weights.sum()
    )

    assert distance.dtype == np.float32
    assert distance[0] == pytest.approx(expected)
    assert distance[1] == pytest.approx(0.0)
    assert np.array_equal(distance, schema.genetic_distance(right, left))


def test_genetic_distance_handles_large_finite_weights():
    schema = GenomeSchema(
        Config.load(
            overrides=["genome.distance_weights.default=3e38"]
        ).genome
    )
    low = schema.founders(1)
    high = schema.founders(1)
    low[:, :, :] = schema.low[None, :, None]
    high[:, :, :] = schema.high[None, :, None]

    distance = schema.genetic_distance(low, high)

    assert np.all(np.isfinite(distance))
    assert distance[0] == pytest.approx(1.0)


def test_recombination_is_mendelian_deterministic_and_non_mutating(
    schema: GenomeSchema,
):
    parent_a = schema.founders(64)
    parent_b = schema.founders(64)
    parent_a[:, :, 0] = schema.low
    parent_a[:, :, 1] = schema.high
    parent_b[:, :, 0] = schema.low + schema.span * np.float32(0.25)
    parent_b[:, :, 1] = schema.low + schema.span * np.float32(0.75)
    original_a = parent_a.copy()
    original_b = parent_b.copy()

    offspring = schema.recombine(
        parent_a,
        parent_b,
        RngBundle(73).recombination,
    )
    repeat = schema.recombine(
        parent_a,
        parent_b,
        RngBundle(73).recombination,
    )

    assert offspring.shape == parent_a.shape
    assert offspring.dtype == np.float32
    assert np.array_equal(offspring, repeat)
    assert np.all(
        (offspring[:, :, 0] == parent_a[:, :, 0])
        | (offspring[:, :, 0] == parent_a[:, :, 1])
    )
    assert np.all(
        (offspring[:, :, 1] == parent_b[:, :, 0])
        | (offspring[:, :, 1] == parent_b[:, :, 1])
    )
    assert np.array_equal(parent_a, original_a)
    assert np.array_equal(parent_b, original_b)


def test_mutation_is_deterministic_bounded_and_preserves_parent(
    schema: GenomeSchema,
):
    parent = schema.founders(32)
    mutation_rate = schema.index_of("mutation_rate")
    parent[:, mutation_rate, :] = schema.high[mutation_rate]
    original = parent.copy()

    mutated = schema.mutate(
        parent,
        RngBundle(401).mutation,
        radiation=1.0,
    )
    repeat = schema.mutate(
        parent,
        RngBundle(401).mutation,
        radiation=1.0,
    )
    other_seed = schema.mutate(
        parent,
        RngBundle(402).mutation,
        radiation=1.0,
    )

    assert mutated.shape == parent.shape
    assert mutated.dtype == np.float32
    assert np.array_equal(mutated, repeat)
    assert not np.array_equal(mutated, other_seed)
    assert not np.array_equal(mutated, parent)
    assert np.all(mutated >= schema.low[None, :, None])
    assert np.all(mutated <= schema.high[None, :, None])
    assert np.array_equal(parent, original)


def test_radiation_tolerance_reduces_radiation_induced_mutation(
    schema: GenomeSchema,
):
    count = 5_000
    low_tolerance = schema.founders(count)
    high_tolerance = schema.founders(count)
    midpoint = (schema.low + schema.high) / np.float32(2.0)
    low_tolerance[:, :, :] = midpoint[None, :, None]
    high_tolerance[:, :, :] = midpoint[None, :, None]
    mutation_rate = schema.index_of("mutation_rate")
    tolerance = schema.index_of("radiation_tolerance")
    low_tolerance[:, mutation_rate, :] = np.float32(0.05)
    high_tolerance[:, mutation_rate, :] = np.float32(0.05)
    low_tolerance[:, tolerance, :] = schema.low[tolerance]
    high_tolerance[:, tolerance, :] = schema.high[tolerance]

    low_result = schema.mutate(
        low_tolerance,
        RngBundle(912).mutation,
        radiation=1.0,
    )
    high_result = schema.mutate(
        high_tolerance,
        RngBundle(912).mutation,
        radiation=1.0,
    )
    low_changes = np.count_nonzero(low_result != low_tolerance)
    high_changes = np.count_nonzero(high_result != high_tolerance)

    assert high_changes < low_changes


def test_zero_sigma_remains_stable_under_extreme_large_effect_scale():
    config = Config.load().genome
    body_index = config.index_of("body_length")
    loci = list(config.loci)
    loci[body_index] = replace(loci[body_index], sigma=0.0)
    mutation = replace(
        config.mutation,
        p_large_effect=1.0,
        large_effect_scale=1e100,
    )
    schema = GenomeSchema(
        replace(config, loci=tuple(loci), mutation=mutation)
    )
    parent = schema.founders(1_000)
    mutation_rate = schema.index_of("mutation_rate")
    parent[:, mutation_rate, :] = schema.high[mutation_rate]

    mutated = schema.mutate(parent, RngBundle(88).mutation)

    assert np.all(np.isfinite(mutated))
    assert np.array_equal(mutated[:, body_index, :], parent[:, body_index, :])


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        ("express", "genomes must have shape"),
        ("distance", "right must have shape"),
        ("recombine", "parent_b must have shape"),
        ("mutate", "genomes must have shape"),
    ],
)
def test_genome_operations_reject_wrong_shapes(
    schema: GenomeSchema,
    operation: str,
    match: str,
):
    valid = schema.founders(2)
    wrong = np.zeros((2, schema.n_loci), dtype=np.float32)

    if operation == "express":
        call = lambda: schema.express(wrong)
    elif operation == "distance":
        call = lambda: schema.genetic_distance(valid, wrong)
    elif operation == "recombine":
        call = lambda: schema.recombine(valid, wrong, RngBundle(1).recombination)
    else:
        call = lambda: schema.mutate(wrong, RngBundle(1).mutation)

    with pytest.raises(ValueError, match=match):
        call()


def test_paired_operations_reject_different_batch_counts(schema: GenomeSchema):
    one = schema.founders(1)
    two = schema.founders(2)

    with pytest.raises(ValueError, match="same number"):
        schema.genetic_distance(one, two)
    with pytest.raises(ValueError, match="same number"):
        schema.recombine(one, two, RngBundle(1).recombination)


def test_genome_operations_reject_non_finite_alleles(schema: GenomeSchema):
    genomes = schema.founders(1)
    genomes[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite alleles"):
        schema.express(genomes)
    with pytest.raises(ValueError, match="finite alleles"):
        schema.mutate(genomes, RngBundle(1).mutation)


@pytest.mark.parametrize("bad_radiation", [-0.1, np.inf, np.nan, "high", True])
def test_mutation_rejects_invalid_radiation(
    schema: GenomeSchema,
    bad_radiation,
):
    with pytest.raises((TypeError, ValueError), match="radiation"):
        schema.mutate(
            schema.founders(1),
            RngBundle(1).mutation,
            radiation=bad_radiation,
        )


def test_random_operations_require_an_explicit_generator(schema: GenomeSchema):
    genomes = schema.founders(1)

    with pytest.raises(TypeError, match="Generator"):
        schema.recombine(genomes, genomes, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Generator"):
        schema.mutate(genomes, None)  # type: ignore[arg-type]
