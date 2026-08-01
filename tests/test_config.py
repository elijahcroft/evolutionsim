"""Tests for configuration loading, validation, and overrides."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from evosim.config import (
    DEFAULT_CONFIG_DIR,
    Config,
    ConfigError,
    apply_override,
)


@pytest.fixture(scope="module")
def raw_config() -> dict:
    """The shipped YAML, loaded but not validated -- the input to mutation tests below."""
    raw = {}
    for section, filename in (
        ("sim", "sim.yaml"),
        ("planet", "planet_default.yaml"),
        ("genome", "genome.yaml"),
        ("energy", "energy.yaml"),
    ):
        with (DEFAULT_CONFIG_DIR / filename).open(encoding="utf-8") as fh:
            raw[section] = yaml.safe_load(fh)
    return raw


@pytest.fixture
def raw(raw_config) -> dict:
    return copy.deepcopy(raw_config)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(DEFAULT_CONFIG_DIR)


# ---------------------------------------------------------------------------------------
# the shipped configuration must be valid
# ---------------------------------------------------------------------------------------


def test_default_config_loads(config):
    assert config.planet.name
    assert config.planet.n_cells == config.planet.grid_width * config.planet.grid_height
    assert config.genome.n_loci >= 20
    assert config.sim.initial_population <= config.sim.max_population


def test_fingerprint_is_stable_and_sensitive(raw):
    a = Config.from_raw(raw)
    b = Config.from_raw(copy.deepcopy(raw))
    assert a.fingerprint() == b.fingerprint()

    changed = copy.deepcopy(raw)
    changed["planet"]["gravity"] = 1.4
    assert Config.from_raw(changed).fingerprint() != a.fingerprint()


def test_seed_is_part_of_the_fingerprint(raw):
    """A run is identified by (config, seed), so the seed must be inside the fingerprint."""
    baseline = Config.from_raw(raw).fingerprint()
    changed = copy.deepcopy(raw)
    changed["sim"]["seed"] += 1
    assert Config.from_raw(changed).fingerprint() != baseline


# ---------------------------------------------------------------------------------------
# genome structure
# ---------------------------------------------------------------------------------------


def test_expected_loci_are_present(config):
    """The loci the design commits to; a rename should break tests loudly, not silently."""
    required = {
        "body_length", "radius_ratio", "fullness", "limb_pairs", "head_size",
        "metabolic_rate", "temp_optimum", "temp_tolerance",
        "aff_autotroph", "aff_detritus", "aff_herbivore", "aff_carnivore",
        "digestion_efficiency", "move_speed", "sense_range", "maturity_age",
        "repro_threshold", "offspring_count", "parental_investment", "sex_bias",
        "camouflage", "armor", "aggression", "mutation_rate",
    }
    missing = required - {locus.name for locus in config.genome.loci}
    assert not missing, missing


def test_locus_arrays_are_consistent(config):
    g = config.genome
    low, high, init, sigma = g.low_array(), g.high_array(), g.init_array(), g.sigma_array()
    assert low.shape == high.shape == init.shape == sigma.shape == (g.n_loci,)
    assert low.dtype == np.float32
    assert np.all(high > low)
    assert np.all((init >= low) & (init <= high))
    assert np.all(sigma > 0), "a locus with zero sigma can never evolve"
    assert np.allclose(g.span_array(), high - low)


def test_index_of_matches_array_order(config):
    g = config.genome
    for i, locus in enumerate(g.loci):
        assert g.index_of(locus.name) == i
        assert locus.name in g
    with pytest.raises(KeyError):
        g.index_of("not_a_locus")


def test_diet_loci_seed_a_mostly_autotrophic_founder(config):
    """The design seeds one autotroph-ish lineage and requires heterotrophy to evolve.

    If the founder already had strong heterotroph affinity, "predation emerged" would be a
    fiction, so assert the starting point explicitly.
    """
    g = config.genome
    logits = np.array(
        [g.loci[g.index_of(n)].init for n in
         ("aff_autotroph", "aff_detritus", "aff_herbivore", "aff_carnivore")]
    )
    fractions = np.exp(logits - logits.max())
    fractions /= fractions.sum()
    assert fractions[0] > 0.5, f"founder is not autotroph-dominant: {fractions}"
    assert fractions[2] < 0.15 and fractions[3] < 0.15, f"founder starts predatory: {fractions}"


def test_distance_weights_resolve_with_overrides(config):
    g = config.genome
    weights = g.distance_weight_array()
    assert weights.shape == (g.n_loci,)
    assert weights.sum() > 0
    # mutation_rate is explicitly excluded from the species concept in genome.yaml.
    assert weights[g.index_of("mutation_rate")] == 0.0


def test_zero_total_distance_weight_is_rejected(raw):
    raw["genome"]["distance_weights"]["default"] = 0.0
    raw["genome"]["distance_weights"]["overrides"] = {}
    with pytest.raises(ConfigError, match="total weight is zero"):
        Config.from_raw(raw)


def test_distance_weight_override_for_unknown_locus_is_rejected(raw):
    raw["genome"]["distance_weights"]["overrides"]["no_such_locus"] = 0.5
    with pytest.raises(ConfigError, match="unknown locus name"):
        Config.from_raw(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100, 1e-100])
def test_distance_weights_must_be_finite_float32_values(raw, value):
    raw["genome"]["distance_weights"]["overrides"]["move_persistence"] = value

    with pytest.raises(ConfigError, match="finite|representable"):
        Config.from_raw(raw)


def test_duplicate_locus_is_rejected(raw):
    raw["genome"]["loci"].append(copy.deepcopy(raw["genome"]["loci"][0]))
    with pytest.raises(ConfigError, match="duplicate locus name"):
        Config.from_raw(raw)


def test_init_outside_locus_range_is_rejected(raw):
    raw["genome"]["loci"][0]["init"] = raw["genome"]["loci"][0]["high"] + 1
    with pytest.raises(ConfigError, match="must lie within"):
        Config.from_raw(raw)


def test_inverted_locus_range_is_rejected(raw):
    locus = raw["genome"]["loci"][0]
    locus["low"], locus["high"] = locus["high"], locus["low"]
    with pytest.raises(ConfigError, match="must exceed low"):
        Config.from_raw(raw)


def test_oversized_sigma_is_rejected(raw):
    """A mutation as large as the locus span destroys heritability; that must not load."""
    locus = raw["genome"]["loci"][0]
    locus["sigma"] = (locus["high"] - locus["low"]) * 0.5
    with pytest.raises(ConfigError, match="swamp inheritance"):
        Config.from_raw(raw)


@pytest.mark.parametrize(
    ("low", "high", "init", "sigma"),
    [
        (0.0, 1e100, 0.4, 0.05),
        (0.0, 1e-100, 0.0, 0.0),
        (-3e38, 3e38, 0.0, 1.0),
    ],
)
def test_locus_values_must_fit_float32_storage(raw, low, high, init, sigma):
    locus = raw["genome"]["loci"][0]
    locus.update(low=low, high=high, init=init, sigma=sigma)

    with pytest.raises(ConfigError, match="float32"):
        Config.from_raw(raw)


def test_locus_interval_must_be_consistently_resolvable_in_float32(raw):
    locus = next(
        item for item in raw["genome"]["loci"] if item["name"] == "temp_optimum"
    )
    locus.update(
        low=-2729808256.000004,
        high=-2729808255.9996295,
        init=-2729808255.9998,
        sigma=1e-5,
    )

    with pytest.raises(ConfigError, match="resolvable consistently in float32"):
        Config.from_raw(raw)


@pytest.mark.parametrize(
    ("name", "low", "match"),
    [
        ("body_length", -1.0, "body_length.low must be >= 0"),
        ("energy_storage", -1.0, "energy_storage.low must be >= 0"),
        ("radiation_tolerance", -1.0, "radiation_tolerance.low must be >= 0"),
        ("radius_ratio", 0.0, "radius_ratio.low must be > 0"),
    ],
)
def test_body_locus_bounds_must_support_valid_geometry(raw, name, low, match):
    locus = next(item for item in raw["genome"]["loci"] if item["name"] == name)
    locus["low"] = low

    with pytest.raises(ConfigError, match=match):
        Config.from_raw(raw)


def test_body_locus_bounds_must_keep_derived_geometry_finite(raw):
    body = next(
        item for item in raw["genome"]["loci"] if item["name"] == "body_length"
    )
    body["high"] = 1e20

    with pytest.raises(ConfigError, match="body volume outside finite float32"):
        Config.from_raw(raw)


def test_body_geometry_validation_uses_joint_float32_arithmetic(raw):
    """Bounds that are individually representable can still overflow when multiplied.

    Volume goes as length cubed, so a body_length whose cube is finite in float32 is fine on its
    own; it is the storage capacity built on top of it that overflows. The validation has to do
    the arithmetic, not check the inputs one at a time.
    """

    body = next(
        item for item in raw["genome"]["loci"] if item["name"] == "body_length"
    )
    storage = next(
        item for item in raw["genome"]["loci"] if item["name"] == "energy_storage"
    )
    body["high"] = float(np.finfo(np.float32).max) ** (1.0 / 3.0)
    storage.update(low=0.0, high=0.1, init=0.05, sigma=0.001)

    with pytest.raises(ConfigError, match="outside finite float32"):
        Config.from_raw(raw)


def test_mutation_rate_bounds_must_be_probabilities(raw):
    locus = next(
        item for item in raw["genome"]["loci"] if item["name"] == "mutation_rate"
    )
    locus["high"] = 1.5

    with pytest.raises(ConfigError, match=r"within \[0, 1\]"):
        Config.from_raw(raw)


# ---------------------------------------------------------------------------------------
# validation: typos and out-of-range values must fail loudly
# ---------------------------------------------------------------------------------------


def test_unknown_key_is_rejected_with_its_path(raw):
    raw["energy"]["costs"]["k_suport"] = 0.01
    with pytest.raises(ConfigError, match=r"energy\.costs: unrecognised key\(s\): k_suport"):
        Config.from_raw(raw)


def test_unknown_top_level_section_is_rejected(raw):
    raw["weather"] = {}
    with pytest.raises(ConfigError, match="unrecognised config section"):
        Config.from_raw(raw)


def test_missing_section_is_rejected(raw):
    del raw["energy"]
    with pytest.raises(ConfigError, match="missing config section"):
        Config.from_raw(raw)


def test_missing_required_key_is_rejected(raw):
    del raw["planet"]["gravity"]
    with pytest.raises(ConfigError, match=r"planet\.gravity: required key is missing"):
        Config.from_raw(raw)


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("planet", "gravity", 0.0, "must be >="),
        ("planet", "gravity", 50.0, "must be <="),
        ("planet", "o2_fraction", 1.5, "must be <="),
        ("planet", "axial_tilt", -5.0, "must be >="),
        ("planet", "grid_width", 2, "must be >="),
        ("sim", "max_population", 0, "must be >="),
        ("sim", "taxonomy_interval", 0, "must be >="),
    ],
)
def test_out_of_range_values_are_rejected(raw, section, key, value, match):
    raw[section][key] = value
    with pytest.raises(ConfigError, match=match):
        Config.from_raw(raw)


def test_wrong_type_is_rejected(raw):
    raw["planet"]["gravity"] = "heavy"
    with pytest.raises(ConfigError, match="expected a number"):
        Config.from_raw(raw)


def test_non_integer_where_integer_required_is_rejected(raw):
    raw["planet"]["grid_width"] = 128.5
    with pytest.raises(ConfigError, match="expected an integer"):
        Config.from_raw(raw)


def test_bad_enum_choice_is_rejected(raw):
    raw["sim"]["initial_habitat"] = "lava"
    with pytest.raises(ConfigError, match="must be one of"):
        Config.from_raw(raw)


def test_nan_is_rejected(raw):
    raw["planet"]["gravity"] = float("nan")
    with pytest.raises(ConfigError, match="must be finite"):
        Config.from_raw(raw)


def test_founders_exceeding_capacity_is_rejected(raw):
    raw["sim"]["initial_population"] = raw["sim"]["max_population"] + 1
    with pytest.raises(ConfigError, match="exceeds"):
        Config.from_raw(raw)


def test_reproduction_overhead_below_one_is_rejected(raw):
    """Overhead < 1 would create energy at every birth, breaking the energy ledger."""
    raw["energy"]["reproduction"]["overhead"] = 0.9
    with pytest.raises(ConfigError, match="must be >="):
        Config.from_raw(raw)


# ---------------------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------------------


def test_override_scalar_preserves_type(raw):
    apply_override(raw, "planet.gravity=1.4")
    apply_override(raw, "sim.seed=7")
    assert raw["planet"]["gravity"] == pytest.approx(1.4)
    assert raw["sim"]["seed"] == 7 and isinstance(raw["sim"]["seed"], int)


def test_override_integer_written_as_int_stays_int(raw):
    apply_override(raw, "planet.year_length_days=100")
    cfg = Config.from_raw(raw)
    assert cfg.planet.year_length_days == 100
    assert isinstance(cfg.planet.year_length_days, int)


def test_override_float_accepts_integer_text(raw):
    apply_override(raw, "planet.gravity=2")
    assert isinstance(raw["planet"]["gravity"], float)


def test_override_nested_section(raw):
    apply_override(raw, "planet.terrain.land_fraction=0.7")
    assert Config.from_raw(raw).planet.terrain.land_fraction == pytest.approx(0.7)


def test_override_locus_by_name(raw):
    """Sweeps need to address loci by name; `genome.loci[3].sigma` would be unreadable."""
    apply_override(raw, "genome.loci.body_length.sigma=0.123")
    cfg = Config.from_raw(raw)
    assert cfg.genome.loci[cfg.genome.index_of("body_length")].sigma == pytest.approx(0.123)


def test_override_string_value(raw):
    apply_override(raw, "sim.initial_habitat=land")
    assert Config.from_raw(raw).sim.initial_habitat == "land"


@pytest.mark.parametrize(
    "override,match",
    [
        ("planet.gravity", "expected the form"),
        ("planet.gravitas=1.0", "no such config key"),
        ("nonexistent.thing=1", "no such config key"),
        ("planet.gravity=heavy", "expected a number"),
        ("planet.year_length_days=1.5", "expected an integer"),
        ("genome.loci.no_such_locus.sigma=1", "no list entry named"),
        ("=1.0", "empty key path"),
        ("planet.gravity.deeper=1", "not a section"),
    ],
)
def test_bad_overrides_are_rejected(raw, override, match):
    with pytest.raises(ConfigError, match=match):
        apply_override(raw, override)


def test_override_via_load(tmp_path, raw_config):
    """End-to-end: the path the sweep tool and biology tests will use."""
    for section, filename in (
        ("sim", "sim.yaml"),
        ("planet", "planet_default.yaml"),
        ("genome", "genome.yaml"),
        ("energy", "energy.yaml"),
    ):
        (tmp_path / filename).write_text(yaml.safe_dump(raw_config[section]), encoding="utf-8")

    cfg = Config.load(tmp_path, overrides=["planet.gravity=1.4", "sim.seed=999"])
    assert cfg.planet.gravity == pytest.approx(1.4)
    assert cfg.sim.seed == 999
    assert cfg.fingerprint() != Config.load(tmp_path).fingerprint()


def test_missing_config_dir_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="config directory not found"):
        Config.load(tmp_path / "nope")


def test_missing_config_file_is_reported(tmp_path):
    (tmp_path / "sim.yaml").write_text("seed: 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="config file not found"):
        Config.load(tmp_path)


def test_alternative_planet_file(tmp_path, raw_config):
    for section, filename in (
        ("sim", "sim.yaml"),
        ("planet", "planet_default.yaml"),
        ("genome", "genome.yaml"),
        ("energy", "energy.yaml"),
    ):
        (tmp_path / filename).write_text(yaml.safe_dump(raw_config[section]), encoding="utf-8")
    heavy = copy.deepcopy(raw_config["planet"])
    heavy["name"] = "Heavy"
    heavy["gravity"] = 2.5
    (tmp_path / "heavy.yaml").write_text(yaml.safe_dump(heavy), encoding="utf-8")

    cfg = Config.load(tmp_path, planet_file="heavy.yaml")
    assert cfg.planet.name == "Heavy"
    assert cfg.planet.gravity == pytest.approx(2.5)


def test_empty_config_file_is_reported(tmp_path, raw_config):
    for section, filename in (
        ("sim", "sim.yaml"),
        ("planet", "planet_default.yaml"),
        ("genome", "genome.yaml"),
        ("energy", "energy.yaml"),
    ):
        (tmp_path / filename).write_text(yaml.safe_dump(raw_config[section]), encoding="utf-8")
    (tmp_path / "energy.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="file is empty"):
        Config.load(tmp_path)


def test_config_objects_are_frozen(config):
    with pytest.raises(Exception):
        config.planet.gravity = 2.0  # type: ignore[misc]
