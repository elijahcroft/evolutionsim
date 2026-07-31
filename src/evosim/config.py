"""Configuration loading and validation.

Design rules this module exists to enforce:

* **No magic numbers in simulation code.** Every tunable lives in `config/*.yaml` and arrives
  here as a validated, frozen dataclass. If a simulation module needs a number, it reads it
  from a config object.

* **Typos must fail loudly.** Unknown keys are rejected rather than ignored. A silently
  ignored `k_suport` would be hours of confused tuning, so every reader checks for leftover
  keys and reports the full dotted path.

* **Overrides are first-class.** Parameter sweeps and the directional-selection tests need to
  vary one value at a time from the command line (`--set planet.gravity=1.4`). That path is
  supported and type-checked here, not bolted on later.

* **A run is identified by (config, seed).** `Config.fingerprint()` hashes the fully-resolved
  configuration so a recorded run can prove what produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

CONFIG_FILES = {
    "sim": "sim.yaml",
    "planet": "planet_default.yaml",
    "genome": "genome.yaml",
    "energy": "energy.yaml",
}


class ConfigError(ValueError):
    """Raised for any malformed, out-of-range, missing, or unrecognised configuration value."""


class _Missing:
    """Marks 'no default supplied', so that `None` remains a usable default value."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def _typename(value: Any) -> str:
    return type(value).__name__


# ---------------------------------------------------------------------------------------
# reader helper
# ---------------------------------------------------------------------------------------


class _Reader:
    """Consumes keys from a mapping, validating as it goes and rejecting leftovers.

    Every accessor records the dotted path so error messages point at the exact YAML key.
    """

    def __init__(self, data: Any, path: str) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path or '<root>'}: expected a mapping, got {_typename(data)}")
        self._data = dict(data)
        self._path = path

    def _at(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _take(self, key: str, default: Any) -> Any:
        if key in self._data:
            return self._data.pop(key)
        if isinstance(default, _Missing):
            raise ConfigError(f"{self._at(key)}: required key is missing")
        return default

    def num(
        self,
        key: str,
        *,
        default: Any = _MISSING,
        low: float | None = None,
        high: float | None = None,
    ) -> float:
        raw = self._take(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ConfigError(f"{self._at(key)}: expected a number, got {_typename(raw)}")
        value = float(raw)
        if not np.isfinite(value):
            raise ConfigError(f"{self._at(key)}: must be finite, got {value}")
        if low is not None and value < low:
            raise ConfigError(f"{self._at(key)}: must be >= {low}, got {value}")
        if high is not None and value > high:
            raise ConfigError(f"{self._at(key)}: must be <= {high}, got {value}")
        return value

    def integer(
        self,
        key: str,
        *,
        default: Any = _MISSING,
        low: int | None = None,
        high: int | None = None,
    ) -> int:
        raw = self._take(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ConfigError(f"{self._at(key)}: expected an integer, got {_typename(raw)}")
        value = int(raw)
        if low is not None and value < low:
            raise ConfigError(f"{self._at(key)}: must be >= {low}, got {value}")
        if high is not None and value > high:
            raise ConfigError(f"{self._at(key)}: must be <= {high}, got {value}")
        return value

    def text(self, key: str, *, default: Any = _MISSING,
             choices: Sequence[str] | None = None) -> str:
        raw = self._take(key, default)
        if not isinstance(raw, str):
            raise ConfigError(f"{self._at(key)}: expected a string, got {_typename(raw)}")
        if choices is not None and raw not in choices:
            raise ConfigError(
                f"{self._at(key)}: must be one of {sorted(choices)}, got {raw!r}"
            )
        return raw

    def section(self, key: str) -> _Reader:
        raw = self._take(key, _MISSING)
        return _Reader(raw, self._at(key))

    def raw_list(self, key: str) -> list[Any]:
        raw = self._take(key, _MISSING)
        if not isinstance(raw, list):
            raise ConfigError(f"{self._at(key)}: expected a list, got {_typename(raw)}")
        return raw

    def raw_mapping(self, key: str, *, default: Any = _MISSING) -> dict[str, Any]:
        raw = self._take(key, default)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{self._at(key)}: expected a mapping, got {_typename(raw)}")
        return dict(raw)

    def ignore(self, *keys: str) -> None:
        """Accept and discard purely documentary keys (e.g. a locus `doc:` string)."""
        for key in keys:
            self._data.pop(key, None)

    def done(self) -> None:
        if self._data:
            where = self._path or "<root>"
            unknown = ", ".join(sorted(self._data))
            raise ConfigError(f"{where}: unrecognised key(s): {unknown}")


# ---------------------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimConfig:
    seed: int
    max_population: int
    initial_population: int
    initial_habitat: str
    taxonomy_interval: int
    sample_interval: int
    extinction_confirm_ticks: int
    snapshot_interval: int

    @classmethod
    def from_mapping(cls, data: Any) -> SimConfig:
        r = _Reader(data, "sim")
        cfg = cls(
            seed=r.integer("seed", low=0),
            max_population=r.integer("max_population", low=1),
            initial_population=r.integer("initial_population", low=1),
            initial_habitat=r.text("initial_habitat", choices=("water", "land", "any")),
            taxonomy_interval=r.integer("taxonomy_interval", low=1),
            sample_interval=r.integer("sample_interval", low=1),
            extinction_confirm_ticks=r.integer("extinction_confirm_ticks", low=1),
            snapshot_interval=r.integer("snapshot_interval", low=0),
        )
        r.done()
        if cfg.initial_population > cfg.max_population:
            raise ConfigError(
                f"sim.initial_population ({cfg.initial_population}) exceeds "
                f"sim.max_population ({cfg.max_population})"
            )
        return cfg


# ---------------------------------------------------------------------------------------
# planet
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TerrainConfig:
    sea_level: float
    octaves: int
    base_frequency: float
    persistence: float
    lacunarity: float
    land_fraction: float
    max_elevation_km: float

    @classmethod
    def from_reader(cls, r: _Reader) -> TerrainConfig:
        cfg = cls(
            sea_level=r.num("sea_level"),
            octaves=r.integer("octaves", low=1, high=12),
            base_frequency=r.num("base_frequency", low=0.1),
            persistence=r.num("persistence", low=0.0, high=1.0),
            lacunarity=r.num("lacunarity", low=1.0),
            land_fraction=r.num("land_fraction", low=0.0, high=1.0),
            max_elevation_km=r.num("max_elevation_km", low=0.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class ClimateConfig:
    base_temperature_c: float
    insolation_amplitude_c: float
    lapse_rate_c_per_km: float
    inertia_land: float
    inertia_water: float
    moisture_ocean: float
    moisture_decay_per_cell: float
    moisture_min: float

    @classmethod
    def from_reader(cls, r: _Reader) -> ClimateConfig:
        cfg = cls(
            base_temperature_c=r.num("base_temperature_c"),
            insolation_amplitude_c=r.num("insolation_amplitude_c", low=0.0),
            lapse_rate_c_per_km=r.num("lapse_rate_c_per_km", low=0.0),
            # Inertia is a per-tick relaxation fraction; 0 would freeze climate forever.
            inertia_land=r.num("inertia_land", low=1e-6, high=1.0),
            inertia_water=r.num("inertia_water", low=1e-6, high=1.0),
            moisture_ocean=r.num("moisture_ocean", low=0.0, high=1.0),
            moisture_decay_per_cell=r.num("moisture_decay_per_cell", low=0.0, high=1.0),
            moisture_min=r.num("moisture_min", low=0.0, high=1.0),
        )
        r.done()
        if cfg.moisture_min > cfg.moisture_ocean:
            raise ConfigError(
                "planet.climate.moisture_min must not exceed climate.moisture_ocean"
            )
        return cfg


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    nutrient_capacity_land: float
    nutrient_capacity_water: float
    nutrient_regen_land: float
    nutrient_regen_water: float
    upwelling_bonus: float
    detritus_decay_rate: float
    detritus_decay_q10: float
    detritus_reference_temp_c: float

    @classmethod
    def from_reader(cls, r: _Reader) -> ResourceConfig:
        cfg = cls(
            nutrient_capacity_land=r.num("nutrient_capacity_land", low=0.0),
            nutrient_capacity_water=r.num("nutrient_capacity_water", low=0.0),
            nutrient_regen_land=r.num("nutrient_regen_land", low=0.0, high=1.0),
            nutrient_regen_water=r.num("nutrient_regen_water", low=0.0, high=1.0),
            upwelling_bonus=r.num("upwelling_bonus", low=0.0),
            detritus_decay_rate=r.num("detritus_decay_rate", low=0.0, high=1.0),
            detritus_decay_q10=r.num("detritus_decay_q10", low=1.0),
            detritus_reference_temp_c=r.num("detritus_reference_temp_c"),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class PlanetConfig:
    name: str
    grid_width: int
    grid_height: int
    gravity: float
    o2_fraction: float
    o2_reference: float
    pressure: float
    solar_constant: float
    radiation: float
    axial_tilt: float
    day_length_hours: float
    year_length_days: int
    terrain: TerrainConfig
    climate: ClimateConfig
    resources: ResourceConfig
    base_toxicity: float
    toxicity_elevation_coupling: float

    @property
    def n_cells(self) -> int:
        return self.grid_width * self.grid_height

    @classmethod
    def from_mapping(cls, data: Any) -> PlanetConfig:
        r = _Reader(data, "planet")
        cfg = cls(
            name=r.text("name"),
            # Width must be even-ish for nothing in particular, but a 2-cell world makes
            # neighbourhood logic degenerate, so require a usable minimum.
            grid_width=r.integer("grid_width", low=8, high=2048),
            grid_height=r.integer("grid_height", low=4, high=1024),
            gravity=r.num("gravity", low=0.01, high=10.0),
            o2_fraction=r.num("o2_fraction", low=0.0, high=1.0),
            o2_reference=r.num("o2_reference", low=1e-4, high=1.0),
            pressure=r.num("pressure", low=1e-3, high=100.0),
            solar_constant=r.num("solar_constant", low=0.0, high=10.0),
            radiation=r.num("radiation", low=0.0, high=100.0),
            axial_tilt=r.num("axial_tilt", low=0.0, high=90.0),
            day_length_hours=r.num("day_length_hours", low=0.1, high=10000.0),
            year_length_days=r.integer("year_length_days", low=1),
            terrain=TerrainConfig.from_reader(r.section("terrain")),
            climate=ClimateConfig.from_reader(r.section("climate")),
            resources=ResourceConfig.from_reader(r.section("resources")),
            base_toxicity=r.num("base_toxicity", low=0.0),
            toxicity_elevation_coupling=r.num("toxicity_elevation_coupling", low=0.0),
        )
        r.done()
        return cfg


# ---------------------------------------------------------------------------------------
# genome
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocusSpec:
    name: str
    low: float
    high: float
    init: float
    sigma: float
    doc: str = ""

    @property
    def span(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class MutationConfig:
    p_large_effect: float
    large_effect_scale: float
    radiation_sensitivity: float

    @classmethod
    def from_reader(cls, r: _Reader) -> MutationConfig:
        cfg = cls(
            p_large_effect=r.num("p_large_effect", low=0.0, high=1.0),
            large_effect_scale=r.num("large_effect_scale", low=1.0),
            radiation_sensitivity=r.num("radiation_sensitivity", low=0.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class GenomeConfig:
    loci: tuple[LocusSpec, ...]
    mutation: MutationConfig
    # Weight of each locus in the standardised genetic-distance metric used for mate
    # compatibility and the speciation split test. Weight 0 excludes a locus from the species
    # concept without excluding it from evolution.
    distance_weight: tuple[float, ...]
    _index: dict[str, int] = field(repr=False, compare=False, default_factory=dict)

    @property
    def n_loci(self) -> int:
        return len(self.loci)

    def index_of(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"no such locus: {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._index

    # Vectorised views, built once. The genome and phenotype modules operate on these rather
    # than iterating LocusSpec objects, per the no-Python-loops-in-the-hot-path rule.
    def low_array(self) -> np.ndarray:
        return np.array([l.low for l in self.loci], dtype=np.float32)

    def high_array(self) -> np.ndarray:
        return np.array([l.high for l in self.loci], dtype=np.float32)

    def init_array(self) -> np.ndarray:
        return np.array([l.init for l in self.loci], dtype=np.float32)

    def sigma_array(self) -> np.ndarray:
        return np.array([l.sigma for l in self.loci], dtype=np.float32)

    def span_array(self) -> np.ndarray:
        return np.array([l.span for l in self.loci], dtype=np.float32)

    def distance_weight_array(self) -> np.ndarray:
        return np.array(self.distance_weight, dtype=np.float32)

    @classmethod
    def from_mapping(cls, data: Any) -> GenomeConfig:
        r = _Reader(data, "genome")
        mutation = MutationConfig.from_reader(r.section("mutation"))
        weights_reader = r.section("distance_weights")
        default_weight = weights_reader.num("default", low=0.0)
        weight_overrides = weights_reader.raw_mapping("overrides", default={})
        weights_reader.done()

        raw_loci = r.raw_list("loci")
        r.done()

        if not raw_loci:
            raise ConfigError("genome.loci: must define at least one locus")

        loci: list[LocusSpec] = []
        seen: set[str] = set()
        for i, entry in enumerate(raw_loci):
            lr = _Reader(entry, f"genome.loci[{i}]")
            name = lr.text("name")
            if name in seen:
                raise ConfigError(f"genome.loci[{i}]: duplicate locus name {name!r}")
            seen.add(name)
            spec = LocusSpec(
                name=name,
                low=lr.num("low"),
                high=lr.num("high"),
                init=lr.num("init"),
                sigma=lr.num("sigma", low=0.0),
                doc=lr.text("doc", default=""),
            )
            lr.done()
            if spec.high <= spec.low:
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): high ({spec.high}) must exceed low ({spec.low})"
                )
            if not (spec.low <= spec.init <= spec.high):
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): init ({spec.init}) must lie within "
                    f"[{spec.low}, {spec.high}]"
                )
            # A sigma comparable to the locus span would make mutation a random reset rather
            # than a perturbation, destroying heritability. Catch it here, not after a
            # confusing run.
            if spec.sigma > 0.25 * spec.span:
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): sigma ({spec.sigma}) exceeds a quarter of the "
                    f"locus span ({spec.span}); mutation would swamp inheritance"
                )
            values = np.asarray(
                [spec.low, spec.high, spec.init, spec.sigma],
                dtype=np.float64,
            )
            with np.errstate(over="ignore", under="ignore"):
                stored = values.astype(np.float32)
                stored_span = np.float32(spec.span)
            if (
                np.any(~np.isfinite(stored))
                or not np.isfinite(stored_span)
                or np.any((values != 0.0) & (stored == 0.0))
            ):
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): bounds, init, sigma, and span must be "
                    "representable as finite float32 values"
                )
            if stored[1] <= stored[0] or stored_span <= 0.0:
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): low and high collapse to the same "
                    "float32 value"
                )
            stored_bound_span = float(stored[1]) - float(stored[0])
            if not np.isclose(
                stored_bound_span,
                float(stored_span),
                rtol=4.0 * float(np.finfo(np.float32).eps),
                atol=float(np.finfo(np.float32).smallest_subnormal),
            ):
                raise ConfigError(
                    f"genome.loci[{i}] ({name}): low/high interval is not resolvable "
                    "consistently in float32"
                )
            loci.append(spec)

        loci_by_name = {spec.name: spec for spec in loci}
        for name in ("body_size", "energy_storage", "radiation_tolerance"):
            spec = loci_by_name.get(name)
            if spec is not None and spec.low < 0.0:
                raise ConfigError(f"genome.loci.{name}.low must be >= 0")
        slenderness = loci_by_name.get("body_slenderness")
        if slenderness is not None and slenderness.low <= 0.0:
            raise ConfigError("genome.loci.body_slenderness.low must be > 0")
        mutation_rate = loci_by_name.get("mutation_rate")
        if mutation_rate is not None and (
            mutation_rate.low < 0.0 or mutation_rate.high > 1.0
        ):
            raise ConfigError(
                "genome.loci.mutation_rate bounds must lie within [0, 1] because it is "
                "a probability"
            )
        body_size = loci_by_name.get("body_size")
        energy_storage = loci_by_name.get("energy_storage")
        if body_size is not None:
            body_high = np.float32(body_size.high)
            with np.errstate(over="ignore", invalid="ignore"):
                maximum_mass = np.float32(body_high * body_high * body_high)
            if not np.isfinite(maximum_mass):
                raise ConfigError(
                    "genome.loci.body_size.high produces body mass outside finite "
                    "float32 range"
                )
            if energy_storage is not None and slenderness is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    maximum_storage = np.float32(
                        maximum_mass
                        * np.float32(energy_storage.high)
                        / np.float32(slenderness.low)
                    )
                if not np.isfinite(maximum_storage):
                    raise ConfigError(
                        "body_size, energy_storage, and body_slenderness bounds produce "
                        "storage capacity outside finite float32 range"
                    )

        unknown = set(weight_overrides) - seen
        if unknown:
            raise ConfigError(
                "genome.distance_weights.overrides: unknown locus name(s): "
                + ", ".join(sorted(unknown))
            )

        weights: list[float] = []
        for spec in loci:
            raw = weight_overrides.get(spec.name, default_weight)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ConfigError(
                    f"genome.distance_weights.overrides.{spec.name}: expected a number, "
                    f"got {_typename(raw)}"
                )
            value = float(raw)
            if not np.isfinite(value):
                raise ConfigError(
                    f"genome.distance_weights.overrides.{spec.name}: must be finite, "
                    f"got {value}"
                )
            if value < 0:
                raise ConfigError(
                    f"genome.distance_weights.overrides.{spec.name}: must be >= 0, got {value}"
                )
            weights.append(value)

        with np.errstate(over="ignore", under="ignore"):
            float32_weights = np.asarray(weights, dtype=np.float32)
        if np.any(~np.isfinite(float32_weights)) or np.any(
            (np.asarray(weights) > 0.0) & (float32_weights == 0.0)
        ):
            raise ConfigError(
                "genome.distance_weights: every positive weight must be representable "
                "as a finite, non-zero float32 value"
            )
        if float32_weights.sum(dtype=np.float64) <= 0:
            raise ConfigError(
                "genome.distance_weights: total weight is zero, so genetic distance would "
                "always be zero and speciation could never occur"
            )

        return cls(
            loci=tuple(loci),
            mutation=mutation,
            distance_weight=tuple(weights),
            _index={spec.name: i for i, spec in enumerate(loci)},
        )


# ---------------------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------------------
# These three dataclasses are deliberately flat mirrors of energy.yaml. They are consumed by
# milestone 3+ modules; validating them now means a tuning typo fails at load rather than
# thousands of ticks into a run.


@dataclass(frozen=True, slots=True)
class CostConfig:
    k_basal: float
    basal_mass_exponent: float
    basal_q10: float
    basal_reference_temp_c: float
    upkeep_temp_tolerance: float
    upkeep_digestion: float
    upkeep_radiation_tolerance: float
    upkeep_longevity: float
    k_support: float
    k_move: float
    move_gravity_exponent: float
    drag_water: float
    drag_land: float
    pressure_buoyancy: float
    k_sense: float
    sense_range_exponent: float
    sense_mass_exponent: float
    k_thermo: float
    thermo_mass_exponent: float
    k_armor: float

    @classmethod
    def from_reader(cls, r: _Reader) -> CostConfig:
        cfg = cls(
            k_basal=r.num("k_basal", low=0.0),
            basal_mass_exponent=r.num("basal_mass_exponent", low=0.0, high=2.0),
            basal_q10=r.num("basal_q10", low=1.0),
            basal_reference_temp_c=r.num("basal_reference_temp_c"),
            upkeep_temp_tolerance=r.num("upkeep_temp_tolerance", low=0.0),
            upkeep_digestion=r.num("upkeep_digestion", low=0.0),
            upkeep_radiation_tolerance=r.num("upkeep_radiation_tolerance", low=0.0),
            upkeep_longevity=r.num("upkeep_longevity", low=0.0),
            k_support=r.num("k_support", low=0.0),
            k_move=r.num("k_move", low=0.0),
            move_gravity_exponent=r.num("move_gravity_exponent", low=0.0, high=2.0),
            drag_water=r.num("drag_water", low=0.0),
            drag_land=r.num("drag_land", low=0.0),
            pressure_buoyancy=r.num("pressure_buoyancy", low=0.0),
            k_sense=r.num("k_sense", low=0.0),
            sense_range_exponent=r.num("sense_range_exponent", low=0.0, high=4.0),
            sense_mass_exponent=r.num("sense_mass_exponent", low=0.0, high=2.0),
            k_thermo=r.num("k_thermo", low=0.0),
            thermo_mass_exponent=r.num("thermo_mass_exponent", low=0.0, high=2.0),
            k_armor=r.num("k_armor", low=0.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class IntakeConfig:
    k_photo: float
    photo_mass_exponent: float
    nutrient_half_saturation: float
    moisture_half_saturation: float
    nutrient_draw_per_energy: float
    k_detritus: float
    detritus_mass_exponent: float
    detritus_half_saturation: float
    k_encounter: float
    capture_bias: float
    capture_size_advantage: float
    capture_speed_advantage: float
    capture_camouflage: float
    capture_armor: float
    capture_aggression: float
    diet_match_floor: float
    aerobic_scope_max: float
    aerobic_scope_o2_exponent: float

    @classmethod
    def from_reader(cls, r: _Reader) -> IntakeConfig:
        cfg = cls(
            k_photo=r.num("k_photo", low=0.0),
            photo_mass_exponent=r.num("photo_mass_exponent", low=0.0, high=2.0),
            nutrient_half_saturation=r.num("nutrient_half_saturation", low=1e-9),
            moisture_half_saturation=r.num("moisture_half_saturation", low=1e-9),
            nutrient_draw_per_energy=r.num("nutrient_draw_per_energy", low=0.0),
            k_detritus=r.num("k_detritus", low=0.0),
            detritus_mass_exponent=r.num("detritus_mass_exponent", low=0.0, high=2.0),
            detritus_half_saturation=r.num("detritus_half_saturation", low=1e-9),
            k_encounter=r.num("k_encounter", low=0.0),
            capture_bias=r.num("capture_bias"),
            capture_size_advantage=r.num("capture_size_advantage"),
            capture_speed_advantage=r.num("capture_speed_advantage"),
            capture_camouflage=r.num("capture_camouflage"),
            capture_armor=r.num("capture_armor"),
            capture_aggression=r.num("capture_aggression"),
            diet_match_floor=r.num("diet_match_floor", low=0.0, high=1.0),
            aerobic_scope_max=r.num("aerobic_scope_max", low=1.0),
            aerobic_scope_o2_exponent=r.num("aerobic_scope_o2_exponent", low=0.0, high=3.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class MortalityConfig:
    background: float
    h_thermal_max: float
    thermal_lethal_margin_c: float
    h_radiation_max: float
    radiation_scale: float
    h_toxicity_max: float
    toxicity_scale: float
    h_pressure_max: float
    pressure_mismatch_scale: float
    senescence_exponent: float

    @classmethod
    def from_reader(cls, r: _Reader) -> MortalityConfig:
        cfg = cls(
            background=r.num("background", low=0.0, high=1.0),
            h_thermal_max=r.num("h_thermal_max", low=0.0, high=1.0),
            thermal_lethal_margin_c=r.num("thermal_lethal_margin_c", low=1e-3),
            h_radiation_max=r.num("h_radiation_max", low=0.0, high=1.0),
            radiation_scale=r.num("radiation_scale", low=0.0),
            h_toxicity_max=r.num("h_toxicity_max", low=0.0, high=1.0),
            toxicity_scale=r.num("toxicity_scale", low=0.0),
            h_pressure_max=r.num("h_pressure_max", low=0.0, high=1.0),
            pressure_mismatch_scale=r.num("pressure_mismatch_scale", low=0.0),
            senescence_exponent=r.num("senescence_exponent", low=0.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class ReproductionConfig:
    overhead: float
    dispersal_radius: float
    mate_search_radius: float
    mate_compatibility_distance: float

    @classmethod
    def from_reader(cls, r: _Reader) -> ReproductionConfig:
        cfg = cls(
            # Overhead below 1 would create energy out of nothing at every birth.
            overhead=r.num("overhead", low=1.0),
            dispersal_radius=r.num("dispersal_radius", low=0.0),
            mate_search_radius=r.num("mate_search_radius", low=0.0),
            mate_compatibility_distance=r.num("mate_compatibility_distance", low=0.0),
        )
        r.done()
        return cfg


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    energy_density: float
    costs: CostConfig
    intake: IntakeConfig
    mortality: MortalityConfig
    reproduction: ReproductionConfig

    @classmethod
    def from_mapping(cls, data: Any) -> EnergyConfig:
        r = _Reader(data, "energy")
        cfg = cls(
            energy_density=r.num("energy_density", low=1e-9),
            costs=CostConfig.from_reader(r.section("costs")),
            intake=IntakeConfig.from_reader(r.section("intake")),
            mortality=MortalityConfig.from_reader(r.section("mortality")),
            reproduction=ReproductionConfig.from_reader(r.section("reproduction")),
        )
        r.done()
        return cfg


# ---------------------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    sim: SimConfig
    planet: PlanetConfig
    genome: GenomeConfig
    energy: EnergyConfig
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Config:
        """Build and validate from already-merged raw mappings (one per config section)."""
        unknown = set(raw) - set(CONFIG_FILES)
        if unknown:
            raise ConfigError(f"unrecognised config section(s): {', '.join(sorted(unknown))}")
        missing = set(CONFIG_FILES) - set(raw)
        if missing:
            raise ConfigError(f"missing config section(s): {', '.join(sorted(missing))}")
        return cls(
            sim=SimConfig.from_mapping(raw["sim"]),
            planet=PlanetConfig.from_mapping(raw["planet"]),
            genome=GenomeConfig.from_mapping(raw["genome"]),
            energy=EnergyConfig.from_mapping(raw["energy"]),
            raw=json.loads(json.dumps(raw, sort_keys=True, default=str)),
        )

    @classmethod
    def load(
        cls,
        config_dir: str | Path = DEFAULT_CONFIG_DIR,
        *,
        overrides: Sequence[str] | None = None,
        planet_file: str | Path | None = None,
    ) -> Config:
        """Load the four config files, apply `key.path=value` overrides, then validate.

        `planet_file` selects an alternative planet definition (an absolute path, or a name
        relative to `config_dir`), so alternative worlds are files rather than edits.
        """
        config_dir = Path(config_dir)
        if not config_dir.is_dir():
            raise ConfigError(f"config directory not found: {config_dir}")

        raw: dict[str, Any] = {}
        for section, filename in CONFIG_FILES.items():
            if section == "planet" and planet_file is not None:
                path = Path(planet_file)
                if not path.is_absolute():
                    path = config_dir / path
            else:
                path = config_dir / filename
            if not path.is_file():
                raise ConfigError(f"config file not found: {path}")
            with path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if loaded is None:
                raise ConfigError(f"{path}: file is empty")
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{path}: expected a mapping at the top level")
            raw[section] = dict(loaded)

        for override in overrides or ():
            apply_override(raw, override)

        return cls.from_raw(raw)

    def fingerprint(self) -> str:
        """Stable hash of the fully-resolved configuration.

        Recorded with every run so a result can be traced back to exactly the parameters that
        produced it -- reproducibility is worthless if you cannot tell which config a saved
        run used.
        """
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a single `dotted.path=value` override to raw config mappings, in place.

    The new value is coerced to the type of the value it replaces, so `planet.gravity=2`
    yields a float and `sim.seed=7` yields an int. Creating new keys is not allowed -- an
    override that does not correspond to an existing key is a typo, and silently accepting it
    would mean the sweep you thought you ran never varied anything.

    Locus fields are addressable by name, e.g. `genome.loci.body_size.sigma=0.1`, because
    `loci` is a YAML list and index-based paths would be unreadable and fragile.
    """
    if "=" not in override:
        raise ConfigError(f"override {override!r}: expected the form key.path=value")
    path_str, _, value_str = override.partition("=")
    path = [part for part in path_str.strip().split(".") if part]
    if not path:
        raise ConfigError(f"override {override!r}: empty key path")

    container: Any = raw
    for depth, key in enumerate(path[:-1]):
        traversed = ".".join(path[: depth + 1])
        if isinstance(container, list):
            container = _locus_by_name(container, key, traversed)
        elif isinstance(container, Mapping):
            if key not in container:
                raise ConfigError(
                    f"override {override!r}: no such config key {traversed!r}"
                )
            container = container[key]
        else:
            raise ConfigError(
                f"override {override!r}: {traversed!r} is a {_typename(container)}, "
                "not a section"
            )

    leaf = path[-1]
    if not isinstance(container, dict):
        # Distinguish "you addressed through a scalar" from "that key does not exist"; the
        # two are different typos and deserve different messages.
        parent = ".".join(path[:-1])
        raise ConfigError(
            f"override {override!r}: {parent!r} is a {_typename(container)}, not a section"
        )
    if leaf not in container:
        raise ConfigError(f"override {override!r}: no such config key {path_str.strip()!r}")

    container[leaf] = _coerce_like(container[leaf], value_str.strip(), override)


def _locus_by_name(entries: list[Any], name: str, traversed: str) -> Any:
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("name") == name:
            return entry
    raise ConfigError(f"override path {traversed!r}: no list entry named {name!r}")


def _coerce_like(current: Any, text: str, override: str) -> Any:
    """Coerce `text` to the type of `current`, rejecting anything ambiguous."""
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
        raise ConfigError(f"override {override!r}: expected a boolean, got {text!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(text)
        except ValueError:
            raise ConfigError(
                f"override {override!r}: expected an integer, got {text!r}"
            ) from None
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError:
            raise ConfigError(
                f"override {override!r}: expected a number, got {text!r}"
            ) from None
    if isinstance(current, str):
        return text
    raise ConfigError(
        f"override {override!r}: cannot override a {_typename(current)} from the command line"
    )
