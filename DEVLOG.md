# Development log

Newest entry first. Each milestone records what was completed, what the known limitations are,
and what the next task is.

---

## Milestone 1 — the world

**Status: complete.** 92 tests passing.

### Completed

- **`src/evosim/world/grid.py` — contained geometry.** Cell-centred latitude/longitude,
  east-west wrapping neighbours, closed polar edges, and cosine latitude area weights live
  behind one API. Resource capacity uses those weights, so geometrically tiny polar cells do
  not receive equatorial resource amounts. Later layers do not need to know which grid
  projection produced an index.

- **`terrain.py` — deterministic procedural topography.** Layered value noise consumes only
  the named `terrain` RNG stream, is periodic across the longitude seam, and is ranked and
  rescaled around sea level to hit `terrain.land_fraction` to within one cell. Elevation and
  local ruggedness are exposed as arrays; rugged ocean cells drive the configured upwelling
  bonus.

- **`climate.py` — seasonal physical fields.** Daily-mean insolation integrates latitude,
  axial tilt, orbital phase, solar constant, and day length, including polar day and night.
  Temperature relaxes toward the radiative/lapse-rate equilibrium using separate configured
  land and water inertia. Ocean moisture is saturated; land moisture decays with distance
  from water and is limited by insolation-driven evaporation.

- **`resources.py` — renewable pools.** Nutrient capacities differ over land and water and
  are area-weighted. Regrowth responds to moisture on land and upwelling at sea. Detritus
  returns to nutrients with the configured temperature-dependent Q10 decay. Every update is
  clipped to preserve non-negative, capacity-bounded pools.

- **`world.py` — the environment boundary.** `World.create(planet_config, rng)` builds every
  field reproducibly; `World.step()` advances climate and resources one simulated day at a
  time. Static toxicity includes the configured highland coupling. Named arrays can be
  consumed without importing terrain or grid internals.

- **Headless use and diagnostics.** `evosim --ticks N` now advances the world, and `--out`
  writes compressed `world.npz` fields plus traceable `world.json` metadata. The new
  `tools/render_world.py` produces six-panel matplotlib maps for elevation, temperature,
  insolation, moisture, nutrients, and detritus.

- **Acceptance coverage.** Tests pin longitude wrapping and polar edges, cosine area weights,
  exact land fraction, terrain seed determinism, cold poles, zero-tilt seasonal invariance,
  stronger continental than maritime seasonality at high tilt, saturated oceans,
  non-negative resources, output files, and whole-world reproducibility.

### Consequences and known limitations

- The world remains a single surface layer on a 128x64 equirectangular grid by design.
  Underwater depth habitats and an icosahedral grid remain contained future changes, but are
  not part of M1.
- Climate is deterministic seasonal climate, not stochastic weather. The reserved `climate`
  RNG stream is intentionally untouched, so later weather can be added without shifting
  terrain or biological random streams.
- Moisture is a distance/insolation proxy, not atmospheric circulation. Terrain does not yet
  create rain shadows, winds, currents, ice sheets, or river transport.
- Nutrient and detritus coefficients now operate as written but cannot be ecologically tuned
  until organisms consume and return material. Their current values remain principled first
  guesses.
- `world.npz` is a diagnostic field dump, not a resume snapshot. Full simulation snapshots
  must also capture populations, history, and every RNG stream in a later milestone.

### Next task — Milestone 2: life substrate

Build the vectorised genome/phenotype/population representation and seed the configured founder
population into valid habitat. Body geometry remains behind `phenotype.py`, and hot-path
population work must operate on arrays rather than per-organism Python loops.

---

## Side spike — 3D morphology (not a milestone)

Requested during M0. ej wants evolvable 3D body plans eventually; this is a **throwaway demo**
to see whether that is viable and which genes are worth having, without committing the
simulation to anything. Lives in `spikes/morphology/index.html`, is imported by nothing, and
has no energy model — selection is the mouse. Full findings in `spikes/README.md`.

Headline results: morphology space is smooth at low mutation strength (7 generations of hand
selection took a featureless blob to a tailed, headed, limbed animal, with offspring visibly
resembling parents), but it **degenerates at high mutation strength** because individually
reasonable gene bounds are jointly absurd — `spine_arch` at maximum is fine on a thick body and
grotesque on a thin one. That joint-constraint problem is a real cost of morphology that the
current scalar-only genome does not have, and it was not in the original estimate. Three of the
19 genes did not earn their place. Conclusion unchanged: this belongs after M5.

**Four constraints this places on M1–M6**, so the later transition is an addition and not a
rewrite:

1. Body geometry stays behind `phenotype.py`; nothing else may assume "body = one number".
   When morphology lands, `body_size` becomes derived and only that module changes.
2. All cost formulas enter through one seam in `energy.py`, so morphology-derived terms plug in
   without touching the tick loop.
3. Organism serialisation for the renderer is a dict, so morphology genes are additive fields
   rather than a format change.
4. `genome.yaml` grows a morphology group only when it is real — dead config is worse than none.

### Decisions taken alongside it

- **World stays a single surface layer.** Ocean depth layers were considered and declined for
  now. Worth recording *why it was tempting*: light attenuating with depth would make the deep
  ocean lightless, so autotrophy becomes physically impossible there and detritivory is forced —
  a real environmental pressure rather than an authored rule, and it would make
  `pressure_tolerance` a live locus instead of a nearly dead one. Cost was ~1 milestone and 3-4x
  the cells. Revisit after M5; it is a contained change to the world layer, not a rewrite.
- **Grid stays 128x64 equirectangular** rather than icosahedral. It projects onto a globe fine
  when 3D rendering arrives, and flat matplotlib dumps make M1 far easier to debug — which
  matters because world bugs must be caught before life exists. The real defect of this grid is
  that polar cells are geometrically tiny but would otherwise hold equatorial resource amounts,
  biasing polar ecology; **M1 must weight per-cell resource capacity by cos(latitude)** to fix
  that. All geometry stays inside `grid.py` so an icosahedral swap stays contained.

---

## Milestone 0 — scaffold, config, RNG discipline

**Status: complete.** 76 tests passing.

### Completed

- **Project scaffold.** `pyproject.toml` with a src layout, `evosim` console script, optional
  dependency groups for `viz` (matplotlib, milestone 1), `server` (FastAPI, milestone 6), and
  `dev`. A `.venv` holds numpy 2.5.1, PyYAML 6.0.3, pytest 9.1.1 — the system Python had no
  PyYAML, and pinning the project to system packages would have been fragile.

- **`src/evosim/rng.py` — the determinism foundation.** One independent
  `np.random.Generator` per subsystem (12 streams), all spawned from a single master seed.
  Two decisions worth noting:
  - Streams are derived from `crc32(stream_name)` as a `SeedSequence` spawn key, *not* from
    spawn order. Adding a new subsystem in a later milestone therefore does not shift the
    numbers any existing subsystem sees. Without this, every recorded run and every tuned
    parameter would be invalidated by an unrelated feature landing.
  - `crc32` rather than `hash()` because CPython randomises string hashing per process, which
    would make runs unreproducible across invocations. A subprocess test with three different
    `PYTHONHASHSEED` values guards this.
  - `get_state`/`set_state` capture every stream's position, for snapshot/resume and for
    determinism tests that assert two runs are at the same point in every stream.

- **`src/evosim/config.py` — all tunables, validated.** Four YAML files load into frozen
  dataclasses. Notable properties:
  - **Unknown keys are rejected with their full dotted path.** A silently ignored `k_suport`
    would be hours of confused tuning.
  - **Dotted-path overrides** (`--set planet.gravity=1.4`), type-coerced to the type of the
    value they replace, refusing to create new keys. Loci are addressable by name
    (`genome.loci.body_size.sigma=0.1`) since `loci` is a YAML list. The parameter sweeps and
    the directional-selection tests depend entirely on this path, so it exists from day one.
  - **`Config.fingerprint()`** — a stable SHA-256 prefix of the resolved config, seed
    included. Reproducibility is worthless if a saved run cannot say which config produced it.
  - Two semantic guards that catch config mistakes the simulation could not: a locus `sigma`
    above a quarter of its span is rejected (mutation would swamp inheritance), and
    `reproduction.overhead < 1` is rejected (it would create energy at every birth).

- **`config/` — every number in the project.** `sim.yaml` (scale and bookkeeping cadence),
  `planet_default.yaml` (an Earth-like reference world), `genome.yaml` (**28 loci** — the
  definitive list of what can evolve), `energy.yaml` (every coefficient in the cost, intake,
  mortality, and reproduction equations). Formulas are documented in the YAML comments
  alongside the coefficients they parameterise, including *why* each exponent was chosen
  (Kleiber 0.75 for basal, 2/3 surface-area scaling for photosynthesis and thermoregulation).

- **`src/evosim/runner.py` — CLI skeleton.** `--config-dir --planet --seed --ticks --out --set`.
  No simulation yet; it resolves and reports config. The surface is fixed now because every
  later milestone drives the sim through it. `--seed` is applied *as an override* so the seed
  lands inside the fingerprint rather than being patched in afterwards.

- **Tests (76).** Config validation (typos, ranges, types, NaN, enums, cross-field
  constraints), override handling, genome structure, RNG reproducibility and independence, and
  two **lint tests** that parse the package AST and fail if any module imports stdlib `random`
  or calls `np.random.<global>`. These are lint rather than convention because the failure mode
  — a subtly unreproducible run — is invisible until you try to reproduce it.

- One test asserts the founder genome is **autotroph-dominant** (softmaxed diet logits give
  ~71% autotrophy, <10% each for herbivory/carnivory). If the founder started predatory,
  "predation emerged" would be a fiction, so the starting point is pinned by a test.

### Two defects the tests caught during this milestone

1. `apply_override` reported "no such config key" when a path traversed *through* a scalar
   (`planet.gravity.deeper=1`). Now distinguishes "that's not a section" from "that key does
   not exist" — different typos, different messages.
2. `RngBundle`'s collision guard was dead code for duplicate names, because the name→key dict
   comprehension deduplicated them before the check ran. Duplicates are now detected
   separately, and the hash-collision guard is proven live by a test that forces a collision.

### Known limitations

- **No world, no life, no simulation.** `--ticks` is accepted and explicitly reported as
  unimplemented; `--out` writes nothing.
- Everything in `energy.yaml` is validated but **unused and untuned** until M3–M4. The
  coefficients are principled first guesses, not calibrated values. Expect them all to move.
- The 28 loci are declared but nothing expresses them yet.
- No performance data at all. The `no-Python-loops-in-the-hot-path` rule is stated in the
  design and in `Population`'s docstring-to-be, but nothing yet exercises it. First real
  numbers arrive with `tools/profile_tick.py` at M2.
- `sim.max_population` throttling behaviour is specified in the YAML comments but not built.

### Next task — Milestone 1: the world

Build `src/evosim/world/` — `grid.py` (geometry, latitude, wrapping neighbours), `terrain.py`
(procedural elevation, renormalised to hit `land_fraction`), `climate.py` (insolation from
latitude × tilt × season, temperature with lapse rate and land/water thermal inertia,
moisture), `resources.py` (nutrient regrowth, detritus decay), and `world.py` as the container.
Plus `tools/render_world.py` for matplotlib map dumps.

Verified by: poles colder than the equator; `axial_tilt=0` produces no seasonal variation; high
tilt produces extreme *continental* seasonality but mild maritime seasonality (the thermal
inertia test); land fraction lands within tolerance of the configured target; nutrients never go
negative. Then look at the PNGs — world bugs must be caught before life exists, or they get
misdiagnosed as evolution bugs later.
