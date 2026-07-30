# Development log

Newest entry first. Each milestone records what was completed, what the known limitations are,
and what the next task is.

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
