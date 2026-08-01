# Evolution Simulator — Design & Roadmap

> **Status of this document.** This is the founding design doc, written before any code existed.
> It is kept verbatim as the record of what was decided and why — so its present tense ("nothing
> exists yet", "implementation starts at Milestone 0") describes the day it was written, not today.
>
> Everything in it still stands **except §11, "Development roadmap", which is superseded by
> [ROADMAP.md](ROADMAP.md)** — M0–M6 shipped as planned here; M7 onward changed direction.
> See [../DEVLOG.md](../DEVLOG.md) for what was actually built, and what it cost.

## Context

ej wants a scientifically grounded evolution sandbox: configure a planet, seed simple life, and watch
genuinely emergent biospheres develop. The appeal is the Spore *sense of discovery* without Spore's
fixed progression — no tech tree, no drive toward intelligence or complexity, and no scripted outcomes.
Two similar planets should produce meaningfully different biospheres, and any outcome should be
*explainable in hindsight* by inspecting the recorded history even though it wasn't predictable up front.

Nothing exists in `/home/ej/evolutionsim` yet. This document defines the vision, the MVP boundary, the
models, the architecture, and the milestone plan. Implementation starts at Milestone 0 after approval.

### Decisions locked in the interview

| Question | Decision |
| --- | --- |
| Experience | Strong / professional — optimize for architecture, not hand-holding |
| Stack | Python (NumPy) sim core, headless & testable; browser UI over a local server |
| Granularity | **Individual organisms** — discrete entities, real genomes, real encounters |
| Seed life | **One lineage; diet is a gene.** Trophic roles must emerge |
| Genetics | **Diploid loci, additive expression, reproduction mode is itself a gene** |
| Planet depth | Grid climate (elevation, temp, moisture, nutrients, toxicity) + global scalars (gravity, O₂, pressure, radiation) |
| Run model | Live, watchable, speed control; logging designed so rewind can be added later without a rewrite |
| MVP priority | **Believable, legible evolution.** Everything else is built to a lower bar first |

### Assumptions I'm making without asking

- **Player control in the MVP is observation only**: planet config + seed at creation, then
  play/pause/speed/step and inspection. God tools (disasters, barriers, translocation) are Milestone 7.
- **UI has no build step initially** — static HTML + ES modules + `<canvas>`, served by the Python
  server. Hand-rolled small charts. If the UI outgrows this, migrate to Vite/React at M8, not before.
- Grid is **128 × 64 cells** on an equirectangular projection, wrapping east–west.
- Target scale: **~20k organisms, 50–200 ticks/sec** headless. One tick = one day; one generation is
  tens to hundreds of ticks depending on evolved life history.

---

## 1. Game vision

*A scientific instrument disguised as a sandbox.* You dial in a planet's physics, drop a single simple
organism into its oceans, and watch what the planet makes of it over deep time. Life spreads, splits,
invents eating each other, specializes, and dies out — all from mutation, inheritance, energy budgets,
and competition. You cannot upgrade anything. You can only change the world and watch the consequences
propagate, then open the history and find out why it went the way it did.

Success test: a player runs the same seed twice with gravity 0.8g vs 1.4g, gets visibly different body
plans, and can *explain the difference* from the trait graphs and the event log.

## 2. Minimum viable prototype

The MVP is done when all of the following are true:

1. A planet is generated from a config file + seed: continents, oceans, latitudinal climate, seasons.
2. One seeded microbe lineage colonizes viable habitat and reaches a stable carrying capacity.
3. Trait distributions **measurably track environmental gradients** — e.g. mean `temp_optimum` differs
   between equatorial and polar populations by more than seed-to-seed noise.
4. **Heterotrophy emerges on its own** (detritivory, then herbivory/predation) and the first occurrence
   is timestamped in the event log.
5. **Speciation and extinction happen and are recorded**, with a parent-pointer phylogeny.
6. The browser UI shows: climate/biome map, population heatmap, live population & trait charts, a
   species list, a species inspector (traits vs. ancestor, current pressures), and an event timeline.
7. Same seed + same config ⇒ **bit-identical run**, asserted by a test.
8. A test suite asserts directional selection on cold / high-gravity / low-O₂ planets.

## 3. Postponed (explicitly out of the MVP)

Deferred to keep M1–M6 achievable. None of these are blocked by the architecture.

- **Atmosphere–life feedback** (photosynthesis raising O₂ and terraforming the planet). Highest-value
  deferred feature; O₂ stays a global constant for now.
- Plate tectonics, continental drift, changing sea level.
- Natural disasters, volcanism, impacts, mass-extinction triggers → M7.
- God-mode intervention → M7.
- Rewind / replay-from-snapshot scrubbing → M8 (the event log is designed for it now).
- ~~Multicellularity, real body plans, symmetry, morphology rendering.~~ **Body plans and morphology
  rendering landed in M8**: thirteen morphology loci, geometry integrated in `phenotype.py`, and a
  WebGL creature viewer in the species panel. Multicellularity proper is still out.
- Parasitism, true symbiosis/mutualism, disease.
- Communication, learning, culture, within-life behavioral plasticity. Behavior is genetic constants.
- Sexual *selection* via ornament/preference coevolution. Sexual *reproduction* is in; mate choice at
  MVP is only assortative-by-genetic-distance.
- Continuous 2-D space. Organisms live at cell resolution with a sub-cell offset used for rendering
  and encounter probability, not for real physics.
- Run-comparison UI → M8.

## 4. Core simulation loop

One tick = one simulated day. Ordering is fixed and deterministic.

1. **Advance clock** → day, year, season phase.
2. **Update environment**: insolation from latitude × axial tilt × season; temperature (with elevation
   lapse rate and ocean thermal inertia); moisture; nutrient regrowth; detritus decay.
3. **Sense & move**: each organism evaluates its own cell and neighbors within `sense_range`, moves
   down-gradient of a genetically weighted desirability (energy availability, crowding, temperature
   mismatch, predator density).
4. **Feed**: autotrophic uptake from light + cell nutrients; detritivory from the corpse pool;
   herbivory and predation resolved as cell-local encounters.
5. **Pay costs**: basal metabolism, locomotion, body support, sensory tissue, thermoregulation, armor.
6. **Apply hazards**: starvation, temperature/pressure/radiation/toxicity mismatch, senescence,
   background mortality. Deaths deposit biomass into the cell's detritus pool.
7. **Reproduce**: mature organisms above their energy threshold spawn offspring; asexual clones with
   mutation, sexual pairs with recombination + mutation.
8. **Compact population arrays** (remove dead, append newborns).
9. **Bookkeeping** (every *K* ticks, not every tick): species assignment & speciation tests, extinction
   detection, time-series sampling, event emission.

## 5. Organism data model

**Structure-of-arrays, NumPy, no per-organism Python loops in the hot path.** This is the single most
important performance commitment in the project.

```python
class Population:
    # capacity-preallocated parallel arrays; `alive` is a boolean mask
    genome:      np.float32  # (N, L, 2)  diploid alleles, L ≈ 24 loci
    trait:       np.float32  # (N, T)     cached expressed phenotype
    cell:        np.int32    # (N,)       flat grid index
    offset:      np.float32  # (N, 2)     sub-cell position, cosmetic + encounter geometry
    energy:      np.float32  # (N,)
    age:         np.int32    # (N,)
    species_id:  np.int32    # (N,)
    parent_id:   np.int64    # (N,)       for ancestry sampling
    uid:         np.int64    # (N,)       stable identity
    alive:       np.bool_    # (N,)
```

Spatial indexing: sort/argsort organisms by `cell` once per tick and build a CSR-style
`cell_start/cell_count` index. All cell-local interactions become vectorized segment operations.

### Loci (initial set, ~24 — all continuous, all in config)

| Group | Loci |
| --- | --- |
| Morphology (M8) | `body_length`, `radius_ratio`, `fullness`, `taper`, `segment_count`, `segment_depth`, `radial_symmetry`, `limb_pairs`, `limb_ratio`, `limb_splay`, `head_size`, `tail_ratio`, `dorsal_fin` |
| Body | `energy_storage`, `armor` |
| Metabolism | `metabolic_rate`, `temp_optimum`, `temp_tolerance`, `radiation_tolerance`, `pressure_tolerance` |
| Diet | `aff_autotroph`, `aff_detritus`, `aff_herbivore`, `aff_carnivore`, `digestion_efficiency` |
| Movement | `move_speed`, `move_persistence` |
| Sensing | `sense_range` |
| Life history | `maturity_age`, `senescence_rate`, `repro_threshold`, `offspring_count`, `parental_investment`, `sex_bias` |
| Defense/behavior | `camouflage`, `aggression`, `fear`, `sociality` |
| Meta | `mutation_rate` (evolvable) |

Diet affinities are softmax-normalized to sum to 1, so becoming a better carnivore *necessarily* costs
autotrophic capability. Specialization is structural, not a bonus.

## 6. Environmental data model

```python
class World:
    # static, generated once from seed
    elevation:   np.float32  # (H, W)  procedural (layered simplex + plate-ish masks)
    is_water:    np.bool_    # elevation < sea_level
    latitude:    np.float32  # (H,)
    base_toxicity: np.float32

    # dynamic, per tick
    temperature: np.float32  # (H, W)  °C
    light:       np.float32  # (H, W)  relative insolation
    moisture:    np.float32  # (H, W)
    nutrients:   np.float32  # (H, W)  regenerating stock
    detritus:    np.float32  # (H, W)  corpse/waste pool, decays into nutrients

    # global scalars from planet config — these enter the energy equations directly
    gravity: float           # g
    o2_fraction: float
    pressure: float          # atm
    solar_constant: float    # relative to 1.0
    radiation: float
    axial_tilt: float        # degrees
    day_length: float        # hours
    year_length: int         # ticks
    sea_level: float
```

Key formulas (all coefficients in `config/`):

- `insolation(lat, day) = solar_constant · max(0, cos(lat − tilt·sin(2π·day/year))) · daylight_fraction`
- `temperature = T_base + T_amp·insolation − lapse_rate·max(0, elevation − sea_level)`, then smoothed
  toward the previous tick with a large time constant over water (thermal inertia → maritime climates,
  continental interiors get extreme seasons for free).
- `nutrients += regen·moisture·(cap − nutrients)/cap` — land nutrients depend on water availability;
  oceans get a separate constant with an upwelling term near steep bathymetry.
- `detritus` decays at `decay_rate·f(temperature)` into `nutrients` — an abiotic decomposer stand-in
  until real decomposers evolve to compete with it.

Biomes are **not** an input. They are read *out* of (temperature, moisture, is_water) for display only.

## 7. Genetics & mutation model

- Genome = fixed-length vector of **L diploid loci**, `float32`, each allele in a per-locus range.
- **Expression is additive**: `trait = clip((a₁ + a₂)/2, lo, hi)`. Heterozygosity is therefore
  measurable and drift is real. Dominance is deliberately deferred.
- **Mutation** per allele per reproduction:
  - with prob `mutation_rate·(1 + radiation_sensitivity·planet_radiation)`, add
    `N(0, σ_locus)` — σ per locus from config.
  - with prob `p_large` (≈2% of mutations), a large-effect jump `N(0, 6σ)`.
  - `mutation_rate` is itself a locus, so mutation rate evolves under selection.
- **Asexual reproduction**: offspring genome = parent genome + mutation.
- **Sexual reproduction**: `sex_bias > 0.5` ⇒ seek a mate in the same cell with genetic distance below
  a compatibility threshold. Offspring takes one allele per locus from each parent — **free
  recombination**, no linkage map (deferred). Asexual and sexual lineages compete directly.
- Radiation raises mutation rate *and* mortality, so a high-radiation planet is genuinely a tradeoff,
  not an accelerator.

## 8. Energy, survival & reproduction model

Every trait must pay in energy or hazard. **No arbitrary stat bonuses anywhere.**

**Costs per tick** (`m` = body mass = `volume · body_density`, `g` = gravity). As of M8 `volume`,
`surface_area`, `cross_section`, `limb_count` and `slenderness` are integrated from the morphology
genome in `phenotype.py`; nothing below knows what an animal looks like, only these five numbers:

```
basal       = k_b · m^0.75 · metabolic_rate · o2_scope · Q10(T)
support     = k_s · m · g · (1 + armor) / slenderness
              · (1 + seg_cost·(segment_count−1) + taper_cost·|taper|)
locomotion  = k_l · m · move_speed² · g^0.5 · medium_drag
              · (cross_section / volume^(2/3) / drag_shape_ref)
              · (1 + limb_drag·limb_count·limb_ratio)
sensory     = k_n · sense_range^1.5 · m^0.25
thermoreg   = k_t · (surface_area / thermo_area_ref) · max(0, |T − temp_optimum| − temp_tolerance)
armor_upkeep= k_a · armor · (surface_area / thermo_area_ref)
```

The speed *ceiling* gains `· (1 + limb_thrust·limb_count·limb_ratio·limb_splay)` on land only —
limbs push against a substrate and are pure drag at sea, which is what makes them a real trade.

Each morphology factor evaluates to exactly 1.0 at founder proportions, so M8 added geometry to
these equations without re-tuning a single constant M3/M7/M7b had calibrated.

- `o2_scope = clip(o2_fraction / o2_ref, ...)` **caps aerobic scope** — low O₂ mechanically limits the
  product of size and metabolic rate. Nothing hardcodes "low O₂ ⇒ small"; it falls out of the budget.
- Gravity enters `support` linearly in mass and `locomotion` in a sublinear power, so heavy planets
  punish big fast bodies without forbidding them.

**Intake per tick:**

```
autotroph = k_p · aff_autotroph · m^0.67 · light · sat(nutrients) · sat(moisture) · digestion_efficiency
detritus  = k_d · aff_detritus  · m^0.67 · sat(cell_detritus)     · digestion_efficiency
herbivory = handled as predation against autotroph-dominant targets
predation : encounter → capture → consume
```

**Encounter resolution** (cell-local, vectorized):

```
encounter_rate ∝ aggression · move_speed · sense_range² · prey_density / cell_area
capture_prob   = σ( α·(attacker_size − prey_size) + β·(attacker_speed − prey_speed)
                    − γ·prey_camouflage·(1/attacker_sense_range) − δ·prey_armor + ε·attacker_aggression )
gain           = prey_mass · energy_density · digestion_efficiency · aff_matching(prey_diet)
```

`aff_matching` means a specialized carnivore extracts little from an autotroph and vice versa. This one
term is what makes the prey-camouflage ⇄ predator-sense arms race and plant-defense ⇄ specialized-gut
coevolution actually close the loop.

**Mortality hazards** (independent, combined as `1 − Π(1 − hᵢ)`):
starvation (`energy ≤ 0`, hard), thermal, radiation, toxicity, pressure mismatch, predation (above),
senescence (`h = senescence_rate·(age/maturity_age)^k`), and small background mortality.
Every death adds `body_mass · energy_density` to the cell's detritus pool — **energy is conserved
through a ledger that tests assert on.**

**Reproduction:** at `age ≥ maturity_age` and `energy ≥ repro_threshold · storage_capacity`, spend
`offspring_count · parental_investment · base_cost`. `parental_investment` sets each offspring's
starting energy, so the r/K spectrum (many cheap vs. few provisioned) is a live tradeoff, not a preset.

## 9. Population & speciation model

- No species are predefined. All organisms start in species 0.
- Every *K* ticks, per species: attempt a **2-means split in standardized trait space**. Split is
  accepted when *(a)* between-cluster separation / within-cluster spread exceeds a threshold, **and**
  *(b)* realized gene flow between clusters over the last window is near zero (spatial separation for
  asexuals, mating incompatibility for sexuals). Both conditions must hold — this is what makes
  geographic isolation and assortative mating the *cause* of speciation rather than a label.
- Accepted split ⇒ new `species_id`s, parent pointer recorded, `SpeciationEvent` logged with the
  driving trait axis and the geographic distribution of each daughter clade.
- Extinction = population 0 for a full window ⇒ `ExtinctionEvent` with last known range and the
  dominant hazard/energy-deficit cause over its final window.
- Drift, bottlenecks, and founder effects need no special code — they are consequences of finite
  individual populations, which is exactly why the individual-based choice was worth its cost.
- Per-species time series sampled every *K* ticks: population, range, mean/variance per trait,
  heterozygosity, energy sources by fraction, birth/death causes. **This table is what makes outcomes
  explainable in hindsight** and is the backbone of the MVP priority.

## 10. Software architecture

Strict layering, dependencies point one way only:

```
config  →  world  →  life  →  evolution  →  history  →  sim  →  server  →  ui
```

- `sim.Simulation` is pure and headless: constructed from `(Config, seed)`, exposes `tick()` and
  `snapshot()`. It imports nothing from `server` or `ui`, and never touches wall-clock time.
- The server owns a **sim thread** running as fast as the speed setting allows, plus a lock-protected
  latest-snapshot buffer. The UI polls JSON. Determinism is preserved because the sim thread's work per
  tick is independent of when it is polled.
- **RNG discipline**: one seeded `np.random.Generator` per subsystem, spawned deterministically from
  the master seed via `SeedSequence`. No global `np.random`, ever. A lint test enforces this.
- Everything tunable lives in `config/*.yaml`, loaded into validated dataclasses. Zero magic numbers in
  the simulation modules.

## 11. Development roadmap

> **Superseded by [ROADMAP.md](ROADMAP.md).** M0–M6 below shipped as written. M7–M8 did not
> happen as written: M6 measured that the reference planet offers only one way to make a living,
> and the project's direction changed toward bodies you can see. Kept here for the record.

Each milestone ends with tests passing and a `DEVLOG.md` entry (done / limitations / next).

| M | Deliverable | Verified by |
| --- | --- | --- |
| **0** | Scaffold, config loader + validation, RNG discipline, CLI skeleton | Config round-trips; RNG spawn is reproducible |
| **1** | World: terrain, climate, seasons, resource fields; matplotlib map dump | Poles colder than equator; tilt=0 ⇒ no seasons; high tilt ⇒ extreme continental seasonality; matplotlib PNGs look like a planet |
| **2** | Genome, phenotype, `Population` SoA, mutation, recombination, spatial index | Mutation variance matches σ; recombination is Mendelian; no-selection run shows neutral drift and heterozygosity decay at ~1/(2Nₑ) |
| **3** | Energy, costs, hazards, reproduction — **autotrophs only** | Population finds carrying capacity, doesn't explode or die out; energy ledger balances; polar vs equatorial populations diverge in `temp_optimum` |
| **4** | Detritivory, herbivory, predation, encounters, movement/foraging | **Heterotrophy emerges from an autotroph-only seed**; predator–prey population cycles appear; camouflage↑ drives sense_range↑ |
| **5** | Species assignment, speciation, extinction, phylogeny, event log | Two isolated continents ⇒ allopatric speciation; a merged single continent ⇒ significantly fewer species, same seed |
| **6** | Server + UI: map, heatmaps, charts, species inspector, timeline, phylogeny | Watch a full run end-to-end in the browser, pause, and explain a trait shift from the UI alone |
| **7** | God tools: disasters, climate shifts, barriers, translocation | A forced mass extinction produces a survivor-clade radiation, not a scripted result |
| **8** | *(superseded — see `docs/ROADMAP.md`)* The morphology genome and the creature viewer | Limbs cost drag and earn speed; the founder's energetics are unchanged; species render as distinguishable animals |

M0–M5 is the real engine work; M6 is where it becomes a thing you can enjoy. I'd rather over-invest in
M3–M5 than ship a pretty M6 over a broken engine — which matches the stated MVP priority.

## 12. Testing & validation strategy

Four tiers, because unit tests alone cannot tell you whether evolution is *believable*:

1. **Unit** — formulas, config validation, grid geometry, genome ops, spatial index correctness.
2. **Determinism** — same seed ⇒ identical SHA-256 of the full state after 5000 ticks. Also asserts a
   different seed differs, and that snapshot/poll timing cannot perturb the run.
3. **Invariants** — energy ledger closes to float tolerance; populations non-negative; no organism
   outside grid bounds; array compaction preserves identity; nutrients never negative.
4. **Biology / scenario tests** — the important tier. Each runs N seeds and asserts a *directional*
   result with tolerance, not an exact number:
   - cold planet ⇒ lower mean `temp_optimum`, higher `temp_tolerance`
   - high gravity ⇒ smaller mean `body_length`, lower `move_speed`
   - low O₂ ⇒ lower `metabolic_rate` × body mass product
   - stable climate ⇒ narrower trait variance; fluctuating climate ⇒ wider (bet-hedging)
   - seasonal extremes ⇒ larger `energy_storage` or shorter `maturity_age`
   - **liveness guard**: a default planet produces ≥2 species and ≥1 heterotroph within N generations —
     so "the planet is dead / nothing diversified" fails CI rather than wasting your afternoon.

Plus `tools/profile_tick.py` from M2 onward, so performance regressions are caught while they're still
cheap to fix.

## 13. Scientific simplifications & their consequences

| Simplification | Consequence |
| --- | --- |
| Additive diploid genetics, no linkage, no dominance | No hybrid vigor, no sheltered deleterious recessives, no selective sweeps dragging neighbors. Drift is real but genome architecture is not. |
| Traits are independent loci (no pleiotropy) | Evolution explores trait space more freely than real organisms can. Constraint comes only from energy budgets, so expect somewhat "too optimal" organisms. |
| Cell-resolution space | No real territories, no fine-grained spatial pattern formation, no chase dynamics. Herding/territoriality will be weak until continuous space arrives. |
| Behavior is genetic constants, no learning | No plasticity, no behavioral adaptation within a lifetime, so `learning_ability` is meaningless for now and is omitted rather than faked. |
| Abiotic detritus decay | Decomposers have a free competitor from day one; the "invention of decomposition" is muted as an event. |
| O₂/CO₂ are constants | Life cannot terraform its planet. The largest single loss of emergent drama in the MVP. |
| One tick = one day, no sub-daily cycle | Day length only modulates insolation; nocturnality cannot evolve. |
| Body plan is a few scalars | Symmetry, limbs, and real morphology are absent; visuals stay abstract. Don't over-read the blobs. |
| Fixed grid, no tectonics | Geographic barriers are static within a run, so vicariance must be introduced by god tools rather than arising naturally. |
| Softmax diet affinities | Omnivory is possible but mediocre by construction; the model has a mild built-in bias toward specialization. |

## 14. Technical risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| **Python speed** — individual-based at 20k organisms | High | Absolute rule: no per-organism Python loops in the tick. Everything vectorized over SoA arrays; cell-local interactions via sort + CSR segment ops. Profile from M2. Escape hatch: move the hot 2–3 functions to Numba, then to a Rust/C extension. The layering makes this a local change. |
| **Dead or degenerate planets** — nothing evolves, or one super-organism wins forever | High | Liveness guard in CI; density-dependent costs and resource depletion cap any single strategy; softmax diet forces tradeoffs; a parameter-sweep tool to find the interesting regime rather than guessing. |
| **Tuning burden** — dozens of coupled coefficients | High | Everything in YAML; sensitivity sweeps as a first-class tool; tune one milestone's subsystem at a time with the rest disabled. |
| **Speciation threshold is arbitrary** | Medium | Require *both* trait separation and near-zero gene flow, and expose the threshold in the UI so the user can see it's a measurement choice, not a biological fact. |
| **Determinism vs. threading** | Medium | Sim thread does all mutation of state; snapshots are deep copies under lock; per-subsystem seeded generators. Guarded by a determinism test that polls at random intervals. |
| **Coupled-feedback instability** (when atmosphere-life lands post-MVP) | Medium | Deferred deliberately. Add with damping and hard bounds, behind a config flag. |
| **UI scope creep** — phylogeny + food web + charts hand-rolled | Medium | M6 ships the map, charts, species inspector, and timeline first; the phylogeny and food-web views are the last two items and may slip to M8. |
| **Snapshot/log volume** over long runs | Low | Sample time series every K ticks, not every tick; events are sparse; full snapshots only at intervals. |

## 15. Folder & file structure

```
evolutionsim/
├── pyproject.toml
├── README.md
├── DEVLOG.md                    # completed / limitations / next task
├── config/
│   ├── sim.yaml                 # tick rate, sampling intervals, capacities
│   ├── planet_default.yaml      # gravity, o2, tilt, sea level, ...
│   ├── genome.yaml              # loci: range, initial value, mutation sigma
│   └── energy.yaml              # every k_ coefficient
├── src/evosim/
│   ├── rng.py                   # SeedSequence discipline
│   ├── config.py                # dataclasses + YAML loader + validation
│   ├── world/{grid,terrain,climate,resources,world}.py
│   ├── life/{genome,phenotype,population,energy,interact,reproduce,mortality,forage}.py
│   ├── evolution/{speciation,taxonomy}.py
│   ├── history/{events,recorder}.py
│   ├── sim.py                   # Simulation.tick() — the loop in §4
│   ├── runner.py                # headless CLI
│   ├── server/{app,state}.py    # FastAPI + sim thread + snapshot buffer
│   └── ui/                      # static, no build step
│       ├── index.html
│       └── js/{main,map,charts,inspector,timeline,tree}.js
├── tests/
│   ├── test_config.py test_rng.py test_determinism.py
│   ├── test_grid.py test_climate.py test_resources.py
│   ├── test_genome.py test_phenotype.py test_population.py
│   ├── test_energy.py test_mortality.py test_reproduce.py
│   ├── test_interact.py test_speciation.py
│   ├── test_invariants.py       # energy ledger, bounds
│   └── test_biology.py          # §12 tier-4 scenario tests
└── tools/
    ├── profile_tick.py
    ├── sweep.py                 # parameter sensitivity sweeps
    └── render_world.py          # matplotlib dumps (pre-UI debugging)
```

## 16. Pseudocode — main simulation update

```python
def tick(self):
    w, pop, cfg = self.world, self.pop, self.config
    self.day += 1

    # ---- 1-2. environment -------------------------------------------------
    w.update_insolation(self.day)          # latitude x tilt x season
    w.update_temperature()                 # lapse rate + ocean thermal inertia
    w.update_moisture()
    w.regrow_nutrients()                   # moisture-limited
    w.decay_detritus()                     # -> nutrients, temperature-dependent

    # ---- spatial index (once; everything below is vectorized on it) -------
    order, cell_start, cell_count = build_cell_index(pop.cell, w.n_cells)

    # ---- 3. sense & move --------------------------------------------------
    desirability = score_neighborhood(w, pop, order, cell_start)  # (N, 9)
    pop.cell, pop.offset = choose_move(pop, desirability, rng.move)
    order, cell_start, cell_count = build_cell_index(pop.cell, w.n_cells)

    # ---- 4. feed ----------------------------------------------------------
    gain  = autotrophic_uptake(w, pop)             # depletes w.nutrients
    gain += detritivory(w, pop, cell_count)        # depletes w.detritus
    kills, predation_gain = resolve_encounters(pop, order, cell_start,
                                               cell_count, rng.encounter)
    gain += predation_gain
    pop.energy += gain

    # ---- 5. costs ---------------------------------------------------------
    cost = (basal(pop, w) + support(pop, w) + locomotion(pop, w)
            + sensory(pop) + thermoregulation(pop, w) + armor_upkeep(pop, w))
    pop.energy -= cost
    pop.energy = np.minimum(pop.energy, pop.storage_capacity)   # cap

    # ---- 6. hazards -------------------------------------------------------
    hazard = combine(starvation(pop), thermal(pop, w), radiation(pop, w),
                     toxicity(pop, w), pressure(pop, w), senescence(pop),
                     background(cfg))
    dead = kills | (rng.death.random(pop.n) < hazard)
    w.detritus += scatter_biomass(pop, dead)       # energy conservation
    self.ledger.record(gain, cost, dead)

    # ---- 7. reproduce -----------------------------------------------------
    parents = ready_to_reproduce(pop, dead)
    children = reproduce(pop, parents, order, cell_start, rng.repro)
    #   asexual -> clone + mutate;  sexual -> mate search, recombine, mutate

    # ---- 8. compaction ----------------------------------------------------
    pop.kill(dead)
    pop.append(children)
    pop.age += 1
    pop.refresh_phenotypes(changed=children.slice)

    # ---- 9. periodic bookkeeping -----------------------------------------
    if self.day % cfg.taxonomy_interval == 0:
        events  = self.taxonomy.update(pop)        # speciation + extinction
        events += detect_milestones(pop, self.history)   # e.g. first carnivore
        self.history.log(self.day, events)
        self.history.sample(self.day, w, pop)      # per-species time series
```

## Verification

- `pytest` — all four tiers; `pytest -m biology` for the slow scenario tests.
- `python -m evosim.runner --config config/planet_default.yaml --seed 42 --ticks 20000 --out runs/a`
  then rerun and `diff` the state hashes to confirm determinism.
- `python tools/render_world.py --seed 42` after M1 — inspect the climate PNGs by eye before any life
  exists, so world bugs never get misdiagnosed as evolution bugs.
- `python tools/profile_tick.py --organisms 20000` after each milestone; record ticks/sec in `DEVLOG.md`.
- From M6: `python -m evosim.server` → open `localhost:8000`, run a planet for a few thousand ticks,
  and confirm a trait shift can be explained from the UI alone.

## First implementation step (after approval)

**Milestone 0 only** — `pyproject.toml`, the four config files, `config.py` with validated dataclasses,
`rng.py` with the SeedSequence discipline, a CLI skeleton, `DEVLOG.md`, and `tests/test_config.py` +
`tests/test_rng.py`. No world, no life. Then I'll report and stop for your review before M1.
