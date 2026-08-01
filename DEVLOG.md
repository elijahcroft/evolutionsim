# Development log

Newest entry first. Each milestone records what was completed, what the known limitations are,
and what the next task is.

---

## Milestone 5 — species and history

**Status: complete.** 378 tests passing.

### Completed

- **`src/evosim/evolution/taxonomy.py` — one metric, two uses.** The split test is 2-means run
  in exactly the space `GenomeSchema.genetic_distance` measures in: each expressed trait divided
  by its locus span and scaled by the square root of its distance weight, which makes Euclidean
  distance between coordinate vectors *equal* genetic distance (a test asserts that to 1e-6). A
  species splits when the two cluster centres are further apart than
  `mate_compatibility_distance` — the same number that decides whether two organisms can breed.
  So assortative mating and the taxonomy cannot disagree, and **no coefficient entered the model
  that was not already in `energy.yaml`**. Splitting recurses until nothing more splits, because
  `taxonomy_interval` should set how often taxonomy is revisited, not how much divergence one
  pass may recognise.

- **The larger half keeps the ancestral identity.** A split is a bifurcation; calling both
  halves new would discard the continuity the lineage tree exists to record. There is
  deliberately **no minimum species size**: a lone organism far enough from everything else is
  reproductively isolated by the same threshold, and inventing a quorum constant would be
  inventing a number `config/` does not contain.

- **`src/evosim/history/records.py` — the run's memory.** A `SpeciesRecord` per species
  (parentage, origin day, origin trait means, current and peak population, extinction), the
  lineage tree they form, and the `sample_interval` time series broken down by species. Nothing
  in the tick ever reads it — a history that fed back into the model would be a hidden selection
  pressure of exactly the kind this project forbids — and a test pins that sampling a run
  changes nothing about it.

- **Extinction is confirmed once, and dated to when it happened.** `extinction_confirm_ticks`
  guards against calling a transient dip an ending, but the ending itself occurred when the last
  member died, so both days are recorded. A species identity is never reused, so a resurrected
  id could only ever be a bug. Counting runs every tick rather than on the taxonomy cadence:
  an extinction date read off a 100-day grid would be wrong by up to 100 days.

- **`sim.py` — an eighth stage.** The tick is now age → move → pay → eat → hunt → die → breed →
  regrow → **record**. Offspring inherit the species of the parent that bore them; only the
  periodic taxonomy reassigns anyone, and it does so on the distribution of a whole group rather
  than on one individual. `TickStats` gained `species`, `species_born`, `species_extinct`; the
  CLI reports living species and writes `history.json` beside the existing run dumps.

- **One calibration finding, deliberately not tuned away: nothing speciates on the reference
  planet, and no threshold value would fix that.** M5 is the first milestone able to measure
  genetic spread, and the measurements are unambiguous:
  - A 2,000-day run holds a maximum pairwise distance of **0.07** against a compatibility
    threshold of **0.15**, with a mean of 0.023. Mate compatibility therefore never bites.
  - Two *completely isolated* lineages (independent runs from the same founder, one on a colder
    planet) diverge to only **0.030** after 3,000 days — no further apart than the spread inside
    a single panmictic population. So there is no gap between "drift" and "divergence" to put a
    threshold in: any value low enough to split isolated lineages would also shatter one
    undivided population into noise.
  - A 10× mutation rate reaches **0.06**, still not the threshold. Mutation-selection balance
    bounds the spread; raising mutation does not raise it much.
  - The root cause is dilution: genetic distance is an RMS over 28 loci, so adaptation on a few
    loci barely moves it. A *full-range* shift of `temp_optimum` alone would be 0.19, but the
    ~50 °C shift a real planet can select is only 0.085. This is the same shape of result as
    M4's predation valley: the mechanism works, the reference planet does not reach it, and
    which of the three available fixes is right (larger sigmas, a distance metric weighted
    toward the loci that actually vary, or stronger geographic isolation) is a question to
    settle with an experiment rather than a guess.

- **Performance.** Recording costs **0.2 ms of a 21.1 ms tick** (~1%) at 40,000 organisms. The
  periodic work is 29.6 ms for a taxonomy pass and 7.9 ms for a sample — about 0.4 ms/tick
  amortised at the default 100-tick cadences.

- **Acceptance coverage.** 37 new tests. The milestone's three criteria are pinned: a lineage
  divided between two habitats is recorded as two species with the right parentage; a species
  that reaches zero is declared extinct exactly once, on the confirmation day, and is never
  revived; and two runs of the same `(config, seed)` produce byte-identical history documents.

### Consequences and known limitations

- **Every split in the test suite is imposed, not evolved**, for the reason measured above. The
  machinery is verified end to end through the ordinary tick, but "two species arose" has not
  been observed happening by itself on any planet tried.
- **The split test compares centroids, not distributions.** Two clusters whose centres are 0.16
  apart are declared separate species even if their tails overlap, and the fraction of pairs
  that could actually still interbreed is not measured. A distributional test would be more
  honest and much more expensive.
- **2-means assumes two.** A species that has genuinely trifurcated is found as three only
  because splitting recurses; a pass sees at most a bisection.
- **History is not yet a resumable snapshot.** `history.json` is a diagnostic dump like
  `world.npz`: it records what happened, not enough to restart from it.
- **The species panel does not exist.** The browser UI reports population and the tick ledger;
  nothing there yet lets you open a species and read its lineage, which is what the recorded
  data is for.
- Every M1–M4 limitation still stands, including the population cap dominating a default run
  and predation sitting across a fitness valley.

### Next task — Milestone 6: the run you can read

The server and browser UI arrived early (they were planned for this milestone and are already
driving the sim). What is missing is that none of what M5 records is visible: surface the
species list, the lineage tree, and the sampled trait time series through the API and the UI, so
that "open a species and read why it became what it is" is something a person can do rather than
something a JSON file permits.

Alongside it, settle the speciation-scale question with an experiment rather than a tuning pass:
run isolated demes under strong divergent selection with varied `sigma` and distance weights, and
record which of them — if any — produces a split that is divergence rather than drift.

Verified by: a species can be opened in the UI and its ancestry and trait history read; and the
speciation experiment produces a written finding, whether or not it produces a species.

---

## Milestone 4 — reproduction, selection, and the food web

**Status: complete.** 325 tests passing.

### Completed

- **`src/evosim/life/reproduction.py` — the loop closes.** Asexual and sexual reproduction
  driven by `maturity_age`, `repro_threshold`, `offspring_count`, `parental_investment` and
  `sex_bias`, finally invoking the M2 recombination and mutation primitives. Four decisions are
  documented at the top of that module; the one worth repeating is that **offspring are endowed
  from the parent's storage capacity, not their own**. `energy.yaml` describes the cost in terms
  of the offspring's capacity, but that depends on a genome which does not exist until the
  parent has committed to making it. Estimating with the parent costs one mutation step of
  accuracy and keeps what was paid exactly equal to what was received; a ledger that balances is
  worth more than a second decimal place.

- **Fecundity is bought, never granted.** A parent that cannot afford its whole brood has a
  smaller one rather than none, so `offspring_count` is a dial and not a cliff. The
  `reproduction.overhead` loss is deposited as detritus rather than deleted, closing what would
  otherwise have been a third hole in the matter ledger.

- **`predation.py` — the two intake channels that need another organism.** Encounter rates scale
  with local density, capture is a logistic on the documented contest, and `diet_match` decides
  what a carcass is worth to a particular gut. Nothing declares trophic levels: a predator is
  any organism whose diet logits carry herbivore or carnivore weight. A prey can only be killed
  once, and what the killer cannot eat stays in the world as carrion.

- **`census.py` and prey-aware movement.** `genome.yaml` defines `sense_range` as habitat *and
  prey* detection, and `fear` as a weight on predator density; both halves were unimplemented,
  which made predation non-functional in space — a hunter would clear its cell and then wander
  off its own food supply. Cells are now also scored by expected hunting yield and by expected
  risk, both computed through the identical predation equations, so no new coefficients enter.
  Risk is priced in the only currency the model has: being eaten costs an organism its body and
  everything it saved, which is what lets `fear` weigh danger against food on one scale.

- **`sim.py` — two new stages.** The tick is now age → move → pay → eat → **hunt** → die →
  **breed** → regrow. Hunting resolves on settled ledgers, so a carcass is worth what its owner
  actually had. Predation kills join hazard and starvation deaths in a single compaction, and a
  predated organism is excluded from the corpse deposit — depositing it again would let a food
  web manufacture matter.

- **Two calibration findings.**
  - **`temp_optimum` init 15.0 → 22.0.** The reference planet's ocean averages ~21.8 °C where
    founders are seeded. At 15 °C the lineage spent its whole life outside its own thermal
    window and went extinct around day 500 — before selection could walk a locus with sigma 0.90
    the seven degrees it needed. A seed lineage that cannot survive its own seeding habitat
    tests nothing. M3 flagged this exact mismatch as an M4 test case; it turned out to be an
    initialization defect instead, and the directional-selection property is now tested
    deliberately rather than relied on by accident.
  - **The population cap was silently selecting.** Capacity throttling dropped births in
    parent-row order, and rows are roughly age-ordered after compaction, so the cap imposed
    selection for the offspring of older parents. It was invisible to every single-tick test and
    it *reversed* the measured direction of thermal selection. Births over the cap are now
    dropped at random, and there is a regression test.

- **Acceptance coverage.** 64 new tests. The selection suite is the one M4 exists for: a
  mismatched `temp_optimum` climbs toward its habitat while a matched one holds; a colder planet
  selects a colder lineage from the *same* founder; and `digestion_efficiency` climbs, which
  confirms the prediction M3 recorded but could not test. Each is a comparison rather than an
  absolute, because a trait mean after 400 ticks is partly drift.

- **Performance.** 33.5 ms per tick at 40,000 organisms, about 1.03 million organism-updates per
  second — roughly half of M3's 2.06 M/s. The food web costs that: a per-cell census every tick,
  an expected-yield and expected-risk evaluation at five candidate cells per mover per step, and
  the encounter/capture resolution itself.

### Consequences and known limitations

- **The reference planet is cap-limited, not resource-limited.** Carrying capacity is above
  1.5 million organisms — far beyond the 40,000 default `max_population`, which `sim.yaml`
  already describes as a memory budget rather than a biological statement. A default run reaches
  the cap around day 400 and stays there. `capacity_throttle` now reports this on every tick and
  in the CLI, as `sim.yaml` requires, but a run at the cap is not measuring natural carrying
  capacity and its results should not be read as if it were.
- **Predation is a fitness valley from the autotroph founder.** A hand-built hunter makes a
  living: body size 3, high aggression and sensing nets about +0.77 energy per tick, and the
  test suite pins that. But the *local* gradient at the founder points away from it — shifting
  diet weight toward herbivory costs more autotrophy than the occasional accidental kill
  returns, because the traits that make hunting pay only pay once several of them are large at
  once. Predation therefore occurs on the reference planet (a few hundred kills per tick at the
  cap) without a predatory lineage arising. Whether `p_large_effect` plus speciation can bridge
  that valley is an M5 question, not something to tune into existence now.
- **Surplus killing is bounded but real.** A sated hunter stops, because the aerobic ceiling is
  spent down across its kills and an attack that yields nothing leaves the prey alive. It can
  still kill far more than it eats when prey are small relative to its metabolism.
- **Mate search is same-cell only.** `energy.reproduction.mate_search_radius > 0` raises rather
  than being silently treated as zero.
- **The cell census uses means, not distributions.** A cell holding one large predator reads the
  same as one holding several small ones of equal total threat. Carrying full distributions into
  a five-cell comparison would cost far more than the decision is worth.
- **`temp_tolerance` did not measurably decay** over 400 ticks despite being pure upkeep for a
  thermally matched lineage — its upkeep is ~8% of basal, likely below what drift resolves at
  these population sizes. "Life gets simpler when simpler is cheaper" is therefore not yet
  demonstrated, only unfalsified.
- Every M1–M3 limitation still stands: single surface layer, deterministic climate, no weather,
  moisture as a distance proxy, and diagnostic output rather than resumable snapshots.

### Next task — Milestone 5: species and history

Give the simulation a memory. Periodic taxonomy over the genetic-distance metric that mate
compatibility already uses, so that assortative mating and a k-means split test agree on what a
species is; lineage trees; extinction confirmation via `extinction_confirm_ticks`; and the
time-series sampling `sample_interval` has been reserving. That is what turns "the population
changed" into "you can open a species and read why it became what it is".

Verified by: a lineage split into two habitats produces two species rather than one; a species
whose population goes to zero is declared extinct once and stays that way; and the recorded
history of a run reproduces from `(config, seed)` alone.

---

## Milestone 3 — ecological tick

**Status: complete.** 261 tests passing.

### Completed

- **`src/evosim/life/energy.py` — the cost and intake seam.** Every expenditure (basal with
  Kleiber scaling and a Q10 on body temperature, tissue upkeep, body support, locomotion,
  sensing, thermoregulation, armour) and both M3 intake channels (autotrophy, detritivory) are
  broadcasting NumPy expressions over explicit arrays rather than methods on an organism.
  That shape is what lets the movement layer evaluate the *same* intake formula at five
  candidate cells at once, so habitat preference cannot drift out of agreement with what
  actually feeds an organism. `Environment.sample` gathers every world field an organism
  experiences in one place, so nothing downstream indexes a grid.

- **Two limits, applied in physiological then environmental order.** Planetary oxygen sets an
  aerobic scope that caps intake at a multiple of basal cost and, by inverting the locomotion
  equation exactly, caps realized speed at what that scope can pay for. A shared cell pool then
  caps what may actually be drawn: `apply_resource_contention` scales every organism in an
  oversubscribed cell by the same factor, in two bin-counts and a gather. This is the whole of
  density dependence in M3 — nothing anywhere counts neighbours, but a cell's pool is finite.

- **`mortality.py` — hazards, senescence, and starvation.** Thermal, radiation, toxicity, and
  pressure hazards share one saturating shape, so a mild excursion is survivable and a large one
  is merely very likely to kill; a threshold would make evolution look like a step function.
  Hazards combine as `1 - prod(1 - h)` rather than by addition, so no combination can exceed
  certainty. Radiation and pressure tolerance *divide* their mismatch, giving diminishing
  returns that stop a hostile planet from selecting every tolerance to its bound. Starvation is
  deterministic: an empty ledger is an outcome, not a risk.

- **`movement.py` — neighbour choice with no coefficients of its own.** Cells are scored with
  the energy model itself, that score is divided by the organism's own basal cost to become a
  choice weight (so "worth moving for" means the same thing at every body size without an
  arbitrary temperature constant), and `sense_range` gates how much of the comparison an
  organism can perceive. Sub-unit speeds round stochastically, because rounding down would
  delete the entire lower half of the `move_speed` locus from selection. The only Python loop is
  over step index, bounded by the fastest organism alive, so cost scales with movement performed
  rather than with population size.

- **`src/evosim/sim.py` — the first real biological tick.** Age, sense and move, pay, eat, die,
  regrow — in that order, and the order is documented as a decision. Moving before feeding is
  what makes movement worth its cost; regrowing last means a cell cannot be harvested and
  replenished within one tick. `TickStats` breaks costs and deaths out by cause, because "500
  died" would make the project's explainability goal unachievable.

- **Energy is not conserved; matter is.** Autotrophy creates energy from light, which is the
  point. But every unit assimilated is removed from a named pool, and every corpse returns its
  body mass *and* its unspent energy to the detritus of the cell it fell in. Detritivores ingest
  `energy / digestion_efficiency` and void the unassimilated remainder back into the same cell,
  so the pool falls by the assimilated amount only. Tests run with regrowth and decay switched
  off and assert the pools move by exactly the reported draws and deposits.

- **`k_photo` calibrated: 0.030 → 0.300.** M3 is the first milestone able to measure this, and
  the M0 guess was an order of magnitude too low. The four factors that erode autotrophic intake
  (light ~0.83, nutrient saturation ~0.67, moisture saturation ~0.87, digestion 0.40) multiply
  to ~0.19, so *no* founder cell on the reference planet returned its own upkeep and the lineage
  always died. At 0.300 about 90% of seeded ocean cells are net-positive for the founder genome
  and the polar 10% are not — a habitable planet with unlivable edges. Reserves now climb from
  20% to 83% of storage over 30 days.

- **Performance.** `tools/profile_tick.py` now times the real thing. The 40,000-organism /
  8,192-cell default runs at **14.7 ms per biological tick, about 2.06 million organism-updates
  per second** (10-tick sample, 15.60 MiB reserved). That is *faster per organism* than M2's
  substrate pass (1.57 M/s) despite doing far more work, because the M2 benchmark re-expressed
  every genome each pass while a real tick reads the cached phenotypes. Throughput counts
  organisms actually stepped, since the population shrinks as the benchmark runs.

- **Acceptance coverage.** 88 new tests. Beyond equation shapes, they pin the *emergent* claims
  the design documents make, which is what would catch one of them quietly becoming a rule:
  high gravity shrinks the largest viable body; large bodies are thermally cheaper per unit
  mass; thin oxygen limits what a large body can earn; diet specialisation costs what it gains
  because the logits are softmaxed; an autotroph size ceiling exists on the reference planet;
  and sensing organisms end up on measurably better ground than blind ones over a 40-day run.

### Consequences and known limitations

- **A tick is now real, but a generation is not.** With no reproduction, the founder cohort ages
  and dies out by roughly day 45. That is the correct M3 outcome, not a failure: the ledger is
  positive throughout, and deaths are senescence and thermal mismatch rather than starvation.
  M4 is what closes the loop.
- **Predation and herbivory are not implemented.** `k_encounter` and the six capture
  coefficients in `energy.yaml` remain validated but unused; both diet channels that require a
  second organism belong with reproduction in M4. `aff_herbivore` and `aff_carnivore` are
  therefore currently pure cost — they take softmax weight away from the two channels that work.
- **Two coefficients look mistuned but cannot be confirmed until selection exists.**
  `upkeep_digestion` appears too cheap: raising `digestion_efficiency` from 0.40 to 0.95
  multiplies intake by 2.4 while adding only ~26% to basal cost, so selection will very likely
  pin it at its upper bound immediately. And the founder's `temp_optimum` of 15 °C sits well
  below the ~22 °C mean of the ocean it is seeded into, which is currently the largest single
  killer at ~5.5% per day. Both are good M4 directional-selection test cases; neither should be
  "fixed" by hand before the selection machinery can show what they do.
- **Support cost, not the surface-area exponent, is what caps body size** at the calibrated
  `k_photo`. The m^0.67 vs m^0.75 mismatch that `energy.yaml` describes is still real but no
  longer binds inside the `body_size` range; the linear-in-mass support term bites first, around
  body size 6. The ceiling is still emergent and still physical, but the comment in that file
  now says which mechanism actually holds.
- **Atmospheric buoyancy relieves support on land only**, exactly as `energy.yaml` specifies.
  Physically an aquatic organism should get *more* relief, not none. Left as written rather than
  invented, and flagged here; adding it is a one-line change to one formula plus a config key.
- **Movement evaluates only the four adjacent cells**, once per step, so `sense_range` above one
  cell buys sharper discrimination between neighbours rather than a wider search. It is enough
  to make sensing earn its cost (measured, not assumed) but it is not real long-range foraging.
- **Climate remains deterministic and the world remains a single surface layer.** Every M1 and
  M2 limitation still stands unchanged.

### Next task — Milestone 4: reproduction and selection

Close the loop. Asexual and sexual reproduction driven by `repro_threshold`, `offspring_count`,
and `parental_investment`, with the M2 recombination and mutation primitives finally invoked;
`capacity_throttle` recorded when `max_population` binds; offspring dispersal. Then predation
and herbivory, which need a living prey population to eat.

Verified by: a founder lineage that persists indefinitely rather than dying out at day 45; the
energy ledger still balancing across births (`reproduction.overhead` > 1 means every birth is
lossy); and the first directional-selection tests — `temp_optimum` should climb toward the
ocean's ~22 °C, and `digestion_efficiency` toward its bound, both of which M3 predicts above and
neither of which it can yet demonstrate.

---

## Milestone 2 — life substrate

**Status: complete.** 171 tests passing.

### Completed

- **`src/evosim/life/genome.py` — diploid genome operations.** `GenomeSchema` converts the
  configured 28 loci into immutable float32 arrays once. Genome batches have shape
  `(organisms, loci, 2)`; founders are homozygous at each configured `init`, and expression is
  the clipped allele mean. Weighted RMS genetic distance, Mendelian recombination, and bounded
  small/large-effect mutation are vectorised and consume only their explicit named RNG streams.

- **`phenotype.py` — the body-model seam.** Expressed traits are cached in a preallocated
  `PhenotypeBuffer`. The four diet logits use a numerically stable softmax. All current body
  geometry lives here: `mass = body_size^3` and, because slender builds trade storage for lower
  support cost, `storage_capacity = mass * energy_storage / body_slenderness`. Nothing in
  population storage derives body quantities itself, so later morphology remains a contained
  replacement. `offspring_count` stays continuous in the cache; stochastic rounding happens
  once per future reproduction event rather than changing whenever a phenotype is read.

- **`population.py` — preallocated struct-of-arrays storage.** Every array is allocated to
  `sim.max_population`, with living organisms kept in one dense prefix. Batch addition and
  stable death compaction operate through NumPy indexing, preserve alignment across genomes,
  phenotypes, positions, energy, ancestry, IDs, and future species fields, and contain no
  per-organism Python loops. Persistent IDs are never recycled. Capacity overflow raises
  explicitly so the reproduction layer can later record `capacity_throttle` rather than drop
  births silently.

- **Deterministic founder seeding.** Exactly `sim.initial_population` identical-genome founders
  are placed with replacement in the configured `water`, `land`, or `any` habitat using only
  the `init` RNG stream. Eligible cells are weighted by physical cell area, avoiding the same
  equirectangular polar bias that M1 removed from resources. A planet with no requested habitat
  fails with a clear error. Founder energy follows the documented newborn ledger:
  `parental_investment * storage_capacity`.

- **Headless diagnostics and renderer boundary.** The CLI now reports living population and
  reserved array memory. `--out` retains `world.npz`/`world.json` and adds
  `population.npz`/`population.json`; both are explicitly diagnostic dumps, not resume
  snapshots. One-organism serialization is a dictionary containing alleles, named traits,
  diet, and a nested body record so future morphology fields can be added without a format
  rewrite.

- **Performance baseline.** `tools/profile_tick.py` times the work M2 can honestly measure:
  refreshing every phenotype plus reducing positions to per-cell occupancy. On the current
  development machine, the default 40,000-organism / 8,192-cell substrate reserves 15.60 MiB
  and ran at 25.5 ms per pass (about 1.57 million organism updates/second, 10-pass sample).
  This is not called a biological tick; M3 adds the ecological work that will dominate it.

- **Acceptance coverage.** 79 new tests pin schema immutability, homozygous founders,
  clipped expression, distance scaling, deterministic recombination and mutation, stable
  softmax, body geometry, habitat validity, named-stream isolation, full-state founder
  determinism, founder energy, capacity overflow, stable aligned compaction, persistent IDs,
  dictionary serialization, hot-path loop absence, diagnostic output, and profiler behavior.
  The cross-process RNG test now also preserves the host environment, fixing its pre-existing
  Windows-only import failure.

### Consequences and known limitations

- Organisms are intentionally static in M2. `evosim --ticks N` advances climate and world
  resources for `N` days, but it does not age, feed, move, kill, or reproduce the founder
  population. Presenting that substrate pass as a simulation tick would give a misleading
  performance number and imply ecology that does not exist yet.
- Founder placement is uniform by physical area within the requested surface habitat. It does
  not yet evaluate temperature, resources, crowding, or trait-specific suitability.
- The scalar body equations above are a documented interim model, not the later 3D morphology
  system. Their containment in `phenotype.py` is the compatibility guarantee.
- Mutation and recombination primitives exist and are tested, but no reproduction scheduler
  invokes them yet. Likewise, species and ancestry arrays are storage for later layers, not a
  claim that taxonomy already exists.
- Population and world output files remain diagnostics. A resumable snapshot must also capture
  every RNG stream plus future history and scheduler state.

### Next task — Milestone 3: ecological tick

Connect the population to the world: vectorised energy costs and autotrophic/detrital intake,
resource draw and return, movement over grid neighbours, environmental mortality, ageing, and
the first real biological tick. Keep every cost routed through the energy-model seam and prove
the energy/resource ledger before adding reproduction and selection.

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

- One test asserts the founder genome is **autotroph-dominant** (the configured diet logits
  softmax to ~81% autotrophy and ~4% each herbivory/carnivory). If the founder started predatory,
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
