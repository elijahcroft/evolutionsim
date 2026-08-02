# Roadmap — the biosphere you can look at

Supersedes §11 of [DESIGN.md](DESIGN.md). Everything else in that document — the vision, the
energy model, the architecture, the three rules — still stands.

`DEVLOG.md` records what each milestone actually cost. This file records what is next and how
we will know it landed.

## Where the project is

M0–M11 are **complete**: 452 tests passing, a run you can watch in a browser, open a species in,
read what has changed in it since it appeared, open one organism and read its own ledger, and see
what kind of place each of them is standing in.

| M | Deliverable | Status |
| --- | --- | --- |
| 0 | Scaffold, config loader + validation, RNG discipline, CLI skeleton | complete |
| 1 | World: terrain, climate, seasons, resource fields | complete |
| 2 | Genome, phenotype, `Population` SoA, mutation, recombination | complete |
| 3 | Energy, costs, hazards, reproduction — autotrophs only | complete |
| 4 | Detritivory, herbivory, predation, encounters, movement | complete |
| 5 | Species assignment, speciation, extinction, phylogeny | complete |
| 6 | Server + UI: map, heatmaps, charts, species inspector | complete |
| 7 | Depth & light: a second way to make a living | complete (took two passes — see `DEVLOG.md`) |
| 8 | The morphology genome, and the creature viewer brought forward from M9 | complete |
| 10 | The organism inspector: open one individual and read its own ledger | complete |
| 11 | Where they live: biomes, habitat lines, the two-species overlap map | complete |

Two things measured at M6 shape everything below, and both are recorded rather than tuned away:
no predatory lineage arises unaided, and **nothing speciates** — two *completely isolated*
lineages end up closer together than two organisms of the same population routinely are, under
every lever tried. The threshold was never the problem. The reference planet offers one way to
make a living. See `DEVLOG.md` for the numbers.

## Where it is going

The direction changed after M6. The original M7–M8 (god tools, then batch experiments) is
deferred in favour of making the biosphere something you can *look at*: real bodies, real
habitats, and creatures whose interactions with each other are worth watching.

**The one thing that does not change:** you cannot steer a creature. There is no tech tree, no
upgrade, no trait editing. You change the planet and you watch. Creatures become far more
legible; they do not become controllable.

| M | Deliverable | Verified by |
| --- | --- | --- |
| ~~7~~ | ~~Depth & light~~ — built, criterion not met. The light gradient produces **zonation without differentiation**: life stops at 2 km and its diet is identical at every depth it reaches. Two experiment variants became separable (best S/N 2.05 → 4.16) but the reference did not, and three 4,000-day seeds each produced one species. Diagnosis in `DEVLOG.md`: the fitness gradient never points down, so no lineage lingers where detritivory would pay. The unfinished piece is a **thermocline** — cold deep water would slow detritus decay where it lands and give the deep a second thermal optimum to adapt to. — **Done on the second pass.** The blocker turned out to be upstream of the light: `k_detritus` was set so low that a detritivore was net-negative at every detritus level on every planet, so no environment could ever have rewarded the alternative. Recalibrating it, plus the thermocline, moved the reference variant from `separable: no` (S/N 1.62) to **`separable: yes` (S/N 2.86)**, and a default run now produces a second species unaided in two seeds of three. | ✅ |
| ~~8~~ | The morphology genome — bodies stop being one number. **Done**, with the M9 viewer brought forward into it at ej's request. Thirteen loci; `body_size` and `body_slenderness` retired in favour of geometry integrated in `phenotype.py`; volume, surface area, frontal area, limb count and slenderness enter the cost equations through the existing `Costs`/`EnergyModel` seam, each factor written to evaluate to exactly 1.0 at founder proportions so nothing calibrated in M3/M7/M7b had to move. | ✅ |
| ~~9~~ | The creature viewer — **folded into M8**. `buildCreature` ported from the spike into `src/evosim/ui/index.html`; body colour comes from diet rather than a `hue` gene. Outstanding: at 4,000 days two real species still look nearly alike, because morphology drifts slowly. | ✅ (see note) |
| ~~10~~ | The organism inspector — **done.** `Simulation.inspect` re-evaluates the tick's own equations for one organism, `GET /api/organism/{id}` and `GET /api/cell/{cell}` expose it, and a map click opens the animal, its itemised ledger, its hazards by cause, and a sentence naming the largest of them. No new state: `_feed` was split so contention is computable without consuming the world, and the creature viewer became a factory so an individual gets its own body beside its species' average one. | ✅ |
| ~~11~~ | Where they live — biomes, ranges, auto-written habitat. **Done.** Ten biomes classified on demand from fields that already exist and stored nowhere; per-species occupancy sampled on the existing `SpeciesSample` cadence; `habitat_line` writes the sentence from the distribution; and the `?species=` range map generalised to two ids in two colours. One design correction on the way, recorded in `DEVLOG.md`: the deep/shallow cut had to be made on `transmittance` rather than on `light`, because a light-based cut called every polar cell "deep ocean" for half the year. | ✅ |
| **12** | Parasites & disease | A host–parasite population cycle appears in the charts and virulence evolves |
| **13** | Ornaments & mate choice | Ornament mean and preference mean covary; a preference-locked control shows no such drift |
| **14** | Symbiosis & cooperation | A mutualist pair outperforms both partners alone; `sociality` stops being a dead locus |
| **15** | The illustrated phylogeny, and run comparison | Read a run's history as a tree with a creature at each tip; 20 seeds × 2 planets compared |

M7–M11 changes what the project is to look at. M12–M14 is what makes it worth looking at twice.
M15 is the payoff view.

### Why M7 came first, and what it settled

Every feature after it is a feature about *variety between creatures*: a gallery of one species
is one animal, host–parasite cycles need two lineages, and "where they live" is only interesting
when different things live in different places. M7 was the enabler, and the only engine milestone
in this list.

It took two passes, and it cleared the bar. **M8 onward now has what it assumed**: more than one
species to put a body on, a gallery with more than one animal in it, and two lineages that could
in principle acquire a relationship. The engine detour is over.

Two loose ends carried forward, neither blocking M8: one planet in three still does not split,
and the splits that do happen are thermal rather than trophic — nothing has yet specialised into
eating what falls into the deep.

---

## M7 — Depth and light: a second way to make a living

**In one line:** light attenuates with water depth, so below a certain depth photosynthesis is
physically impossible and detritivory is the only living available. That is an environmental
pressure, not an authored rule — which is the point.

This was considered and declined during M0; the reasoning, and the note that it is "a contained
change to the world layer, not a rewrite", is in `DEVLOG.md` under the morphology spike.

**Changes**

- `src/evosim/world/world.py` — a `depth_km` field derived from the existing
  `terrain.elevation_km` and `planet.terrain.sea_level` (no new terrain generation), and a
  `light` field = surface `insolation × exp(-attenuation × depth_km)` on water, `insolation` on
  land. Register both in `World.arrays()`.
- `src/evosim/life/energy.py` — `Environment.sample` gains `light` and `depth_km`;
  `EnergyModel.photosynthesis` reads `light` rather than `insolation`. Pressure cost and hazard
  finally read a real depth, which makes `pressure_tolerance` a live locus instead of the nearly
  dead one DESIGN.md admits it is.
- `src/evosim/life/movement.py` — the neighbour-scoring gather in `_take_one_step` picks the new
  field up through `foraging_yield`. No new scoring term.
- `config/planet_default.yaml` — `climate.light_attenuation_per_km`. One number.
- `src/evosim/server/app.py` — add `light` and `depth_km` to `LAYERS`.

**Reuse, don't rebuild:** `Grid.cell_area_weights`, `Resources.upwelling_bonus` (already scales
water nutrient regrowth by ruggedness — deep-water nutrient supply is half-modelled already), and
`World.arrays()`.

**Verified by:** `tools/speciation_experiment.py`, with the two lineages seeded on opposite sides
of the photic boundary, reporting `separable: yes` — the between-lineage gap clearing the
within-lineage p99, which is the criterion M6 wrote. Then the real test: a default run producing a
second species without being told to. Plus unit tests — light falls monotonically with depth and
equals insolation on land (`test_world.py`); photosynthesis is zero below the photic depth
(`test_energy.py`).

## M8 — The morphology genome

The backbone of everything after it. "See what they look like" requires something to look at, and
`mass = body_size ** 3` is not it.

This was spiked first. `spikes/morphology/index.html` walked a lineage from a featureless blob to
a tailed, headed, limbed animal in seven generations of hand selection, and `spikes/README.md`
records which genes earned their place and which did not. **M1–M6 were built to four constraints
specifically so this lands as an addition rather than a rewrite** (see the spike entry in
`DEVLOG.md`): body geometry stays behind `phenotype.py`, all cost formulas enter through one seam
in `energy.py`, organism serialisation is a dict, and `genome.yaml` grows a morphology group only
when it is real. Those constraints held. Collect on them.

**The twelve genes that earned their place** — `body_length`, `body_radius`, `fullness`, `taper`,
`segment_count`, `segment_depth`, `radial_symmetry`, `limb_pairs`, `limb_length`, `limb_splay`,
`head_size`, `tail_length`, `dorsal_fin`. Cut `limb_droop`, `limb_segments` and `eye_size`: the
spike found them invisible or decorative, and every extra locus is genome size, mutation surface
and UI clutter.

**Changes**

- `config/genome.yaml` — a `morphology:` group. `body_size` becomes **derived**, not a locus:
  volume falls out of the geometry. This is the one breaking change in the roadmap, and it is the
  one `phenotype.py` was designed to absorb.
- `src/evosim/life/phenotype.py` — `PhenotypeBuffer.update` is the single documented place scalar
  body geometry is derived. It gains `volume`, `surface_area`, `limb_count` and `cross_section`,
  computed vectorised over the whole batch. `mass = volume × density`; `storage_capacity` reads
  real volume.
- `src/evosim/life/energy.py` — morphology enters the cost equations, or it is decoration:
  `limb_count`/`limb_length` → locomotion cost and `realized_speed`; `surface_area / volume` →
  thermoregulation (the existing 2/3 exponent becomes a real ratio); `cross_section` → drag,
  differing in water and on land; `segment_count`/`taper` → support cost under gravity. All
  through the existing `Costs`/`EnergyModel` seam. No tick-loop changes.

**The hard part, named in advance.** The spike found that individually reasonable gene bounds are
*jointly* absurd — an arched spine is fine on a thick body and grotesque on a thin one. Two
mitigations in order: keep per-locus sigma small relative to span (`config.py` already enforces
`sigma ≤ span/4`, and that guard is exactly right here), and express the worst offenders as
*ratios of* other genes rather than absolutes, so the constraint is structural rather than
checked. Explicit joint-constraint validation in `config.py` is the fallback.

**Verified by:** a heritability test — offspring geometry vectors closer to their parent's than to
the population mean, the property the spike confirmed by eye, now asserted. A cost test: two
organisms identical but for `limb_pairs`, where the limbed one pays strictly more locomotion and
moves faster, and neither is free. And the existing `tests/test_selection.py` scenarios still
passing — a body-plan change must not silently break the M4 selection results.

## M9 — The creature viewer

Port `buildCreature` out of the spike and into the product. It is ~120 lines of dependency-free
WebGL2, which suits a UI that is one self-contained `index.html` with no build step.

**Changes**

- `src/evosim/ui/creature.js` — the genotype→geometry function plus the matrix helpers, served as
  package data alongside `index.html`. It becomes a real module with a test, not a spike. Once
  this lands, `spikes/morphology/` is superseded and should be deleted per `spikes/README.md`.
- `src/evosim/ui/index.html` — a creature panel beside the map: a rotating animal drawn from the
  selected species' trait means, plus a **gallery strip** of every living species as a small
  portrait. The gallery is nearly free once one creature renders, and it is the single biggest
  change in what the project feels like.
- `src/evosim/server/app.py` — `/api/species/{id}` gains the species' morphology trait means. It
  already returns the record, lineage and drift table. No new endpoint.

Colour comes from the genome too — diet mix, and `camouflage` against the local substrate — so a
carnivore and an autotroph read differently at a glance without anything being authored.

**Verified by:** three species, three visibly different animals; a portrait's silhouette changing
when its genome's morphology means change; and a headless test that the geometry builder produces
a valid non-degenerate mesh for the founder genome and for both extremes of every morphology
locus. The degeneracy the spike hit at 3× mutation strength should be caught by a test, not by
looking at it.

## M10 — The organism inspector

The energy ledger currently exists only in aggregate — `TickStats`, 38 fields, rendered in the UI.
Every number in it is computed per organism and then summed. Expose one organism's row.

**Changes**

- `src/evosim/server/app.py` — `GET /api/organism/{organism_id}`. `Population.organism_dict`
  already exists, and `organism_id` is persistent and never reused, which is exactly what a stable
  inspector selection needs.
- The panel shows: its body (the M9 renderer, on this individual's genome rather than a species
  mean); its energy budget itemised the way the tick ledger already itemises it — basal, support,
  locomotion, sensory, thermoregulation, armor against photosynthesis, detritivory, predation; its
  diet softmax; age against `maturity_age`; energy against `repro_threshold` and
  `storage_capacity`; its parents' ids; and its current hazard rates by cause from
  `MortalityModel.hazards`.
- Map click → nearest organism in that cell → inspector. This is also how "where they live"
  becomes navigable rather than statistical.

**The design constraint:** this is a *read*. No new state, no new arrays, nothing recorded for it.
If a number is not already computed during the tick, the inspector does not show it.

**Verified by:** picking a starving organism and explaining, from its panel alone, why it is
dying — which channel is short and what it is spending on. That is M6's "explain a trait shift
from the UI alone", moved down to the individual.

## M11 — Where they live

**Built as specified, with one correction: the deep/shallow cut is on `transmittance`, not on
`light`.** See `DEVLOG.md`. The rest of this section is what was planned and what landed.

**Changes**

- `src/evosim/world/` — a `biome` classification derived from fields that already exist:
  temperature, moisture, light/depth, land mask. A pure function of the world, no new state, added
  to `World.arrays()` and `LAYERS`.
- `src/evosim/history/records.py` — per-species habitat occupancy sampled at the existing
  `SpeciesSample` cadence: the distribution of biomes its members occupy. Cold path only, like
  everything else in `history/`.
- The species panel writes a habitat line from that distribution — *"cold shallow water, 62% of
  its range; deep ocean, 30%"* — generated from the numbers, never authored.
- A **two-species overlap view**: M6's `?species=N` range-map filter generalised to two ids in two
  colours, which is how you see at a glance whether a split was geographic.

**Reuse:** `Session._density` and the `?species=` filter path exist, and already refuse to apply
the filter to layers where it would be meaningless. Extend that; don't duplicate it.

**Verified by:** two species at different depths producing different habitat lines, and the
overlap view showing disjoint ranges — with the text derived from occupancy, asserted in
`tests/test_history.py`.

## M12 — Parasites and disease

The most dramatic thing here to watch, and the cheapest of the three biology milestones, because
it needs no new lineage machinery: a parasite is a lineage.

**Changes**

- `src/evosim/life/infection.py` — a new tick phase between hunt and reap. Transmission is
  cell-local contact, and `CellNeighbourhood.sample_other` already exists for exactly this shape
  of draw; it is what mate search and attack draws use.
- New loci: `virulence` and `transmissibility` on the parasite side, `immune_investment` on the
  host side. Immunity costs energy through the existing `Costs` seam, so resistance is never free.
- Infection state is one `int32` array on `Population`, following the standard SoA discipline in
  `add`/`remove`.
- A new named RNG stream. Streams are keyed by `crc32(name)`, not spawn order, so adding one **does
  not shift the numbers any existing subsystem sees** — a property built in at M0 for this.
- `TickStats` gains infection fields. There are exactly three call sites to update:
  `runner._print_tick_stats`, `server._history_sample`, `server._stats_payload`.

**Verified by:** a host–parasite cycle visible in the UI charts, with the parasite peak lagging
the host peak; virulence evolving away from both extremes when transmission depends on host
survival; and a control run with transmission disabled showing no cycle.

## M13 — Ornaments and mate choice

Where genuinely strange-looking, non-optimal creatures come from. It is only possible after M8,
because an ornament needs a body part to be an ornament *of*.

**Changes**

- Ornament loci reuse M8 morphology genes — `dorsal_fin`, `tail_length`, plus a colour locus.
  They are already costly through the energy seam, which is what makes an ornament an honest
  handicap rather than a free flourish.
- A `preference` locus per ornament, and mate choice in `ReproductionModel._choose_mates` weighting
  candidates by preference-vs-ornament match. Mate choice today is assortative-by-genetic-distance
  only, so this is an additional weighting on an existing search inside `mate_search_radius`.
- `config/genome.yaml`'s `distance_weights:` needs a decision per new locus. The file already
  supports weighting a locus out of the *species concept* without excluding it from evolution —
  `mutation_rate: 0.0` does this today.

**Verified by:** ornament mean and preference mean covarying over a run — the signature of runaway
selection — against a control with the preference locus's sigma set to zero, which is a one-flag
`--set genome.loci.preference.sigma=0` run the override system already supports.

## M14 — Symbiosis and cooperation

**Changes**

- `sociality` is currently a declared locus with **no consumer anywhere in `life/`**. Confirm that,
  then give it one: a grouping term in movement scoring (attraction to conspecific density, which
  `CellCensus` already reduces per cell), with a real payoff (predator dilution against
  `PredationModel.encounter_rate`) and a real cost (local resource contention, which
  `apply_resource_contention` already models).
- Mutualism as the inverse of M12's infection machinery: the same cell-local association, with an
  energy transfer positive for both parties. Reusing that association state rather than adding a
  second mechanism is the whole reason this milestone comes after M12.

**Verified by:** a mutualist pair whose members each do better paired than alone, measured in the
energy ledger; and `sociality` showing directional selection under at least one predation regime —
the locus finally doing something.

## M15 — The illustrated phylogeny, and run comparison

- A phylogeny tree rendered from `History.summary()` — which already returns every species with its
  parent, origin day, extinction day and peak — with an M9 creature portrait at each tip. This is
  the "read the whole run" view the project has been building toward since M5.
- Run comparison: N seeds × M planet configs, over the `--set` override path and
  `Config.fingerprint()`, both of which have existed since M0.
  `tools/speciation_experiment.py` already demonstrates the pattern.

**Verified by:** 20 seeds × 2 planets showing a statistically significant difference in a
morphology trait — DESIGN.md's original success test, now with bodies to measure.

---

## Notes for whoever implements this

- **`snapshot_interval` in `config/sim.yaml` is 0 and unimplemented.** It is the hook for rewind.
  Nothing here needs it; do not implement it speculatively.
- **`sociality` has no consumer.** Verify before M14 builds on it.
- **There is no generic event log** — only `SpeciesRecord` plus samples. M12–M14 will want discrete
  events ("first infection", "first mutualism"). Add an append-only event list to
  `history/records.py` when M12 needs it, not before.
- **New RNG streams are free**, by name, by design. New `TickStats` fields cost three call sites.
  New world fields cost two lines: `World.arrays()` and `LAYERS`.
- **`config.py` rejects unknown keys with their full dotted path**, so every new config key must be
  added to its validating dataclass — and a typo will say so.

## What "done" means, per milestone

Unchanged from M0: tests pass, and `DEVLOG.md` gets an entry recording done / limitations / next.

```sh
.venv/bin/python -m pytest                 # all green, biology markers included
.venv/bin/python tools/profile_tick.py     # throughput recorded in DEVLOG — the
                                           # no-Python-loops rule is a performance claim
                                           # and has to keep being measured
.venv/bin/evosim-ui                        # then look at it
```

Plus the milestone's own acceptance criterion above. Each one is falsifiable and most are a number
rather than an opinion.

That last step is not a formality. Watchability is the stated priority of M7–M15, and a milestone
that passes its tests and looks wrong has not landed.
