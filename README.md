# evosim

An individual-based evolution sandbox. Configure a planet's physics, seed it with one simple
organism, and watch what the planet makes of that lineage over deep time.

There is no tech tree and no progression. Every organism is a discrete individual with a diploid
genome; body plans, diets, life histories, and behaviours change only because some variants
leave more descendants than others under the current conditions. Life is not pushed toward
intelligence, complexity, or size — a lineage will get simpler when simpler is cheaper.

The design goal is that outcomes are **unpredictable beforehand but explainable afterwards**:
you should be able to pause, open a species, and read why it became what it is from the recorded
history.

## Status

**Milestone 5 of 8 — species and history.** The simulation now has a memory. Organisms age,
move, feed, hunt, die, and breed; selection is real and measured; and a run records which
species existed, which species each one came from, when each went extinct, and how their traits
moved over time. A species is not a label applied from outside — it is the same genetic-distance
threshold that already decides whether two organisms can breed, so the taxonomy and assortative
mating cannot disagree.

Two things do not happen by themselves on the reference planet, both recorded rather than tuned
away: no predatory lineage arises (hunting only pays once several traits are large at once), and
**nothing speciates** — fully isolated lineages diverge to 0.03 against a compatibility
threshold of 0.15, and no threshold value separates real divergence from drift. See
[DEVLOG.md](DEVLOG.md) for exactly what is and is not implemented, and for the measurements
behind that claim.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,viz]'
.venv/bin/python -m pytest
```

## Usage

```sh
.venv/bin/evosim                                  # resolve and report the default planet
.venv/bin/evosim --seed 7                         # a different run of the same planet
.venv/bin/evosim --set planet.gravity=1.4         # override any config value
.venv/bin/evosim --set genome.loci.body_size.sigma=0.1
.venv/bin/evosim --planet heavy.yaml              # an alternative world file
.venv/bin/evosim --ticks 365 --out runs/year_one  # run; save world, population, and history
.venv/bin/python tools/render_world.py --day 90 --out world-day-90.png
.venv/bin/python tools/profile_tick.py             # time the biological tick at 40k organisms
```

A run written with `--out` leaves `history.json` beside the world and population dumps: every
species that existed, the species it split from, its origin and extinction days, and the
per-species trait time series sampled every `sim.sample_interval` days. It reproduces exactly
from `(config, seed)`.

### Watching a run in the browser

```sh
.venv/bin/python -m pip install -e '.[server]'
.venv/bin/evosim-ui                                # http://127.0.0.1:8000
.venv/bin/evosim-ui --seed 7 --set planet.gravity=1.4   # same flags as the CLI
```

Play, pause, or single-step the world; switch the map between population density, temperature,
elevation, nutrients, detritus, moisture, insolation, and toxicity; read the tick's energy
ledger and cause-of-death breakdown beside it. Space toggles play, `s` steps.

Time only advances when the browser asks for it — there is no background thread — so a session
driven from the UI produces exactly the run `evosim --ticks N` produces from the same
`(config, seed)`. A test asserts that.

## How it is organised

```
config  →  world  →  life  →  evolution  →  history  →  sim  →  server  →  ui
```

Dependencies point one way only. `sim.Simulation` is pure and headless — it never reads the wall
clock and never imports from `server` or `ui`, which is what makes a run reproducible from
`(config, seed)` alone.

### Three rules the code holds itself to

1. **No magic numbers in simulation code.** Every tunable lives in `config/*.yaml`. If you want
   to know what selection favours, those four files plus the formulas they document are the
   complete answer.

2. **No hidden bonuses.** A trait may only be advantageous by earning energy, and may only be
   costly by spending energy or raising a mortality hazard. "High gravity favours small bodies"
   is not a rule anywhere — it falls out of body-support cost scaling with mass × gravity.

3. **No global randomness, and no per-organism Python loops in the hot path.** The first is
   enforced by lint tests that parse the package AST; the second is what makes an
   individual-based simulation viable in Python at all.

## Configuration files

| File | Contents |
| --- | --- |
| `config/sim.yaml` | Scale, population cap, bookkeeping cadence |
| `config/planet_default.yaml` | An Earth-like reference world; every value is meant to be varied |
| `config/genome.yaml` | The 28 loci — this file *is* the list of things that can evolve |
| `config/energy.yaml` | Every coefficient in the cost, intake, mortality, and reproduction equations |
