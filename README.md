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

**Milestone 4 of 8 — reproduction, selection, and the food web.** The loop is closed. Organisms
age, move, feed, hunt, die, and breed; a founder cohort is now a lineage that outlives it by
hundreds of generations. Selection is real and measured: seed a lineage two degrees below the
temperature of the water it lives in and its descendants climb toward it, and the same founder
on a colder planet ends up colder — with nothing in the code naming a fitness function.

Predation and herbivory work, but no predatory lineage arises on the reference planet by itself:
the traits that make hunting pay only pay once several of them are large at once, so it sits
across a fitness valley. Species, lineage history, and extinction records are Milestone 5. See
[DEVLOG.md](DEVLOG.md) for exactly what is and is not implemented, including two calibration
findings and one bug where the population cap was quietly doing selection of its own.

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
.venv/bin/evosim --ticks 365 --out runs/year_one  # run the simulation; save world + population
.venv/bin/python tools/render_world.py --day 90 --out world-day-90.png
.venv/bin/python tools/profile_tick.py             # time the biological tick at 40k organisms
```

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
