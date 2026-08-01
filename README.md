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

**Milestone 7b — a second way to make a living.** Organisms age, move, feed, hunt, die, and breed;
selection is real and measured; a run records which species existed, which species each came
from, when each went extinct, and how their traits moved; and you can now open a species in the
browser and read its ancestry, its population curve, and — ranked by how far each trait moved
relative to what its locus can do — what has changed in it since it appeared.

**A run now produces a species by itself.** On the default planet, two seeds in three split a
second lineage off the founder without being told to — one of them separated by 7 °C of thermal
optimum, both daughter populations in the thousands. That is the thing this project was built to
see, and it took until now.

What made it possible was giving the ocean a second way to live in it. Light falls off
exponentially with water depth, so the abyss is dark and photosynthesis there is not poor but
impossible. Water gets colder with depth toward a fixed floor, so the deep is a genuinely
different place to be adapted to rather than merely a dark one. Detritus sinks downhill as
marine snow, conserving matter exactly. Pressure is the water column overhead rather than a
planetary constant, which makes `pressure_tolerance` a locus selection can act on for the first
time. And detritivory was recalibrated — it had been set so low that a detritivore was
net-negative at *every* detritus level on *every* planet, which meant no environment could ever
have rewarded it.

Two things still do not happen by themselves, both measured and recorded rather than tuned away.
No predatory lineage arises: hunting only pays once several traits are large at once. And the
splits that do occur are thermal rather than trophic — the deep is livable and colder-adapted,
but nothing has yet specialised into eating what falls into it. See [DEVLOG.md](DEVLOG.md) for
the numbers and for exactly what is and is not implemented.

## Where it is going

[docs/ROADMAP.md](docs/ROADMAP.md) is the plan: bodies that are real evolvable geometry rather
than one number, a creature you can look at and an organism you can open, habitats and ranges,
and then parasites, ornaments and symbiosis. [docs/DESIGN.md](docs/DESIGN.md) is the founding
design doc — the models, the architecture, and why each was chosen.

You still cannot steer a creature, and that does not change. There is no tech tree and no
upgrade. You change the planet and you watch.

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
.venv/bin/evosim --set genome.loci.body_length.sigma=0.1
.venv/bin/evosim --planet heavy.yaml              # an alternative world file
.venv/bin/evosim --ticks 365 --out runs/year_one  # run; save world, population, and history
.venv/bin/python tools/render_world.py --day 90 --out world-day-90.png
.venv/bin/python tools/profile_tick.py             # time the biological tick at 40k organisms
.venv/bin/python tools/speciation_experiment.py    # is a split divergence, or drift?
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
elevation, water depth, nutrients, detritus, moisture, surface insolation, light at depth, and
toxicity; read the tick's energy ledger and cause-of-death breakdown beside it. Space toggles
play, `s` steps. Comparing the insolation and light layers is the quickest way to see why life
sits where it does.

Open a species from the species panel to see its ancestry, when it appeared and from what, its
population over time, and — ranked by how far each trait moved relative to what its locus can
do — **what has changed in it since it appeared**. Selecting a species also restricts the map to
that species, which is how you see whether a split is geographic or only numerical.

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
| `config/genome.yaml` | The 39 loci — this file *is* the list of things that can evolve |
| `config/energy.yaml` | Every coefficient in the cost, intake, mortality, and reproduction equations |
