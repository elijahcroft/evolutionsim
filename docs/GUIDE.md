# Running and configuring evosim

A practical reference for running simulations and changing what they do. For what the
simulation *is* and how it works internally, see [DESIGN.md](DESIGN.md). For current status,
see the top-level [README.md](../README.md) and [DEVLOG.md](../DEVLOG.md).

## Two ways to run

**Headless — for real runs, especially long ones.** Runs as fast as your CPU allows, no
wall-clock throttling, writes results to disk.

```sh
.venv/bin/evosim --ticks 5000 --out runs/mytest
```

Writes `world.json`, `population.json`, and `history.json` into `runs/mytest/`.
`history.json` is the important one: every species that existed, its parent species, when it
appeared/went extinct, and its trait time series.

**Browser UI — for watching one unfold interactively.**

```sh
.venv/bin/pip install -e '.[server]'   # once
.venv/bin/evosim-ui
```

Opens `http://127.0.0.1:8000`. Play/pause/step, switch the map layer, open species to see
ancestry/population/trait-shift. Time only advances when you ask for it from the UI — there's
no background thread — so it's not meant for unattended long runs. Use the headless CLI for
those.

Both commands accept the same `--seed`, `--planet`, `--config-dir`, and `--set` flags.

## Changing anything: `--set KEY.PATH=VALUE`

Repeatable, works identically on `evosim` and `evosim-ui`. It walks straight into the config
YAML structure:

```sh
.venv/bin/evosim --ticks 5000 --out runs/cold \
  --set planet.gravity=1.4 \
  --set planet.o2_fraction=0.15 \
  --set genome.loci.body_length.sigma=0.1
```

| Prefix | File |
| --- | --- |
| `sim.X` | `config/sim.yaml` — scale & bookkeeping |
| `planet.X` | `config/planet_default.yaml` — world physics |
| `genome.loci.<name>.X` | `config/genome.yaml` — what can evolve, how fast |
| `energy.X` | `config/energy.yaml` — every cost/gain coefficient |

You can also point at a whole different planet file with `--planet somefile.yaml` instead of
overriding piece by piece, and `--seed N` to rerun the same planet with different founder
noise (same config + seed = bit-identical run, always).

## Knobs worth knowing about

**Planet physics** (`planet.*`):
- `gravity` (1.0 = Earth) — bigger/faster bodies cost more to support and move
- `o2_fraction` — caps the size × metabolic-rate product (aerobic scope)
- `pressure`, `pressure_per_km_depth` — makes deep water survivable only for a lineage that
  evolves `pressure_tolerance`
- `radiation` — raises mutation rate *and* mortality together; a genuine tradeoff, not a free
  "evolve faster" dial
- `axial_tilt` — 0 = no seasons; high = violent continental seasons
- `terrain.land_fraction` — ocean vs. land coverage
- `terrain.base_frequency` — few large continents vs. many small ones; your main lever for
  forcing allopatric (geographic) speciation
- `climate.deep_temperature_c`, `climate.light_attenuation_per_km` — how distinct the deep
  ocean is from the shelf

**Genome** (`genome.loci.<name>.sigma` / `.min` / `.max`) — every evolvable trait lives in
`config/genome.yaml`; that file *is* the list of things that can evolve. Set a locus's `sigma`
near 0 to effectively freeze it.

**Energy coefficients** (`energy.*`) — every `k_` constant in the cost/intake/mortality
formulas documented in `config/energy.yaml` and §8 of DESIGN.md. This is where you'd tune, say,
predation profitability if you're trying to coax a predator lineage into existing.

**Sim scale & bookkeeping** (`sim.*`, see `config/sim.yaml`):
- `max_population` — hard cap (default 40000); raise it if you have CPU/RAM headroom
- `initial_population`, `initial_habitat` (`water` / `land` / `any`)
- `taxonomy_interval`, `sample_interval` — ticks between speciation checks / trait sampling;
  lower = finer-grained history, slower run
- `extinction_confirm_ticks` — ticks at zero population before declaring extinct

## Running long

```sh
.venv/bin/evosim --ticks 50000 --out runs/long_run
```

Notes:
- No checkpoint/resume yet — `snapshot_interval` exists in `sim.yaml` but resuming from a
  snapshot isn't wired up. A run is one shot from tick 0 to `--ticks`.
- `.venv/bin/python tools/profile_tick.py` reports ticks/sec at ~40k organisms — use it to
  estimate wall-clock time before committing to a large `--ticks`.
- `.venv/bin/python tools/render_world.py --day N --out world-day-N.png` dumps a map PNG from a
  finished run without needing the browser UI.
- `.venv/bin/python tools/speciation_experiment.py` — checks whether an observed split is real
  divergence or just drift.

## Full CLI reference

```
$ .venv/bin/evosim --help
--config-dir CONFIG_DIR   directory containing sim.yaml, planet_default.yaml, genome.yaml, energy.yaml
--planet PLANET           alternative planet YAML (absolute, or relative to --config-dir)
--seed SEED                master RNG seed; overrides sim.seed from the config
--ticks TICKS               number of simulated days to run
--out OUT                   directory to write run output into
--set KEY.PATH=VALUE   override a config value; repeatable

$ .venv/bin/evosim-ui --help
--host HOST
--port PORT
--config-dir CONFIG_DIR
--planet PLANET
--seed SEED
--set KEY.PATH=VALUE
```
