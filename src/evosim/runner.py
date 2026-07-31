"""Headless command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evosim import __version__
from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.life import DIET_NAMES, HabitatError, Population
from evosim.rng import STREAM_NAMES
from evosim.sim import Simulation, TickStats
from evosim.world import World


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evosim",
        description="Run an evolution simulation headlessly.",
    )
    parser.add_argument("--version", action="version", version=f"evosim {__version__}")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="directory containing sim.yaml, planet_default.yaml, genome.yaml, energy.yaml",
    )
    parser.add_argument(
        "--planet",
        type=Path,
        default=None,
        help="alternative planet YAML (absolute, or relative to --config-dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master RNG seed; overrides sim.seed from the config",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="number of simulated days to run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write run output into",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "override a config value; repeatable. "
            "e.g. --set planet.gravity=1.4 --set genome.loci.body_size.sigma=0.1"
        ),
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Load config with CLI overrides applied.

    `--seed` is translated into an override rather than mutated afterwards, so that the seed is
    part of the resolved config and therefore part of its fingerprint.
    """
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"sim.seed={args.seed}")
    return Config.load(args.config_dir, overrides=overrides, planet_file=args.planet)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ticks < 0:
        print("argument error: --ticks must be non-negative", file=sys.stderr)
        return 2

    try:
        config = resolve_config(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        simulation = Simulation.create(config)
    except HabitatError as exc:
        print(f"initialization error: {exc}", file=sys.stderr)
        return 2
    stats = simulation.run(args.ticks)
    rng, world, population = simulation.rng, simulation.world, simulation.population

    planet = config.planet
    print(f"evosim {__version__}")
    print(f"  config fingerprint : {config.fingerprint()}")
    print(f"  seed               : {config.sim.seed}")
    print(f"  rng streams        : {len(rng)} ({', '.join(sorted(STREAM_NAMES))})")
    print(f"  planet             : {planet.name}")
    print(f"  grid               : {planet.grid_width} x {planet.grid_height} "
          f"({planet.n_cells} cells)")
    print(f"  gravity            : {planet.gravity} g")
    print(f"  atmosphere         : {planet.o2_fraction:.3f} O2, {planet.pressure} atm")
    print(f"  insolation         : {planet.solar_constant} x solar, "
          f"tilt {planet.axial_tilt} deg")
    print(f"  year               : {planet.year_length_days} days of "
          f"{planet.day_length_hours} h")
    print(f"  target land        : {planet.terrain.land_fraction:.0%}")
    print(f"  genome             : {config.genome.n_loci} loci")
    print(f"  founders           : {config.sim.initial_population} in "
          f"{config.sim.initial_habitat}")
    print(f"  living organisms   : {population.size} / {population.capacity}")
    print(f"  population storage : {population.memory_bytes / (1024 ** 2):.1f} MiB")
    print(f"  simulated day      : {world.day}")
    print(f"  actual land        : {world.terrain.land.mean():.1%}")
    print(
        "  temperature        : "
        f"{world.climate.temperature_c.min():.1f} to "
        f"{world.climate.temperature_c.max():.1f} C"
    )
    if stats is not None:
        _print_tick_stats(stats)

    if args.out:
        _write_run(args.out, config, world, population)
        print(f"  output             : {args.out.resolve()}")
    return 0


def _print_tick_stats(stats: TickStats) -> None:
    """Report the final tick's ledger.

    Intake and cost are shown beside each other because their difference is the single number
    that says whether the planet can currently support the lineage at all.
    """
    print(f"  last tick births   : {stats.births} from {stats.breeding_parents} parents "
          f"({stats.sexual_births} sexual)")
    print(f"  last tick deaths   : {stats.deaths} "
          f"({stats.deaths_starvation} starved, {stats.deaths_hazard} hazard)")
    if stats.capacity_throttle:
        # sim.yaml is explicit that a run which spends time at the cap is not measuring a
        # natural carrying capacity and must say so.
        print(f"  capacity throttle  : {stats.capacity_throttle} births dropped at the "
              f"population cap")
    print(f"  last tick intake   : {stats.energy_intake:.4g} "
          f"({stats.intake_autotrophy:.4g} autotrophy, "
          f"{stats.intake_detritivory:.4g} detritivory)")
    print(f"  last tick cost     : {stats.energy_cost:.4g} "
          f"(basal {stats.cost_basal:.4g}, support {stats.cost_support:.4g}, "
          f"thermo {stats.cost_thermoregulation:.4g})")
    print(f"  last tick net      : {stats.net_energy:+.4g}")
    if stats.attacks:
        print(f"  last tick hunting  : {stats.kills} kills from {stats.attacks} attacks by "
              f"{stats.hunters} hunters ({stats.intake_predation:.4g} eaten, "
              f"{stats.carrion_returned:.4g} left as carrion)")
    print(f"  cells moved        : {stats.cells_moved}")
    print(f"  mean energy        : {stats.mean_energy:.4g} "
          f"({stats.mean_energy_fullness:.1%} of storage)")


def _write_run(
    output_dir: Path,
    config: Config,
    world: World,
    population: Population,
) -> None:
    """Write diagnostic world and population arrays.

    These files intentionally are not called snapshots: full resume support must also persist
    every RNG stream and future evolution/history state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "world.npz", day=world.day, **world.arrays())
    metadata = {
        "format": 1,
        "config_fingerprint": config.fingerprint(),
        "seed": config.sim.seed,
        "planet": config.planet.name,
        "day": world.day,
        "shape": list(world.grid.shape),
    }
    with (output_dir / "world.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    np.savez_compressed(
        output_dir / "population.npz",
        **population.active_arrays(),
    )
    population_metadata = {
        "format": 1,
        "kind": "diagnostic_population_dump",
        "config_fingerprint": config.fingerprint(),
        "seed": config.sim.seed,
        "day": world.day,
        "size": population.size,
        "capacity": population.capacity,
        "n_loci": population.schema.n_loci,
        "locus_names": list(population.schema.names),
        "trait_names": list(population.phenotypes.trait_names),
        "diet_names": list(DIET_NAMES),
        "genome_shape": [population.size, population.schema.n_loci, 2],
    }
    with (output_dir / "population.json").open("w", encoding="utf-8") as handle:
        json.dump(population_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
