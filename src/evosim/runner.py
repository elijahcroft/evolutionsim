"""Headless command-line entry point.

At milestone 0 there is no world and no life yet, so this only resolves and reports the
configuration. It exists now so that the argument surface (config dir, planet file, seed,
overrides) is fixed early -- every later milestone, the parameter sweeps, and the
directional-selection tests all drive the simulation through this interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evosim import __version__
from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.rng import STREAM_NAMES, RngBundle


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
        help="number of simulated days to run (milestone 0: accepted but not yet simulated)",
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

    try:
        config = resolve_config(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    rng = RngBundle(config.sim.seed)

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

    if args.ticks:
        print(
            f"\nmilestone 0: no world or life is implemented yet, so --ticks {args.ticks} "
            "was not simulated.",
            file=sys.stderr,
        )
    if args.out:
        print(f"milestone 0: --out {args.out} was accepted but nothing is written yet.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
