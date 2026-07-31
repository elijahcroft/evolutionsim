"""Profile the biological tick.

From Milestone 3 this measures the real thing: ageing, sensing and movement over
grid neighbours, every cost and intake term, resource draw and return, and
mortality with corpse deposition.

The population shrinks as the benchmark runs, because a tick kills and Milestone
3 has no reproduction to replace the dead.  Throughput is therefore reported
against the organisms actually stepped rather than against the starting count,
and the starting and ending populations are both shown so a suspiciously fast
number is visible as a small one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.life.population import HabitatError
from evosim.sim import Simulation

DEFAULT_ITERATIONS = 10
BENCHMARK_NAME = "m3_biological_tick"
BENCHMARK_DESCRIPTION = (
    "age, sense and move, pay all costs, feed and draw resources, then apply mortality"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Milestone 3 has no reproduction, so the population falls as the "
            "benchmark runs; throughput counts the organisms actually stepped."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="directory containing the four simulation YAML files",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        metavar="COUNT",
        help="active organisms to profile (default: sim.max_population)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        metavar="COUNT",
        help=f"timed biological ticks (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master RNG seed (default: sim.seed from configuration)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable JSON object instead of the human report",
    )
    return parser


def run_benchmark(
    config: Config,
    *,
    population_count: int,
    iterations: int,
    seed: int,
    config_dir: Path,
) -> dict[str, Any]:
    """Set up and time the biological tick."""
    benchmark_config = replace(
        config,
        sim=replace(
            config.sim,
            seed=seed,
            initial_population=population_count,
        ),
    )

    setup_started = perf_counter()
    simulation = Simulation.create(benchmark_config)
    # One untimed tick removes one-off cache and allocation effects from the measurement.
    simulation.step()
    setup_seconds = perf_counter() - setup_started

    population = simulation.population
    stepped = 0
    timed_started = perf_counter()
    for _ in range(iterations):
        stepped += population.size
        simulation.step()
    elapsed_seconds = perf_counter() - timed_started

    # perf_counter is monotonic, but guard the division for synthetic clocks in tests.
    measured_seconds = max(elapsed_seconds, np.finfo(np.float64).eps)
    occupancy = np.bincount(
        population.cell[population.active], minlength=population.n_cells
    )

    return {
        "benchmark": BENCHMARK_NAME,
        "description": BENCHMARK_DESCRIPTION,
        "biological_tick": True,
        "config_dir": str(config_dir.resolve()),
        "config_fingerprint": benchmark_config.fingerprint(),
        "seed": seed,
        "population": population_count,
        "final_population": population.size,
        "organisms_stepped": stepped,
        "capacity": population.capacity,
        "iterations": iterations,
        "world_cells": population.n_cells,
        "setup_ms": setup_seconds * 1_000.0,
        "memory_bytes": population.memory_bytes,
        "memory_mib": population.memory_bytes / (1024.0**2),
        "total_ms": measured_seconds * 1_000.0,
        "ms_per_pass": measured_seconds * 1_000.0 / iterations,
        "organisms_per_second": stepped / measured_seconds,
        "occupancy_total": int(occupancy.sum()),
        "occupied_cells": int(np.count_nonzero(occupancy)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.iterations <= 0:
        return _argument_error("--iterations must be positive")
    if args.population is not None and args.population <= 0:
        return _argument_error("--population must be positive")
    if args.seed is not None and args.seed < 0:
        return _argument_error("--seed must be non-negative")

    try:
        config = Config.load(args.config_dir)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    population_count = (
        config.sim.max_population if args.population is None else args.population
    )
    if population_count > config.sim.max_population:
        return _argument_error(
            f"--population ({population_count}) exceeds configured "
            f"sim.max_population ({config.sim.max_population})"
        )
    seed = config.sim.seed if args.seed is None else args.seed

    try:
        result = run_benchmark(
            config,
            population_count=population_count,
            iterations=args.iterations,
            seed=seed,
            config_dir=args.config_dir,
        )
    except HabitatError as exc:
        print(f"benchmark setup error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        _print_human_report(result)
    return 0


def _argument_error(message: str) -> int:
    print(f"argument error: {message}", file=sys.stderr)
    return 2


def _print_human_report(result: dict[str, Any]) -> None:
    print("Biological tick performance diagnostic")
    print(
        "  scope              : age, move, cost, intake and resource draw, "
        "mortality and corpse return"
    )
    print("  biological tick    : yes")
    print(f"  config fingerprint : {result['config_fingerprint']}")
    print(f"  seed               : {result['seed']}")
    print(
        f"  population         : {result['population']:,} -> "
        f"{result['final_population']:,} of {result['capacity']:,} "
        "(M3 has no reproduction)"
    )
    print(f"  world cells        : {result['world_cells']:,}")
    print(f"  iterations         : {result['iterations']:,}")
    print(f"  setup              : {result['setup_ms']:.3f} ms")
    print(
        f"  reserved memory    : {result['memory_mib']:.2f} MiB "
        f"({result['memory_bytes']:,} bytes)"
    )
    print(f"  biological tick    : {result['ms_per_pass']:.3f} ms/tick")
    print(
        f"  throughput         : {result['organisms_per_second']:,.0f} organisms/s "
        f"({result['organisms_stepped']:,} stepped)"
    )
    print(
        f"  occupied cells     : {result['occupied_cells']:,} "
        f"({result['occupancy_total']:,} organisms counted)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
