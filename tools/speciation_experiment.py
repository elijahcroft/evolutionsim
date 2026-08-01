"""Measure whether a species split would be divergence or drift.

Milestone 5 recorded that nothing speciates on the reference planet: fully isolated lineages
diverged no further than the spread inside a single interbreeding population, so any threshold
low enough to separate them would also shatter one population into noise.  That was a
measurement of the default configuration.  This tool asks the follow-up question -- *is there a
configuration where the answer changes?* -- and it asks it the only way that means anything:

For each variant it runs two lineages that are **completely isolated** -- independent runs from
the same founder genome, one of them on a colder planet, so the pair gets both allopatry and
divergent selection -- and compares them against the spread inside one of them, reporting

    signal-to-noise = distance between the two isolated centroids
                      -------------------------------------------
                      typical distance between two organisms in one population

A species concept can only be meaningful where that ratio is comfortably above 1: it is exactly
the statement "two organisms from different lineages are more different than two organisms from
the same lineage".  Both numbers are in the units the mating threshold uses, so
``mate_compatibility_distance`` can be read straight off the same scale.

The variants are the three levers available without redesigning anything: how fast mutation
supplies variation, how hard selection pulls the two lineages apart, and which loci the species
concept is allowed to look at.

    .venv/bin/python tools/speciation_experiment.py --days 3000
    .venv/bin/python tools/speciation_experiment.py --days 500 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.evolution.taxonomy import TaxonomyModel
from evosim.life.population import HabitatError
from evosim.sim import Simulation

DEFAULT_DAYS = 2000
DEFAULT_PAIRS = 4000

#: A world small enough to run several thousand days of, and large enough to have climate.
SCALE = [
    "planet.grid_width=48",
    "planet.grid_height=24",
    "sim.initial_population=400",
    "sim.max_population=6000",
]

#: A colder planet for the second lineage, so isolation comes with divergent selection rather
#: than with drift alone.  This is the same lever milestone 4 used to demonstrate that thermal
#: selection is real, so its effect on trait means is already established.
COLD = ["planet.solar_constant=0.82"]


@dataclass(frozen=True, slots=True)
class Variant:
    """One question, expressed as config changes applied to both isolated lineages."""

    name: str
    overrides: list[str]
    question: str
    weights: dict[str, float] | None = None
    """Distance weights to use instead of the configured ones; loci not named get zero.

    This is not expressible as a ``--set`` override, because overrides deliberately refuse to
    create config keys that the YAML does not already contain, and
    ``genome.distance_weights.overrides`` only names the two loci it currently adjusts.  The
    species concept is one of the things under test here, so it is applied to the resolved
    config directly.
    """


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "reference",
        [],
        "does the default configuration separate isolated lineages at all?",
    ),
    Variant(
        "mutation x10",
        ["genome.loci.mutation_rate.init=0.05"],
        "does more variation to select from widen the gap, or just the noise?",
    ),
    Variant(
        "mutation x50",
        ["genome.loci.mutation_rate.init=0.25"],
        "does an implausibly high mutation rate reach the threshold?",
    ),
    Variant(
        "wide thermal sigma",
        ["genome.loci.temp_optimum.sigma=8.0"],
        "does a locus that can move fast under selection carry the whole distance?",
    ),
    Variant(
        "thermal species concept",
        [],
        "does measuring distance over the loci selection acts on undo the 28-locus dilution?",
        weights={"temp_optimum": 1.0, "body_length": 1.0, "metabolic_rate": 1.0},
    ),
)


def run(config: Config, days: int) -> Simulation | None:
    """Advance a lineage, returning ``None`` if it died on the way."""

    simulation = Simulation.create(config)
    while simulation.day < days and simulation.population.size:
        simulation.step()
    return simulation if simulation.population.size else None


def centroid(taxonomy: TaxonomyModel, simulation: Simulation) -> np.ndarray:
    population = simulation.population
    return taxonomy.coordinates(population.genomes[population.active]).mean(axis=0)


def spread(
    taxonomy: TaxonomyModel,
    simulation: Simulation,
    rng: np.random.Generator,
    pairs: int,
) -> dict[str, float]:
    """Typical and extreme distance between two organisms of one interbreeding population."""

    population = simulation.population
    coordinates = taxonomy.coordinates(population.genomes[population.active])
    draw = rng.integers(0, coordinates.shape[0], size=(2, pairs))
    distances = np.sqrt(
        np.sum((coordinates[draw[0]] - coordinates[draw[1]]) ** 2, axis=1)
    )
    return {
        "mean": float(distances.mean()),
        "p99": float(np.percentile(distances, 99)),
        "max": float(distances.max()),
    }


def measure(variant: Variant, days: int, pairs: int, config_dir: Path) -> dict[str, Any]:
    """Run one variant's two isolated lineages and compare them with themselves."""

    def load(extra: list[str]) -> Config:
        config = Config.load(config_dir, overrides=[*SCALE, *variant.overrides, *extra])
        if variant.weights is None:
            return config
        weights = tuple(
            variant.weights.get(locus.name, 0.0) for locus in config.genome.loci
        )
        return replace(config, genome=replace(config.genome, distance_weight=weights))

    warm = run(load(["sim.seed=1"]), days)
    cold = run(load([*COLD, "sim.seed=2"]), days)
    if warm is None or cold is None:
        return {
            "variant": variant.name,
            "question": variant.question,
            "extinct": True,
        }

    # The taxonomy of the variant being tested: a species concept is part of what varies here,
    # so the distances must be measured with the same weights the split test would use.
    taxonomy = warm.taxonomy
    rng = np.random.default_rng(0)
    within = spread(taxonomy, warm, rng, pairs)
    between = float(
        np.sqrt(np.sum((centroid(taxonomy, warm) - centroid(taxonomy, cold)) ** 2))
    )
    return {
        "variant": variant.name,
        "question": variant.question,
        "extinct": False,
        "days": days,
        "population": warm.population.size,
        "threshold": taxonomy.split_distance,
        "between_lineages": between,
        "within_lineage_mean": within["mean"],
        "within_lineage_p99": within["p99"],
        "within_lineage_max": within["max"],
        "signal_to_noise": between / within["mean"] if within["mean"] > 0.0 else 0.0,
        "reaches_threshold": between > taxonomy.split_distance,
        # The decisive question is not whether the lineages differ on average, but whether any
        # threshold could tell them apart: one must sit above almost every within-lineage pair
        # and below the between-lineage gap.  If the p99 is above that gap, no value exists.
        "separable": between > within["p99"],
        "species_in_one_lineage": len(warm.history.living()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"simulated days per lineage (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help=f"organism pairs sampled for within-lineage spread (default: {DEFAULT_PAIRS})",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="run only the named variant; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.days <= 0:
        print("argument error: --days must be positive", file=sys.stderr)
        return 2
    if args.pairs <= 0:
        print("argument error: --pairs must be positive", file=sys.stderr)
        return 2

    chosen = VARIANTS
    if args.variant:
        names = {variant.name for variant in VARIANTS}
        unknown = sorted(set(args.variant) - names)
        if unknown:
            print(
                f"argument error: no such variant(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(names))}",
                file=sys.stderr,
            )
            return 2
        chosen = tuple(v for v in VARIANTS if v.name in set(args.variant))

    try:
        results = [measure(v, args.days, args.pairs, args.config_dir) for v in chosen]
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except HabitatError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, sort_keys=True))
    else:
        _print_report(results, args.days)
    return 0


def _print_report(results: list[dict[str, Any]], days: int) -> None:
    print(f"Speciation scale experiment — {days} days per lineage, two isolated lineages each")
    print()
    print(
        f"  {'variant':<24}{'between':>10}{'within':>10}{'p99':>10}{'S/N':>7}"
        f"{'separable':>11}{'threshold':>11}{'reached':>9}"
    )
    for result in results:
        if result["extinct"]:
            print(f"  {result['variant']:<24}{'lineage died':>47}")
            continue
        print(
            f"  {result['variant']:<24}"
            f"{result['between_lineages']:>10.4f}"
            f"{result['within_lineage_mean']:>10.4f}"
            f"{result['within_lineage_p99']:>10.4f}"
            f"{result['signal_to_noise']:>7.2f}"
            f"{'yes' if result['separable'] else 'no':>11}"
            f"{result['threshold']:>11.3f}"
            f"{'yes' if result['reaches_threshold'] else 'no':>9}"
        )
    print()
    print("  between  : distance between the centroids of two completely isolated lineages")
    print("  within   : mean distance between two organisms of the same lineage")
    print("  p99      : 99th percentile of that same within-lineage distance")
    print("  S/N      : between / within. A species concept needs this comfortably above 1.")
    print("  separable: is between > p99? If not, no threshold value can tell them apart.")
    for result in results:
        print(f"\n  {result['variant']}: {result['question']}")


if __name__ == "__main__":
    raise SystemExit(main())
