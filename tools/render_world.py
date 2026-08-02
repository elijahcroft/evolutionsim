"""Render deterministic world fields to a diagnostic PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evosim.config import DEFAULT_CONFIG_DIR, Config  # noqa: E402
from evosim.rng import RngBundle  # noqa: E402
from evosim.world import World  # noqa: E402
from evosim.world.biome import BIOME_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--planet", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("world.png"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.day < 0:
        raise SystemExit("--day must be non-negative")
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"sim.seed={args.seed}")
    config = Config.load(
        args.config_dir, overrides=overrides, planet_file=args.planet
    )
    world = World.create(config.planet, RngBundle(config.sim.seed))
    world.step(args.day)
    render(world, args.out, config.fingerprint())
    print(args.out.resolve())
    return 0


def render(world: World, output: Path, fingerprint: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("Elevation (km)", world.terrain.elevation_km, "terrain"),
        ("Water depth (km)", world.depth_km, "Blues"),
        ("Temperature (C)", world.climate.temperature_c, "coolwarm"),
        ("Insolation (surface)", world.climate.insolation, "inferno"),
        # The field that decides where an autotroph can earn anything, and the one to read
        # first when a run's life is not where it was expected to be.
        ("Light (at depth)", world.light, "inferno"),
        ("Moisture", world.climate.moisture, "Blues"),
        ("Nutrients", world.resources.nutrients, "YlGn"),
        ("Detritus", world.resources.detritus, "copper"),
        # Not a field but a reading of them, which is why it comes last: it is the summary the
        # eight panels above it add up to.
        ("Biome", world.biomes(), "tab10"),
    )
    extent = (-180.0, 180.0, -90.0, 90.0)
    figure, axes = plt.subplots(3, 3, figsize=(19, 10), constrained_layout=True)
    for axis, (title, values, colour_map) in zip(axes.flat, arrays, strict=True):
        categorical = title == "Biome"
        image = axis.imshow(
            values,
            extent=extent,
            aspect="auto",
            cmap=colour_map,
            interpolation="nearest",
            **({"vmin": -0.5, "vmax": len(BIOME_NAMES) - 0.5} if categorical else {}),
        )
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        bar = figure.colorbar(image, ax=axis, shrink=0.82)
        if categorical:
            bar.set_ticks(range(len(BIOME_NAMES)))
            bar.set_ticklabels(BIOME_NAMES)
    figure.suptitle(
        f"{world.config.name} — day {world.day} — seed/config {fingerprint}",
        fontsize=14,
    )
    figure.savefig(output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
