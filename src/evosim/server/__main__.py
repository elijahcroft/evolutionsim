"""Serve the UI: ``python -m evosim.server`` or ``evosim-ui``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evosim.config import DEFAULT_CONFIG_DIR, Config, ConfigError
from evosim.server.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evosim-ui",
        description="Serve the evosim browser UI over a live simulation.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--planet", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override a config value; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)

    def reload(seed: int | None) -> Config:
        overrides = list(args.overrides)
        chosen = args.seed if seed is None else seed
        if chosen is not None:
            overrides.append(f"sim.seed={chosen}")
        return Config.load(args.config_dir, overrides=overrides, planet_file=args.planet)

    try:
        config = reload(None)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"evosim ui on http://{args.host}:{args.port}  (planet: {config.planet.name}, "
          f"seed: {config.sim.seed})")
    uvicorn.run(create_app(config, reload=reload), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
