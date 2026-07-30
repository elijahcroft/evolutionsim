"""Seeded randomness discipline.

Determinism is a hard requirement of this project: the same config plus the same seed must
produce a bit-identical run. That is only achievable if *every* random draw comes from a
generator whose position in the stream is a function of the simulation's own logic and
nothing else.

Two rules follow, and both are enforced by tests:

1. **Never touch the global RNG.** `np.random.random()`, `random.random()`, and friends draw
   from process-global state that any other module (or a debugger, or a test) can perturb.
   Only generators obtained from an `RngBundle` may be used.

2. **One independent stream per subsystem.** If mortality and reproduction shared a
   generator, adding a single extra draw to the mortality code would shift every subsequent
   reproduction draw, silently changing every run. Independent streams make each subsystem's
   consumption invisible to the others.

Streams are derived from a *stable hash of the stream name*, not from spawn order. This means
adding a new subsystem later does not change the numbers any existing subsystem sees --
otherwise every recorded run and every tuned parameter would be invalidated by an unrelated
feature. See `_spawn_key`.
"""

from __future__ import annotations

import zlib
from typing import Iterator

import numpy as np

# Every random-consuming subsystem declares its stream here. Order is irrelevant by design
# (see module docstring), but the set is explicit so that a typo raises rather than silently
# creating a fresh, undocumented stream.
STREAM_NAMES: tuple[str, ...] = (
    "terrain",      # procedural elevation / world generation
    "climate",      # any stochastic weather component
    "resources",    # resource field noise
    "init",         # seeding the founder population
    "mutation",     # per-allele mutation
    "recombination",  # meiosis / allele choice
    "repro",        # mate search, offspring placement
    "move",         # movement and foraging choices
    "encounter",    # predation encounters and capture rolls
    "death",        # mortality hazard rolls
    "speciation",   # k-means initialisation for the split test
    "misc",         # deliberate catch-all; prefer a named stream
)

_MAX_SPAWN_KEY = 1 << 32


def _spawn_key(name: str) -> int:
    """Map a stream name to a stable 32-bit spawn key.

    `zlib.crc32` is used rather than `hash()` because CPython randomises string hashing per
    process (PYTHONHASHSEED), which would make runs non-reproducible across invocations --
    exactly the bug this module exists to prevent.
    """
    return zlib.crc32(name.encode("utf-8")) % _MAX_SPAWN_KEY


class RngBundle:
    """A collection of independent, reproducibly-seeded generators.

    Access a stream as an attribute or by item:

        rng = RngBundle(seed=42)
        rng.mutation.normal(size=10)
        rng["death"].random(100)
    """

    __slots__ = ("_seed", "_streams")

    def __init__(self, seed: int, stream_names: tuple[str, ...] = STREAM_NAMES) -> None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        duplicates = sorted({n for n in stream_names if stream_names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate stream name(s): {duplicates}")

        keys = {name: _spawn_key(name) for name in stream_names}
        collisions = _find_collisions(keys)
        if collisions:
            # Astronomically unlikely, but a silent collision would give two subsystems the
            # same number stream, which is a subtle and horrible bug. Fail loudly instead.
            raise ValueError(f"stream name spawn-key collision: {collisions}")

        self._seed = seed
        self._streams: dict[str, np.random.Generator] = {
            name: np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))
            )
            for name, key in keys.items()
        }

    @property
    def seed(self) -> int:
        return self._seed

    def __getattr__(self, name: str) -> np.random.Generator:
        # Only reached when normal attribute lookup fails, so __slots__ members are safe.
        try:
            return self._streams[name]
        except KeyError:
            raise AttributeError(
                f"unknown RNG stream {name!r}; add it to evosim.rng.STREAM_NAMES"
            ) from None

    def __getitem__(self, name: str) -> np.random.Generator:
        try:
            return self._streams[name]
        except KeyError:
            raise KeyError(
                f"unknown RNG stream {name!r}; add it to evosim.rng.STREAM_NAMES"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._streams

    def __iter__(self) -> Iterator[str]:
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    def __repr__(self) -> str:
        return f"RngBundle(seed={self._seed}, streams={len(self._streams)})"

    # -- state capture ---------------------------------------------------------------
    # Needed so a run can be snapshotted and resumed, and so determinism tests can assert
    # that two runs are at the same point in every stream rather than merely producing the
    # same visible state.

    def get_state(self) -> dict[str, object]:
        """Return a deep-copyable snapshot of every stream's internal position."""
        return {
            "seed": self._seed,
            "streams": {name: gen.bit_generator.state for name, gen in self._streams.items()},
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore stream positions previously captured by `get_state`."""
        streams = state["streams"]
        assert isinstance(streams, dict)
        missing = set(self._streams) - set(streams)
        extra = set(streams) - set(self._streams)
        if missing or extra:
            raise ValueError(
                f"RNG state does not match this bundle (missing={sorted(missing)}, "
                f"unexpected={sorted(extra)})"
            )
        for name, gen_state in streams.items():
            self._streams[name].bit_generator.state = gen_state
        self._seed = int(state["seed"])  # type: ignore[arg-type]


def _find_collisions(keys: dict[str, int]) -> list[tuple[str, str]]:
    seen: dict[int, str] = {}
    out: list[tuple[str, str]] = []
    for name, key in keys.items():
        if key in seen:
            out.append((seen[key], name))
        else:
            seen[key] = name
    return out
