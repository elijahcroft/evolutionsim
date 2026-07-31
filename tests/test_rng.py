"""Tests for the seeded-randomness discipline.

These are the foundation of every later determinism guarantee: if the RNG layer is not
reproducible, no amount of care elsewhere makes a run repeatable.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evosim.rng import STREAM_NAMES, RngBundle, _spawn_key

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "evosim"


def test_same_seed_reproduces_every_stream():
    a = RngBundle(seed=123)
    b = RngBundle(seed=123)
    for name in STREAM_NAMES:
        assert np.array_equal(a[name].random(64), b[name].random(64)), name


def test_different_seeds_differ():
    a = RngBundle(seed=1)
    b = RngBundle(seed=2)
    for name in STREAM_NAMES:
        assert not np.array_equal(a[name].random(64), b[name].random(64)), name


def test_streams_are_mutually_independent():
    """Draws from one stream must not be correlated with, or identical to, another's."""
    rng = RngBundle(seed=7)
    draws = {name: rng[name].random(256) for name in STREAM_NAMES}
    names = list(draws)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not np.array_equal(draws[first], draws[second]), (first, second)


def test_consuming_one_stream_does_not_disturb_others():
    """The core reason subsystems get separate streams.

    Adding a random draw to (say) mortality must not shift the numbers reproduction sees,
    otherwise any change to one subsystem silently changes every run.
    """
    baseline = RngBundle(seed=99)
    expected = baseline.repro.random(32)

    perturbed = RngBundle(seed=99)
    perturbed.death.random(1_000)  # simulate an extra draw added by unrelated code
    assert np.array_equal(perturbed.repro.random(32), expected)


def test_stream_keys_are_stable_across_processes():
    """Guards against PYTHONHASHSEED-style non-determinism in stream derivation.

    A subprocess is used deliberately: string hashing is randomised per process, so a
    same-process check could not detect the bug this test exists to prevent.
    """
    code = (
        "from evosim.rng import RngBundle, STREAM_NAMES\n"
        "r = RngBundle(seed=4242)\n"
        "print(' '.join(f'{r[n].random():.17g}' for n in STREAM_NAMES))\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"stream derivation is not process-stable: {outputs}"


def test_adding_a_stream_does_not_perturb_existing_streams():
    """Streams are keyed by name hash, not spawn order, so the genome of a run survives
    the addition of a new subsystem in a later milestone."""
    original = RngBundle(seed=5, stream_names=STREAM_NAMES)
    extended = RngBundle(seed=5, stream_names=STREAM_NAMES + ("a_brand_new_subsystem",))
    for name in STREAM_NAMES:
        assert np.array_equal(original[name].random(16), extended[name].random(16)), name


def test_spawn_keys_are_unique():
    keys = {name: _spawn_key(name) for name in STREAM_NAMES}
    assert len(set(keys.values())) == len(keys), keys


def test_unknown_stream_raises():
    rng = RngBundle(seed=0)
    with pytest.raises(AttributeError, match="unknown RNG stream"):
        _ = rng.nonexistent_stream
    with pytest.raises(KeyError, match="unknown RNG stream"):
        _ = rng["nonexistent_stream"]


def test_state_round_trip_restores_stream_positions():
    rng = RngBundle(seed=11)
    rng.move.random(50)
    state = rng.get_state()
    expected = rng.move.random(20)

    rng.move.random(500)  # advance far away
    rng.set_state(state)
    assert np.array_equal(rng.move.random(20), expected)


def test_state_from_a_mismatched_bundle_is_rejected():
    rng = RngBundle(seed=1)
    state = rng.get_state()
    state["streams"].pop("move")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="does not match"):
        rng.set_state(state)


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "42", None, True])
def test_invalid_seeds_are_rejected(bad_seed):
    with pytest.raises((TypeError, ValueError)):
        RngBundle(seed=bad_seed)


def test_duplicate_stream_name_is_detected():
    with pytest.raises(ValueError, match="duplicate stream name"):
        RngBundle(seed=0, stream_names=("dup", "dup"))


def test_spawn_key_collision_is_detected(monkeypatch):
    """A hash collision would silently give two subsystems the same number stream.

    It cannot be triggered with real names, so the hash is forced to collide here to prove the
    guard actually fires rather than being dead code.
    """
    monkeypatch.setattr("evosim.rng._spawn_key", lambda name: 1234)
    with pytest.raises(ValueError, match="collision"):
        RngBundle(seed=0, stream_names=("alpha", "beta"))


# ---------------------------------------------------------------------------------------
# lint: nothing in the package may touch global randomness
# ---------------------------------------------------------------------------------------

_FORBIDDEN_MODULES = {"random"}
_FORBIDDEN_NP_RANDOM_CALLS = {
    # np.random.<name> functions that read or write process-global state.
    "random", "rand", "randn", "randint", "choice", "shuffle", "permutation",
    "normal", "uniform", "seed", "poisson", "binomial", "exponential", "beta",
    "gamma", "standard_normal", "sample", "random_sample", "bytes",
}


def _package_sources() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py"))


def test_no_module_imports_the_stdlib_random_module():
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _FORBIDDEN_MODULES:
                    offenders.append(f"{path.name}:{node.lineno} from {node.module} import ...")
    assert not offenders, (
        "the stdlib `random` module uses process-global state and would break "
        f"reproducibility; use an RngBundle stream instead. Offenders: {offenders}"
    )


def test_no_module_calls_the_numpy_global_rng():
    """`np.random.foo(...)` draws from global state; only `np.random.Generator` is allowed.

    This is a lint test rather than a convention because the failure mode -- a run that is
    subtly unreproducible -- is invisible until you try to reproduce it.
    """
    offenders = []
    for path in _package_sources():
        if path.name == "rng.py":
            continue  # the one place np.random.Generator/PCG64/SeedSequence are constructed
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            value = func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "random"
                and isinstance(value.value, ast.Name)
                and value.value.id == "np"
                and func.attr in _FORBIDDEN_NP_RANDOM_CALLS
            ):
                offenders.append(f"{path.name}:{node.lineno} np.random.{func.attr}(...)")
    assert not offenders, (
        "numpy's global RNG must never be used in simulation code; take a stream from an "
        f"RngBundle instead. Offenders: {offenders}"
    )
