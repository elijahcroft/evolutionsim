"""Tests for the CLI surface.

The argument surface is fixed at milestone 0 because every later milestone, the parameter
sweeps, and the directional-selection tests all drive the simulation through it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evosim.runner import build_parser, main, resolve_config


def test_default_invocation_succeeds(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "config fingerprint" in out
    assert "loci" in out
    assert "living organisms   : 800 / 40000" in out


def test_seed_flag_overrides_config_seed():
    args = build_parser().parse_args(["--seed", "1234"])
    assert resolve_config(args).sim.seed == 1234


def test_seed_flag_changes_the_fingerprint():
    """The seed must be inside the resolved config, not applied after the fact."""
    base = resolve_config(build_parser().parse_args([]))
    seeded = resolve_config(build_parser().parse_args(["--seed", "4321"]))
    assert base.fingerprint() != seeded.fingerprint()


def test_set_flag_is_repeatable():
    args = build_parser().parse_args(
        ["--set", "planet.gravity=1.4", "--set", "planet.o2_fraction=0.10"]
    )
    cfg = resolve_config(args)
    assert cfg.planet.gravity == pytest.approx(1.4)
    assert cfg.planet.o2_fraction == pytest.approx(0.10)


def test_bad_override_exits_with_code_2_and_a_message(capsys):
    assert main(["--set", "planet.gravitas=1.0"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_ticks_advance_the_world(capsys):
    assert main(["--ticks", "100"]) == 0
    assert "simulated day      : 100" in capsys.readouterr().out


def test_ticks_are_biological_and_report_their_ledger(capsys):
    """From milestone 3 a tick feeds, moves, and kills, so the CLI must say what it did."""
    assert main(["--ticks", "5"]) == 0
    out = capsys.readouterr().out
    assert "last tick intake" in out
    assert "last tick cost" in out
    assert "last tick deaths" in out
    assert "cells moved" in out


def test_negative_ticks_fail(capsys):
    assert main(["--ticks", "-1"]) == 2
    assert "must be non-negative" in capsys.readouterr().err


def test_missing_requested_founder_habitat_fails_cleanly(capsys):
    assert main(
        [
            "--set",
            "planet.terrain.land_fraction=0",
            "--set",
            "sim.initial_habitat=land",
        ]
    ) == 2
    assert "initialization error" in capsys.readouterr().err


def test_out_writes_world_state(tmp_path):
    assert main(["--ticks", "3", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "world.npz").is_file()
    assert (tmp_path / "world.json").is_file()
    assert (tmp_path / "population.npz").is_file()
    assert (tmp_path / "population.json").is_file()

    metadata = json.loads((tmp_path / "population.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "diagnostic_population_dump"
    assert metadata["day"] == 3
    assert len(metadata["locus_names"]) == len(metadata["trait_names"]) == 28

    # The dump is whoever survived three ticks of ecology, so its size is an outcome rather
    # than the founder count. What must hold is that every array still describes the same
    # organisms in the same order.
    size = metadata["size"]
    assert 0 < size <= 800
    with np.load(tmp_path / "population.npz") as state:
        assert state["genome"].shape == (size, 28, 2)
        assert state["cell"].shape == (size,)
        assert state["diet"].shape == (size, 4)
        assert state["age"].shape == (size,)
        assert np.all(state["age"] == 3)
    assert metadata["diet_names"] == [
        "autotroph",
        "detritus",
        "herbivore",
        "carnivore",
    ]
