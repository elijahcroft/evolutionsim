"""Tests for the CLI surface.

The argument surface is fixed at milestone 0 because every later milestone, the parameter
sweeps, and the directional-selection tests all drive the simulation through it.
"""

from __future__ import annotations

import pytest

from evosim.runner import build_parser, main, resolve_config


def test_default_invocation_succeeds(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "config fingerprint" in out
    assert "loci" in out


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


def test_ticks_is_accepted_but_reported_as_unimplemented(capsys):
    assert main(["--ticks", "100"]) == 0
    assert "not simulated" in capsys.readouterr().err
