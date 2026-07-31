"""Tests for the Milestone 2 life-substrate performance diagnostic."""

from __future__ import annotations

import json

import pytest

from tools.profile_tick import main


def test_profile_tick_json_smoke(capsys):
    assert main(
        [
            "--population",
            "8",
            "--iterations",
            "1",
            "--seed",
            "7",
            "--json",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["benchmark"] == "m2_life_substrate_pass"
    assert result["biological_tick"] is False
    assert result["ecology_starts"] == "M3"
    assert result["population"] == result["occupancy_total"] == 8
    assert result["iterations"] == 1
    assert result["seed"] == 7
    assert result["memory_bytes"] > 0
    assert result["ms_per_pass"] >= 0.0
    assert result["organisms_per_second"] > 0.0


@pytest.mark.parametrize(
    "args,message",
    [
        (["--population", "0"], "--population must be positive"),
        (["--iterations", "0"], "--iterations must be positive"),
        (["--seed", "-1"], "--seed must be non-negative"),
        (["--population", "40001"], "exceeds configured sim.max_population"),
    ],
)
def test_profile_tick_rejects_invalid_positive_arguments(args, message, capsys):
    assert main(args) == 2
    assert message in capsys.readouterr().err
