"""Tests for the speciation-scale experiment.

The tool exists to answer a question about the model, so what matters here is that the numbers
it reports are the ones it claims: distances measured with the *variant's own* species concept,
in the same units as the mating threshold, and a signal-to-noise ratio that is genuinely the
quotient of the two.  A tool that silently measured something else would produce a finding that
reads fine and means nothing.
"""

from __future__ import annotations

import json

import pytest

from tools.speciation_experiment import VARIANTS, main


def test_a_short_run_reports_both_scales(capsys):
    assert main(["--days", "20", "--pairs", "200", "--variant", "reference", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 1
    result = results[0]
    assert result["variant"] == "reference"
    assert result["days"] == 20
    assert result["between_lineages"] >= 0.0
    assert result["within_lineage_mean"] > 0.0
    assert result["within_lineage_mean"] <= result["within_lineage_max"]
    assert result["signal_to_noise"] == pytest.approx(
        result["between_lineages"] / result["within_lineage_mean"]
    )
    assert result["reaches_threshold"] == (
        result["between_lineages"] > result["threshold"]
    )
    # Separability is the decisive claim: it must be the tail comparison, not the mean one.
    assert result["separable"] == (
        result["between_lineages"] > result["within_lineage_p99"]
    )


def test_a_variant_that_changes_the_species_concept_changes_the_threshold_scale(capsys):
    """Weights are part of what varies, so distances must be measured with the variant's own."""
    assert main(
        ["--days", "20", "--pairs", "200", "--variant", "thermal species concept", "--json"]
    ) == 0
    weighted = json.loads(capsys.readouterr().out)[0]

    assert main(["--days", "20", "--pairs", "200", "--variant", "reference", "--json"]) == 0
    reference = json.loads(capsys.readouterr().out)[0]

    # Concentrating the metric on three loci must not leave the numbers identical to a metric
    # spread over 28 -- if it did, the overrides were not reaching the taxonomy.
    assert weighted["within_lineage_mean"] != reference["within_lineage_mean"]


def test_every_declared_variant_is_runnable(capsys):
    for variant in VARIANTS:
        assert main(["--days", "2", "--pairs", "50", "--variant", variant.name, "--json"]) == 0
        result = json.loads(capsys.readouterr().out)[0]
        assert result["variant"] == variant.name


def test_the_human_report_names_what_it_measured(capsys):
    assert main(["--days", "5", "--pairs", "50", "--variant", "reference"]) == 0
    out = capsys.readouterr().out
    assert "between" in out and "within" in out and "S/N" in out
    assert "does the default configuration separate isolated lineages" in out


@pytest.mark.parametrize(
    "args,message",
    [
        (["--days", "0"], "--days must be positive"),
        (["--pairs", "0"], "--pairs must be positive"),
        (["--variant", "nonsense"], "no such variant"),
    ],
)
def test_bad_arguments_fail_cleanly(capsys, args, message):
    assert main(args) == 2
    assert message in capsys.readouterr().err
