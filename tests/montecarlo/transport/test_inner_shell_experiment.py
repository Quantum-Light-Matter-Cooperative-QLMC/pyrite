"""Transcribed inner-shell ionization measurements and their model comparison (#92).

Validation: bote-salvat-ionization-backend
"""

import numpy as np
import pytest

from pyrite import datasets
from pyrite.validation import inner_shell_experiment as experiment


def test_every_measurement_is_cited_and_physical():
    references = experiment.load_references()
    k_shell = experiment.load_k_shell_measurements()
    l_shell = experiment.load_l_subshell_measurements()

    assert (len(k_shell), len(l_shell)) == (362, 59)
    assert {m.ref for m in k_shell + l_shell} <= set(references)
    assert all(references[ref]["doi"] for ref in references)
    for m in k_shell + l_shell:
        assert m.sigma_b > 0.0 and m.overvoltage > 1.0
        assert m.dsigma_b is None or m.dsigma_b > 0.0
    assert {m.shell for m in l_shell} == {"L1", "L2", "L3"}


def test_summary_bands_ratios_and_one_sigma_hits():
    rows = [
        experiment.Measurement(6, "C", "K", 1.0, u, 10.0, 1.0, "Li12")
        for u in (1.2, 2.0, 3.0, 30.0)
    ]
    model = np.array([[0.0, 10.5], [5.0, 20.0], [20.0, 10.0], [10.0, 12.0]])
    low, peak, bethe = experiment.summarise(rows, model)

    assert (low.n, peak.n, bethe.n) == (1, 2, 1)
    assert np.isnan(low.eedl_ratio) and low.bote_salvat_within == 1.0
    assert peak.eedl_ratio == pytest.approx(1.0) and peak.bote_salvat_ratio == pytest.approx(
        np.sqrt(2.0)
    )
    assert peak.eedl_within == 0.0 and peak.bote_salvat_within == 0.5
    assert bethe.band == (20.0, np.inf)


def test_pinned_models_bracket_the_k_shell_near_threshold_data():
    k_shell = experiment.load_k_shell_measurements()
    try:
        model = experiment.model_cross_sections_b(k_shell)
    except datasets.DatasetNotFoundError as exc:
        pytest.skip(str(exc))
    near = experiment.summarise(k_shell, model)[0]

    # Measured K data at U < 1.5 sit about 24 % above EEDL and 2 % below
    # Bote-Salvat (docs/validation/radiation-physics/bote-salvat-ionization-backend.md).
    assert near.band == (1.0, 1.5)
    assert 0.70 < near.eedl_ratio < 0.82
    assert 0.95 < near.bote_salvat_ratio < 1.10
