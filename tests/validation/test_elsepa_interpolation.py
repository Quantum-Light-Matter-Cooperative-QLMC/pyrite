"""Analytic anchors for proposed node-mixture moments; no sampler changes.

Validation: elsepa-energy-mixture-moments
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrite.validation._elsepa_interpolation import _pair_moment_bounds


def test_uniform_and_triangular_nodes_enclose_the_exact_quantile_distance():
    # Q0(u)=u, Q1(u)=sqrt(u): int (Q1-Q0)^2 du = 1/30.
    mu = np.array([0.0, 1.0])
    coarse = _pair_moment_bounds(mu, [1, 1], [0, 2], subdivisions=1)
    fine = _pair_moment_bounds(mu, [1, 1], [0, 2], subdivisions=128)
    for bounds in (coarse, fine):
        assert bounds["max_second_moment_difference_lower"] <= 1 / 30
        assert bounds["max_second_moment_difference_upper"] >= 1 / 30
        assert bounds["first_moment_difference"] == 0
        np.testing.assert_allclose(bounds["node_first_moments"], [1, 4 / 3], rtol=1e-14)
        np.testing.assert_allclose(bounds["node_second_moments"], [4 / 3, 2], rtol=1e-14)
    # The interior extremum of sqrt(u)-u at u=1/4 must not be missed.
    assert coarse["max_second_moment_difference_upper"] == pytest.approx(1 / 16)
    assert fine["max_second_moment_difference_upper"] < coarse["max_second_moment_difference_upper"]
    assert fine["max_second_moment_difference_lower"] > coarse["max_second_moment_difference_lower"]


def test_bound_applies_to_all_energy_weights_and_both_node_orders():
    bounds = _pair_moment_bounds([0, 1], [1, 1], [0, 2], subdivisions=64)
    # Analytic square of (1-f)u+f*sqrt(u); no production inversion helper.
    f = np.r_[np.linspace(0, 1, 101), bounds["relative_max_weight"]]
    mixture = 4 * ((1 - f) / 3 + f / 2)
    quantile = 4 * ((1 - f) ** 2 / 3 + f**2 / 2 + 4 * f * (1 - f) / 5)
    difference = mixture - quantile
    assert np.min(difference) >= -1e-15
    assert np.max(difference) <= bounds["max_second_moment_difference_upper"]
    assert np.max(difference / quantile) <= bounds["max_relative_second_moment_difference_upper"]
    # Both interpolation laws reproduce either node, including the top endpoint.
    np.testing.assert_array_equal(mixture[[0, 100]], quantile[[0, 100]])
    reverse = _pair_moment_bounds([0, 1], [0, 2], [1, 1], subdivisions=64)
    assert reverse["max_second_moment_difference_upper"] == pytest.approx(
        bounds["max_second_moment_difference_upper"], rel=1e-14
    )
    assert reverse["relative_max_weight"] == pytest.approx(1 - bounds["relative_max_weight"])


def test_identical_nodes_and_density_rescaling_leave_the_bound_unchanged():
    mu = np.array([0, 0.1, 0.6, 1])
    dcs = np.array([40, 2, 1, 0.5])
    same = _pair_moment_bounds(mu, dcs, dcs)
    assert same["max_second_moment_difference_upper"] == 0
    assert same["max_second_moment_difference_lower"] == 0
    original = _pair_moment_bounds(mu, dcs, dcs[::-1])
    scaled = _pair_moment_bounds(mu, 1e-20 * dcs, 1e-12 * dcs[::-1])
    assert scaled["max_second_moment_difference_upper"] == pytest.approx(
        original["max_second_moment_difference_upper"], rel=1e-13
    )


@pytest.mark.parametrize(
    "mu,p0,p1",
    [
        ([0, 0, 1], [1, 1, 1], [1, 1, 1]),
        ([0, 1], [-1, 2], [1, 1]),
        ([0, 0.5, 1], [0, 0, 1], [1, 1, 1]),
    ],
)
def test_invalid_or_zero_mass_panels_are_refused(mu, p0, p1):
    with pytest.raises(ValueError):
        _pair_moment_bounds(mu, p0, p1)


@pytest.mark.slow
def test_all_released_tables_and_solid_crossovers_keep_the_moment_envelopes(tmp_path, monkeypatch):
    """A table-wide physics anchor, not a stochastic sampler smoke test.

    Absolute second-moment changes stay below .01 (electrons) / .003
    (positrons), but HOPG's joined crossover exceeds 10% relative for both.
    This prevents an assertion that the mixture is uniformly a 2% change.
    """
    from pyrite.xsgen.elsepa.release import load_release_index
    from pyrite.xsgen.store import resolve

    # conftest deliberately clears ambient PYRITE_HOME. Opt in to an isolated,
    # read-only data audit explicitly, rather than reaching into a user workspace.
    if audit_home := os.environ.get("PYRITE_ELSEPA_AUDIT_HOME"):
        monkeypatch.setenv("PYRITE_HOME", audit_home)
    for species in ("electron", "positron"):
        release = load_release_index(projectile=species)
        assert release is not None
        for entry in release.tables:
            stored = resolve(entry.key)
            if stored is None or stored.digest != entry.manifest_sha256:
                pytest.skip(f"pinned {species} ELSEPA release tables must be installed")
    output = tmp_path / "moments.json"
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "checks/elsepa_interpolation_moments.py", "--output", str(output)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(output.read_text())
    for species, absolute_ceiling in (("electron", 0.01), ("positron", 0.003)):
        report = data["projectiles"][species]
        assert report["first_moment_difference"] == 0
        assert (
            report["max_absolute"]["max_absolute"]["max_second_moment_difference_upper"]
            < absolute_ceiling
        )
        worst = report["max_relative"]["max_relative"]
        assert worst["max_relative_second_moment_difference_upper"] < 0.11
        hopg = next(r for r in report["crossovers"] if r["label"] == "hopg (joined crossover)")
        assert hopg["max_relative"]["max_relative_second_moment_difference_lower"] > 0.10
        # Every reported D enclosure must be ordered and nonnegative.
        for table in report["records"] + report["crossovers"]:
            for kind in ("max_absolute", "max_relative"):
                bound = table[kind]
                assert (
                    0
                    <= bound["max_second_moment_difference_lower"]
                    <= bound["max_second_moment_difference_upper"]
                )
