"""Per-primary cone-detector scoring plumbing around the production estimator."""

import math

import numpy as np
import pytest

from pyrite.perf import detector_scoring
from pyrite.perf.comparison import cone_solid_angle, validate_primary_scores


def test_band_panels_split_at_edges_and_cap_width():
    panels = detector_scoring.band_panels(10.0, 800.0, [5.0, 10.2, 69.5, 900.0])
    assert panels[0] == (10.0, 10.2) and panels[1] == (10.2, 69.5)
    assert panels[-1][1] == 800.0
    assert all(b - a <= detector_scoring.MAX_PANEL_KEV + 1e-9 for a, b in panels)
    assert all(p[1] == q[0] for p, q in zip(panels, panels[1:], strict=False))


def test_band_rule_integrates_polynomials_exactly_in_eV():
    nodes, weights = detector_scoring.band_rule([(10.0, 20.0), (20.0, 50.0)], 3)
    assert weights.sum() == pytest.approx(40e3)
    # Degree 5 is exact for 3-node Gauss-Legendre on each panel.
    exact = ((50e3) ** 6 - (10e3) ** 6) / 6
    assert weights @ nodes**5 == pytest.approx(exact, rel=1e-12)


def test_detector_axis_lies_in_the_plane_of_incidence():
    assert detector_scoring.detector_axis(0) == pytest.approx([0, 0, 1])
    assert detector_scoring.detector_axis(90) == pytest.approx([1, 0, 0], abs=1e-15)
    assert detector_scoring.detector_axis(135) == pytest.approx([0.5**0.5, 0, -(0.5**0.5)])


@pytest.fixture
def fake_estimator(monkeypatch):
    """Per-electron density c_e photons/eV/sr; half survives a nonzero absorber."""
    calls = []

    def estimator(segments, energies, *, layers, electron_band_weights, n_hat, **kwargs):
        calls.append(kwargs)
        assert kwargs["cutoff_eV"] == 1000.0 and np.isclose(np.linalg.norm(n_hat), 1.0)
        transmission = 1.0 if all(n == 0.0 for _el, n in layers[0][2]) else 0.5
        # Sum each primary's segments before returning, as the estimator does.
        per_row = segments["density"] * np.sum(electron_band_weights)
        totals = np.bincount(segments["elec_id"], weights=per_row, minlength=segments["Ne"])
        return transmission * totals

    monkeypatch.setattr(
        "pyrite.montecarlo.spectrum.brem_events.mc_coupled_brem_spectrum", estimator
    )
    return calls


def segments(n=3):
    # Primary 1 has no segment that emits: its zero score must be kept.
    return {
        "Ne": n,
        "elec_id": np.array([0, 0, 2, 2, 2]),
        "density": np.array([1e-6, 1e-6, 2e-6, 0.0, 1e-6]),
        "thickness_ang": 1e5,
        "n_layers": 1,
        "radiative": {"model": "bremslib-soft-hard", "cutoff_eV": 1000.0},
    }


def test_scores_are_per_incident_primary_cone_and_band_integrals(fake_estimator):
    out = detector_scoring.cone_detector_scores(
        segments(),
        composition=[("Si", 0.05)],
        energy_keV=90.0,
        center_deg=90,
        half_angle_deg=5,
        bremslib_tables={},
        rtol=1e-9,
    )
    omega = cone_solid_angle(math.radians(5))
    density = np.array([2e-6, 0.0, 3e-6])
    scores = out["scores"]
    assert scores["cone_source_photons_10_50kev"] == pytest.approx(density * omega * 40e3)
    # The 50-100 keV band stops at the 90 keV endpoint; >=100 keV is exactly zero.
    assert scores["cone_source_photons_50_100kev"] == pytest.approx(density * omega * 40e3)
    assert not np.any(scores["cone_source_photons_ge100kev"])
    assert scores["cone_escape_photons_ge10kev"] == pytest.approx(
        0.5 * scores["cone_source_photons_ge10kev"]
    )
    assert scores["cone_source_photons_ge10kev"][1] == 0.0
    acceptance = out["acceptance"]
    assert acceptance["converged"] and acceptance["solid_angle_sr"] == pytest.approx(omega)
    assert acceptance["fields"]["cone_escape_photons_ge100kev"]["status"] == "exact-zero"
    transport = {
        "transmitted": [1, 0, 1],
        "backscattered": [0, 1, 0],
        "side_exited": [0, 0, 0],
        "stopped": [0, 0, 0],
        "transmitted_energy_keV_per_primary": [50.0, 0, 20.0],
    }
    validate_primary_scores(transport | scores, 3, 90.0, kind="detector")
    assert {tuple(c["composition"]) for c in fake_estimator} == {(("Si", 0.05),)}


def test_uncoupled_or_layered_rows_are_refused(fake_estimator):
    kwargs = dict(
        composition=[("Si", 0.05)], energy_keV=90.0, center_deg=0, half_angle_deg=1,
        bremslib_tables={}, rtol=1e-3,
    )  # fmt: skip
    with pytest.raises(ValueError, match="coupled single-layer"):
        detector_scoring.cone_detector_scores(segments() | {"radiative": None}, **kwargs)
    with pytest.raises(ValueError, match="coupled single-layer"):
        detector_scoring.cone_detector_scores(segments() | {"n_layers": 2}, **kwargs)
    assert not fake_estimator
