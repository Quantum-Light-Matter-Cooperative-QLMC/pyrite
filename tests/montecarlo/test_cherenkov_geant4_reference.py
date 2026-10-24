"""Frank--Tamm yield against binned Geant4 11.4.2 G4Cerenkov references (#587).

Each reference (``checks/cherenkov_geant4``) records the primary's in-slab
path length grouped by step-mean beta and the energy/polar-angle histograms
of the Cherenkov photons Geant4 created on those steps. The prediction sums
``frank_tamm_yield`` over the same steps with the same tabulated index, so
the comparison isolates the source term from tracking. Geant4 counts are
Poisson in every bin.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cherenkov import ChargedPassage, frank_tamm_yield

REFERENCE = Path(__file__).resolve().parents[2] / "checks" / "cherenkov_geant4" / "reference"
CASES = ("water_mu_b099", "silica_mu_threshold", "n150_alpha_b095", "water_p_below")
#: Quadrature midpoints per Geant4 energy bin.
_SUB = 16


def _edges(spec):
    lo, hi, n = spec
    return np.linspace(lo, hi, int(n) + 1)


def _predict(ref):
    """Expected energy and polar-angle histograms from the recorded steps."""
    e_edges = _edges(ref["energy_edges"])
    width = (e_edges[1] - e_edges[0]) / _SUB
    energy = e_edges[0] + width * (np.arange(_SUB * (len(e_edges) - 1)) + 0.5)
    # Geant4 interpolates n linearly in energy (G4PhysicsFreeVector).
    eps = np.interp(energy, ref["rindex_energy_eV"], ref["rindex_n"]) ** 2
    e_counts = np.zeros(len(e_edges) - 1)
    theta, weight = [], []
    for beta, path_mm in zip(ref["step_beta"], ref["step_path_mm"], strict=True):
        passage = ChargedPassage(beta, ref["charge"], path_length_m=path_mm * 1e-3)
        result = frank_tamm_yield(passage, energy, eps)
        per_point = result.photons_per_eV() * width
        e_counts += per_point.reshape(-1, _SUB).sum(axis=1)
        theta.append(np.degrees(result.theta_rad[result.above_threshold]))
        weight.append(per_point[result.above_threshold])
    t_counts, _ = np.histogram(
        np.concatenate(theta), _edges(ref["theta_edges_deg"]), weights=np.concatenate(weight)
    )
    return e_counts, t_counts


def _reduced_chi2(observed, expected, floor=10.0):
    keep = expected >= floor
    return float(np.sum((observed[keep] - expected[keep]) ** 2 / expected[keep]) / keep.sum())


@pytest.fixture(scope="module", params=CASES)
def case(request):
    ref = json.loads((REFERENCE / f"{request.param}.json").read_text())
    return ref, *_predict(ref)


def test_reference_is_self_consistent(case):
    ref, _, _ = case
    assert ref["photons"] == ref["photons_from_steps"] == sum(ref["energy_counts"])
    assert ref["angle_outside"] == 0
    assert sum(ref["theta_counts"]) == ref["photons"]


def test_total_photon_count_agrees_within_poisson(case):
    ref, e_expected, _ = case
    expected = e_expected.sum()
    if ref["case"] == "water_p_below":
        # Below threshold everywhere in the band: both sides emit nothing.
        assert expected == 0.0
        assert ref["photons"] == 0
        return
    assert abs(ref["photons"] - expected) < 5.0 * np.sqrt(expected)
    assert ref["photons"] / expected == pytest.approx(1.0, abs=5e-3)


def test_energy_spectrum_agrees(case):
    ref, e_expected, _ = case
    if e_expected.sum() == 0.0:
        pytest.skip("zero-yield case")
    observed = np.asarray(ref["energy_counts"], float)
    assert _reduced_chi2(observed, e_expected) < 1.5
    # Threshold: no photon below the lowest energy where beta n > 1.
    np.testing.assert_array_equal(observed[e_expected == 0.0], 0.0)


def test_angular_distribution_agrees(case):
    ref, _, t_expected = case
    if t_expected.sum() == 0.0:
        pytest.skip("zero-yield case")
    observed = np.asarray(ref["theta_counts"], float)
    centers = 0.5 * (_edges(ref["theta_edges_deg"])[1:] + _edges(ref["theta_edges_deg"])[:-1])
    mean_obs = np.average(centers, weights=observed)
    mean_exp = np.average(centers, weights=t_expected)
    spread = np.sqrt(np.average((centers - mean_exp) ** 2, weights=t_expected) + 0.1**2 / 12)
    assert abs(mean_obs - mean_exp) < 5.0 * spread / np.sqrt(observed.sum())
    assert _reduced_chi2(observed, t_expected) < 2.0
