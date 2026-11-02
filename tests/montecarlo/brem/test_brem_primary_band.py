"""Primary identities and FP64 moments for fused band quadrature (#938)."""

import os
import warnings

import numpy as np
import pytest

from pyrite._backend import BACKEND, REAL
from pyrite.montecarlo.spectrum import brem
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
from tests.helpers.bremslib import synthetic_bremslib_arrays
from tests.montecarlo.brem.test_brem_photon_blocking import _LAYERS, _segments

# FP32 cells and portable GEMV have different operation/reduction order.
# 128 eps bounds < 500 positive quadrature terms with tree reduction; moments
# double the relative score perturbation. FP64 primary accumulation is required.
_RTOL = 128 * np.finfo(np.float32).eps


def test_requested_backend_precision():
    requested = os.environ.get("PYRITE_TEST_BACKEND", "cpu")
    assert BACKEND.name == requested
    if requested == "cuda":
        expected = np.float64 if os.environ.get("PYRITE_FP64") == "1" else np.float32
        assert np.dtype(REAL) == np.dtype(expected)
    print(f"band-test backend={BACKEND.name} REAL={np.dtype(REAL)}")


@pytest.fixture
def table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)


@pytest.mark.parametrize("nodes", [64, 176, 496])
@pytest.mark.parametrize("layered,plasma", [(False, 0.0), (True, 0.0), (True, 30.0)])
def test_band_primary_parity_and_moments(monkeypatch, table, nodes, layered, plasma):
    segments = _segments(n=53, n_electrons=7)
    # Unsorted original IDs, missing histories 1/4/6, and excluded ID 8.
    segments["elec_id"] = np.resize([5, 0, 2, 8, 3], 53)
    segments["E_repr_keV"] = segments["E_keV"] * 0.99
    grid = np.geomspace(1000.0, 190000.0, nodes)
    weights = np.gradient(grid)
    weights[::7] = 0
    kwargs = dict(
        composition=[("C", 0.1)],
        cross_section_model="bremslib",
        bremslib_tables={"C": table},
        electron_band_weights=weights,
        layers=_LAYERS if layered else None,
        dielectric_plasma_eV=plasma,
        electron_limit=7,
        chunk=11,
    )
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", False)
    reference = brem.mc_brem_spectrum(segments, grid, **kwargs)
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", True)
    if BACKEND.name == "cuda" and np.dtype(REAL) == np.dtype(np.float32):

        def dense_forbidden(*args, **kwargs):
            raise AssertionError("supported fused band path allocated dense cells")

        monkeypatch.setattr(brem, "evaluate_bremslib", dense_forbidden)
    candidate = brem.mc_brem_spectrum(segments, grid, **kwargs)
    assert candidate.dtype == np.float64
    np.testing.assert_array_equal(candidate[[1, 4, 6]], 0)
    np.testing.assert_allclose(candidate, reference, rtol=_RTOL)
    np.testing.assert_allclose(candidate.sum(), reference.sum(), rtol=_RTOL)
    np.testing.assert_allclose(candidate @ candidate, reference @ reference, rtol=2 * _RTOL)
    np.testing.assert_allclose(candidate.mean(), reference.mean(), rtol=_RTOL)

    def rse(x):
        return np.std(x, ddof=1) / np.sqrt(x.size) / x.mean()

    np.testing.assert_allclose(rse(candidate), rse(reference), rtol=4 * _RTOL)


def test_coupled_band_fallback_and_empty(monkeypatch, table):
    segments = _segments(n=23, n_electrons=4)
    segments["E_keV"][0] = 400.0  # outside staged table: EEDL fallback
    segments["radiative"] = {
        "model": "bremslib-soft-hard",
        "cutoff_eV": 20000.0,
        "bremslib_tables": (("C", 6, table.key, table.digest),),
    }
    grid = np.geomspace(1000, 180000, 64)
    kwargs = dict(
        cutoff_eV=20000,
        composition=[("C", 0.1)],
        bremslib_tables={"C": table},
        electron_band_weights=np.gradient(grid),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", False)
        reference = mc_coupled_brem_spectrum(segments, grid, **kwargs)
        monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", True)
        candidate = mc_coupled_brem_spectrum(segments, grid, **kwargs)
    np.testing.assert_allclose(candidate, reference, rtol=_RTOL)
    empty = {k: v[:0] if isinstance(v, np.ndarray) else v for k, v in segments.items()}
    np.testing.assert_array_equal(mc_coupled_brem_spectrum(empty, grid, **kwargs), np.zeros(4))


@pytest.mark.parametrize("geometry", ["finite", "groove"])
def test_band_escape_splitting_cutoff_and_tiny_endpoint(monkeypatch, table, geometry):
    from tests.montecarlo.brem.test_brem_photon_blocking import _GROOVE, _GROOVE_N_HAT

    segments = _segments(n=37, thickness=20000.0)
    segments["layer"] = np.zeros(37, dtype=int)
    segments["E_keV"][0] = 190.0
    kwargs = dict(
        composition=[("C", 0.1)],
        cross_section_model="bremslib",
        bremslib_tables={"C": table},
        E_cut_keV=30.0,
    )
    if geometry == "finite":
        segments.update(crystal_width_ang=100.0, crystal_height_ang=100.0)
        kwargs["n_hat"] = np.array([0.6, 0.0, 0.8])
    else:
        kwargs.update(groove=_GROOVE, n_hat=_GROOVE_N_HAT)
    grid = np.r_[1000.0, 30000.0, 189999.99, 190000.0, 190000.01, 300000.0]
    weights = np.array([1e-25, 1.0, 1e-15, 1.0, 1.0, 1.0])
    kwargs["electron_band_weights"] = weights
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", False)
    reference = brem.mc_brem_spectrum(segments, grid, **kwargs)
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", True)
    candidate = brem.mc_brem_spectrum(segments, grid, **kwargs)
    np.testing.assert_allclose(candidate, reference, rtol=_RTOL)
