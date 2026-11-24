"""Fused float64 CPU BremsLib reducer against the portable dense path (#1429).

The fused reducer evaluates every cell with the portable expressions and only
reorders the sums (per photon bin over rows; per row over photon bins), so it
agrees with the dense NumPy/BLAS path to a few ulps of those sums.

Validation: bremslib-angular-model
"""

import numpy as np
import pytest

from pyrite._backend import REAL, xp
from pyrite.montecarlo.spectrum import _brem_cpu_fused, brem
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from tests.helpers.bremslib import synthetic_bremslib_arrays
from tests.montecarlo.brem.test_brem_photon_blocking import _LAYERS, _segments

pytestmark = pytest.mark.skipif(
    getattr(xp, "__name__", "") != "numpy" or np.dtype(REAL) != np.dtype(np.float64),
    reason="the fused reducer scores the float64 NumPy backend only",
)

#: Reordered float64 sums over <= 60 rows or <= 100 bins of positive terms.
_RTOL = 1.0e-12
#: Photon grid with nodes at and around the 190 keV kinematic tip of row 0.
_GRID = np.r_[np.geomspace(800.0, 150_000.0, 60), 189_999.99, 190_000.0, 190_000.01, 2.0e5]


def _table(Z):
    arrays = synthetic_bremslib_arrays()
    return prepare_bremslib_table(arrays, atomic_number=Z, key=f"k{Z}", digest=f"d{Z}")


@pytest.fixture(scope="module")
def tables():
    return {"C": _table(6), "N": _table(7)}


def _case(tables, *, layered, plasma, band, n_hat=None):
    segments = _segments(n=57, n_electrons=5, seed=1429)
    segments["E_keV"][0] = 190.0  # tip at a grid node
    segments["E_repr_keV"] = segments["E_keV"] * 0.995
    segments["E_repr_keV"][0] = 190.0
    kwargs = dict(
        composition=[("C", 0.1), ("N", 0.03)],
        cross_section_model="bremslib",
        bremslib_tables=tables,
        layers=_LAYERS if layered else None,
        dielectric_plasma_eV=plasma,
        chunk=13,
    )
    if n_hat is not None:
        kwargs["n_hat"] = np.asarray(n_hat, dtype=float)
    if band:
        weights = np.gradient(_GRID)
        weights[::5] = 0.0
        kwargs["electron_band_weights"] = weights
    return segments, kwargs


def _score(monkeypatch, segments, kwargs, *, fused):
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", fused)
    if fused:

        def dense_forbidden(*args, **kw):
            raise AssertionError("covered CPU BremsLib scoring built dense cells")

        monkeypatch.setattr(brem, "evaluate_bremslib", dense_forbidden)
    return brem.mc_brem_spectrum(segments, _GRID, **kwargs)


@pytest.mark.parametrize("band", [False, True], ids=["spectrum", "band"])
@pytest.mark.parametrize("layered,plasma", [(False, 0.0), (True, 0.0), (True, 30.0)])
@pytest.mark.parametrize("n_hat", [None, (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
def test_fused_matches_portable(monkeypatch, tables, band, layered, plasma, n_hat):
    segments, kwargs = _case(tables, layered=layered, plasma=plasma, band=band, n_hat=n_hat)
    with monkeypatch.context() as patch:
        reference = _score(patch, segments, kwargs, fused=False)
    candidate = _score(monkeypatch, segments, kwargs, fused=True)
    assert candidate.dtype == np.float64
    assert np.count_nonzero(reference) > reference.size // 2
    np.testing.assert_array_equal(candidate == 0.0, reference == 0.0)
    np.testing.assert_allclose(candidate, reference, rtol=_RTOL, atol=0.0)


@pytest.mark.parametrize("band", [False, True], ids=["spectrum", "band"])
def test_fused_is_thread_and_block_invariant(monkeypatch, tables, band):
    import numba

    segments, kwargs = _case(tables, layered=True, plasma=0.0, band=band)
    monkeypatch.setattr(brem, "_USE_JIT_BREM_REDUCTION", True)
    monkeypatch.setattr(_brem_cpu_fused, "FUSED_ROW_BLOCK", 7)
    monkeypatch.setattr(_brem_cpu_fused, "_ROW_TILE", 3)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = brem.mc_brem_spectrum(segments, _GRID, **kwargs)
    finally:
        numba.set_num_threads(previous)
    monkeypatch.setattr(_brem_cpu_fused, "FUSED_ROW_BLOCK", 1 << 16)
    monkeypatch.setattr(_brem_cpu_fused, "_ROW_TILE", 256)
    threaded = brem.mc_brem_spectrum(segments, _GRID, **kwargs)
    blocked = brem.mc_brem_spectrum(segments, _GRID, photon_block=9, **kwargs)
    np.testing.assert_array_equal(serial, threaded)
    if band:
        # Per-electron band values add per-block partial sums (#819).
        np.testing.assert_allclose(blocked, threaded, rtol=_RTOL, atol=0.0)
    else:
        np.testing.assert_array_equal(blocked, threaded)


def test_uncovered_rows_keep_the_portable_path(monkeypatch, tables):
    segments, kwargs = _case(tables, layered=False, plasma=0.0, band=False)
    segments["E_keV"][3] = 400.0  # above the synthetic table: EEDL fallback rows
    segments["E_repr_keV"][3] = 400.0

    def fused_forbidden(*args, **kw):
        raise AssertionError("uncovered rows must use the portable path")

    monkeypatch.setattr(_brem_cpu_fused, "accumulate_bremslib_fused", fused_forbidden)
    with pytest.warns(Warning, match="outside the BremsLib range"):
        spectrum = brem.mc_brem_spectrum(segments, _GRID, **kwargs)
    assert np.all(np.isfinite(spectrum)) and spectrum.max() > 0.0
