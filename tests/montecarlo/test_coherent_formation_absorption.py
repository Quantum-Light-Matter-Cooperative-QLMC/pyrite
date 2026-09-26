"""Coherent line fields under absorption: the per-piece formation integral.

A segment's coherent field is the formation integral of its emitter phase,
damped along the flight by ``exp(-tau/2)``. On a linear escape piece the
integrand is one exponential, so ``t_L F`` with
``F = exp(-tau_c/2) sinh(w)/w`` is exact (issue #181). The checks here pin
the closed form to quadrature, its Parseval integral to the segment-mean
escape the incoherent route scores, and the spectrum to split invariance under
absorption and refraction together.

Validation: coherent-formation-absorption
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import mc_spectrum
from pyrite.montecarlo.spectrum.lines import _setup as lines_setup
from pyrite.montecarlo.spectrum.lines._formation import (
    formation_coefficients,
    formation_factor,
)
from pyrite.montecarlo.spectrum.segment_escape import mean_transmission
from pyrite.montecarlo.transport import beta_from_keV
from tests.helpers import scaled_rtol

CRYSTAL = "hopg"
HKL = (0, 0, 2)
E_KEV = 100.0
E_GRID = np.arange(1550.0, 1800.0, 0.05)
N_HAT = np.array([np.sin(np.deg2rad(119.0)), 0.0, np.cos(np.deg2rad(119.0))])
SPLITS = (1, 4, 16)

# (v, tau_start, tau_end): undamped, uniform damping, into and out of the
# crystal, the series branch, and pieces running into opaque depth.
FORMATION_CASES = [
    (0.0, 0.0, 0.0),
    (3.0, 0.1, 0.1),
    (0.5, 0.2, 3.0),
    (-7.0, 5.0, 0.3),
    (1.0e-4, 1.0e-5, 2.0e-5),
    (40.0, 0.0, 60.0),
    (0.0, 0.0, 200.0),
]


def _formation_quadrature(v, tau_start, tau_end, n=400_001):
    """``(1/2) int_{-1}^{1} exp(i v s - tau(s)/2) ds``, tau affine along travel."""
    s = np.linspace(-1.0, 1.0, n)
    tau = tau_start + 0.5 * (tau_end - tau_start) * (s + 1.0)
    return 0.5 * np.trapezoid(np.exp(1j * v * s - 0.5 * tau), s)


@pytest.mark.parametrize(("v", "tau_start", "tau_end"), FORMATION_CASES)
def test_formation_factor_matches_quadrature(v, tau_start, tau_end):
    F = formation_factor(np.float64(v), *formation_coefficients(tau_start, tau_end))
    np.testing.assert_allclose(F, _formation_quadrature(v, tau_start, tau_end), atol=1e-9)


def test_transparent_formation_factor_is_the_sinc():
    v = np.linspace(-30.0, 30.0, 6001)
    F = formation_factor(v, *formation_coefficients(np.zeros(()), np.zeros(())))
    np.testing.assert_allclose(F.real, np.sinc(v / np.pi), atol=1e-15)
    np.testing.assert_array_equal(F.imag, 0.0)


@pytest.mark.parametrize(
    ("tau_start", "tau_end"), [(0.0, 0.0), (0.2, 3.0), (5.0, 0.3), (0.0, 60.0)]
)
def test_formation_parseval_is_the_segment_mean_escape(tau_start, tau_end):
    """``int |F|^2 dv = pi <exp(-tau)>``: the coherent self-term integrates to
    exactly the incoherent route's segment-mean escape weight."""
    V, dv = 1.0e4, 0.01
    v = np.arange(-V, V + 0.5 * dv, dv)
    F = formation_factor(v, *formation_coefficients(tau_start, tau_end))
    a, b = np.exp(-0.5 * tau_start), np.exp(-0.5 * tau_end)
    # |F|^2 -> (a^2 + b^2 - 2ab cos 2v) / (4 v^2): the cos term averages out,
    # so the two tails beyond |v| = V add (a^2 + b^2) / (2 V).
    integral = np.trapezoid(np.abs(F) ** 2, v) + (a * a + b * b) / (2.0 * V)
    expected = np.pi * float(mean_transmission(np.float64(tau_start), np.float64(tau_end)))
    np.testing.assert_allclose(integral, expected, rtol=1e-6)


def test_formation_factor_stays_below_the_sinc_envelope():
    """``|F| <= 1/|v|``, so a cutoff on ``|v|`` stays a conservative tail cut."""
    v = np.linspace(0.5, 200.0, 4000)
    for tau_start, tau_end in [(0.0, 0.0), (0.0, 60.0), (5.0, 0.3), (0.2, 3.0)]:
        F = formation_factor(v, *formation_coefficients(tau_start, tau_end))
        assert np.all(np.abs(F) <= 1.0 / v * (1.0 + 1e-12))


def _track(pieces, *, length=40000.0, z_mid=25000.0, direction=(0.0, 0.0, 1.0), **extra):
    """One straight segment, re-cut into ``pieces`` collinear pieces."""
    v = np.asarray(direction, dtype=float)
    v /= np.linalg.norm(v)
    beta = float(beta_from_keV(np.array([E_KEV]))[0])
    piece = length / pieces
    s_start = -0.5 * length + piece * np.arange(pieces)
    segments = {
        "r_mid": np.array([0.0, 0.0, z_mid]) + (s_start + 0.5 * piece)[:, None] * v,
        "v_hat": np.tile(v, (pieces, 1)),
        "L_ang": np.full(pieces, piece),
        "E_keV": np.full(pieces, E_KEV),
        "t_ang": (s_start + 0.5 * length) / beta,
        "t0_ang": np.zeros(pieces),
        "elec_id": np.zeros(pieces, dtype=int),
        "layer": np.zeros(pieces, dtype=int),
        "Ne": 1,
        "thickness_ang": 1.0e5,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }
    segments.update(extra)
    return segments


def _transparent(monkeypatch):
    real = lines_setup.expand_escape_pieces

    def zero_paths(*args, **kwargs):
        pieces, owner, start, end = real(*args, **kwargs)
        return pieces, owner, start * 0.0, end * 0.0

    monkeypatch.setattr(lines_setup, "expand_escape_pieces", zero_paths)


def _spectrum(segments, **kwargs):
    return mc_spectrum(segments, E_GRID, CRYSTAL, [HKL], B_ang2=0.4, n_hat=N_HAT, **kwargs)


CASES = {
    # Batched route, laterally infinite slab: exit through the entrance face,
    # the escape path growing ~1 C attenuation length along the segment.
    "slab": ({}, {}),
    # Batched route, finite footprint: the exit face switches partway along.
    "box-face-switch": (
        {
            "direction": (0.6, 0.0, 0.8),
            "z_mid": 20000.0,
            "crystal_width_ang": 1.0e5,
            "crystal_height_ang": 1.0e5,
        },
        {},
    ),
    # Per-reflection route (finite sinc cutoff, wide enough for this grid).
    "per-hkl": ({}, {"sinc_cutoff": 1.0e4}),
}


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_coherent_spectrum_is_invariant_under_segment_splitting(case):
    """Pieces of a straight flight sum to its field exactly, absorption and the
    in-medium escape-leg phase included -- the midpoint escape amplitude and
    sinc it replaced moved at first order in both."""
    geometry, kwargs = CASES[case]
    spectra = [_spectrum(_track(k, **geometry), coherent=True, **kwargs) for k in SPLITS]
    peak = float(spectra[0].max())
    assert peak > 0.0
    for spectrum in spectra[1:]:
        np.testing.assert_allclose(
            spectrum, spectra[0], rtol=0, atol=scaled_rtol(1e-9, eps_multiple=1e4) * peak
        )


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_single_segment_coherent_yield_is_the_incoherent_yield(monkeypatch, case):
    """Parseval at spectrum level: the coherent self-term of one absorbed
    segment integrates to the incoherent route's segment-mean escape yield."""
    geometry, kwargs = CASES[case]
    segments = _track(1, **geometry)
    coherent = _spectrum(segments, coherent=True, **kwargs)
    incoherent = _spectrum(segments, **kwargs)
    with monkeypatch.context() as m:
        _transparent(m)
        clear = _spectrum(segments, coherent=True, **kwargs)

    assert float(coherent.sum() / clear.sum()) < 0.8  # strongly absorbed
    np.testing.assert_allclose(coherent.sum(), incoherent.sum(), rtol=2e-3)
