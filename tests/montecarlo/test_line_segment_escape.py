"""Segment-mean escape on the incoherent PXR/CBS line route (issue #181).

Splitting a segment into collinear pieces keeps its velocity, so every piece
radiates at the same resonance energy with the same amplitude, and only the
finite-time width and the escape weight change. Against a transparent control
run on the same pieces (escape paths zeroed), the absorbed spectrum is then
``<T> * control`` point by point, and ``<T>`` -- the segment mean of
``exp(-tau)`` -- must not depend on the split.

Validation: segment-escape-average
"""

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS
from pyrite.montecarlo.spectrum import mc_spectrum
from pyrite.montecarlo.spectrum.lines import _setup as lines_setup
from pyrite.montecarlo.spectrum.segment_escape import (
    mean_transmission,
    piece_mean_transmission,
    segment_escape_paths,
    segment_escape_pieces,
)
from pyrite.montecarlo.transport import beta_from_keV

CRYSTAL = "hopg"
HKL = (0, 0, 2)
E_KEV = 100.0
E_GRID = np.arange(1550.0, 1800.0, 0.05)
N_HAT = np.array([np.sin(np.deg2rad(119.0)), 0.0, np.cos(np.deg2rad(119.0))])
SPLITS = (1, 4, 16)


def _n_atoms():
    info = CRYSTALS[CRYSTAL]
    return len(info["basis"]) / info["V_cell"]


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
    real = lines_setup.segment_escape_pieces

    def zero_paths(*args, **kwargs):
        fraction, start, end = real(*args, **kwargs)
        return fraction, start * 0.0, end * 0.0

    monkeypatch.setattr(lines_setup, "segment_escape_pieces", zero_paths)


def _escape_ratio(monkeypatch, segments, **kwargs):
    # Isolate attenuation: with dispersion, face switches have distinct roots
    # and absorption can reshape their superposition (issue #187).
    monkeypatch.setattr(
        lines_setup,
        "refractive_index",
        lambda crystal, energy, *args: np.ones_like(energy, dtype=complex),
    )
    absorbed = mc_spectrum(segments, E_GRID, CRYSTAL, [HKL], B_ang2=0.4, n_hat=N_HAT, **kwargs)
    with monkeypatch.context() as m:
        _transparent(m)
        clear = mc_spectrum(segments, E_GRID, CRYSTAL, [HKL], B_ang2=0.4, n_hat=N_HAT, **kwargs)
    peak = float(clear.max())
    assert peak > 0.0
    ratio = float(absorbed.sum() / clear.sum())
    # One resonance energy: absorption rescales the line, never reshapes it.
    np.testing.assert_allclose(absorbed, ratio * clear, rtol=1e-9, atol=1e-12 * peak)
    return ratio


CASES = {
    # Batched route, laterally infinite slab: exit through the entrance face;
    # the escape path grows by ~8 um along the segment, ~1 C attenuation
    # length at 1.6 keV.
    "slab": ({}, {}),
    # Batched route, finite footprint: the exit face switches from the
    # entrance face to the +x side face partway along the tilted segment.
    "box-face-switch": (
        {
            "direction": (0.6, 0.0, 0.8),
            "z_mid": 20000.0,
            "crystal_width_ang": 1.0e5,
            "crystal_height_ang": 1.0e5,
        },
        {},
    ),
    # Per-hkl route, strongly absorbing Cu film on the crystal: the segment
    # starts inside the film and crosses the interface.
    "layered": (
        {"length": 8000.0, "z_mid": 4500.0},
        {
            "layers": [
                (0.0, 2500.0, [("Cu", 0.085)]),
                (2500.0, 1.0e5, [("C", _n_atoms())]),
            ]
        },
    ),
}


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_incoherent_line_escape_is_invariant_under_segment_splitting(monkeypatch, case):
    geometry, kwargs = CASES[case]
    ratios = [_escape_ratio(monkeypatch, _track(k, **geometry), **kwargs) for k in SPLITS]

    assert 0.0 < ratios[0] < 0.8  # strongly absorbed, so the check has teeth
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)


def test_midpoint_escape_would_drift_under_splitting(monkeypatch):
    """The split ladder above is not trivially flat: the midpoint rule moves."""
    whole = _track(1)
    pieces = segment_escape_pieces(whole, N_HAT)
    p0, p1 = float(pieces[1][0, 0, 0]), float(pieces[2][0, 0, 0])
    mu = 1.0 / 3000.0
    midpoint = np.exp(-mu * 0.5 * (p0 + p1))
    mean = float(mean_transmission(mu * p0, mu * p1))
    assert mean > 1.05 * midpoint  # Jensen, at Delta tau ~ 5


def test_padded_pieces_match_the_flat_piece_sum():
    segments = _track(
        3,
        direction=(0.6, 0.3, 0.7),
        z_mid=2500.0,
        crystal_width_ang=6000.0,
        crystal_height_ang=6000.0,
    )
    layers = [(0.0, 2000.0, None), (2000.0, 1.0e5, None)]
    mu = np.array([1.0 / 900.0, 1.0 / 4000.0])

    owner, fraction, start, end = segment_escape_paths(segments, np.arange(3), N_HAT, layers=layers)
    flat = np.bincount(
        owner, weights=fraction * mean_transmission(start @ mu, end @ mu), minlength=3
    )
    frac, p_start, p_end = segment_escape_pieces(segments, N_HAT, layers=layers)
    padded = piece_mean_transmission(frac, p_start, p_end, np.broadcast_to(mu, (3, 1, 2)))

    assert frac.shape[1] > 1  # the box and layer cuts produced several pieces
    np.testing.assert_allclose(frac.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(padded[:, 0], flat, rtol=1e-6)
