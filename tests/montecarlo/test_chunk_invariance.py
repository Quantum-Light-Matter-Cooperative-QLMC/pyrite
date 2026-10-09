"""A1 chunk-size invariance gate (docs/repo-design/compute/compute-performance-optimization.md).

The ``spec_chunk`` / ``brem_chunk`` knobs are memory/performance controls, not
physics controls.  Eager CPU/CuPy fallbacks use them to partition the segment
reduction; optimized JIT reductions may bypass part or all of that dense-matrix
chunking.  Either way, changing the requested chunk must not change the spectrum.
Where the reduction order does move, equality is therefore tolerance-based rather
than bit-for-bit.  This gate lets the implementation retune or replace chunked
work without silently perturbing results.

The tolerance must track the backend's accumulation precision, which
``pyrite._backend.REAL`` selects at import: float64 on a CPU box (or under
``PYRITE_FP64=1``), float32 on a GPU. Reordering an N-term sum perturbs it by
roughly ``sqrt(N)*eps`` relative, so a fixed float64 rtol spuriously fails on
any GPU machine -- the historical bug this parametrization fixes. Set
``CUDA_VISIBLE_DEVICES=""`` to exercise the strict float64 path anywhere.
"""

import numpy as np
import pytest

from pyrite._backend import REAL
from pyrite.materials.crystal import CRYSTALS
from pyrite.montecarlo import (
    _brem_dsigma_dk,
    mc_brem_spectrum,
    mc_spectrum,
    simulate_trajectories,
    spectrum,
)
from pyrite.montecarlo.runner import _env_chunk

xp = spectrum.xp

THETA = np.deg2rad(119.0)
E0_KEV = 25.0
B_002 = 0.8
HKL = ((0, 0, 2), (0, 0, -2))
E_LINE = np.arange(700.0, 1200.0, 4.0)
E_BREM = np.arange(700.0, 20000.0, 200.0)
NE = 12
TINY_CHUNK = 8  # << segment count, so the chunk loop runs many iterations
BIG_CHUNK = 10**9  # one shot: the whole segment set in a single matmul

_info = CRYSTALS["hopg"]
_n_atoms = len(_info["basis"]) / _info["V_cell"]

# Reordering an N-term float sum moves it by ~sqrt(N)*eps relative. The chunk
# loop here reduces over many segments, so allow a few hundred eps of the
# backend's working precision: 1e-10 under float64 (far above the ~1e-13 the
# reorder actually costs), ~2e-5 under the GPU's float32.
_EPS = float(np.finfo(REAL).eps)
RTOL = max(1e-10, 200.0 * _EPS)


@pytest.fixture(scope="module")
def segments():
    """A single small transported case shared by the chunk-invariance checks."""
    segs = simulate_trajectories(
        E0_KEV, NE, 1e7, element="C", n_atoms_per_ang3=_n_atoms, E_cut_keV=1.0, seed=7
    )
    # The gate is meaningless unless the tiny chunk actually splits the segments.
    assert segs["L_ang"].size > TINY_CHUNK
    return segs


def _assert_chunk_invariant(one_shot, chunked):
    peak = float(np.max(np.abs(one_shot)))
    assert peak > 0.0  # a degenerate all-zero spectrum would pass vacuously
    np.testing.assert_allclose(chunked, one_shot, rtol=RTOL, atol=RTOL * 1e-2 * peak)


@pytest.fixture(scope="module")
def finite_side_segments():
    """Finite-footprint segments whose far-field photons leave through +x."""
    return {
        "r_mid": np.array([[4.0, 0.0, 5.0], [3.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        "v_hat": np.tile([0.0, 0.0, 1.0], (3, 1)),
        "L_ang": np.full(3, 10.0),
        "E_keV": np.full(3, 30.0),
        "t_ang": np.zeros(3),
        "t0_ang": np.zeros(3),
        "elec_id": np.arange(3),
        "layer": np.zeros(3, dtype=int),
        "Ne": 3,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
    }


def test_line_spectrum_chunk_invariant(segments):
    kw = dict(crystal="hopg", hkl_list=HKL, theta_obs_rad=THETA, B_ang2=B_002)
    one_shot = mc_spectrum(segments, E_LINE, chunk=BIG_CHUNK, **kw)
    chunked = mc_spectrum(segments, E_LINE, chunk=TINY_CHUNK, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_coherent_line_spectrum_chunk_invariant(segments):
    """Coherent field accumulation is invariant to the spectrum chunk request.

    On the eager fallback this exercises a different partitioning of the complex
    field sum.  On the CUDA JIT reduction the chunk may be bypassed entirely;
    that is also a valid implementation as long as the public knob cannot alter
    the physics result.
    """
    kw = dict(
        crystal="hopg",
        hkl_list=HKL,
        theta_obs_rad=THETA,
        B_ang2=B_002,
        coherent=True,
    )
    one_shot = mc_spectrum(segments, E_LINE, chunk=BIG_CHUNK, **kw)
    chunked = mc_spectrum(segments, E_LINE, chunk=TINY_CHUNK, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_electron_grouped_coherent_chunk_invariant(segments):
    """The per-electron grouped term slices energy when one electron outgrows ``chunk``.

    Bunch offsets activate the decoherence blend and the sinc cutoff keeps the
    eager per-reflection route, so ``chunk=1`` leaves every electron a block
    larger than the chunk; the energy slicing that bounds its memory must not
    change the spectrum.
    """
    active = dict(segments)
    rng = np.random.default_rng(3)
    active["initial_t0_ang"] = rng.normal(0.0, 300.0, NE)
    active["initial_r_ang"] = np.zeros((NE, 3))
    kw = dict(
        crystal="hopg",
        hkl_list=HKL,
        theta_obs_rad=THETA,
        B_ang2=B_002,
        coherent=True,
        sinc_cutoff=4.0,
    )
    one_shot = mc_spectrum(active, E_LINE, chunk=BIG_CHUNK, **kw)
    chunked = mc_spectrum(active, E_LINE, chunk=1, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_energy_slices_bound_an_oversized_group_block():
    from pyrite.montecarlo.spectrum.lines._kernels import _energy_slices

    assert list(_energy_slices(5, 8, 100)) == [slice(0, 100)]
    slices = list(_energy_slices(40, 8, 100))
    assert all((s.stop - s.start) * 40 <= 8 * 100 for s in slices)
    assert [i for s in slices for i in range(s.start, s.stop)] == list(range(100))
    assert list(_energy_slices(10**6, 1, 3)) == [slice(0, 1), slice(1, 2), slice(2, 3)]


def test_brem_spectrum_chunk_invariant(segments):
    kw = dict(element="C", n_atoms_per_ang3=_n_atoms, theta_obs_rad=THETA)
    one_shot = mc_brem_spectrum(segments, E_BREM, chunk=BIG_CHUNK, **kw)
    chunked = mc_brem_spectrum(segments, E_BREM, chunk=TINY_CHUNK, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_finite_line_spectrum_chunk_invariant(finite_side_segments):
    kw = dict(
        crystal="hopg",
        hkl_list=((0, 0, 2),),
        B_ang2=B_002,
        n_hat=np.array([1.0, 0.0, 0.01]),
    )
    one_shot = mc_spectrum(finite_side_segments, E_LINE, chunk=BIG_CHUNK, **kw)
    chunked = mc_spectrum(finite_side_segments, E_LINE, chunk=1, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_finite_coherent_line_spectrum_chunk_invariant(finite_side_segments):
    kw = dict(
        crystal="hopg",
        hkl_list=((0, 0, 2),),
        B_ang2=B_002,
        n_hat=np.array([1.0, 0.0, 0.01]),
        coherent=True,
    )
    one_shot = mc_spectrum(finite_side_segments, E_LINE, chunk=BIG_CHUNK, **kw)
    chunked = mc_spectrum(finite_side_segments, E_LINE, chunk=1, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_finite_brem_spectrum_chunk_invariant(finite_side_segments):
    kw = dict(composition=[("C", 0.176)], n_hat=np.array([1.0, 0.0, 0.01]))
    one_shot = mc_brem_spectrum(finite_side_segments, E_BREM, chunk=BIG_CHUNK, **kw)
    chunked = mc_brem_spectrum(finite_side_segments, E_BREM, chunk=1, **kw)
    _assert_chunk_invariant(one_shot, chunked)


def test_brem_dsigma_gpu_scalar_z_preserves_real_precision():
    dsig = _brem_dsigma_dk(
        6,
        np.array([30.0]),
        np.array([700.0, 1000.0, 5000.0]),
    )

    assert dsig.dtype == REAL
    assert xp.all(dsig > 0)


# ---- the env-var override that drives the A1 spike ---------------------------
def test_env_chunk_parses_positive_override(monkeypatch):
    monkeypatch.setenv("PYRITE_MC_SPEC_CHUNK", "250000")
    assert _env_chunk("PYRITE_MC_SPEC_CHUNK", 40000) == 250000


@pytest.mark.parametrize("bad", ["", "0", "-5", "12.5", "lots", "  "])
def test_env_chunk_falls_back_on_bad_values(monkeypatch, bad):
    monkeypatch.setenv("PYRITE_MC_SPEC_CHUNK", bad)
    assert _env_chunk("PYRITE_MC_SPEC_CHUNK", 40000) == 40000


def test_env_chunk_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_SPEC_CHUNK", raising=False)
    assert _env_chunk("PYRITE_MC_SPEC_CHUNK", 40000) == 40000
