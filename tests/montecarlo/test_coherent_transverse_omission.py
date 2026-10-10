"""Flat omission certified by the row's transverse factor ``F_z F_perp`` (#365).

The reducer blends each finite-footprint row with ``F_z F_perp``. Certifying
omission by ``F_z`` alone is conservative under ``F_perp <= 1``; a short bunch
with a wide spot then never omits although its cross-electron term vanishes.
Every comparison reuses one segment set, so differences are the omission alone.

Validation: coherent-transverse-flat-omission
"""

from types import SimpleNamespace

import numpy as np
import pytest
from mpmath import mp

from pyrite._backend import REAL, xp
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from pyrite.montecarlo import mc_spectrum
from pyrite.montecarlo.geometry import project_beam_entry
from pyrite.montecarlo.spectrum.coherent_transverse import (
    transverse_form_factor,
    transverse_form_factor_upper,
    transverse_slope_sign,
    transverse_spot,
)
from pyrite.montecarlo.spectrum.lines._per_hkl import _flat_energy_keep
from pyrite.montecarlo.transport.beam_entry import beam_entry_record, face_arrival_delay_ang

TILT, AZIM = np.deg2rad(45.0), np.deg2rad(135.0)
E0_KEV = 30.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
ENERGY = np.arange(700.0, 1500.0, 2.0)
N_HAT = np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1])
KWARGS = {"crystal": "hopg", "hkl_list": [(0, 0, 2)], "B_ang2": 0.8, "n_hat": N_HAT}
G_002 = np.asarray(reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])[0], dtype=float)
SUBGRID_RTOL = max(1e-12, 100.0 * float(np.finfo(REAL).eps))


def _record(fwhm_ang, *, tilt=TILT, azim=AZIM):
    return beam_entry_record(
        transverse_distribution=None,
        beam_fwhm_mm=fwhm_ang * 1e-7,
        beam_fwhm_y_mm=None,
        tilt_polar_rad=tilt,
        tilt_azim_rad=azim,
        E0_keV=E0_KEV,
        energy_spread_frac=None,
        groove=None,
    )


def _exact_perp(spot, omega, n_hat, g_vec):
    """Independent 60-digit evaluation of the production law at binary64 inputs."""
    mp.dps = 60
    S = [[mp.mpf(float(v)) for v in row] for row in spot.face_covariance]
    a = [mp.mpf(float(n_hat[i])) - mp.mpf(float(spot.beam_xy_over_beta[i])) for i in (0, 1)]
    k = [mp.mpf(float(omega)) * a[i] + mp.mpf(float(g_vec[i])) for i in (0, 1)]
    q = k[0] ** 2 * S[0][0] + 2 * k[0] * k[1] * S[0][1] + k[1] ** 2 * S[1][1]
    return mp.exp(-max(q, 0))


def _stub(spot, omega, limit, population, *, rms_fs=1e-12, row_vectors_known=True):
    return SimpleNamespace(
        request=SimpleNamespace(
            coherent_flat_omission_limit=limit,
            physical_electrons=population,
            longitudinal_rms_fs=rms_fs,
        ),
        Ne=6,
        E_grid=omega * HBARC_EV_ANG,
        omega_grid=omega,
        finite_footprint_now=True,
        transverse=spot,
        n_hat=N_HAT,
    )


@pytest.mark.parametrize("fwhm", [2.0, 40.0, 1.0e4])
def test_upper_bound_encloses_the_production_factor(fwhm):
    spot = transverse_spot(_record(fwhm))
    omega = ENERGY[::37] / HBARC_EV_ANG
    production = np.asarray(transverse_form_factor(spot, omega, N_HAT, G_002))
    for om, value in zip(omega, production, strict=True):
        upper = transverse_form_factor_upper(spot, om, N_HAT, G_002)
        assert 0.0 < upper <= 1.0
        assert upper >= _exact_perp(spot, om, N_HAT, G_002)
        # Tight up to relative rounding; subnormals have only absolute precision.
        assert upper <= value * (1 + 1e-12) + 1e-310
    whole = transverse_form_factor_upper(spot, (omega[0], omega[-1]), N_HAT, G_002)
    assert whole >= max(float(_exact_perp(spot, om, N_HAT, G_002)) for om in omega)


def _resonant_row(spot, omega_match):
    """Row vector with ``K(omega_match) = 0``: ``F_perp = 1`` there, decaying both ways."""
    g = G_002.copy()
    g[:2] = -omega_match * (N_HAT[:2] - spot.beam_xy_over_beta)
    return g


def test_slope_sign_matches_the_quadratic_and_abstains_at_its_minimum():
    spot = transverse_spot(_record(20.0))
    g = _resonant_row(spot, 0.5)
    assert transverse_slope_sign(spot, 0.4, N_HAT, g) == -1
    assert transverse_slope_sign(spot, 0.6, N_HAT, g) == 1
    # At exact phase matching the slope interval must contain zero.
    assert transverse_slope_sign(spot, 0.5, N_HAT, g) == 0


def test_mask_certifies_both_monotone_sides_and_keeps_the_resonance():
    """Omitted nodes satisfy the exact certificate; F_z = 1 omits none by itself."""
    spot = transverse_spot(_record(20.0))
    omega = ENERGY / HBARC_EV_ANG
    g = _resonant_row(spot, float(omega[omega.size // 2]))
    population, limit = 50.0, 0.05 * 49.0
    F = np.asarray(transverse_form_factor(spot, omega, N_HAT, g))
    st = _stub(spot, omega, limit, population)
    keep = _flat_energy_keep(st, F, row_vectors=g)
    omitted = np.setdiff1d(np.arange(omega.size), keep)
    assert omitted.size and keep.size
    exact = np.array([float(_exact_perp(spot, om, N_HAT, g)) for om in omega])
    # Pair weight is (N-1)/(M-1) * (M-1) = N-1; certified nodes need F (N-1) <= limit.
    assert np.all(exact[omitted] * (population - 1) <= limit)
    # Kept nodes are uncertified, up to the outward rounding at the threshold.
    assert np.all(np.isin(np.flatnonzero(exact * (population - 1) > limit), keep))
    kept_without_reason = np.setdiff1d(keep, np.flatnonzero(exact * (population - 1) > limit))
    assert np.all(np.abs(exact[kept_without_reason] * (population - 1) / limit - 1) < 1e-9)
    # Both monotone sides contribute: omitted nodes lie below and above the kept band.
    assert omitted.min() < keep.min() and omitted.max() > keep.max()
    assert omega.size // 2 in keep
    # Without the transverse spot the conservative F_z certificate omits nothing.
    st.transverse = None
    assert _flat_energy_keep(st, F, row_vectors=g).size == omega.size


def test_mask_without_row_vectors_or_with_a_point_spot_stays_longitudinal():
    omega = ENERGY / HBARC_EV_ANG
    wide = transverse_spot(_record(2.0))
    F = np.asarray(transverse_form_factor(wide, omega, N_HAT, G_002))
    st = _stub(wide, omega, 0.05 * 49.0, 50.0)
    assert _flat_energy_keep(st, F, row_vectors=None).size == omega.size
    point = transverse_spot(_record(1e-30))
    st.transverse = point
    assert _flat_energy_keep(st, np.zeros(omega.size), row_vectors=G_002).size == omega.size


def test_one_failing_row_retains_its_coordinates_for_every_row():
    spot = transverse_spot(_record(20.0))
    omega = ENERGY / HBARC_EV_ANG
    # Second row phase-matched at the middle of the axis: F_perp = 1 there.
    matched = _resonant_row(spot, float(omega[omega.size // 2]))
    rows = np.stack([G_002, matched])
    F = np.stack([np.asarray(transverse_form_factor(spot, omega, N_HAT, g)) for g in rows])
    st = _stub(spot, omega, 0.05 * 49.0, 50.0)
    joint = _flat_energy_keep(st, F, row_vectors=rows)
    alone = _flat_energy_keep(st, F[0], row_vectors=G_002)
    assert alone.size == 0 and omega.size // 2 in joint
    assert joint.size < omega.size


def test_flat_row_uses_the_whole_range_enclosure():
    """``a = 0`` makes ``F_perp`` constant: no slope sign, one interval bound."""
    record = _record(2.0, tilt=0.0, azim=0.0)
    spot = transverse_spot(record)
    n_hat = np.array([0.0, 0.0, 1.0])
    g = np.array([3.0, 0.0, 1.87])
    omega = ENERGY / HBARC_EV_ANG
    F = np.asarray(transverse_form_factor(spot, omega, n_hat, g))
    assert np.ptp(F) == 0.0 and F[0] * 49.0 < 0.05 * 49.0
    st = _stub(spot, omega, 0.05 * 49.0, 50.0)
    st.n_hat = n_hat
    assert _flat_energy_keep(st, F, row_vectors=g).size == 0


def _tilted_identical_tracks(count, fwhm_ang):
    """Identical straight tracks translated by sampled face points and delays."""
    record = _record(fwhm_ang)
    w = np.random.default_rng(7).normal(size=(count, 2)) * fwhm_ang * FWHM_TO_SIGMA
    p0 = np.zeros((count, 3))
    p0[:, :2] = project_beam_entry(w, TILT, AZIM)
    t0 = face_arrival_delay_ang(record, p0, np.full(count, E0_KEV))
    return {
        "r_mid": np.array([4.0, 0.0, 5.0]) + p0,
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, E0_KEV),
        "t_ang": np.zeros(count),
        "t0_ang": t0,
        "elec_id": np.arange(count),
        "layer": np.zeros(count, dtype=int),
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 1.0e4,
        "crystal_height_ang": 1.0e4,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
        "initial_r_ang": p0,
        "initial_t0_ang": t0,
        "beam_entry": record,
    }


@pytest.fixture(params=["auto", "eager", "jit"] if xp.__name__ == "cupy" else ["auto"])
def coherent_route(request, monkeypatch):
    """Exercise each CUDA reducer; CPU has one implementation per geometry."""
    from pyrite.montecarlo.spectrum.lines import _policy

    if request.param != "auto":
        monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_STREAM", False)
        monkeypatch.setattr(_policy, "_USE_JIT_COHERENT_REDUCTION", request.param == "jit")
    return request.param


@pytest.mark.parametrize("extra", [{}, {"sinc_cutoff": 4.0}], ids=["exact", "windowed"])
def test_short_bunch_omits_where_the_transverse_factor_certifies(extra, coherent_route):
    """Aligned offset-free fields attain the bound: Y = G [1 + (N-1) F_z F_perp].

    With a 1e-12 fs bunch ``F_z = 1`` to rounding, so every omitted node is
    licensed by ``F_perp`` alone; the error must stay inside ``limit * G``.
    """
    segments = _tilted_identical_tracks(6, 3.0)
    population = 50.0
    limit = 0.05 * (population - 1)
    common = dict(
        coherent=True, longitudinal_rms_fs=1e-12, physical_electrons=population, **KWARGS, **extra
    )
    full = mc_spectrum(segments, ENERGY, **common)
    disabled = mc_spectrum(segments, ENERGY, coherent_flat_omission_limit=0, **common)
    floor = mc_spectrum(segments, ENERGY, coherent_flat_omission_limit=1e300, **common)
    masked = mc_spectrum(segments, ENERGY, coherent_flat_omission_limit=limit, **common)
    np.testing.assert_array_equal(disabled, full)
    spot = transverse_spot(segments["beam_entry"])
    omega = ENERGY / HBARC_EV_ANG
    exact = np.array([float(_exact_perp(spot, om, N_HAT, G_002)) for om in omega])
    peak = float(np.max(full))
    np.testing.assert_allclose(
        full, (1 + (population - 1) * exact) * floor, rtol=1e-6, atol=SUBGRID_RTOL * peak
    )
    lit = floor > 1e-6 * peak
    omitted = lit & np.isclose(masked, floor, rtol=SUBGRID_RTOL, atol=0)
    assert np.all(np.isclose(masked, full, rtol=SUBGRID_RTOL)[lit & ~omitted])
    assert omitted.any() and (exact * (population - 1) > limit).any()
    assert np.all(exact[omitted] * (population - 1) <= limit * (1 + 1e-9))
    assert np.all(np.abs(full - masked) <= limit * floor + SUBGRID_RTOL * peak)


def test_underflowed_bound_product_is_never_a_certified_zero():
    """Two positive bounds can multiply to binary64 zero; it must round outward.

    Fresh-context counterexample: true ``F W`` 5.75e-315 above a 1e-318 limit.
    """
    from pyrite.montecarlo.spectrum.coherent_transverse import TransverseSpot
    from pyrite.montecarlo.spectrum.lines._per_hkl import _row_transverse_certified

    energy = np.array([1000.0, 1001.0, 1002.0])
    omega = energy / HBARC_EV_ANG
    spot = TransverseSpot(np.diag([1.6 / omega[0] ** 2, 0.0]), np.zeros(2))
    n_hat = np.array([1.0, 0.0, 0.0])
    sigma = np.sqrt(743.5) * HBARC_EV_ANG / energy[0]
    st = SimpleNamespace(n_hat=n_hat)
    out = _row_transverse_certified(st, spot, np.zeros(3), omega, energy, sigma, 1e10, 1e-318)
    mp.dps = 60
    true = [
        mp.exp(-((mp.mpf(e) * mp.mpf(sigma) / mp.mpf(HBARC_EV_ANG)) ** 2))
        * mp.exp(-(mp.mpf(om) ** 2) * mp.mpf(1.6 / omega[0] ** 2))
        * 1e10
        for e, om in zip(energy, omega, strict=True)
    ]
    assert all(value > 1e-318 for value in true)
    assert not out.any()
