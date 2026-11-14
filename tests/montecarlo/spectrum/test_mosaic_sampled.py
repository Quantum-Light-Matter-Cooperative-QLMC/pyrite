"""Per-history sampled crystallite orientations for the exact mosaic route (#1185).

The sampled estimator draws one crystallite (``K`` stratified tilts) per
electron history instead of evaluating every row at the ``nodes**2``
Gauss-Hermite tilts. It must be an unbiased estimator of the same incoherent
ensemble average: its yield, centroid and second moment match the converged
quadrature within statistical error, its draws depend only on
``(seed, elec_id)``, and a zero mosaic width is the perfect crystal bit-for-bit.

Validation: mosaic-mc-sampled
"""

import numpy as np
import pytest

from pyrite.montecarlo import _small_tilt_R
from pyrite.montecarlo.spectrum import mc_cxr_spectrum, subdivide_flights
from pyrite.montecarlo.spectrum.cxr._kernels import _polarization_pair
from pyrite.montecarlo.spectrum.cxr._mosaic_sampled import (
    MOSAIC_CHILD,
    history_tilts,
    lattice_points,
    polarization_pairs,
    tilt_rotate,
)
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories
from tests.helpers import host_backend_only, to_host

_host_only = host_backend_only(
    "sampled-mosaic statistics are float64 host statements on the grouped "
    "reduction; device parity is a separate gate"
)

FWHM = np.deg2rad(3.5)
SIGMA = FWHM / 2.3548200450309493
CARBON = [("C", 0.1136)]
CXR_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "theta_obs_rad": np.deg2rad(119.0),
    "composition": CARBON,
    # psi = angle(v, g) != 0, so a tilt moves the line at first order
    "recip_miscut_rad": (np.deg2rad(30.0), 0.0),
}
GRID = np.arange(400.0, 1600.0, 1.0)


# ---- draws -------------------------------------------------------------------
def test_tilts_are_the_mosaic_gaussian():
    tilts = history_tilts(11, np.arange(200_000), FWHM, 1)[:, 0, :]
    assert tilts.mean(axis=0) == pytest.approx([0.0, 0.0], abs=5.0 * SIGMA / np.sqrt(2e5))
    assert tilts.std(axis=0) / SIGMA == pytest.approx([1.0, 1.0], abs=0.01)
    assert abs(np.corrcoef(tilts.T)[0, 1]) < 0.01


def test_tilts_depend_only_on_seed_and_history_id():
    full = history_tilts(3, np.arange(1000), FWHM, 4)
    # any subset, any order, any block: the same history keeps its crystallite
    ids = np.array([999, 5, 500, 0])
    np.testing.assert_array_equal(history_tilts(3, ids, FWHM, 4), full[ids])
    np.testing.assert_array_equal(history_tilts(3, np.arange(400, 1000), FWHM, 4), full[400:])
    assert not np.array_equal(history_tilts(4, np.arange(1000), FWHM, 4), full)


def test_draws_are_a_shifted_lattice():
    points = lattice_points(5)
    assert sorted(points[:, 0]) == pytest.approx(np.arange(5) / 5)
    assert sorted(points[:, 1]) == pytest.approx(np.arange(5) / 5)  # stratified per axis
    tilts = history_tilts(1, np.arange(20_000), FWHM, 5)
    # each draw slot is marginally the mosaic Gaussian
    assert tilts.std(axis=0).ravel() / SIGMA == pytest.approx(np.ones(10), abs=0.03)


def test_child_stream_is_new():
    assert MOSAIC_CHILD == (8, 7)


def test_vector_rotation_matches_small_tilt_matrix():
    g = np.array([0.3, -0.2, 1.7])
    tilts = history_tilts(2, np.arange(64), FWHM, 1)[:, 0, :]
    tilts[0] = 0.0  # identity branch
    expected = np.stack([_small_tilt_R(dx, dy) @ g for dx, dy in tilts])
    got = tilt_rotate(g, tilts[:, 0], tilts[:, 1])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(got[0], g)


def test_vector_polarization_pairs_match_scalar():
    k_hat = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
    g_rows = tilt_rotate([0.0, 0.0, 1.0], *history_tilts(2, np.arange(16), FWHM, 1)[:, 0, :].T)
    e_s, e_p = polarization_pairs(k_hat, g_rows)
    for i, g in enumerate(g_rows):
        s, p = _polarization_pair(k_hat, g)
        np.testing.assert_allclose(e_s[i], s, atol=1e-15)
        np.testing.assert_allclose(e_p[i], p, atol=1e-15)


# ---- spectrum ----------------------------------------------------------------
def _flights(Ne, seed=7, thickness_ang=500.0):
    """Thin film: the mosaic, not multiple-scattering Doppler, sets the width."""
    return simulate_trajectories(
        E0_keV=25.0,
        Ne=Ne,
        thickness_ang=thickness_ang,
        composition=CARBON,
        seed=seed,
        transport_core="lockstep",
        energy_model="midpoint",
        E_cut_keV=1.0,
    )


def _moments(spec):
    spec = np.asarray(to_host(spec), dtype=float)
    total = spec.sum()
    mean = (GRID * spec).sum() / total
    var = ((GRID - mean) ** 2 * spec).sum() / total
    return np.array([total, mean, np.sqrt(var)])


def test_zero_width_is_the_perfect_crystal_bit_for_bit():
    segs = _flights(8)
    perfect = mc_cxr_spectrum(segs, GRID, **CXR_KWARGS)
    for fwhm in (None, 0.0):
        sampled = mc_cxr_spectrum(
            segs,
            GRID,
            **CXR_KWARGS,
            mosaic_fwhm_rad=fwhm,
            mosaic_nodes=3,
            mosaic_sampling="history",
            mosaic_seed=5,
        )
        np.testing.assert_array_equal(to_host(sampled), to_host(perfect))


def test_sampled_needs_a_seed_and_refuses_coherent():
    segs = _flights(2)
    kw = dict(CXR_KWARGS, mosaic_fwhm_rad=FWHM, mosaic_nodes=2, mosaic_sampling="history")
    with pytest.raises(ValueError, match="mosaic_seed"):
        mc_cxr_spectrum(segs, GRID, **kw)
    with pytest.raises(NotImplementedError, match="incoherent-only"):
        mc_cxr_spectrum(segs, GRID, **kw, mosaic_seed=1, coherent=True)
    with pytest.raises(ValueError, match="mosaic_sampling"):
        mc_cxr_spectrum(segs, GRID, **dict(kw, mosaic_sampling="row"), mosaic_seed=1)


@_host_only
def test_sampled_is_reproducible_and_seed_sensitive():
    segs = _flights(16)
    kw = dict(CXR_KWARGS, mosaic_fwhm_rad=FWHM, mosaic_nodes=2, mosaic_sampling="history")
    a = mc_cxr_spectrum(segs, GRID, **kw, mosaic_seed=9)
    b = mc_cxr_spectrum(segs, GRID, **kw, mosaic_seed=9)
    c = mc_cxr_spectrum(segs, GRID, **kw, mosaic_seed=10)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@_host_only
def test_sampled_mean_matches_converged_quadrature_moments():
    """Yield, centroid and RMS width agree with GH N=33 within 5 standard errors.

    The trajectories are fixed; only the crystallite draws vary across the
    mosaic seeds, so the seed-to-seed scatter is the estimator's own error.
    GH is converged here (N=17 vs 49 agree to 4e-5 in width).
    """
    segs = _flights(120)
    reference = _moments(
        mc_cxr_spectrum(segs, GRID, **CXR_KWARGS, mosaic_fwhm_rad=FWHM, mosaic_nodes=33)
    )
    samples = np.array(
        [
            _moments(
                mc_cxr_spectrum(
                    segs,
                    GRID,
                    **CXR_KWARGS,
                    mosaic_fwhm_rad=FWHM,
                    mosaic_nodes=4,
                    mosaic_sampling="history",
                    mosaic_seed=seed,
                )
            )
            for seed in range(16)
        ]
    )
    mean = samples.mean(axis=0)
    stderr = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(np.abs(mean - reference) <= 5.0 * stderr + 1e-12 * np.abs(reference)), (
        mean,
        reference,
        stderr,
    )
    # the mosaic is resolved: it broadens the line by many standard errors
    perfect = _moments(mc_cxr_spectrum(segs, GRID, **CXR_KWARGS))
    assert mean[2] - perfect[2] > 10.0 * stderr[2]


@_host_only
def test_substeps_of_a_flight_share_its_history_crystallite():
    """Refining flights into substeps moves the sampled spectrum no more than GH's.

    Substeps carry their history's ``elec_id``, so they keep its crystallite and
    still add as one flight's field in the grouped reduction; the residual is
    the substep quadrature error both routes share.
    """
    segs = _flights(40)
    rows, _ = subdivide_flights(segs, composition=CARBON, max_dE_frac=3e-4)
    assert rows["L_ang"].size > 2 * segs["L_ang"].size

    def l1(**kw):
        base = mc_cxr_spectrum(segs, GRID, **CXR_KWARGS, mosaic_fwhm_rad=FWHM, **kw)
        fine = mc_cxr_spectrum(rows, GRID, **CXR_KWARGS, mosaic_fwhm_rad=FWHM, **kw)
        return float(np.abs(fine - base).sum() / np.abs(base).sum())

    sampled = l1(mosaic_nodes=2, mosaic_sampling="history", mosaic_seed=3)
    assert sampled < 1.5 * l1(mosaic_nodes=3)
    assert sampled < 3e-3


# ---- case wiring -------------------------------------------------------------
def _sweep(**kw):
    from pyrite.campaign.sweep import BeamSpec, Sweep
    from pyrite.detectors import EnergyBins

    return Sweep(
        material="hopg",
        thickness_ang=10000.0,
        beam=BeamSpec(energy_keV=30),
        tilt_deg=30.0,
        mosaic=True,
        **kw,
        energy_bins=EnergyBins(cxr=GRID, brem=np.arange(0.0, 1000.0, 100.0)),
    )


def test_route_mc_sampled_lowers_to_the_exact_route_with_history_draws():
    from pyrite.campaign.sweep import build_cases

    sampled = build_cases(_sweep(mosaic_route="mc-sampled", mosaic_nodes=3))[0]
    quadrature = build_cases(_sweep(mosaic_route="mc", mosaic_nodes=3))[0]
    assert sampled["mosaic_mc_sampling"] == "history"
    assert sampled["mosaic_mc_fwhm_rad"] == pytest.approx(np.deg2rad(0.8))
    assert sampled["mosaic_fwhm_rad"] is None  # analytic term off
    assert "mosaic_mc_sampling" not in quadrature  # GH case keys unchanged
    # the sampled rule changes no transport input: only the scoring key differs
    differ = {k for k in sampled if not np.array_equal(sampled[k], quadrature.get(k))}
    assert differ == {"mosaic_mc_sampling"}


def test_sampled_case_refuses_coherent_emission():
    from pyrite.montecarlo.case import Case

    case = _sweep_case(mosaic_route="mc-sampled")
    with pytest.raises(ValueError, match="incoherent-only"):
        Case(**dict(case, coherent_emission=True))


def _sweep_case(**kw):
    from pyrite.campaign.sweep import build_cases

    return dict(build_cases(_sweep(**kw))[0])
