"""Transverse phase-space policy: resolution, sampling, and inertness.

Validation: beam-phase-space-injection
Validation: beam-energy-spread-injection
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from cxr_mc.beam_metrics import sampled_beam_metrics
from cxr_mc.montecarlo.geometry import beam_frame_basis
from cxr_mc.montecarlo.transport import simulate_trajectories
from cxr_mc.sweep import BeamSpec, Sweep, build_cases
from cxr_mc.transverse import (
    TransverseDistribution,
    resolve_transverse_distribution,
    sample_transverse,
)

_POLICY = TransverseDistribution(
    normalized_emittance_x_mm_mrad=1.0,
    beta_twiss_x_m=0.5,
    alpha_twiss_x=-0.8,
)


def test_alpha_twiss_accepts_negative_values():
    """A diverging beam past its waist has alpha < 0; it is not a magnitude."""
    assert _POLICY.alpha_twiss_x == -0.8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_emittance_x_mm_mrad", 0.0),
        ("normalized_emittance_x_mm_mrad", -1.0),
        ("beta_twiss_x_m", 0.0),
        ("beta_twiss_y_m", -2.0),
    ],
)
def test_positive_magnitudes_are_rejected_when_nonpositive(field, value):
    with pytest.raises(ValueError, match="positive"):
        TransverseDistribution(**{**_POLICY.__dict__, field: value})


def test_y_plane_mirrors_x_when_unset():
    resolved = resolve_transverse_distribution(_POLICY, energy_keV=30.0)
    assert resolved.y == resolved.x


def test_geometric_emittance_scales_inversely_with_beta_gamma():
    """The normalized-emittance convention: eps_n is the sweep invariant.

    At fixed eps_n, geometric emittance must fall as 1/(beta*gamma), so a
    single BeamSpec describes ONE beam across the whole energy axis.
    """
    low = resolve_transverse_distribution(_POLICY, energy_keV=30.0)
    high = resolve_transverse_distribution(_POLICY, energy_keV=3000.0)
    emittance_ratio = low.x.geometric_emittance_mm_rad / high.x.geometric_emittance_mm_rad
    beta_gamma_ratio = high.beta_gamma / low.beta_gamma
    assert emittance_ratio == pytest.approx(beta_gamma_ratio, rel=1e-12)


def test_sampled_moments_round_trip_through_beam_metrics():
    """The strongest single check: what goes in comes back out.

    Emittance was previously measurable on the way out but not specifiable on
    the way in. This closes that asymmetry at two energies differing in
    beta*gamma, which is what makes the normalized convention observable.
    """
    for energy_keV in (30.0, 3000.0):
        resolved = resolve_transverse_distribution(_POLICY, energy_keV=energy_keV)
        rng = np.random.default_rng(20260806)
        x, xp, y, yp = sample_transverse(resolved, 400_000, rng)
        zeros = np.zeros_like(x)
        metrics = sampled_beam_metrics(
            x,
            xp,
            y,
            yp,
            zeros,
            zeros,
            energy_keV=energy_keV,
            bunch_charge_pc=1.0,
            rep_rate_hz=5000.0,
        )
        # mm*rad out vs mm*mrad in; m in vs mm/rad out.
        assert metrics.x.normalized_emittance_mm_rad * 1e3 == pytest.approx(1.0, rel=2e-2)
        assert metrics.x.beta_mm_per_rad / 1e3 == pytest.approx(0.5, rel=2e-2)
        assert metrics.x.alpha == pytest.approx(-0.8, rel=2e-2)


def test_sampled_correlation_sign_follows_alpha():
    """alpha = -<x x'>/eps, so a negative alpha gives a positive correlation."""
    resolved = resolve_transverse_distribution(_POLICY, energy_keV=30.0)
    x, xp, _, _ = sample_transverse(resolved, 200_000, np.random.default_rng(7))
    covariance = float(np.mean((x - x.mean()) * (xp - xp.mean())))
    assert covariance > 0.0


def test_transverse_policy_is_mutually_exclusive_with_legacy_spot():
    """Three numbers for two moments plus a correlation is over-determined.

    Silent precedence would hand back a plausible beam that is not the one
    requested, so build_cases refuses rather than picking a winner.
    """
    with pytest.raises(ValueError, match="incompatible"):
        build_cases(Sweep(material="hopg", beam=BeamSpec(transverse=_POLICY)))


def test_with_transverse_clears_the_default_spot():
    beam = BeamSpec.with_transverse(_POLICY)
    assert beam.transverse_fwhm_x_mm is None
    assert beam.transverse_fwhm_y_mm is None
    cases = build_cases(Sweep(material="hopg", beam=beam))
    assert cases[0]["transverse_distribution"]["x"]["alpha_twiss"] == -0.8


def test_resolution_joins_case_identity_only_when_set():
    """Inert default stays bit-for-bit: no policy, no case key."""
    cases = build_cases(Sweep(material="hopg", beam=BeamSpec()))
    assert "transverse_distribution" not in cases[0]


def test_derived_divergence_falls_with_energy():
    """divergence_mrad cannot be a stored input across a swept energy axis."""
    beam = BeamSpec.with_transverse(_POLICY)
    low = beam.derived_divergence_mrad(30.0)
    high = beam.derived_divergence_mrad(3000.0)
    assert low is not None and high is not None
    assert high[0] < low[0]
    assert BeamSpec().derived_divergence_mrad(30.0) is None


def test_charge_and_rep_rate_stay_inert_on_the_sampler():
    """Charge is normalization, not phase space (critique 6).

    Pinned by a test rather than a docstring because coherent emission
    genuinely scales with N_phys, so a future change here is legitimate --
    it just has to be a deliberate, ledgered decision instead of drift.
    """
    resolved = resolve_transverse_distribution(_POLICY, energy_keV=30.0)
    baseline = sample_transverse(resolved, 5_000, np.random.default_rng(3))
    beam = BeamSpec.with_transverse(_POLICY, bunch_charge_pc=250.0, rep_rate_hz=1.0)
    charged = resolve_transverse_distribution(beam.transverse, energy_keV=30.0)
    assert charged == resolved
    repeat = sample_transverse(charged, 5_000, np.random.default_rng(3))
    for lhs, rhs in zip(baseline, repeat, strict=True):
        assert np.array_equal(lhs, rhs)


def test_bundled_emittance_demo_profile_resolves_a_twiss_beam():
    """The shipped `hopg_emittance_demo` is the worked example of the block.

    It also pins the one interaction a profile author cannot see from the TOML:
    a `beam.transverse` table has to clear the spot FWHM that every profile beam
    otherwise defaults to, or the profile could not build a single case.
    """
    from cxr_mc.config import material_sweep
    from cxr_mc.materials import CATALOG

    assert CATALOG.profile_materials("hopg_emittance_demo") == ("hopg",)
    beam = material_sweep("hopg", profile="hopg_emittance_demo").beam
    assert beam.transverse_fwhm_x_mm is None
    assert beam.transverse_fwhm_y_mm is None
    assert beam.transverse.normalized_emittance_x_mm_mrad == 0.1
    assert beam.transverse.alpha_twiss_x == 0.0  # a waist on the entrance face
    assert beam.energy_spread_frac == 0.001

    cases = build_cases(material_sweep("hopg", profile="hopg_emittance_demo"))
    assert [case["E0_keV"] for case in cases] == [30.0, 100.0]
    for case in cases:
        assert case["beam_fwhm_mm"] is None  # the policy owns <x^2>, not a spot
        assert case["energy_spread_frac"] == 0.001
    # One beam, two energies: the spot and divergence both shrink with beta*gamma
    # because the STORED emittance is normalized.
    low, high = (case["transverse_distribution"]["x"] for case in cases)
    assert high["sigma_position_mm"] < low["sigma_position_mm"]
    assert high["sigma_slope_rad"] < low["sigma_slope_rad"]
    assert low["sigma_position_mm"] == pytest.approx(0.12, abs=0.01)
    assert low["sigma_slope_rad"] * 1e3 == pytest.approx(2.4, abs=0.1)


def _transport(**beam):
    """A small collimated-geometry transport run, varying only the beam block."""
    return simulate_trajectories(
        30.0,
        64,
        thickness_ang=2.0e4,
        element="C",
        n_atoms_per_ang3=0.1128,
        seed=11,
        **beam,
    )


def test_unset_policy_transport_is_bit_for_bit_with_the_collimated_run():
    """No policy, no change: the default path must not move at all.

    The draws live on their own RNG child streams, so an unset beam block has
    to reproduce the pre-BeamSpec run EXACTLY, not to within Monte Carlo error.
    """
    baseline = _transport()
    for key, expected in _transport(transverse_distribution=None).items():
        actual = baseline[key]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), key
        else:
            assert actual == expected, key


def test_zero_emittance_transport_converges_to_the_collimated_run():
    """The limiting case: eps_n -> 0 recovers the collimated beam.

    Positions scale as sqrt(eps_n) and slopes as sqrt(eps_n), so neither can
    reach exactly zero from a strictly positive emittance -- what is asserted
    is that they shrink without bound while every array the transport actually
    integrates comes back bit-for-bit, which is the physically meaningful half.
    """
    baseline = _transport()
    tiny = TransverseDistribution(
        normalized_emittance_x_mm_mrad=1.0e-300,
        beta_twiss_x_m=0.5,
        alpha_twiss_x=-0.8,
    )
    resolved = asdict(resolve_transverse_distribution(tiny, energy_keV=30.0))
    collimated = _transport(transverse_distribution=resolved)
    # The three geometry arrays inherit the vanishing offset; everything else,
    # including every energy, length, clock and exit tally, is exact.
    vanishing = ("initial_r_ang", "initial_v_hat", "r_mid", "v_hat")
    for key in vanishing:
        assert np.abs(collimated[key] - baseline[key]).max() < 1e-100, key
    for key, expected in baseline.items():
        if key in vanishing:
            continue
        actual = collimated[key]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), key
        else:
            assert actual == expected, key


def test_transport_directions_carry_the_sampled_slopes():
    resolved = asdict(resolve_transverse_distribution(_POLICY, energy_keV=30.0))
    out = _transport(transverse_distribution=resolved)
    v_hat = out["initial_v_hat"]
    assert np.allclose(np.linalg.norm(v_hat, axis=1), 1.0)
    # A finite emittance must actually spread the directions; the collimated
    # run has every row identical.
    assert v_hat[:, 0].std() > 0.0
    # Slopes are dx/dz about +z, so the RMS of v_x/v_z is the sampled sigma_x'.
    slope_rms = float(np.sqrt(np.mean((v_hat[:, 0] / v_hat[:, 2]) ** 2)))
    expected = resolve_transverse_distribution(_POLICY, energy_keV=30.0).x.sigma_slope_rad
    assert slope_rms == pytest.approx(expected, rel=0.2)


def test_transverse_policy_and_spot_fwhm_cannot_both_reach_transport():
    resolved = asdict(resolve_transverse_distribution(_POLICY, energy_keV=30.0))
    with pytest.raises(ValueError, match="incompatible"):
        _transport(transverse_distribution=resolved, beam_fwhm_mm=1.0)


def test_energy_spread_broadens_initial_energies_about_the_nominal():
    baseline = _transport()
    assert np.all(baseline["initial_E_keV"] == 30.0)
    spread = _transport(energy_spread_frac=0.02)
    energies = spread["initial_E_keV"]
    assert energies.mean() == pytest.approx(30.0, rel=0.02)
    assert energies.std() / 30.0 == pytest.approx(0.02, rel=0.35)


def test_energy_spread_zero_is_bit_for_bit_monoenergetic():
    baseline = _transport()
    inert = _transport(energy_spread_frac=0.0)
    for key, expected in baseline.items():
        actual = inert[key]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), key
        else:
            assert actual == expected, key


def test_energy_spread_refuses_to_transport_a_non_positive_energy():
    with pytest.raises(ValueError, match="non-positive electron energy"):
        _transport(energy_spread_frac=5.0)


def test_beam_frame_basis_is_identity_on_axis_and_maps_z_onto_the_beam():
    assert np.array_equal(beam_frame_basis((0.0, 0.0, 1.0)), np.eye(3))
    for direction in ((0.3, 0.0, 1.0), (-0.2, 0.5, 2.0)):
        d = np.asarray(direction) / np.linalg.norm(direction)
        basis = beam_frame_basis(d)
        assert np.allclose(basis @ np.array([0.0, 0.0, 1.0]), d)
        # Orthonormal, so the transverse columns stay perpendicular to the beam.
        assert np.allclose(basis.T @ basis, np.eye(3))
