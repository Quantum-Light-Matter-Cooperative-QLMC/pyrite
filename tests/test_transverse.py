"""Transverse phase-space policy: resolution, sampling, and inertness.

Validation: beam-phase-space-injection
"""

from __future__ import annotations

import numpy as np
import pytest

from cxr_mc.beam_metrics import sampled_beam_metrics
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
