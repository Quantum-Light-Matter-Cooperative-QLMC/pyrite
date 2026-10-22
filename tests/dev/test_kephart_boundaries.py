"""Exploratory #522 sinks preserve energy references and resolve real band gaps."""

from dataclasses import replace

import numpy as np
import pytest

from checks.kephart_occupation import boundary_candidates
from pyrite.materials.channeling import planar_profile


@pytest.fixture(scope="module")
def profile():
    return planar_profile("silicon", (1, 1, 0), 293.15)


def test_angle_sink_has_barrier_limit_and_is_invariant_to_potential_zero(profile):
    base, _ = boundary_candidates(profile, 16.92e6, 16, [300.0], [1.0, 4.0], [])
    shifted = profile.potential_eV.copy()
    shifted[0] += 37.0
    moved, _ = boundary_candidates(
        replace(profile, potential_eV=shifted), 16.92e6, 16, [300.0], [1.0, 4.0], []
    )
    _, barrier = profile.extrema()
    assert base[1]["cutoff_eV"] == pytest.approx(barrier)
    for a, b in zip(base, moved, strict=True):
        assert b["margin_eV"] == pytest.approx(a["margin_eV"])
        assert b["cutoff_eV"] == pytest.approx(a["cutoff_eV"] + 37.0)
        assert b["critical_angle_rad"] == pytest.approx(a["critical_angle_rad"])
    # Relativistic limit: quadrupling energy halves the critical angle.
    high, _ = boundary_candidates(profile, 4 * 16.92e6, 16, [], [4.0], [])
    assert high[0]["critical_angle_rad"] / base[2]["critical_angle_rad"] == pytest.approx(
        0.5, rel=0.012
    )


def test_free_band_threshold_converges_in_basis_and_tightening_moves_it_up(profile):
    small, _ = boundary_candidates(profile, 16.92e6, 16, [], [], [0.1, 0.01])
    large, _ = boundary_candidates(profile, 16.92e6, 24, [], [], [0.1, 0.01])
    assert large[1]["margin_eV"] > large[0]["margin_eV"] > 0
    for a, b in zip(small, large, strict=True):
        # The onset is near the barrier; basis error must be small on the eV scale.
        assert a["cutoff_eV"] == pytest.approx(b["cutoff_eV"], abs=0.01)


def test_free_particle_spectrum_has_no_resolved_gaps(profile):
    flat = replace(profile, potential_eV=np.zeros_like(profile.potential_eV))
    boundaries, spectrum = boundary_candidates(flat, 16.92e6, 16, [], [], [0.01])
    assert spectrum is not None
    assert boundaries[0]["margin_eV"] > 0
    # Folded free parabolas touch exactly; tiny residuals are eigensolver roundoff.
    assert max(spectrum["gap_to_width"][:10]) < 1e-12


@pytest.mark.parametrize("angles,gaps", [([0.9], []), ([np.nan], []), ([], [0]), ([], [1])])
def test_invalid_threshold_is_rejected(profile, angles, gaps):
    with pytest.raises(ValueError):
        boundary_candidates(profile, 16.92e6, 16, [], angles, gaps)


def test_unresolved_threshold_requires_larger_basis(profile):
    with pytest.raises(ValueError, match="raise H"):
        boundary_candidates(profile, 16.92e6, 4, [], [], [0.01])
