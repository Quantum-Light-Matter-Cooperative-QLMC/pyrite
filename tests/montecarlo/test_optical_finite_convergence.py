"""Independent pitch/window convergence diagnostic for finite-PEC collection (#616)."""

import pytest

from pyrite.montecarlo.spectrum.optical_finite_collection import PlaneAcceptance
from pyrite.montecarlo.spectrum.optical_finite_convergence import (
    finite_screen_collection_convergence,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import PECCircularScreen

_SCREEN = PECCircularScreen(outer_radius_m=2e-6)
_APERTURE = PlaneAcceptance("rectangle", width_m=2e-6, height_m=2e-6)


def _run(nx, dx, **kw):
    return finite_screen_collection_convergence(
        1.0, _SCREEN, nx=nx, ny=nx, dx_m=dx, dy_m=dx, plane_z_m=0.5e-6, aperture=_APERTURE, **kw
    )


def test_pitch_converges_but_tight_window_is_flagged():
    result = _run(32, 0.2e-6, refinements=2)
    assert len(result.pitch_refined) == len(result.window_refined) == 2
    assert result.pitch_refined[-1][:2] == (128, 128)
    assert result.window_refined[-1][2] == pytest.approx(0.2e-6)
    # Source pitch converges at fixed window; the light-circle k-sampling does not.
    assert result.pitch_relative_change < 2e-3
    assert abs(result.hemisphere_fft_bias) > 0.05
    assert not result.converged


def test_refuses_bad_refinement_controls():
    with pytest.raises(ValueError):
        _run(32, 0.2e-6, refinements=0)
    with pytest.raises(ValueError):
        _run(32, 0.2e-6, rtol=0.0)
    with pytest.raises(ValueError, match="remote"):
        _run(1024, 0.2e-6, refinements=3)
