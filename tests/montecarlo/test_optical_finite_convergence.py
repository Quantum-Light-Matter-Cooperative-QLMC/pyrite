"""Independent pitch/window convergence diagnostic for finite-PEC collection (#616)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    _green_fields,
    collect_screen_field,
    collect_screen_green,
)
from pyrite.montecarlo.spectrum.optical_finite_convergence import (
    finite_screen_collection_convergence,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    _HBAR_C_EV_M,
    _Z0_OHM,
    PECCircularScreen,
    circular_pec_transition_field,
    observe_screen_green,
    propagate_screen_field,
)

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


def _compact_source():
    return circular_pec_transition_field(1.0, _SCREEN, nx=48, ny=48, dx_m=0.1e-6, dy_m=0.1e-6)


def test_direct_green_h_is_the_curl_of_e_by_faraday():
    field = _compact_source()
    k0 = 1.0 / _HBAR_C_EV_M
    step = 1e-10
    for point in ([0.25e-6, -0.15e-6, 1e-6], [1.6e-6, 0.8e-6, 0.4e-6], [-3e-6, 2e-6, 2e-6]):
        point = np.array(point)
        grad = [
            (
                observe_screen_green(field, (point + step * axis)[None])[0]
                - observe_screen_green(field, (point - step * axis)[None])[0]
            )
            / (2 * step)
            for axis in np.eye(3)
        ]
        curl = np.array([grad[1][2] - grad[2][1], grad[2][0] - grad[0][2], grad[0][1] - grad[1][0]])
        electric, magnetic = _green_fields(field, point[None], 10**6)
        np.testing.assert_allclose(electric[0], observe_screen_green(field, point[None])[0])
        expected = curl / (1j * k0 * _Z0_OHM)
        assert np.max(np.abs(magnetic[0] - expected)) < 1e-6 * np.max(np.abs(expected))


def test_direct_green_collection_matches_large_window_fft_without_window_error():
    field = _compact_source()
    direct = collect_screen_green(field, plane_z_m=0.5e-6, aperture=_APERTURE, pitch_m=0.1e-6)
    finer = collect_screen_green(field, plane_z_m=0.5e-6, aperture=_APERTURE, pitch_m=0.05e-6)
    assert direct.photons_per_eV == pytest.approx(finer.photons_per_eV, rel=2e-3)
    padded = circular_pec_transition_field(1.0, _SCREEN, nx=1024, ny=1024, dx_m=0.1e-6, dy_m=0.1e-6)
    fft = collect_screen_field(padded, plane_z_m=0.5e-6, aperture=_APERTURE)
    assert direct.photons_per_eV == pytest.approx(fft.photons_per_eV, rel=1e-3)
    tight = _run(32, 0.2e-6, green_pitch_m=0.1e-6)
    assert tight.green_relative_difference > 0.1
    assert direct.configuration["model"] == "finite-pec-detector-plane-green-v1"
    with pytest.raises(ValueError, match="compact"):
        collect_screen_green(
            propagate_screen_field(field, 0.2e-6),
            plane_z_m=0.5e-6,
            aperture=_APERTURE,
            pitch_m=0.1e-6,
        )
