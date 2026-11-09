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


# --- Spectral (energy-node) and angular convergence of a band (#616) --------------


def _band(radius_m, n=48, **kw):
    from pyrite.montecarlo.spectrum.optical_finite_convergence import (
        finite_screen_spectral_convergence,
    )

    screen = PECCircularScreen(outer_radius_m=radius_m, beta_z=-0.95)
    return finite_screen_spectral_convergence(
        (1.0, 2.0), screen, nx=n, ny=n, dx_m=0.1e-6, dy_m=0.1e-6, n_u=16, n_phi=32, **kw
    )


def _gf_hemisphere_photons_per_eV(energy_eV):
    from pyrite.montecarlo.spectrum.optical_finite_screen import ginzburg_frank_photons_per_eV_sr

    t, w = np.polynomial.legendre.leggauss(400)
    theta = np.arccos((t + 1) / 2)
    return np.pi * np.array(
        [
            np.sum(w * ginzburg_frank_photons_per_eV_sr(e, theta, 0.95))
            for e in np.atleast_1d(energy_eV)
        ]
    )


def test_band_converges_in_energy_and_angle_to_the_planar_far_field():
    result = _band(3e-6, n=64, n_energy=3, refinements=1)
    assert [row[0] for row in result.energy_rows] == [3, 5]
    assert [row[0] for row in result.angular_rows] == [(16, 32), (32, 64)]
    assert result.energy_relative_change < 3e-3  # Simpson 3 -> 5 nodes on ~1/E
    assert result.angular_relative_change < 1e-5 and result.angular_pointwise_change < 1e-5
    assert result.converged
    assert not _band(3e-6, n=64, n_energy=3, refinements=1, rtol=1e-6).converged
    # Large disc -> planar Ginzburg--Frank far field: photons/eV ~ 1/E, band ln 2.
    gf = _gf_hemisphere_photons_per_eV(result.energy_eV)
    np.testing.assert_allclose(result.photons_per_eV, gf, rtol=2e-3)
    assert result.band_photons == pytest.approx(gf[0] * np.log(2.0), rel=2e-4)


def test_finite_disc_changes_the_spectrum_and_needs_resolved_energy_nodes():
    # Radius ~ lambda: edge interference rings around the planar yield by several %.
    ring = _band(1e-6, n_energy=9, refinements=1)
    ratio = ring.photons_per_eV / _gf_hemisphere_photons_per_eV(ring.energy_eV)
    assert np.ptp(ratio) > 0.08 and ring.energy_relative_change < 1e-4
    coarse = _band(1e-6, n_energy=3, refinements=1)
    assert coarse.energy_relative_change > 10 * ring.energy_relative_change
    # Radius << gamma beta lambda / 2 pi: below the planar yield, recovering with E.
    small = _band(0.2e-6, n_energy=3, refinements=1)
    ratio = small.photons_per_eV / _gf_hemisphere_photons_per_eV(small.energy_eV)
    assert np.all(ratio < 0.95) and np.all(np.diff(ratio) > 0)
    assert small.band_photons < ring.band_photons


def test_spectral_convergence_refuses_bad_controls():
    for kw in ({"n_energy": 4}, {"n_energy": 1}, {"refinements": 0}, {"rtol": 1.0}):
        with pytest.raises(ValueError):
            _band(1e-6, **kw)


# --- Finite-aperture band convergence (direct Green, no window axis) (#616) -------


def _aperture_band(**kw):
    from pyrite.montecarlo.spectrum.optical_finite_convergence import (
        finite_aperture_spectral_convergence,
    )

    screen = PECCircularScreen(outer_radius_m=1e-6, beta_z=-0.95)
    return finite_aperture_spectral_convergence(
        (1.0, 2.0),
        screen,
        nx=32,
        ny=32,
        dx_m=0.1e-6,
        dy_m=0.1e-6,
        plane_z_m=0.5e-6,
        aperture=_APERTURE,
        detector_pitch_m=0.1e-6,
        **{"n_energy": 3, **kw},
    )


def test_aperture_band_converges_and_is_a_lambda_dependent_hemisphere_fraction():
    result = _aperture_band()
    assert [row[0] for row in result.energy_rows] == [3, 5]
    assert [row[0] for row in result.detector_rows] == pytest.approx([1e-7, 5e-8])
    assert [row[0] for row in result.source_rows] == pytest.approx([1e-7, 5e-8])
    assert result.detector_pointwise_change < 2e-3 and result.source_pointwise_change < 1e-3
    assert result.energy_relative_change < 1e-2 and result.converged
    assert not _aperture_band(rtol=1e-4).converged
    # Each node is the window-free direct Green collection of that energy.
    screen = PECCircularScreen(outer_radius_m=1e-6, beta_z=-0.95)
    node = circular_pec_transition_field(1.5, screen, nx=32, ny=32, dx_m=0.1e-6, dy_m=0.1e-6)
    direct = collect_screen_green(node, plane_z_m=0.5e-6, aperture=_APERTURE, pitch_m=0.1e-6)
    assert result.photons_per_eV[2] == pytest.approx(direct.photons_per_eV, rel=1e-12)
    # A finite aperture collects a wavelength-dependent part of the hemisphere.
    from pyrite.montecarlo.spectrum.optical_finite_convergence import (
        finite_screen_spectral_convergence,
    )

    hemi = finite_screen_spectral_convergence(
        (1.0, 2.0), screen, nx=32, ny=32, dx_m=0.1e-6, dy_m=0.1e-6, n_energy=3, n_u=16, n_phi=32
    )
    fraction = result.photons_per_eV / hemi.photons_per_eV
    assert np.all((fraction > 0.5) & (fraction < 1.0)) and np.ptp(fraction) > 0.01


def test_aperture_band_refuses_bad_controls():
    for kw in ({"n_energy": 4}, {"refinements": 0}, {"rtol": 0.0}):
        with pytest.raises(ValueError):
            _aperture_band(**kw)
    with pytest.raises(TypeError):
        _aperture_band(aperture=None)
    with pytest.raises(ValueError, match="remote"):
        _aperture_band(refinements=8)
