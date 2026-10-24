"""CUDA vs CPU parity of the optical finite-screen/wavefront/instrument path (#616, #617).

Device paths keep complex128 (fp64) fields: these operators are numerical
references, so parity is limited by FFT/summation order, not by precision.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import BACKEND
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.detectors.optical_interferometer import IdealMartinPuplett, polarized_interferogram
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
    collect_screen_green,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    ScreenSpectralField,
    circular_pec_transition_field,
    far_field_photons_per_eV_sr,
    hemisphere_energy_per_eV,
    observe_screen_green,
    propagate_screen_field,
    spectral_energy_per_eV,
)
from pyrite.montecarlo.spectrum.optical_interferometer_handoff import (
    coherency_from_screen_fields,
)
from pyrite.montecarlo.spectrum.optical_wavefront import (
    OpticalWavefront,
    forward_power_w,
    propagate_vacuum,
)

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = [
    pytest.mark.skipif(_NDEV < 1, reason="CUDA device required"),
    pytest.mark.skipif(BACKEND.name != "cuda", reason="run with PYRITE_TEST_BACKEND=cuda"),
]

_SCREEN = PECCircularScreen(outer_radius_m=2e-6, beta_z=-0.95)
_APERTURE = PlaneAcceptance("rectangle", width_m=2e-6, height_m=2e-6)


def _pair(energy_eV=1.0, n=128, pitch=0.1e-6, screen=_SCREEN):
    host = circular_pec_transition_field(energy_eV, screen, nx=n, ny=n, dx_m=pitch, dy_m=pitch)
    device = replace(host, electric_xy_v_s_m=cp.asarray(host.electric_xy_v_s_m))
    assert isinstance(device.electric_xy_v_s_m, cp.ndarray)
    return host, device


@pytest.mark.parametrize(
    "screen", [_SCREEN, PECCircularScreen(2e-6, 0.5e-6, beta_z=-0.8, beta_x=0.2)]
)
def test_propagated_field_and_fft_collection_match_on_device(screen):
    host, device = _pair(screen=screen)
    h = propagate_screen_field(host, 0.7e-6)
    d = propagate_screen_field(device, 0.7e-6)
    assert isinstance(d.electric_xy_v_s_m, cp.ndarray)
    scale = np.max(np.abs(h.electric_xy_v_s_m))
    np.testing.assert_allclose(
        cp.asnumpy(d.electric_xy_v_s_m), h.electric_xy_v_s_m, rtol=1e-11, atol=1e-12 * scale
    )
    assert spectral_energy_per_eV(device) == pytest.approx(
        spectral_energy_per_eV(host), rel=1e-12, abs=0
    )
    for jones in (None, [[1.0, 0.0], [0.0, 0.0]]):
        kw = dict(plane_z_m=0.5e-6, aperture=_APERTURE, jones_sp=jones)
        assert collect_screen_field(device, **kw).photons_per_eV == pytest.approx(
            collect_screen_field(host, **kw).photons_per_eV, rel=1e-11, abs=0
        )


def test_direct_sums_match_on_device():
    host, device = _pair(n=48)
    points = np.array([[0.2e-6, -0.3e-6, 0.5e-6], [-0.4e-6, 0.1e-6, 1.2e-6]])
    h = observe_screen_green(host, points)
    d = observe_screen_green(device, cp.asarray(points))
    assert isinstance(d, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(d), h, rtol=1e-11, atol=1e-12 * np.max(np.abs(h)))
    for h, d in zip(
        far_field_photons_per_eV_sr(host), far_field_photons_per_eV_sr(device), strict=True
    ):
        assert isinstance(d, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(d), h, rtol=1e-11, atol=1e-12)
    assert hemisphere_energy_per_eV(device) == pytest.approx(
        hemisphere_energy_per_eV(host), rel=1e-12, abs=0
    )
    kw = dict(plane_z_m=0.5e-6, aperture=_APERTURE, pitch_m=0.1e-6)
    assert collect_screen_green(device, **kw).photons_per_eV == pytest.approx(
        collect_screen_green(host, **kw).photons_per_eV, rel=1e-12, abs=0
    )


def test_monochromatic_wavefront_matches_on_device():
    rng = np.random.default_rng(616)
    field = rng.normal(size=(2, 64, 64)) + 1j * rng.normal(size=(2, 64, 64))
    host = OpticalWavefront(1.2, 0.2e-6, 0.2e-6, field)
    device = OpticalWavefront(1.2, 0.2e-6, 0.2e-6, cp.asarray(field))
    h = propagate_vacuum(host, 3e-6)
    d = propagate_vacuum(device, 3e-6, xp=cp)
    np.testing.assert_allclose(
        cp.asnumpy(d.electric_xy_v_m),
        h.electric_xy_v_m,
        rtol=1e-11,
        atol=1e-12 * np.max(np.abs(field)),
    )
    assert forward_power_w(d, xp=cp) == pytest.approx(forward_power_w(h), rel=1e-12, abs=0)


def _beam(energy_eV, phase):
    coords = (np.arange(128) - 63.5) * 0.25e-6
    x, y = np.meshgrid(coords, coords)
    envelope = np.exp(-(x * x + y * y) / (2 * (5e-6) ** 2))
    field = np.stack([envelope, envelope * np.exp(1j * phase)]).astype(complex) * 1e-16
    host = ScreenSpectralField(energy_eV, 0.25e-6, 0.25e-6, field)
    return host, replace(host, electric_xy_v_s_m=cp.asarray(field))


def test_end_to_end_interferogram_matches_from_device_fields():
    energies = np.linspace(0.8, 1.2, 9)
    pairs = [_beam(e, phase=0.3 * k) for k, e in enumerate(energies)]
    kw = dict(plane_z_m=1.2e-6, provenance="cuda parity: synthetic paraxial complex fields")
    coh_h = coherency_from_screen_fields([h for h, _ in pairs], **kw)
    coh_d = coherency_from_screen_fields([d for _, d in pairs], **kw)
    np.testing.assert_allclose(
        coh_d.photons_per_eV_per_pulse,
        coh_h.photons_per_eV_per_pulse,
        rtol=1e-11,
        atol=1e-12 * np.max(np.abs(coh_h.photons_per_eV_per_pulse)),
    )
    detector = CalibratedPulseEnergyDetector([0.5, 1.5], [0.7, 0.7], 1e12, "synthetic")
    args = dict(instrument=IdealMartinPuplett("parity"), detectors=(detector, detector))
    shifts = np.linspace(-2e-6, 2e-6, 41)
    v_h = polarized_interferogram(coh_h, shifts, **args).expected_voltage_v_ports
    v_d = polarized_interferogram(coh_d, shifts, **args).expected_voltage_v_ports
    np.testing.assert_allclose(v_d, v_h, rtol=1e-11, atol=1e-12 * np.max(np.abs(v_h)))
