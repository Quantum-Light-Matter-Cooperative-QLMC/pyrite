"""Native CUDA source generation and streaming bunch moments (#616/#617).

Float64/complex128 arithmetic parity; source/acceptance convergence and a
published physics anchor are separate. No precomputed CPU field is uploaded.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._backend import BACKEND, CUPY_BACKENDS
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.montecarlo.spectrum.optical_finite_bunch import (
    finite_screen_bunch_angular,
    finite_screen_bunch_band,
    finite_screen_realization_field,
)
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
    finite_screen_bunch_plane_band,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    propagate_screen_field,
)
from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource

cp = pytest.importorskip("cupy")
try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - runtime dependent
    _NDEV = 0
pytestmark = [
    pytest.mark.skipif(_NDEV < 1, reason="CUDA device required"),
    pytest.mark.skipif(
        BACKEND.name not in CUPY_BACKENDS, reason="run with PYRITE_TEST_BACKEND=cuda or rocm"
    ),
]

_GRID = dict(nx=64, ny=64, dx_m=0.1e-6, dy_m=0.1e-6)
_SCREEN = PECCircularScreen(2e-6, beta_z=-0.95)
_APERTURE = PlaneAcceptance("rectangle", width_m=2e-6, height_m=2e-6)
_E_CHARGE_PC = 1.602176634e-7


def _same(device, host, *, rtol=1e-10):
    assert isinstance(device, cp.ndarray)
    # Peak-scaled floor covers Fourier zeros/cancellation, not model error.
    np.testing.assert_allclose(
        cp.asnumpy(device), host, rtol=rtol, atol=1e-12 * np.max(np.abs(host)), equal_nan=True
    )


@pytest.mark.parametrize(
    "energy,screen",
    [
        (1.0, _SCREEN),
        (0.2, PECCircularScreen(2e-6, 0.5e-6, beta_z=-0.8, beta_x=0.2)),
        (5.0, replace(_SCREEN, electron_x_m=0.3e-6, electron_y_m=-0.2e-6, arrival_time_s=-1.3e-15)),
    ],
)
def test_native_source_and_propagated_flux_match_cpu(energy, screen, monkeypatch):
    from pyrite.montecarlo.spectrum import optical_finite_screen as module

    host = circular_pec_transition_field(energy, screen, **_GRID)

    def refuse_cpu_bessel(*args):
        raise AssertionError("device source must not evaluate CPU Bessel kernels")

    monkeypatch.setattr(module, "k0", refuse_cpu_bessel)
    monkeypatch.setattr(module, "k1", refuse_cpu_bessel)
    device = circular_pec_transition_field(energy, screen, xp=cp, **_GRID)
    _same(device.electric_xy_v_s_m, host.electric_xy_v_s_m)
    h = propagate_screen_field(host, 0.7e-6)
    d = propagate_screen_field(device, 0.7e-6)
    _same(d.electric_xy_v_s_m, h.electric_xy_v_s_m)
    args = dict(plane_z_m=0.7e-6, aperture=_APERTURE)
    assert collect_screen_field(device, **args).photons_per_eV == pytest.approx(
        collect_screen_field(host, **args).photons_per_eV, rel=1e-10, abs=0
    )


def _histories(law):
    if law == "gaussian":
        source = OpticalBunchSource.gaussian(
            20 * _E_CHARGE_PC,
            "synthetic temporal Gaussian",
            rms_t_fs=0.2,
            rms_x_um=0.0,
            rms_y_um=0.0,
            weights=[1.0, 2.0, 4.0],
        )
        return [_SCREEN] * 3, source
    screens = [
        replace(_SCREEN, electron_x_m=k * 0.05e-6, arrival_time_s=k * 0.1e-15) for k in range(3)
    ]
    if law == "realization":
        return screens, OpticalBunchSource.realization(3 * _E_CHARGE_PC, "three physical electrons")
    return screens, OpticalBunchSource.sampled(
        20 * _E_CHARGE_PC, "weighted histories", weights=[1.0, 2.0, 4.0]
    )


@pytest.mark.parametrize("law", ["sampled", "gaussian", "realization"])
def test_device_angular_and_plane_source_moments(law):
    screens, source = _histories(law)
    host = finite_screen_bunch_angular(1.0, screens, source=source, **_GRID)
    device = finite_screen_bunch_angular(1.0, screens, source=source, xp=cp, **_GRID)
    for name in ("self_yield", "pair_yield", "yield_per_electron", "bunch_yield"):
        _same(getattr(device, name), getattr(host, name))
    scalar = device.collect_cone(theta_max_rad=0.8, to_host=False)
    assert isinstance(scalar, cp.ndarray)
    assert float(scalar) == pytest.approx(host.collect_cone(theta_max_rad=0.8), rel=1e-10)
    energies = [0.8, 1.0, 1.2]
    hband = finite_screen_bunch_band(energies, screens, source=source, theta_max_rad=0.8, **_GRID)
    dband = finite_screen_bunch_band(
        energies, screens, source=source, theta_max_rad=0.8, xp=cp, **_GRID
    )
    _same(dband.collected_photons_per_eV_per_pulse, hband.collected_photons_per_eV_per_pulse)
    kwargs = dict(source=source, plane_z_m=0.5e-6, aperture=_APERTURE, **_GRID)
    hplane = finite_screen_bunch_plane_band(energies, screens, **kwargs)
    dplane = finite_screen_bunch_plane_band(energies, screens, xp=cp, **kwargs)
    for name in (
        "collected_photons_per_eV_per_pulse",
        "self_photons_per_eV_per_electron",
        "pair_photons_per_eV_per_electron_pair",
    ):
        _same(getattr(dplane, name), getattr(hplane, name))
    detector = CalibratedPulseEnergyDetector(
        [0.5, 1.5], [0.7, 0.7], 1e12, "synthetic source channel"
    )
    voltage = detector.predict_batch(
        dplane.energy_eV, dplane.collected_photons_per_eV_per_pulse
    ).expected_voltage_v
    assert isinstance(voltage, cp.ndarray)
    assert float(voltage) == pytest.approx(hplane.detect(detector).expected_voltage_v, rel=1e-10)


def test_native_realization_n_squared_and_phase_cancellation():
    source = OpticalBunchSource.realization(2 * _E_CHARGE_PC, "two physical electrons")
    one = circular_pec_transition_field(1.0, _SCREEN, xp=cp, **_GRID)
    total = finite_screen_realization_field(1.0, [_SCREEN, _SCREEN], source=source, xp=cp, **_GRID)
    _same(total.electric_xy_v_s_m, 2 * cp.asnumpy(one.electric_xy_v_s_m), rtol=1e-12)
    a = collect_screen_field(one, plane_z_m=0.5e-6, aperture=_APERTURE).photons_per_eV
    b = collect_screen_field(total, plane_z_m=0.5e-6, aperture=_APERTURE).photons_per_eV
    assert b == pytest.approx(4 * a, rel=1e-12)
    opposite = replace(_SCREEN, arrival_time_s=np.pi * 6.582119569e-16)
    cancelled = finite_screen_realization_field(
        1.0, [_SCREEN, opposite], source=source, xp=cp, **_GRID
    )
    assert float(cp.max(cp.abs(cancelled.electric_xy_v_s_m))) < 1e-14 * float(
        cp.max(cp.abs(one.electric_xy_v_s_m))
    )


@pytest.mark.slow
def test_real_case_transport_to_native_device_source():
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.finite_screen_emission import (
        case_finite_screen_plane_band,
        case_finite_screen_realization_field,
    )

    scene = pr.Scene(
        pr.Beam(
            energy_keV=30.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()
        ),
        pr.geometry.Slab(
            "hopg",
            1e4,
            tilt_deg=0.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
            allow_normal_incidence=True,
        ),
        pr.Detector(),
        physics=pr.Physics(
            transition_radiation="history", radiative_model="uncoupled", bremsstrahlung_model="eedl"
        ),
    )
    numerics = pr.Numerics(
        cxr_trials=4,
        brem_trials=2,
        energy_bins=EnergyBins(cxr=np.array([150.0, 1500.0]), brem=np.array([500.0, 5000.0])),
    )
    case = replace(api.build_case(scene, numerics), seed=11)
    transport = runner._transport_case(case)
    source = OpticalBunchSource.realization(2 * _E_CHARGE_PC, "two transported primaries")
    kwargs = dict(source=source, outer_radius_m=2e-6, **_GRID)
    host = case_finite_screen_realization_field(case, transport, 1.0, **kwargs)
    device = case_finite_screen_realization_field(case, transport, 1.0, xp=cp, **kwargs)
    _same(device.electric_xy_v_s_m, host.electric_xy_v_s_m)
    band = case_finite_screen_plane_band(
        case, transport, [0.8, 1.0], plane_z_m=0.5e-6, aperture=_APERTURE, xp=cp, **kwargs
    )
    assert isinstance(band.collected_photons_per_eV_per_pulse, cp.ndarray)
    assert bool(cp.all(band.collected_photons_per_eV_per_pulse > 0))
