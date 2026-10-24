"""Real Case transport to the opt-in PEC diagnostic and pulse channel (#616/#617).

These are plumbing/normalization checks, not a material or experimental anchor.
Validation: optical-finite-boundary-field, optical-instrument-observable
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.detectors import EnergyBins
from pyrite.detectors.optical_energy import CalibratedPulseEnergyDetector
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.finite_screen_emission import (
    case_finite_screen_band,
    case_finite_screen_plane_band,
    case_finite_screen_realization_field,
)
from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    spectral_energy_per_eV,
)
from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource
from pyrite.montecarlo.spectrum.transition_radiation_crossings import boundary_crossings

_CHARGE_C = 1.602176634e-19
_ENERGY = np.array([0.8, 1.0, 1.2])
_GRID = dict(nx=64, ny=64, dx_m=0.1e-6, dy_m=0.1e-6)
_RADIUS_M = 2e-6

# Includes real transport and first-use table/kernel setup (>5 s cold).
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def transported():
    # Ne > Ne_brem exercises selection of scored primaries from real segments.
    # No planar TR is scored; the separately specified PEC is a diagnostic.
    scene = pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab(
            "hopg",
            thickness_ang=1e4,
            tilt_deg=0.0,
            footprint=pr.Footprint(5.0, 5.0),
            allow_normal_incidence=True,
        ),
        pr.Detector(),
        physics=pr.Physics(
            transition_radiation="history",
            radiative_model="uncoupled",
            bremsstrahlung_model="eedl",
        ),
    )
    numerics = pr.Numerics(
        cxr_trials=12,
        brem_trials=8,
        energy_bins=EnergyBins(cxr=np.linspace(150.0, 1500.0, 28), brem=np.array([500.0, 5000.0])),
    )
    case = replace(api.build_case(scene, numerics), seed=11)
    transport = runner._transport_case(case)
    crossings = boundary_crossings(transport["segs"], [(0, case.thickness_ang, case.composition)])
    entry = (crossings.layer_before == -1) & (crossings.layer_after == 0)
    np.testing.assert_allclose(crossings.kinetic_energy_keV[entry], 30.0)
    np.testing.assert_allclose(crossings.position_ang[entry, :2], 0.0, atol=1e-8)
    count = int(transport["Ne_brem"])
    source = OpticalBunchSource.realization(
        count * _CHARGE_C * 1e12, "seed 11, point bunch of transported physical electrons"
    )
    return case, transport, source


def _single(energy):
    # Independent launch-state expectation: 30 keV, on axis, t=0. No crossing
    # converter is shared with the result under test.
    beta = np.sqrt(1 - (1 + 30.0 / 510.99895069) ** -2)
    screen = PECCircularScreen(outer_radius_m=_RADIUS_M, beta_z=-beta)
    return circular_pec_transition_field(energy, screen, **_GRID)


def test_real_transport_preserves_entrance_fields_and_population(transported):
    case, transport, source = transported
    count = transport["Ne_brem"]
    assert count == 8
    field = case_finite_screen_realization_field(
        case, transport, 1.0, source=source, outer_radius_m=_RADIUS_M, **_GRID
    )
    # Shared mesh, float64/complex128: allow summation/FFT roundoff only.
    # The absolute floor handles vanishing field components, not model error.
    np.testing.assert_allclose(
        field.electric_xy_v_s_m,
        count * _single(1.0).electric_xy_v_s_m,
        rtol=1e-12,
        atol=1e-30,
    )
    band = case_finite_screen_band(
        case,
        transport,
        _ENERGY,
        source=source,
        outer_radius_m=_RADIUS_M,
        theta_max_rad=np.pi / 2,
        **_GRID,
    )
    expected = np.array([spectral_energy_per_eV(_single(e)) / (e * _CHARGE_C) for e in _ENERGY])
    np.testing.assert_allclose(
        band.collected_photons_per_eV_per_pulse, count**2 * expected, rtol=1e-12
    )


def test_real_transport_plane_collection_to_pulse_voltage(transported):
    case, transport, source = transported
    aperture = PlaneAcceptance("rectangle", width_m=2e-6, height_m=2e-6)
    band = case_finite_screen_plane_band(
        case,
        transport,
        _ENERGY,
        source=source,
        outer_radius_m=_RADIUS_M,
        plane_z_m=0.5e-6,
        aperture=aperture,
        **_GRID,
    )
    single = np.array(
        [
            collect_screen_field(_single(e), plane_z_m=0.5e-6, aperture=aperture).photons_per_eV
            for e in _ENERGY
        ]
    )
    expected = transport["Ne_brem"] ** 2 * single
    np.testing.assert_allclose(band.collected_photons_per_eV_per_pulse, expected, rtol=1e-12)
    detector = CalibratedPulseEnergyDetector(
        _ENERGY, np.full(3, 0.25), 1e9, "synthetic constant absorptance; plumbing only"
    )
    reading = band.detect(detector, repetition_hz=1e3)
    joules = np.trapezoid(expected * _ENERGY, _ENERGY) * _CHARGE_C
    assert joules > 0
    assert reading.incident_joule_per_pulse == pytest.approx(joules, rel=1e-12)
    assert reading.expected_voltage_v == pytest.approx(joules * 0.25e9, rel=1e-12)
    assert reading.average_incident_power_w == pytest.approx(joules * 1e3, rel=1e-12)
    assert reading.response_sha256 == detector.response_sha256


def test_real_transport_formation_guard_refuses_short_straight_runs(transported):
    case, transport, source = transported
    with pytest.raises(ValueError, match="formation-scale"):
        case_finite_screen_band(
            case,
            transport,
            _ENERGY,
            source=source,
            outer_radius_m=_RADIUS_M,
            theta_max_rad=0.8,
            min_formation_lengths=1e6,
            **_GRID,
        )
