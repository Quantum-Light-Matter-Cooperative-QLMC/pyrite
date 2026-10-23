"""Transport entrance handoff to a distinct optical PEC diagnostic (#616)."""

import sys
from dataclasses import dataclass, replace
from types import ModuleType

import numpy as np
import pytest

from pyrite.montecarlo.runner.finite_screen_emission import (
    case_finite_screen_band,
    case_finite_screen_realization_field,
)
from pyrite.montecarlo.spectrum.optical_finite_bunch import (
    finite_screen_bunch_band,
    finite_screen_realization_field,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    _E_CHARGE_C,
    circular_pec_transition_field,
)
from pyrite.montecarlo.spectrum.optical_finite_transport import (
    entrance_pec_screens_from_crossings,
)

_GRID = dict(nx=64, ny=64, dx_m=1.5e-7, dy_m=1.5e-7)


@dataclass
class _Crossings:
    electron_id: np.ndarray
    interface: np.ndarray
    layer_before: np.ndarray
    layer_after: np.ndarray
    position_ang: np.ndarray
    direction: np.ndarray
    kinetic_energy_keV: np.ndarray
    time_ang: np.ndarray
    charge: np.ndarray
    z_interfaces_ang: np.ndarray
    straight_after_ang: np.ndarray | None = None


def _entries():
    """Two physical primaries, deliberately out of order, plus one exit."""
    return _Crossings(
        electron_id=np.array([1, 0, 1]),
        interface=np.array([0, 0, 0]),
        layer_before=np.array([-1, -1, 0]),
        layer_after=np.array([0, 0, -1]),
        position_ang=np.array([[1000.0, 2000.0, 0.0], [0.0, 0.0, 0.0], [2000.0, 3000.0, 0.0]]),
        direction=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        kinetic_energy_keV=np.array([30.0, 50.0, 10.0]),
        time_ang=np.array([2997.92458, 0.0, 4000.0]),
        charge=np.array([-1, -1, -1]),
        z_interfaces_ang=np.array([0.0, 1e4]),
        straight_after_ang=np.array([1e4, 2e4, np.inf]),
    )


def _source(*, n=2, law="realization", weights=None):
    class Source:
        physical_electrons = n
        history_weights = weights
        inter_electron = "correlated" if law == "realization" else "independent"
        charge_pC = n * _E_CHARGE_C * 1e12

        def __init__(self):
            self.law = law

        def record(self):
            return {"law": law, "physical_electrons": n}

        def form_factor_squared(self, energy_eV, direction):
            return np.ones(np.shape(energy_eV))

    return Source()


def _screens(c=None, **kwargs):
    return entrance_pec_screens_from_crossings(
        _entries() if c is None else c,
        n_histories=2,
        outer_radius_m=3e-6,
        **kwargs,
    )


def test_selects_only_first_ne_brem_even_when_transport_has_more_primaries():
    c = _entries()
    # A third independent primary is present in the transport, but the optical
    # diagnostic uses only Ne_brem=2 histories, as planar TR scoring does.
    c.electron_id = np.r_[c.electron_id, 2]
    c.interface = np.r_[c.interface, 0]
    c.layer_before = np.r_[c.layer_before, -1]
    c.layer_after = np.r_[c.layer_after, 0]
    c.position_ang = np.vstack((c.position_ang, [0.0, 0.0, 0.0]))
    c.direction = np.vstack((c.direction, [0.0, 0.0, 1.0]))
    c.kinetic_energy_keV = np.r_[c.kinetic_energy_keV, 25.0]
    c.time_ang = np.r_[c.time_ang, 1.0]
    c.charge = np.r_[c.charge, -1]
    c.straight_after_ang = np.r_[c.straight_after_ang, 1e4]
    assert _screens(c) == _screens()


def test_selection_sorting_coordinates_beta_sign_and_time_units():
    a, b = _screens(centre_xy_ang=(100.0, 200.0), time_origin_ang=100.0)
    # By history ID (0 then 1), not by input crossing record order.
    assert a.electron_x_m == pytest.approx(-100e-10)
    assert a.electron_y_m == pytest.approx(200e-10)
    assert b.electron_x_m == pytest.approx(900e-10)
    assert b.electron_y_m == pytest.approx(-1800e-10)
    assert a.beta_z == pytest.approx(-np.sqrt(1 - (1 + 50 / 510.99895069) ** -2))
    assert b.beta_z == pytest.approx(-np.sqrt(1 - (1 + 30 / 510.99895069) ** -2))
    assert a.charge_c == b.charge_c == -_E_CHARGE_C
    assert b.arrival_time_s - a.arrival_time_s == pytest.approx(1e-15, rel=1e-8)
    # The extra backscattered vacuum exit did not become a third screen source.
    assert len((a, b)) == 2


def test_realization_resolved_fields_and_source_phase():
    c = _entries()
    c.position_ang[0, :2] = 0.0
    c.time_ang[0] = 0.0
    c.kinetic_energy_keV[0] = c.kinetic_energy_keV[1]
    screens = _screens(c)
    one = circular_pec_transition_field(1.0, screens[0], **_GRID)
    total = finite_screen_realization_field(1.0, screens, source=_source(), **_GRID)
    np.testing.assert_allclose(total.electric_xy_v_s_m, 2 * one.electric_xy_v_s_m, rtol=1e-12)
    # Physical bunch time offsets are not overwritten by the screen converter.
    shifted = _screens(c, time_origin_ang=2997.92458)
    shifted_field = finite_screen_realization_field(1.0, shifted, source=_source(), **_GRID)
    phase = np.exp(-1j * (1.0 / 6.582119569e-16) * 1e-15)
    np.testing.assert_allclose(
        shifted_field.electric_xy_v_s_m, phase * total.electric_xy_v_s_m, rtol=1e-10, atol=1e-20
    )


def test_band_scores_transported_position_and_velocity_changes():
    c = _entries()
    screens = _screens(c)
    e = [0.8, 1.0, 1.2]
    result = finite_screen_bunch_band(
        e,
        screens,
        source=_source(),
        theta_max_rad=1.0,
        **_GRID,
    )
    assert result.collected_photons_per_eV_per_pulse.shape == (3,)
    assert np.all(result.collected_photons_per_eV_per_pulse >= 0)
    baseline = _entries()
    baseline.position_ang[0, :2] = baseline.position_ang[1, :2]
    baseline.kinetic_energy_keV[0] = baseline.kinetic_energy_keV[1]
    baseline.time_ang[0] = baseline.time_ang[1]
    other = finite_screen_bunch_band(
        e,
        _screens(baseline),
        source=_source(),
        theta_max_rad=1.0,
        **_GRID,
    )
    assert not np.allclose(
        result.collected_photons_per_eV_per_pulse,
        other.collected_photons_per_eV_per_pulse,
        rtol=0.001,
        atol=0,
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: setattr(c, "electron_id", np.array([1, 1, 1])), "one primary entrance"),
        (lambda c: setattr(c, "layer_before", np.array([0, -1, 0])), "one primary entrance"),
        (
            lambda c: setattr(
                c,
                "direction",
                np.array([[0.0, 0.1, np.sqrt(0.99)], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
            ),
            "normal-incidence",
        ),
        (lambda c: setattr(c, "charge", np.array([+1, -1, -1])), "electron primaries"),
        (
            lambda c: setattr(c, "kinetic_energy_keV", np.array([0.0, 50.0, 10.0])),
            "energy must be positive",
        ),
        (lambda c: setattr(c, "z_interfaces_ang", np.array([2.0, 1e4])), "z=0 entrance"),
        (
            lambda c: setattr(
                c, "position_ang", np.array([[0.0, 0.0, 1e-2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            ),
            "z=0 entrance",
        ),
    ],
)
def test_invalid_crossings_are_not_silently_reinterpreted(mutate, match):
    c = _entries()
    mutate(c)
    with pytest.raises(ValueError, match=match):
        _screens(c)


def test_straight_run_guard_and_hole_miss():
    with pytest.raises(ValueError, match="shorter"):
        _screens(min_straight_after_ang=1.1e4)
    with pytest.raises(ValueError, match="missing"):
        _screens(replace(_entries(), straight_after_ang=None), min_straight_after_ang=1e3)
    with pytest.raises(ValueError, match="inside the hole"):
        _screens(inner_radius_m=0.5e-7)  # second history 1000,2000 Å off centre
    assert len(_screens(inner_radius_m=3e-7)) == 2


def test_option_refusals():
    with pytest.raises(ValueError, match="positive integer"):
        entrance_pec_screens_from_crossings(_entries(), n_histories=0, outer_radius_m=3e-6)
    with pytest.raises(ValueError, match="centre_xy_ang"):
        _screens(centre_xy_ang=(np.nan, 0.0))
    with pytest.raises(ValueError, match="max_angle_rad"):
        _screens(max_angle_rad=0)
    with pytest.raises(ValueError, match="min_straight_after_ang"):
        _screens(min_straight_after_ang=-1)


def _stub_runner_dependencies(monkeypatch, c):
    # In full PyRITE these names resolve to the actual transport interfaces.
    # Test the adapter separately from heavy material/transport imports here.
    crossings_mod = ModuleType("pyrite.montecarlo.spectrum.transition_radiation_crossings")
    crossings_mod.boundary_crossings = lambda segs, layers: c
    monkeypatch.setitem(sys.modules, crossings_mod.__name__, crossings_mod)
    case_mod = ModuleType("pyrite.montecarlo.runner.transition_emission")
    case_mod.case_layers = lambda case: [(0.0, 1e4, "synthetic")]
    monkeypatch.setitem(sys.modules, case_mod.__name__, case_mod)


def test_runner_adapter_accepts_real_case_interface_and_refuses_charge_mismatch(monkeypatch):
    _stub_runner_dependencies(monkeypatch, _entries())
    source = _source()
    transport = {"segs": {"placeholder": True}, "Ne_brem": 2}
    kw = dict(source=source, outer_radius_m=3e-6, **_GRID)
    band = case_finite_screen_band(
        {},
        transport,
        [0.8, 1.0, 1.2],
        theta_max_rad=1.0,
        **kw,
    )
    assert band.collected_photons_per_eV_per_pulse.shape == (3,)
    field = case_finite_screen_realization_field({}, transport, 1.0, **kw)
    assert field.electric_xy_v_s_m.shape == (2, 64, 64)
    with pytest.raises(ValueError, match="charge disagrees"):
        case_finite_screen_band(
            {"bunch_charge_pc": 999},
            transport,
            [0.8, 1.0],
            theta_max_rad=1.0,
            **kw,
        )
    with pytest.raises(ValueError, match="Ne_brem"):
        case_finite_screen_realization_field(
            {},
            {"segs": {}, "Ne_brem": 0},
            1.0,
            **kw,
        )


def test_native_crossings_and_source_class_when_full_pyrite_available():
    """Runs in actual PyRITE; self-contained source-only environments skip."""
    from importlib.util import find_spec

    if find_spec("pyrite.materials") is None:
        pytest.skip("source-only reconstruction lacks full PyRITE materials")
    from pyrite.montecarlo.spectrum.optical_transition import OpticalBunchSource
    from pyrite.montecarlo.spectrum.transition_radiation_crossings import BoundaryCrossings

    c = _entries()
    real = BoundaryCrossings(
        electron_id=c.electron_id,
        track_id=c.electron_id,
        charge=c.charge,
        position_ang=c.position_ang,
        direction=c.direction,
        kinetic_energy_keV=c.kinetic_energy_keV,
        time_ang=c.time_ang,
        interface=c.interface,
        normal=np.tile([0, 0, 1], (3, 1)),
        layer_before=c.layer_before,
        layer_after=c.layer_after,
        z_interfaces_ang=c.z_interfaces_ang,
        n_matched_dropped=0,
        n_side_exits=0,
        straight_before_ang=np.array([np.inf, np.inf, 1e3]),
        straight_after_ang=c.straight_after_ang,
    )
    screen_sources = _screens(real)
    native = OpticalBunchSource.realization(
        2 * _E_CHARGE_C * 1e12, "two physical transported electrons"
    )
    result = finite_screen_bunch_band(
        [0.8, 1.0, 1.2],
        screen_sources,
        source=native,
        theta_max_rad=1.0,
        **_GRID,
    )
    assert np.all(result.collected_photons_per_eV_per_pulse >= 0)


def test_runner_oblique_and_formation_scale_options(monkeypatch):
    c = _entries()
    tilt = 0.12
    c.direction[:2] = [np.sin(tilt), 0.0, np.cos(tilt)]
    _stub_runner_dependencies(monkeypatch, c)
    source = _source()
    transport = {"segs": {"placeholder": True}, "Ne_brem": 2}
    options = dict(
        source=source,
        outer_radius_m=3e-6,
        nx=64,
        ny=64,
        dx_m=1e-7,
        dy_m=1e-7,
        max_angle_rad=0.2,
        min_formation_lengths=1.0,
    )
    band = case_finite_screen_band({}, transport, [0.8, 1.0], theta_max_rad=0.9, **options)
    assert np.all(np.isfinite(band.collected_photons_per_eV_per_pulse))
    exact = case_finite_screen_realization_field({}, transport, 1.0, **options)
    assert np.all(np.isfinite(exact.electric_xy_v_s_m))
    with pytest.raises(ValueError, match="formation-scale"):
        case_finite_screen_band(
            {},
            transport,
            [0.8, 1.0],
            theta_max_rad=0.9,
            **dict(options, min_formation_lengths=1e3),
        )
