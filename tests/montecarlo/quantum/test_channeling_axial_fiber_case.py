"""Admitted axial cases build exactly the calibrated #992/#993 fibre model (#994).

The production builder must reproduce ``checks/axial_fiber_source.py::build``
for the numerics held with each admitted case: identical channels, entrance,
carrier and boundary, and therefore identical per-primary source estimates.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_fiber_case import axial_fiber_inputs
from pyrite.montecarlo.channeling.axial_fiber_regression import fiber_on_axis_source
from pyrite.quantum._axial_envelope import ADMITTED_AXIAL_CASES

CHECKS = Path(__file__).resolve().parents[3] / "checks"


def _research_build(case):
    driver = runpy.run_path(str(CHECKS / "axial_fiber_source.py"))
    n = case.numerics
    options = {
        "axis": list(case.axis_uvw),
        "energy_eV": case.kinetic_energy_eV,
        "thickness_ang": case.thickness_ang,
        "step_ang": n.step_ang,
        "max_order": n.max_order,
        "outgoing_order_max": n.outgoing_order_max,
        "electronic": n.electronic,
        "boundary": n.boundary,
        "tilt_deg": case.tilt_deg,
        "tilt_azimuth_deg": case.tilt_azimuth_deg,
        "harmonic_max": n.harmonic_max,
        "grid_points": n.grid_points,
        "cut_multiple": n.cut_multiple,
    }
    return driver["build"](options)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(ADMITTED_AXIAL_CASES))
def test_builder_reproduces_the_calibrated_research_build(key):
    case = ADMITTED_AXIAL_CASES[key]
    ours = axial_fiber_inputs(case)
    channels, entry, hbar_v, info, settings = _research_build(case)
    assert ours.hbar_v_eV_ang == hbar_v
    assert ours.target_eV == settings["target_eV"]
    assert ours.boundary_eV == settings["boundary_eV"]
    assert list(ours.carrier_inv_ang) == settings["carrier_inv_ang"]
    assert ours.k_inv_ang == info["k_inv_ang"]
    np.testing.assert_array_equal(ours.entry_orders, entry)
    a, b = ours.channels, channels
    assert (a.order_max, a.outgoing_order_max) == (b.order_max, b.outgoing_order_max)
    assert a.reciprocal_inv_ang == b.reciprocal_inv_ang
    np.testing.assert_array_equal(a.kick_columns, b.kick_columns)
    np.testing.assert_array_equal(a.kick_q_inv_ang, b.kick_q_inv_ang)
    assert len(a.tds_strings) == len(b.tds_strings)
    for s, t in zip(a.tds_strings, b.tds_strings, strict=True):
        assert (s.position_ang, s.displacement_ang2, s.linear_density_inv_ang) == (
            t.position_ang,
            t.displacement_ang2,
            t.linear_density_inv_ang,
        )
        q = np.linspace(0.1, 8.0, 7)
        np.testing.assert_array_equal(s.phase(q), t.phase(q))
    energies, _ = a.spectrum(ours.carrier_inv_ang)
    assert a.cut(energies) == b.cut(energies)
    depths = np.linspace(0.0, case.thickness_ang, int(round(case.thickness_ang / 15.0)) + 1)
    np.testing.assert_array_equal(ours.depths_ang, depths)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(ADMITTED_AXIAL_CASES))
def test_builder_reproduces_research_source_estimates(key):
    case = ADMITTED_AXIAL_CASES[key]
    ours = axial_fiber_inputs(case)
    channels, entry, hbar_v, _info, settings = _research_build(case)
    energies = np.linspace(800.0, 10000.0, 47)
    common = dict(
        kinetic_energy_eV=case.kinetic_energy_eV,
        depths_ang=ours.depths_ang,
        photon_energy_eV=energies,
        incident_ids=range(3),
        search_step_ang=200.0,
    )
    a = fiber_on_axis_source(
        ours.channels,
        ours.entry_orders,
        hbar_v_eV_ang=ours.hbar_v_eV_ang,
        carrier_inv_ang=ours.carrier_inv_ang,
        boundary_eV=ours.boundary_eV,
        **common,
    )
    b = fiber_on_axis_source(
        channels,
        entry,
        hbar_v_eV_ang=hbar_v,
        carrier_inv_ang=tuple(settings["carrier_inv_ang"]),
        boundary_eV=settings["boundary_eV"],
        **common,
    )
    np.testing.assert_array_equal(a.spectrum, b.spectrum)
