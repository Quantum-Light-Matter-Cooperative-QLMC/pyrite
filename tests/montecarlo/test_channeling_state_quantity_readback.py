"""Quantity read-back on numeric transverse channeling states (#816)."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling import states as st
from pyrite.units import Q_, ureg


def test_preset_hints_centre_and_shift_are_detached_quantities():
    state = st.gaussian(waist_ang=2 * ureg.nm, dimension=1)
    assert state.centre_position.shape == (1,)
    assert state.centre_position.to("nm").magnitude.tolist() == [0.0]
    assert state.momentum_extent.units == ureg.Unit("1/angstrom")
    assert state.momentum_resolution.units == ureg.Unit("1/angstrom")
    assert np.all(np.isfinite(state.momentum_extent.magnitude))
    original_extent = tuple(state.k_extent)
    state.momentum_extent.magnitude[0] = 0
    assert state.k_extent == original_extent
    shifted = st.translate(state, 3 * ureg.nm)
    np.testing.assert_allclose(shifted.displacement.to("nm").magnitude, [3.0])
    np.testing.assert_allclose(shifted.centre_position.to("nm").magnitude, [3.0])
    shifted.displacement.magnitude[0] = -50
    assert shifted.offset == pytest.approx((30.0,))


def test_sampled_axes_and_energy_metadata_round_trip_without_mutation():
    state = st.from_array(
        np.ones(8),
        spacing_ang=Q_(0.2, "nm"),
        origin_ang=Q_(-5, "angstrom"),
        kinetic_energy_eV=Q_(60, "keV"),
    )
    assert state.sample_positions[0].to("nm").magnitude[0] == pytest.approx(-0.5)
    assert state.kinetic_energy.to("keV").magnitude == pytest.approx(60)
    original = state.axes[0].copy()
    state.sample_positions[0].magnitude[:] = -999
    np.testing.assert_array_equal(state.axes[0], original)
    assert st.from_array(np.ones(4), spacing_ang=1).kinetic_energy is None


def test_aperture_parameters_are_reported_with_correct_units_and_finite_energy():
    probe = st.aperture_probe(
        Q_(60, "keV"),
        semiangle_rad=Q_(20, "mrad"),
        inner_semiangle_rad=Q_(5, "mrad"),
        aberrations={"C10": 2 * ureg.nm, "phi12": 30 * ureg.degree},
    )
    assert probe.kinetic_energy.to("keV").magnitude == pytest.approx(60, rel=1e-12)
    assert probe.semiangle.to("mrad").magnitude == pytest.approx(20)
    assert probe.inner_semiangle.to("mrad").magnitude == pytest.approx(5)
    assert probe.aberration_quantities["C10"].to("nm").magnitude == pytest.approx(2)
    assert probe.aberration_quantities["phi12"].to("degree").magnitude == pytest.approx(30)
    probe.aberration_quantities["C10"].ito("nm")
    assert probe.aberrations["C10"] == pytest.approx(20.0)
    np.testing.assert_allclose(probe.centre_position.magnitude, [0, 0])
