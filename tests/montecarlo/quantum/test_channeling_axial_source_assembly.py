"""Positive incident-normalized source assembly, not a radiation-physics proof."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_source_assembly import (
    AxialRadiationOwnership,
    integrate_incident_axial_source,
)


def _setup():
    return dict(
        depths_ang=[0.0, 5.0, 10.0],
        photon_energy_eV=[200.0, 300.0],
        rate_per_eV_sr_ang=np.ones((3, 2, 4)) * 0.02,
        absorption_inv_ang=[0.0, 0.3],
        path_to_exit_ang=np.array(
            [[10.0, 5.0, 10.0, 5.0], [5.0, 2.5, 5.0, 2.5], [0.0, 0.0, 0.0, 0.0]]
        ),
        ownership=AxialRadiationOwnership(frozenset({"bound-CR"}), frozenset({"CBS", "brems"})),
    )


def test_source_constant_limit_azimuth_dependent_absorption_and_finiteness():
    a = _setup()
    output = integrate_incident_axial_source(**a)
    assert output.shape == (2, 4)
    np.testing.assert_allclose(output[0], np.full(4, 0.2), atol=1e-15)
    # Independent direct trapezoid oracle for each ray, no survivor renormalization.
    for ray in range(4):
        expected = np.trapezoid(
            0.02 * np.exp(-0.3 * a["path_to_exit_ang"][:, ray]), x=a["depths_ang"]
        )
        assert output[1, ray] == pytest.approx(expected)
    assert output[1, 0] != pytest.approx(output[1, 1])


def test_zero_source_zero_result():
    a = _setup()
    a["rate_per_eV_sr_ang"][:] = 0
    assert np.count_nonzero(integrate_incident_axial_source(**a)) == 0


def test_overlapping_processes_refused():
    with pytest.raises(ValueError, match="duplicate"):
        AxialRadiationOwnership(frozenset({"CBS"}), frozenset({"CBS"}))
    with pytest.raises(ValueError, match="explicit owned"):
        AxialRadiationOwnership(frozenset(), frozenset())


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("absorption_inv_ang", [-0.2, 0.0], "nonnegative"),
        ("path_to_exit_ang", np.zeros((3, 3)), "dimensions"),
        ("photon_energy_eV", [0.0, 100.0], "positive energy"),
        ("depths_ang", [0.0, 5.0, 5.0], "increasing"),
        ("rate_per_eV_sr_ang", np.full((3, 2, 4), np.nan), "nonnegative"),
        ("max_working_bytes", 2, "budget"),
    ],
)
def test_bad_input_refused(name, value, message):
    a = _setup()
    a[name] = value
    with pytest.raises(ValueError, match=message):
        integrate_incident_axial_source(**a)
