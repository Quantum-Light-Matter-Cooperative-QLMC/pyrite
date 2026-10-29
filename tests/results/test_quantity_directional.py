"""Directional quantity views preserve native measures, support and ownership."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from pyrite.results import QuantifiedDirectionalSpectrum
from pyrite.results.directional import DirectionalSpectrum
from pyrite.units import ureg


def source() -> DirectionalSpectrum:
    return DirectionalSpectrum(
        angle_deg=np.array([-90.0, 0.0, 90.0]),
        energy_eV=np.array([100.0, 300.0, 1000.0]),
        direction_lab=np.eye(3),
        direction_sample=np.eye(3),
        density=np.array([[1.0, 2.0, 4.0], [np.nan, np.nan, np.nan], [4.0, 3.0, 2.0]]),
        supported=np.array([True, False, True]),
        case={"bunch_charge_pc": 2.0, "rep_rate_hz": 3000.0},
        provenance={"seed": 42},
        component="bremsstrahlung",
    )


def test_native_directional_axes_energy_integrals_and_angular_measure() -> None:
    numeric = source()
    rate_before = numeric.rate()
    view = numeric.quantify().to_energy("keV")
    assert isinstance(view, QuantifiedDirectionalSpectrum)
    assert view.axis_names == ("angle", "photon_energy")
    assert view.component == "bremsstrahlung"
    assert view.count_basis == "photons_per_incident_electron"
    assert view.normalization == "source"
    assert view.sampling == "nodes"
    assert view.measure == "per_energy_per_steradian"
    assert view.density.units == ureg("1 / keV / sr").units
    np.testing.assert_array_equal(view.photon_energy.magnitude, [0.1, 0.3, 1.0])
    np.testing.assert_allclose(view.density.magnitude, numeric.density * 1000)
    np.testing.assert_allclose(
        np.trapezoid(view.density.magnitude, view.photon_energy.magnitude, axis=1),
        [2400.0, np.nan, 2450.0],
    )
    np.testing.assert_allclose(view.angle.to("radian").magnitude, [-np.pi / 2, 0, np.pi / 2])
    np.testing.assert_array_equal(view.supported, numeric.supported)
    np.testing.assert_array_equal(
        view.density.magnitude[view.supported], [[1000, 2000, 4000], [4000, 3000, 2000]]
    )
    np.testing.assert_array_equal(view.to_energy("eV").density.magnitude, numeric.density)
    np.testing.assert_array_equal(numeric.rate(), rate_before)


def test_arrays_are_snapshots_and_every_readback_is_isolated() -> None:
    numeric = source()
    view = numeric.quantify()
    converted = view.to_energy("keV")
    numeric.density[0, 0] = 99
    numeric.angle_deg[0] = 99
    numeric.supported[0] = False
    numeric.energy_eV[0] = 99
    view.density.magnitude[:] = 0
    view.angle.magnitude[:] = 0
    view.photon_energy.magnitude[:] = 0
    view.supported[:] = False
    assert view.density.magnitude[0, 0] == 1
    assert converted.density.magnitude[0, 0] == 1000
    assert converted.angle.magnitude[0] == -90
    assert converted.photon_energy.magnitude[0] == 0.1
    assert converted.supported[0]
    assert np.isnan(converted.density.magnitude[1]).all()


@pytest.mark.parametrize("unit", ["s", "radian", "sr", "1/keV/sr", "dimensionless"])
def test_conversion_refuses_nonenergy_and_normalization_requests(unit: str) -> None:
    with pytest.raises(ValueError, match="energy_unit.*energy"):
        source().quantify().to_energy(unit)


def test_nonstandard_numeric_units_cannot_be_silently_reinterpreted() -> None:
    with pytest.raises(ValueError, match="source density"):
        replace(source(), units="ph/s/eV/sr").quantify()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"density": np.ones((3, 2))}, "shape"),
        ({"supported": np.ones(2, dtype=bool)}, "mask"),
        ({"supported": np.ones(3, dtype=int)}, "boolean"),
        ({"angle_deg": np.ones((3, 1))}, "one-dimensional"),
        ({"energy_eV": np.array([1j, 2j, 3j])}, "real"),
        ({"density": [[1, 2, 3]]}, "CPU NumPy"),
    ],
)
def test_malformed_or_non_cpu_arrays_rejected(changes: dict, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        replace(source(), **changes).quantify()


def test_numeric_result_import_and_rate_remain_pint_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'pint':
        raise AssertionError('Pint imported on numeric path')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pyrite
import numpy as np
from pyrite.results import SpectralDensity, QuantifiedDirectionalSpectrum
from pyrite.results.directional import DirectionalSpectrum
SpectralDensity(np.array([1., 2.]), np.ones(2))
result = DirectionalSpectrum(np.zeros(1), np.ones(2), np.zeros((1, 3)),
    np.zeros((1, 3)), np.ones((1, 2)), np.ones(1, dtype=bool), {}, {})
assert result.rate(0).shape == (1, 2)
assert 'pint' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
