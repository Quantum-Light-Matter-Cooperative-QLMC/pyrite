"""Regression contracts for resolved material scan axes."""

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from pyrite._cxr_vocabulary import catalog_key
from pyrite.materials._catalog_decode import _Errors, _readonly
from pyrite.materials._parse import _scan
from pyrite.materials._schema import DEFAULT_BREM_GRID, CrystalInfo, CrystalSpec


def _values() -> dict[str, object]:
    return {
        "thickness_ang": 500,
        "energy_keV": {"values": [60, 30, 60, 90]},
        "tilt_deg": 0,
        "tilt_azim_deg": 360,
    }


def test_scan_partial_line_store_preserves_declared_order() -> None:
    errors = _Errors()
    store = {30.0: _readonly([100]), 60.0: _readonly([200]), 120.0: _readonly([300])}
    scan = _scan(_values(), "scan", None, errors, cxr_grid_store=store)

    assert errors.items == []
    assert scan is not None
    np.testing.assert_array_equal(scan.energy_keV, [60, 30, 60, 90])
    assert scan.E_grid_line is None
    assert isinstance(scan.E_grid_line_by_energy, MappingProxyType)
    assert tuple(scan.E_grid_line_by_energy) == (60.0, 30.0)
    assert scan.E_grid_line_by_energy[60.0] is store[60.0]
    assert not scan.E_grid_brem_declared
    np.testing.assert_array_equal(scan.E_grid_brem, np.arange(*DEFAULT_BREM_GRID))
    assert not scan.E_grid_brem.flags.writeable


def test_scan_fixed_line_grid_takes_precedence_over_store() -> None:
    errors = _Errors()
    values = {**_values(), "E_grid_line": 150, "E_grid_brem": 0, "n_electrons": 12}
    scan = _scan(values, "scan", None, errors, cxr_grid_store={60.0: _readonly([200])})

    assert errors.items == []
    assert scan is not None
    assert scan.E_grid_line_by_energy is None
    np.testing.assert_array_equal(scan.E_grid_line, [150])
    assert scan.E_grid_brem_declared
    np.testing.assert_array_equal(scan.E_grid_brem, [0])
    np.testing.assert_array_equal(scan.n_electrons, [12])


def test_scan_layer_count_conversion(tmp_path: Path) -> None:
    crystal = CrystalSpec(
        key="example",
        cif=tmp_path / "example.cif",
        validation_id="example",
        formula="C",
        full_name=None,
        phase=None,
        cod_id=None,
        mp_id=None,
        B_ang2=0.0,
        beam_uvw=None,
        hkl_families=(),
        hkl_reason=None,
        layers_per_cell=2,
        info=CrystalInfo({"c": 6.0}, (), 1.0, (), None),
        surface_hkl=(0, 0, 1),
    )
    values = _values()
    del values["thickness_ang"]
    values["thickness_layers"] = {"values": [1, 3]}
    errors = _Errors()
    scan = _scan(values, "scan", crystal, errors)

    assert errors.items == []
    assert scan is not None
    np.testing.assert_array_equal(scan.thickness_ang, [3, 9])
    np.testing.assert_array_equal(scan.thickness_layers, [1, 3])
    assert not scan.thickness_ang.flags.writeable
    assert scan.E_grid_line_by_energy == {}


def test_scan_collects_errors_in_validation_order() -> None:
    errors = _Errors()
    values = {
        "thickness_ang": -1,
        "thickness_layers": 1,
        "energy_keV": -2,
        "E_grid_line": -3,
        "E_grid_brem": -4,
        "tilt_deg": 90,
        "tilt_azim_deg": 361,
        "n_electrons": 0,
        "n_electrons_brem": 1.5,
    }
    assert _scan(values, "scan", None, errors) is None

    assert errors.items == [
        "scan: requires exactly one of thickness_ang or thickness_layers",
        "scan.thickness_ang: values must be positive",
        "scan.energy_keV: values must be positive",
        "scan.E_grid_line: values must be positive",
        "scan.E_grid_brem: values must be nonnegative",
        "scan.tilt_deg: values must satisfy 0 <= tilt_deg < 90",
        "scan.tilt_azim_deg: values must satisfy 0 <= tilt_azim_deg <= 360",
        f"scan.{catalog_key('n_electrons')}: values must be positive integers",
        f"scan.{catalog_key('n_electrons_brem')}: values must be positive integers",
    ]


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        (1.5, "values must be positive integers"),
        (2, "requires the referenced crystal to define layers_per_cell"),
    ],
)
def test_scan_invalid_layer_counts(layers, message) -> None:
    values = _values()
    del values["thickness_ang"]
    values["thickness_layers"] = layers
    errors = _Errors()

    assert _scan(values, "scan", None, errors) is None
    assert errors.items == [f"scan.thickness_layers: {message}"]
