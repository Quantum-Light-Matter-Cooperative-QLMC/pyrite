"""Configuration-only coverage for #680's first production adapter."""

from dataclasses import FrozenInstanceError, replace

import pytest

from pyrite import Physics, Numerics
from pyrite.quantum import QuantumDynamics, QuantumNumerics, QuantumRegion, QuantumSetup


def _planar():
    return QuantumSetup(
        region=QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
        dynamics=QuantumDynamics(kind="channeling"),
    )


def test_planar_adapter_matches_existing_legacy_controls():
    physics, numerics = _planar().lower(Physics(radiative_model="uncoupled"), Numerics())
    assert physics == Physics(channeling_model="planar", channeling_plane=(1, 1, 0),
                              radiative_model="uncoupled")
    assert numerics == Numerics()


def test_custom_planar_grid_lowering():
    setup = replace(_planar(), numerics=QuantumNumerics(basis_size=12, bloch_cells=8,
                                                       step_ang=100.0))
    _, numerics = setup.lower(Physics(radiative_model="uncoupled"), Numerics())
    assert (numerics.channeling_basis_size, numerics.channeling_bloch_cells,
            numerics.channeling_dz_ang) == (12, 8, 100.0)


def test_immutable_and_defensive_tuple_conversion():
    region = QuantumRegion(kind="planar", plane_hkl=[1, 1, 0])
    assert region.plane_hkl == (1, 1, 0)
    with pytest.raises(FrozenInstanceError):
        region.kind = "free"


@pytest.mark.parametrize("kwargs", [
    {"kind": "planar"},
    {"kind": "free", "plane_hkl": (1, 1, 0)},
    {"kind": "axial", "axis_uvw": (0, 0, 0)},
    {"kind": "planar", "plane_hkl": (1, 1, 0), "zone_max_ang": -10.0},
])
def test_invalid_regions(kwargs):
    with pytest.raises(ValueError):
        QuantumRegion(**kwargs)


def test_unsupported_quantum_methods_are_not_silent_noops():
    setup = replace(_planar(), numerics=QuantumNumerics(method="mcwf"))
    with pytest.raises(ValueError, match="diagnostic-only"):
        setup.lower(Physics(), Numerics())
    with pytest.raises(ValueError, match="diagnostic-only"):
        QuantumSetup().lower(Physics(), Numerics())


def test_legacy_conflicts_refused():
    with pytest.raises(ValueError, match="Physics.channeling_model"):
        _planar().lower(Physics(channeling_model="planar", channeling_plane=(1, 1, 0),
                                radiative_model="uncoupled"), Numerics())
    with pytest.raises(ValueError, match="Numerics.channeling_basis_size"):
        _planar().lower(Physics(), Numerics(channeling_basis_size=22))
