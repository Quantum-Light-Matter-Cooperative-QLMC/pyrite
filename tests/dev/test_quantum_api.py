"""Quantum facade compatibility, import isolation, and executable reference contracts."""

from __future__ import annotations

import importlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyrite
from pyrite import quantum


def test_deleted_root_modules_are_gone_and_root_names_keep_identity():
    for old in ("states", "quantum_state", "planar_source", "_channeling_admission"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"pyrite.{old}")
    for name in ("Channeling", "ChannelingNumerics", "TransverseStateSpec"):
        assert getattr(quantum, name) is getattr(pyrite, name)


def test_spec_and_configuration_keep_wire_identity():
    from pyrite.quantum.state import SCHEMA, TransverseStateSpec

    spec = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    decoded = quantum.TransverseStateSpec.from_dict(json.loads(spec.canonical_json()))
    assert decoded.to_dict() == spec.to_dict()
    assert decoded.digest == spec.digest
    assert SCHEMA == "pyrite.quantum.transverse-state.v1"
    values = [
        spec,
        quantum.Channeling("planar", plane=(1, 1, 0)),
        quantum.ChannelingNumerics(step_ang=100.0),
    ]
    for value in values:
        cls = type(value)
        expected_module = "pyrite.quantum.state" if value is spec else "pyrite.quantum._channeling"
        assert cls.__module__ == expected_module
        assert getattr(importlib.import_module(expected_module), cls.__name__) is cls
        assert getattr(quantum, cls.__name__) is cls
        assert pickle.loads(pickle.dumps(value)) == value


def test_light_imports_and_discovery_never_attempt_solver_imports():
    script = """
import importlib.abc
import sys

class BlockEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("pyrite.montecarlo", "pyrite.quantum.planar_source",
                                "cupy", "numba", "matplotlib", "plotly",
                                "distributed", "dask")):
            raise AssertionError("unexpected eager import: " + fullname)

sys.meta_path.insert(0, BlockEngines())
from pyrite import quantum
from pyrite.quantum import Channeling, ChannelingNumerics, TransverseStateSpec
assert "MCWFSolver" in dir(quantum)
assert "MCWFSolver" not in vars(quantum)
assert "pyrite.materials.catalog" not in sys.modules
assert set(quantum.__all__) <= set(dir(quantum))
spec = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
assert TransverseStateSpec.from_dict(spec.to_dict()) == spec
Channeling("planar", plane=(1, 1, 0), numerics=ChannelingNumerics())
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "first", ["pyrite.quantum", "pyrite.quantum.state", "pyrite.quantum.planar_source"]
)
def test_import_order_keeps_projection_and_state_bindings(first):
    script = f"""
import importlib
importlib.import_module({first!r})
from pyrite import quantum
from pyrite.quantum import state, planar_source
from pyrite.montecarlo.channeling import states
assert quantum.TransverseStateSpec is state.TransverseStateSpec
assert quantum.gaussian is states.gaussian
assert quantum.PlanarStateProjection is planar_source.PlanarStateProjection
assert quantum.Channeling is planar_source.Channeling
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_experimental_quantum_exports_are_pinned_to_pyrite_quantum():
    from pyrite.experimental import quantum as experimental

    assert set(experimental.__all__) == set(experimental._EXPORTS)
    for name, target in experimental._EXPORTS.items():
        assert quantum._EXPORTS[name] == target, name
        assert getattr(experimental, name) is getattr(quantum, name)


_CHANNELING_HELPERS = {
    "planar_profile": "pyrite.materials.channeling.potential",
    "axial_profile": "pyrite.materials.channeling.axial",
    "axial_states": "pyrite.montecarlo.channeling.axial_bloch",
    "AxialStates": "pyrite.montecarlo.channeling.axial_bloch",
    "transverse_states": "pyrite.montecarlo.channeling.bloch",
    "entry_populations": "pyrite.montecarlo.channeling.bloch",
    "longitudinal_kinematics": "pyrite.montecarlo.channeling.bloch",
    "line_energy_eV": "pyrite.montecarlo.channeling.bloch",
    "channeling_spectrum": "pyrite.montecarlo.channeling.emission",
    "axial_line_catalog": "pyrite.montecarlo.channeling.axial_emission",
    "classical_exit_angles": "pyrite.montecarlo.channeling.exit_angular",
    "coherent_exit_distribution": "pyrite.montecarlo.channeling.exit_angular",
    "planar_fiber_inputs": "pyrite.montecarlo.channeling.planar_fiber",
    "unravel_fiber_outgoing": "pyrite.montecarlo.channeling.axial_fiber_unraveling",
    "M_E_EV": "pyrite.materials.crystal",
}


@pytest.mark.parametrize(("name", "module"), sorted(_CHANNELING_HELPERS.items()))
def test_channeling_helpers_resolve_from_quantum_with_identity(name, module):
    assert name in quantum.__all__
    assert getattr(quantum, name) is getattr(importlib.import_module(module), name)


def test_m_e_ev_is_the_plain_float_rest_energy():
    assert type(quantum.M_E_EV) is float
    assert quantum.M_E_EV == pytest.approx(510998.95)


def test_axial_states_left_materials_without_alias():
    from pyrite.montecarlo.channeling import axial_bloch

    assert quantum.axial_states.__module__ == axial_bloch.__name__
    with pytest.raises(ImportError, match="axial_states"):
        from pyrite.materials.channeling.axial import axial_states  # noqa: F401
    with pytest.raises(ImportError, match="AxialStates"):
        from pyrite.materials.channeling.axial import AxialStates  # noqa: F401


@pytest.mark.parametrize(
    "name", ["AxialDiagnosticConfig", "AxialDiagnosticResult", "axial_diagnostic"]
)
def test_pre_rename_axial_names_are_gone_without_alias(name):
    from pyrite.experimental import quantum as experimental

    for namespace in (quantum, experimental):
        assert name not in namespace.__all__
        with pytest.raises(AttributeError, match=name):
            getattr(namespace, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyrite.montecarlo.channeling.axial_diagnostic")


def test_facade_rejects_unreviewed_engine_helpers():
    assert "zone_solver" not in quantum.__all__
    with pytest.raises(AttributeError, match="zone_solver"):
        quantum.__getattr__("zone_solver")


def test_quantum_reference_examples_keep_normalization_and_result_meanings():
    """Execute bounded facade examples, excluding the larger transport recipes."""
    path = Path(__file__).resolve().parents[2] / "docs/reference/quantum-api.md"
    examples, _ = path.read_text().split("\n## Planar transport and source contracts\n", 1)
    snippets = re.findall(r"```python\n(.*?)```", examples, flags=re.DOTALL)
    assert len(snippets) == 4
    scope = {}
    for snippet in snippets:
        exec(compile(snippet, str(path), "exec"), scope)
    assert scope["coherent_norm"] == pytest.approx(1.0, abs=1e-12)
    density = scope["density_result"]
    mcwf = scope["mcwf_result"]
    assert isinstance(density, quantum.DensityMatrixResult)
    assert isinstance(mcwf, quantum.MCWFResult)
    np.testing.assert_array_equal(density.z_ang, [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(mcwf.z_ang, density.z_ang)
    dx = scope["profile"].period_ang / len(scope["x_ang"])
    np.testing.assert_allclose(density.spatial_density_per_ang.sum(axis=1) * dx, density.trace)
    for result in (density, mcwf):
        total = (
            result.trace
            + result.entry_flux
            + result.free_flux
            + result.overflow_flux
            + result.reprojection_flux
        )
        np.testing.assert_allclose(total, 1.0, atol=1e-12)
