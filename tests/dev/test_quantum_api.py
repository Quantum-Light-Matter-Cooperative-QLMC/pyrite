"""Quantum facade compatibility, import isolation, and executable guide contracts."""

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


def test_legacy_state_and_projection_aliases_keep_identity():
    from pyrite import planar_source, states

    for name in states.__all__:
        assert getattr(quantum, name) is getattr(states, name)
    for name in (
        "PlanarStateProjection",
        "project_planar_source",
        "propagate_planar_source",
        "score_planar_source",
    ):
        assert getattr(quantum, name) is getattr(planar_source, name)
    for name in (
        "QuantumRegion",
        "QuantumDynamics",
        "QuantumNumerics",
        "QuantumSetup",
        "TransverseStateSpec",
    ):
        assert getattr(quantum, name) is getattr(pyrite, name)


def test_spec_and_configuration_keep_wire_identity():
    from pyrite.quantum_state import SCHEMA, TransverseStateSpec

    spec = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
    decoded = quantum.TransverseStateSpec.from_dict(json.loads(spec.canonical_json()))
    assert decoded.to_dict() == spec.to_dict()
    assert decoded.digest == spec.digest
    assert SCHEMA == "pyrite.quantum.transverse-state.v1"
    values = [
        spec,
        quantum.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
        quantum.QuantumDynamics(kind="channeling"),
        quantum.QuantumNumerics(),
        quantum.QuantumSetup(),
    ]
    for value in values:
        cls = type(value)
        expected_module = "pyrite.quantum_state" if value is spec else "pyrite.quantum"
        assert cls.__module__ == expected_module
        assert getattr(importlib.import_module(expected_module), cls.__name__) is cls
        assert pickle.loads(pickle.dumps(value)) == value


def test_light_imports_and_discovery_never_attempt_solver_imports():
    script = """
import importlib.abc
import sys

class BlockEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("pyrite.montecarlo", "pyrite.planar_source",
                                "cupy", "numba", "matplotlib", "plotly",
                                "distributed", "dask")):
            raise AssertionError("unexpected eager import: " + fullname)

sys.meta_path.insert(0, BlockEngines())
from pyrite import quantum
from pyrite.quantum import (QuantumSetup, QuantumRegion, QuantumDynamics,
                           QuantumNumerics, TransverseStateSpec)
assert "MCWFSolver" in dir(quantum)
assert "MCWFSolver" not in vars(quantum)
assert "pyrite.materials.catalog" not in sys.modules
assert set(quantum.__all__) <= set(dir(quantum))
spec = TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
assert TransverseStateSpec.from_dict(spec.to_dict()) == spec
QuantumSetup()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "first", ["pyrite.quantum", "pyrite.quantum_state", "pyrite.planar_source", "pyrite.states"]
)
def test_import_order_keeps_projection_and_state_bindings(first):
    script = f"""
import importlib
importlib.import_module({first!r})
from pyrite import quantum, quantum_state, planar_source, states
assert quantum.TransverseStateSpec is quantum_state.TransverseStateSpec
assert quantum.gaussian is states.gaussian
assert quantum.PlanarStateProjection is planar_source.PlanarStateProjection
assert quantum.QuantumSetup is planar_source.QuantumSetup
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_facade_rejects_unreviewed_engine_helpers():
    assert "zone_solver" not in quantum.__all__
    with pytest.raises(AttributeError, match="zone_solver"):
        quantum.__getattr__("zone_solver")


def test_quantum_guide_examples_keep_normalization_and_result_meanings():
    """Execute the actual guide so public snippets cannot drift from the API."""
    path = Path(__file__).resolve().parents[2] / "docs/guides/quantum-api.md"
    snippets = re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.DOTALL)
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
