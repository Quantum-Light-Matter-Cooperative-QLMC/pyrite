"""Independent run arguments retain legacy values, identities and refusals (#695)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import Scene, Sweep, api
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.config import Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.quantum import QuantumDynamics, QuantumNumerics, QuantumRegion, QuantumSetup


@pytest.fixture
def scene():
    return Scene(
        Beam(5_000.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("silicon", 1_000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
        physics=Physics(bremsstrahlung_model="eedl", radiative_model="uncoupled"),
    )


@pytest.fixture
def numerics():
    return Numerics(
        cxr_trials=1,
        brem_trials=1,
        energy_bins=EnergyBins(cxr=(1_000.0, 1_500.0), brem=(1_000.0, 1_500.0)),
    )


@pytest.fixture
def quantum():
    return QuantumSetup(
        region=QuantumRegion(kind="planar", plane_hkl=(1, -1, 0)),
        dynamics=QuantumDynamics(kind="channeling"),
        numerics=QuantumNumerics(basis_size=12, bloch_cells=8, step_ang=100.0),
    )


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def run(case, *, transport_core):
        calls.append((case, transport_core))
        return {
            "E_grid": np.array([1_000.0, 1_500.0]),
            "spec": np.array([2.0, 3.0]),
            "brem": np.array([4.0, 5.0]),
        }

    monkeypatch.setattr(api, "run_case", run)
    return calls


def test_one_scene_runs_under_independent_configurations(scene, numerics, transport):
    original = replace(scene)
    first = api.simulate(scene, numerics=numerics)
    second = api.simulate(
        scene,
        physics=replace(scene.physics, brem_source="none"),
        numerics=replace(numerics, cxr_trials=2, transport_core="per-electron"),
    )

    np.testing.assert_array_equal(first.brem.density, [4.0, 5.0])
    np.testing.assert_array_equal(second.brem.density, [0.0, 0.0])
    assert first.case["Ne"] == 1
    assert second.case["Ne"] == 2
    assert first.provenance.identity_digest != second.provenance.identity_digest
    assert [core for _, core in transport] == ["auto", "per-electron"]
    assert scene == original


def test_quantum_component_scene_and_legacy_calls_are_equivalent(
    scene, numerics, quantum, transport
):
    independent = api.simulate(scene, numerics=numerics, quantum=quantum)
    components = api.simulate(
        scene.beam,
        scene.target,
        scene.detector,
        physics=scene.physics,
        numerics=numerics,
        quantum=quantum,
    )
    embedded = api.simulate(replace(scene, numerics=numerics, quantum=quantum))
    # Compare against the preexisting explicit planar fields, independently
    # of QuantumSetup.lower or the new resolver.
    legacy = api.simulate(
        scene,
        physics=replace(scene.physics, channeling_model="planar", channeling_plane=(1, -1, 0)),
        numerics=replace(
            numerics, channeling_basis_size=12, channeling_bloch_cells=8, channeling_dz_ang=100.0
        ),
    )

    for result in (independent, components, embedded):
        assert result.case.to_dict() == legacy.case.to_dict()
        assert result.provenance.identity_digest == legacy.provenance.identity_digest
        np.testing.assert_array_equal(result.cxr.incoherent.density, legacy.cxr.incoherent.density)
        assert result.provenance.numerics.channeling_dz_ang == 100.0
        assert result.provenance.scene.quantum is quantum
    assert scene.quantum is None
    assert scene.numerics.channeling_dz_ang == 250.0


def test_quantum_omission_override_and_explicit_disable(scene, numerics, quantum):
    embedded = replace(scene, quantum=quantum)
    inherited = api.build_case(embedded, numerics)
    disabled = api.build_case(embedded, numerics, quantum=None)
    changed = api.build_case(
        embedded,
        numerics,
        quantum=replace(quantum, numerics=replace(quantum.numerics, step_ang=125.0)),
    )

    assert inherited["channeling"]["dz_ang"] == 100.0
    assert changed["channeling"]["dz_ang"] == 125.0
    assert disabled.to_dict() == api.build_case(scene, numerics).to_dict()
    assert embedded.quantum is quantum


@pytest.mark.parametrize("mode", ["legacy-conflict", "unsupported", "invalid-type"])
def test_quantum_errors_precede_material_preparation(scene, numerics, quantum, monkeypatch, mode):
    from pyrite.campaign import lowering

    def forbidden(*args, **kwargs):
        pytest.fail("invalid composition reached case/material preparation")

    monkeypatch.setattr(lowering, "build_cases", forbidden)
    physics = scene.physics
    error, message = ValueError, "diagnostic-only"
    if mode == "legacy-conflict":
        physics = replace(physics, channeling_model="planar", channeling_plane=(1, -1, 0))
        message = "conflicts with legacy Physics.channeling_model"
    elif mode == "unsupported":
        quantum = replace(quantum, numerics=replace(quantum.numerics, method="mcwf"))
    else:
        quantum = object()
        error, message = TypeError, "quantum must be a QuantumSetup or None"
    with pytest.raises(error, match=message):
        api.simulate(scene, numerics=numerics, physics=physics, quantum=quantum)
    with pytest.raises(error, match=message):
        api.build_case(scene, numerics, physics=physics, quantum=quantum)


def test_sweep_keeps_quantum_controls_and_seed_order(scene, numerics, quantum):
    sweep = Sweep(
        replace(scene, numerics=numerics, quantum=quantum), {"beam.energy_keV": (5000, 6000)}
    )
    cases = sweep.cases()
    assert [case["seed"] for case in cases] == [1, 2]
    assert [case["E0_keV"] for case in cases] == [5000, 6000]
    assert all(case["channeling"]["dz_ang"] == 100.0 for case in cases)
