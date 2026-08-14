from __future__ import annotations

import numpy as np

import pyrite as pr
from pyrite import api
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.campaign.sweep import build_cases
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import run_case


def _inputs():
    beam = pr.Beam(energy_keV=30.0)
    target = pr.Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0)
    detector = pr.Detector(
        energy_bins=EnergyBins(
            line=np.array([1_000.0, 1_050.0]),
            brem=np.array([1_000.0, 1_050.0]),
        )
    )
    numerics = pr.Numerics(n_electrons=1, n_electrons_brem=1)
    return beam, target, detector, numerics


def test_build_case_is_the_existing_case_builder_boundary() -> None:
    beam, target, detector, numerics = _inputs()
    scene = pr.Scene(beam, target, detector)

    canonical = api.build_case(scene, numerics)
    legacy = build_cases(
        LegacySweep(material="hopg", beam=beam, target=target, detector=detector),
        n_electrons=1,
        n_electrons_brem=1,
    )[0]

    assert canonical == legacy


def test_simulate_returns_intrinsic_result_and_provenance_without_store(monkeypatch) -> None:
    beam, target, detector, numerics = _inputs()
    seen = {}

    def fake_run_case(case, *, transport_core):
        seen["case"] = case
        seen["transport_core"] = transport_core
        return {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "E_grid_brem": np.array([1.0, 2.0, 3.0]),
            "brem_wide": np.array([5.0, 6.0, 7.0]),
            "brem": np.array([5.0, 6.0]),
        }

    monkeypatch.setattr(api, "run_case", fake_run_case)

    result = pr.simulate(beam, target, detector, numerics=numerics)

    np.testing.assert_array_equal(result.spectrum, [3.0, 4.0])
    np.testing.assert_array_equal(result.background, [5.0, 6.0, 7.0])
    assert result.case is seen["case"]
    assert seen["transport_core"] == "auto"
    assert result.provenance["scene"].target == target
    assert len(result.provenance["identity_digest"]) == 64


def test_simulate_is_bit_for_bit_the_existing_single_case_path() -> None:
    beam, target, detector, numerics = _inputs()
    expected = run_case(api.build_case(pr.Scene(beam, target, detector), numerics))

    result = pr.simulate(beam, target, detector, numerics=numerics)

    np.testing.assert_array_equal(result.energy_eV, expected["E_grid"])
    np.testing.assert_array_equal(result.spectrum, expected["spec"])
    np.testing.assert_array_equal(result.background, expected["brem_wide"])


def test_scene_model_switches_select_returned_arrays(monkeypatch) -> None:
    beam, target, detector, numerics = _inputs()

    monkeypatch.setattr(
        api,
        "run_case",
        lambda case, *, transport_core: {
            "E_grid": np.array([1.0, 2.0]),
            "spec": np.array([3.0, 4.0]),
            "spec_coherent": np.array([7.0, 8.0]),
            "E_grid_brem": np.array([1.0, 2.0]),
            "brem_wide": np.array([5.0, 6.0]),
            "brem": np.array([5.0, 6.0]),
        },
    )

    result = pr.simulate(
        beam,
        target,
        detector,
        numerics=numerics,
        emission="coherent",
        brem_source="none",
    )

    np.testing.assert_array_equal(result.spectrum, [7.0, 8.0])
    np.testing.assert_array_equal(result.coherent_spectrum, [7.0, 8.0])
    np.testing.assert_array_equal(result.background, [0.0, 0.0])
