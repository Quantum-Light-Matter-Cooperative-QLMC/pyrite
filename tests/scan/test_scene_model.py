import pickle
from dataclasses import asdict, fields

import numpy as np
import pytest

from pyrite.api import build_sweep_cases
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.geometry import Footprint, Layer, Slab, Stack
from pyrite.campaign.legacy import (
    adapt_configured_sweep,
    analysis_from_legacy,
    numerics_from_legacy,
)
from pyrite.campaign.model import Analysis, CxrReflections, Numerics, Physics, Scene, Sweep
from pyrite.campaign.sweep import BeamSpec
from pyrite.campaign.sweep import build_cases as build_legacy_cases
from pyrite.detectors import (
    Detector,
    IdealPhotonCounter,
    PixelGrid,
    PlanarDetector,
    PlanarPose,
)
from pyrite.materials import CATALOG
from pyrite.observations import Acquisition, Observation, PixelScorer
from pyrite.results import Settings


def _scene() -> Scene:
    return Scene(
        beam=BeamSpec(energy_keV=30.0),
        target=Stack(
            layers=(Layer("mos2", 2_000.0), Layer("sio2", 2_850.0)),
            tilt_deg=30.0,
            footprint=Footprint(5.0, 5.0),
        ),
        detector=Detector(),
    )


def _case_bytes(cases) -> list[bytes]:
    return [pickle.dumps(case.to_dict(), protocol=5) for case in cases]


def test_scene_rejects_implicit_sweep_values() -> None:
    with pytest.raises(ValueError, match="put multiple values in Sweep.axes"):
        Scene(
            beam=BeamSpec(energy_keV=[30.0, 60.0]),
            target=Slab("hopg", footprint=Footprint(5.0, 5.0)),
        )


def test_scene_keeps_legacy_run_settings_outside_physical_value_and_round_trips_them():
    scene = Scene(
        beam=BeamSpec(energy_keV=30.0),
        target=Slab("hopg", footprint=Footprint(5.0, 5.0)),
        detector=Detector(),
        physics=Physics(brem_source="none"),
        numerics=Numerics(cxr_trials=7),
    )

    assert {item.name for item in fields(scene)} == {"beam", "target", "detector", "filters"}
    assert set(scene.__dict__) == {"beam", "target", "detector", "filters"}
    assert set(asdict(scene)) == {"beam", "target", "detector", "filters"}
    assert scene.physics.brem_source == "none"
    assert scene.numerics.cxr_trials == 7

    restored = pickle.loads(pickle.dumps(scene))
    assert restored.physics == scene.physics
    assert restored.numerics == scene.numerics
    assert set(restored.__dict__) == {"beam", "target", "detector", "filters"}
    first = Scene(beam=scene.beam, target=scene.target)
    second = Scene(beam=scene.beam, target=scene.target)
    assert first.numerics is not second.numerics


def test_scene_and_planar_detector_dataclass_values_contain_only_physical_inputs():
    detector = PlanarDetector(
        PlanarPose((0.0, 0.0, 100.0)),
        pixels=PixelGrid((4, 4), (0.5, 0.5)),
        response=IdealPhotonCounter(),
        scorer=PixelScorer((2, 2)),
    )
    scene = Scene(
        beam=BeamSpec(energy_keV=30.0),
        target=Slab("hopg", footprint=Footprint(5.0, 5.0)),
        detector=detector,
        acquisition=Acquisition(1.0, (0.0, 1000.0)),
        physics=Physics(brem_source="none"),
        numerics=Numerics(cxr_trials=7),
    )

    assert set(asdict(detector)) == {"pose", "size_mm", "pixels"}
    assert detector.response == IdealPhotonCounter()
    assert detector.scorer == PixelScorer((2, 2))
    assert set(asdict(scene)) == {"beam", "target", "detector", "filters"}
    assert set(asdict(scene)["detector"]) == {"pose", "size_mm", "pixels"}
    assert "acquisition" not in scene.__dict__

    restored = pickle.loads(pickle.dumps(scene))
    assert restored.acquisition == scene.acquisition
    assert restored.detector.response == detector.response
    assert restored.detector.scorer == detector.scorer


def test_path_axes_expand_nested_and_indexed_fields_mechanically() -> None:
    sweep = Sweep(
        base=_scene(),
        axes={
            "beam.energy_keV": np.array([30.0, 45.0]),
            "target.layers[1].thickness_ang": [2_850.0, 5_000.0],
        },
    )

    expanded = sweep.expand()

    assert [scene.beam.energy_keV for _, scene in expanded] == [30.0, 30.0, 45.0, 45.0]
    assert [scene.target.layers[1].thickness_ang for _, scene in expanded] == [
        2_850.0,
        5_000.0,
        2_850.0,
        5_000.0,
    ]
    assert expanded[-1][0] == ("beam.energy_keV=45 target.layers[1].thickness_ang=5000")


@pytest.mark.parametrize(
    "path, message",
    [
        ("target.tlit_deg", "has no field"),
        ("target.layers[9].thickness_ang", "out of range"),
        ("target.layers.one", "has no field"),
    ],
)
def test_axis_paths_fail_at_construction(path: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Sweep(base=_scene(), axes={path: [1.0]})


def test_object_defaults_are_separate_by_lifetime() -> None:
    numerics = Numerics(cxr_reflections=CxrReflections(max_reflections=4))
    analysis = Analysis()

    assert numerics.cxr_trials == 450
    assert numerics.cxr_reflections.max_reflections == 4
    assert analysis.convolve_with_det is False


def test_legacy_pair_distributes_settings_and_swept_scene_fields() -> None:
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=[30.0, 60.0]),
        thickness_ang=[1_000.0, 2_000.0],
        tilt_deg=30.0,
    )
    settings = Settings(n_electrons=12, n_electrons_brem=6, emission="both")

    sweep = adapt_configured_sweep(old, settings)

    assert len(sweep.expand()) == 4
    assert sweep.base.physics.emission_policy == "both"
    assert numerics_from_legacy(old, settings).cxr_trials == 12
    assert analysis_from_legacy(settings).beam_current_na == settings.beam_current_na


def test_configured_adapter_preserves_expanded_cases_exactly() -> None:
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=[30.0, 60.0]),
        thickness_ang=[1_000.0, 2_000.0],
        tilt_deg=[20.0, 30.0],
        n_electrons=[2, 3],
    )
    settings = Settings(n_electrons=12, n_electrons_brem=6, emission="both")

    converted = adapt_configured_sweep(old, settings)

    expected = build_legacy_cases(
        old,
        settings.n_electrons,
        settings.n_electrons_brem,
        coherent_emission=settings.coherent_emission,
    )
    assert _case_bytes(build_sweep_cases(converted)) == _case_bytes(expected)


def test_configured_sweep_accepts_independent_run_and_observation_overrides():
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=30.0),
        thickness_ang=1_000.0,
        tilt_deg=30.0,
    )
    converted = adapt_configured_sweep(old, Settings(n_electrons=12, n_electrons_brem=6))
    numerics = Numerics(cxr_trials=9, brem_trials=4)
    observation = Observation(detector=Detector())

    cases = build_sweep_cases(
        converted,
        numerics,
        physics=Physics(brem_source="none"),
        observation=observation,
    )

    assert len(cases) == 1
    assert cases[0]["Ne"] == 9
    assert cases[0]["Ne_brem"] == 4


def test_every_catalog_profile_round_trips_a_resolved_case_list() -> None:
    settings = default_settings("survey")
    for profile in CATALOG.profile_names:
        members = CATALOG.profile_materials(profile)
        material = members[0] if members else "hopg"
        old = material_sweep(material, fidelity="survey", catalog_profile=profile)
        converted = adapt_configured_sweep(old, settings)
        expected = build_legacy_cases(
            old,
            settings.n_electrons,
            settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
        )
        assert _case_bytes(build_sweep_cases(converted)) == _case_bytes(expected), profile


def test_configured_profile_path_preserves_fractional_angles_and_case_identity():
    from pyrite.campaign.profiles import case_content_key
    from pyrite.campaign.sweep import Sweep as LegacySweep

    old = LegacySweep(
        material="hopg",
        beam=BeamSpec(energy_keV=30.0),
        thickness_ang=1000.0,
        tilt_deg=[30.123456789, 30.2],
        tilt_azim_deg=-75.123456789,
    )
    settings = Settings(n_electrons=1, n_electrons_brem=1)
    cases = build_sweep_cases(adapt_configured_sweep(old, settings))
    assert [(case.tilt_deg, case.tilt_azim_deg) for case in cases] == [
        (30.123456789, -75.123456789),
        (30.2, -75.123456789),
    ]
    from dataclasses import replace

    same_seed = replace(cases[1], seed=cases[0].seed, name=cases[0].name)
    assert case_content_key(cases[0]) != case_content_key(same_seed)
