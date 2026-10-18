"""Transition radiation through the production surface: API, identity, storage, replay (#103).

Validation: tr-crossing-superposition
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._physics_provenance import resolved_physics_view
from pyrite._spectral_components import cxr_spectrum
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import case_content_key, dataset_identity
from pyrite.checkpoints import _checkpoint_store, recompute
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner
from pyrite.montecarlo.spectrum.transition_radiation_crossings import TRANSITION_MODEL
from pyrite.montecarlo.trajectories import (
    TrajectoryArtifactError,
    read_trajectory_artifact,
    write_trajectory_artifact,
)
from pyrite.results.store import Settings, store_result

_NUMERICS = pr.Numerics(
    cxr_trials=12,
    brem_trials=8,
    energy_bins=EnergyBins(cxr=np.linspace(150.0, 1500.0, 28), brem=np.array([500.0, 5000.0])),
)


def _scene(mode="history"):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=1e4, footprint=pr.Footprint(5.0, 5.0)),
        pr.Detector(),
        physics=pr.Physics(
            transition_radiation=mode, radiative_model="uncoupled", bremsstrahlung_model="eedl"
        ),
    )


@pytest.fixture(scope="module")
def tr_run():
    case = api.build_case(_scene(), _NUMERICS).to_dict()
    transport = runner._transport_case(case)
    return case, transport, runner._spectrum_case(case, transport)


def test_physics_selector_and_lowering():
    assert pr.Physics().transition_radiation == "off"
    with pytest.raises(ValueError, match="transition_radiation must be one of"):
        pr.Physics(transition_radiation="crossing")
    with pytest.raises(ValueError, match="channeling_model='planar'"):
        pr.Physics(
            transition_radiation="history",
            channeling_model="planar",
            channeling_plane=(0, 0, 2),
            radiative_model="uncoupled",
        )
    with pytest.raises(ValueError, match="energy_model='midpoint'"):
        pr.Physics(transition_radiation="history").validate_transport("frozen")
    off = api.build_case(replace(_scene(), physics=pr.Physics()), _NUMERICS)
    on = api.build_case(_scene("bunch"), _NUMERICS)
    assert off.to_dict().get("transition_radiation") is None
    assert on["transition_radiation"] == {"model": "planar-stack", "coherence": "bunch"}
    with pytest.raises(ValueError, match="energy_model must be 'frozen' or 'midpoint'|midpoint"):
        Settings(energy_model="frozen", transition_radiation="history")


def test_identity_and_provenance_diverge_only_when_on():
    sweep = material_sweep("hopg")
    base = default_settings()
    off = dataset_identity("hopg", "full", base, sweep)
    on = dataset_identity("hopg", "full", replace(base, transition_radiation="history"), sweep)
    bunch = dataset_identity("hopg", "full", replace(base, transition_radiation="bunch"), sweep)
    assert "transition_model" not in off["resolved_parameters"]
    assert on["resolved_parameters"]["transition_model"] == TRANSITION_MODEL
    assert len({off["parameter_sha256"], on["parameter_sha256"], bunch["parameter_sha256"]}) == 3
    plain = api.build_case(replace(_scene(), physics=pr.Physics()), _NUMERICS)
    case = api.build_case(_scene(), _NUMERICS)
    assert case_content_key(case) != case_content_key(plain)
    view = resolved_physics_view(case, table_markers={})
    assert dict(view.model_markers)["transition_model"] == TRANSITION_MODEL
    assert any(descriptor.startswith("emission.transition") for descriptor in view.descriptor_ids)
    assert "transition_model" not in dict(
        resolved_physics_view(plain, table_markers={}).model_markers
    )


def test_simulate_returns_the_transition_component():
    result = pr.simulate(
        _scene().beam, _scene().target, pr.Detector(), physics=_scene().physics, numerics=_NUMERICS
    )
    assert result.transition is not None
    assert np.all(result.transition.density > 0.0)
    total = result.combined_cxr().density
    without = result.combined_cxr(include_transition=False).density
    np.testing.assert_allclose(total - without, result.transition.density, rtol=1e-9, atol=0)
    assert "transition" in set(result.to_dataframe()["component"].to_list())
    assert "transition" in result.spectral_flux().columns


def test_checkpoint_round_trips_a_separate_transition_component(tr_run, tmp_path):
    case, _transport, live = tr_run
    records = {}
    store_result(records, dict(case), live)
    _checkpoint_store.save("hopg", tmp_path, records)
    assert _checkpoint_store.component_path("hopg", "transition", tmp_path).is_file()
    loaded = _checkpoint_store.load("hopg", tmp_path)
    restored = next(iter(next(iter(loaded.values())).values()))
    np.testing.assert_array_equal(restored["spec_transition"], live["spec_transition"])
    np.testing.assert_allclose(cxr_spectrum(restored), cxr_spectrum(live), rtol=1e-14)
    # A store without TR keeps the historical layout.
    plain = {name: {e: {k: v for k, v in r.items() if k != "spec_transition"}
                    for e, r in by.items()} for name, by in records.items()}  # fmt: skip
    for record in (r for by in plain.values() for r in by.values()):
        record["case"] = {k: v for k, v in record["case"].items() if k != "transition_radiation"}
    _checkpoint_store.save("plain", tmp_path, plain)
    assert not _checkpoint_store.component_path("plain", "transition", tmp_path).is_file()


@pytest.mark.parametrize("block_size", [1, 1 << 20])
def test_full_and_streamed_replay_match_the_live_component(tr_run, tmp_path, block_size):
    case, transport, live = tr_run
    inputs = {k: v for k, v in transport.items() if k != "segs"}
    path = write_trajectory_artifact(
        tmp_path / "tr.h5", transport["segs"], case=case, spectrum_inputs=inputs
    )
    full = runner.spectrum_from_artifact(read_trajectory_artifact(path))
    streamed = runner.stream_spectrum_from_artifact(path, max_segments=block_size)
    np.testing.assert_array_equal(full["spec_transition"], live["spec_transition"])
    # Whole-history blocks add; only the summation order differs.
    np.testing.assert_allclose(streamed["spec_transition"], live["spec_transition"], rtol=1e-12)
    bunch = {**case, "transition_radiation": {"model": "planar-stack", "coherence": "bunch"}}
    bunch_path = write_trajectory_artifact(
        tmp_path / "bunch.h5", transport["segs"], case=bunch, spectrum_inputs=inputs
    )
    with pytest.raises(TrajectoryArtifactError, match="every history at once"):
        runner.stream_spectrum_from_artifact(bunch_path)


def test_reline_rescores_transition_from_the_same_transport(tr_run):
    case, transport, live = tr_run
    records = {}
    store_result(records, dict(case), live)
    record = next(iter(next(iter(records.values())).values()))
    assert (
        recompute.repair_cxr_spec(
            records, "hopg", from_config=False, line_step_eV=25.0, redo_all=True
        )
        == 1
    )
    target = np.asarray(record["E_grid"])
    reference = runner._spectrum_case(case, {**transport, "E_grid": target})["spec_transition"]
    np.testing.assert_allclose(record["spec_transition"], reference, rtol=1e-12)
    assert target.size != live["E_grid"].size


def test_unsupported_transport_options_refuse(tr_run):
    case, _transport, _live = tr_run
    with pytest.raises(ValueError, match="electron-block transport does not support transition"):
        runner._transport_case(case, block_electrons=8)
