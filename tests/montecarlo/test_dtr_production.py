"""Opt-in two-beam production scoring, identities and replay."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._physics_provenance import resolved_physics_view
from pyrite.checkpoints import _checkpoint_store
from pyrite.montecarlo import runner
from pyrite.montecarlo.spectrum.dtr_history import DTR_MODEL
from pyrite.montecarlo.trajectories import read_trajectory_artifact, write_trajectory_artifact
from pyrite.results.store import store_result


def _scene(mode="two-beam"):
    return pr.Scene(
        pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        pr.Slab("hopg", thickness_ang=100.0, tilt_deg=30.0, footprint=None),
        pr.Detector(),
        physics=pr.Physics(
            transition_radiation=mode, radiative_model="uncoupled", bremsstrahlung_model="eedl"
        ),
    )


def _numerics():
    return pr.Numerics(
        cxr_trials=3,
        brem_trials=3,
        cxr_reflections=pr.CxrReflections(max_reflections=1),
        energy_bins=pr.EnergyBins(
            cxr=np.linspace(500.0, 1500.0, 9), brem=np.array([500.0, 5000.0])
        ),
    )


def test_selector_is_explicit_and_provenance_is_distinct():
    off = api.build_case(_scene("off"), _numerics())
    on = api.build_case(_scene(), _numerics())
    assert "transition_radiation" not in off
    assert on["transition_radiation"]["model"] == "two-beam"
    assert (
        dict(resolved_physics_view(on, table_markers={}).model_markers)["transition_model"]
        == DTR_MODEL
    )


def test_live_two_beam_path_bypasses_the_legacy_line_amplitude(monkeypatch):
    case = api.build_case(_scene(), _numerics()).to_dict()
    transport = runner._transport_case(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("the opt-in model must replace the legacy PXR/CBS scorer")

    monkeypatch.setattr(runner, "mc_cxr_spectrum", forbidden)
    output = runner._spectrum_case(case, transport)
    total = output["spec"] + output["spec_transition"]
    assert np.all(np.isfinite(total))
    assert np.all(total >= 0)
    assert np.any(total > 0)


def test_turning_off_does_not_invoke_two_beam_solver(monkeypatch):
    from pyrite.montecarlo.runner import dtr_emission

    def forbidden(*args, **kwargs):
        raise AssertionError("off must not score two-beam radiation")

    monkeypatch.setattr(dtr_emission, "case_dtr_spectrum", forbidden)
    case = api.build_case(_scene("off"), _numerics()).to_dict()
    output = runner._spectrum_case(case, runner._transport_case(case))
    assert "spec_transition" not in output


@pytest.fixture(scope="module")
def dtr_run():
    case = api.build_case(_scene(), _numerics()).to_dict()
    transport = runner._transport_case(case)
    return case, transport, runner._spectrum_case(case, transport)


def test_zero_harmonics_leave_only_the_separate_tr_component(dtr_run, monkeypatch):
    from pyrite.montecarlo.runner import dtr_emission

    case, transport, _ = dtr_run
    original = dtr_emission._row_parameters

    def zero(case, energy):
        return original(case, energy) | dict(
            chi_transfer=0.0,
            chi_reverse=0.0,
            potential_transfer_over_m=0.0,
            potential_reverse_over_m=0.0,
        )

    monkeypatch.setattr(dtr_emission, "_row_parameters", zero)
    output = runner._spectrum_case(case, transport)
    np.testing.assert_array_equal(output["spec"], 0.0)
    assert np.any(output["spec_transition"] > 0)


def test_two_beam_checkpoint_and_full_streamed_replay(dtr_run, tmp_path):
    case, transport, live = dtr_run
    records = {}
    store_result(records, dict(case), live)
    _checkpoint_store.save("hopg", tmp_path, records)
    restored = next(iter(next(iter(_checkpoint_store.load("hopg", tmp_path).values())).values()))
    for key in ("spec", "spec_transition"):
        np.testing.assert_array_equal(restored[key], live[key])
    inputs = {key: value for key, value in transport.items() if key != "segs"}
    path = write_trajectory_artifact(
        tmp_path / "dtr.h5", transport["segs"], case=case, spectrum_inputs=inputs
    )
    full = runner.spectrum_from_artifact(read_trajectory_artifact(path))
    streamed = runner.stream_spectrum_from_artifact(path, max_segments=1)
    for key in ("spec", "spec_transition"):
        np.testing.assert_array_equal(full[key], live[key])
        np.testing.assert_allclose(streamed[key], live[key], rtol=1e-9, atol=1e-18)


def test_two_beam_reline_uses_the_same_amplitude_path(dtr_run):
    case, transport, live = dtr_run
    spec, coherent, characteristic, tr = runner._cxr_pair_for_case(
        case,
        transport["E_grid"],
        want_coherent=False,
        return_characteristic=True,
        return_transition=True,
    )
    assert coherent is None
    np.testing.assert_array_equal(spec, live["spec"])
    np.testing.assert_array_equal(tr, live["spec_transition"])


def test_public_result_includes_the_signed_correction(dtr_run):
    result = pr.simulate(
        _scene().beam,
        _scene().target,
        _scene().detector,
        physics=_scene().physics,
        numerics=_numerics(),
    )
    np.testing.assert_allclose(
        result.combined_cxr().density,
        dtr_run[2]["spec"] + dtr_run[2]["spec_transition"] + dtr_run[2]["spec_characteristic"],
    )


def test_coherent_companion_uses_the_joint_field_population():
    from pyrite.campaign.beam import PointSpot

    scene = _scene()
    scene = replace(
        scene,
        beam=replace(scene.beam, transverse=PointSpot()),
        physics=replace(scene.physics, emission=("incoherent", "coherent")),
    )
    case = api.build_case(scene, _numerics()).to_dict()
    output = runner._spectrum_case(case, runner._transport_case(case))
    assert np.all(np.isfinite(output["spec_coherent"]))
    assert not np.array_equal(output["spec_coherent"], output["spec"])
    from pyrite.montecarlo.runner.dtr_emission import _power

    fields = np.ones((2, 3, 3), complex)
    np.testing.assert_allclose(_power(fields, 3, population=5), 15.0)
    np.testing.assert_allclose(_power(fields, 3, population=1), 3.0)


def test_dataset_identity_tracks_model_and_termination():
    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.profiles import dataset_identity

    base = default_settings()
    sweep = material_sweep("hopg")
    off = dataset_identity("hopg", "full", base, sweep)
    on = dataset_identity("hopg", "full", replace(base, transition_radiation="two-beam"), sweep)
    shifted = dataset_identity(
        "hopg",
        "full",
        replace(
            base, transition_radiation="two-beam", transition_lattice_origin_ang=(0.0, 0.0, 1.0)
        ),
        sweep,
    )
    assert "transition_lattice_origin_ang" not in off["resolved_parameters"]["settings"]
    assert on["resolved_parameters"]["transition_model"] == DTR_MODEL
    assert len({row["parameter_sha256"] for row in (off, on, shifted)}) == 3


def test_unsupported_configs_refuse_before_transport():
    with pytest.raises(ValueError, match="one reflection"):
        api.build_case(_scene(), replace(_numerics(), cxr_reflections=pr.CxrReflections()))
    with pytest.raises(ValueError, match="equal line and continuum"):
        api.build_case(_scene(), replace(_numerics(), brem_trials=2))
    with pytest.raises(ValueError, match="three finite"):
        pr.Physics(
            transition_radiation="two-beam", transition_lattice_origin_ang=(float("nan"), 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="requires transition_radiation"):
        pr.Physics(transition_lattice_origin_ang=(0.0, 0.0, 1.0))
