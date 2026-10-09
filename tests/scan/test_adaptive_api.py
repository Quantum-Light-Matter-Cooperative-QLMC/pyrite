"""Adaptive policy lowering, stable identity, and realized provenance (#361)."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._line_grid_policy import LineYieldStatisticsWarning
from pyrite.campaign.profiles import case_content_key, dataset_identity
from pyrite.campaign.sweep import BeamSpec, build_cases
from pyrite.campaign.sweep import Sweep as LegacySweep
from pyrite.checkpoints import _checkpoint_io, _checkpoint_store
from pyrite.montecarlo import Case, runner
from pyrite.results import Settings, store_result


@pytest.fixture
def scene():
    return pr.Scene(
        pr.Beam(energy_keV=30.0),
        pr.Slab("hopg", thickness_ang=1e4, tilt_deg=30.0),
        pr.Detector(),
    )


@pytest.fixture
def precision():
    return pr.Precision(0.1, 40, 60, 20, observables=("line", "brem"))


@pytest.fixture
def numerics(precision):
    return pr.Numerics(
        precision=precision, bremsstrahlung_model="eedl", transport_core="per-electron"
    )


def test_precision_roundtrips_json_and_process_pickle(precision):
    assert pr.Precision.from_dict(json.loads(json.dumps(precision.to_dict()))) == precision
    assert pickle.loads(pickle.dumps(precision)) == precision
    assert "realized_electrons" not in precision.to_dict()


def test_precision_freezes_caller_owned_sequences():
    observables = ["line"]
    band = [1000.0, 3000.0]
    precision = pr.Precision(0.1, 40, 60, 20, observables=observables, band_eV=band)
    observables.append("brem")
    band[1] = 5000.0
    assert precision.observables == ("line",)
    assert precision.band_eV == (1000.0, 3000.0)


def test_numerics_requires_typed_precision():
    with pytest.raises(TypeError, match="precision must be a Precision"):
        pr.Numerics(precision={"target_rse": 0.1})
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        pr.Numerics(precision=pr.Precision(0.1, 40, 60, 20), transport_core="lockstep")


def test_lowering_uses_policy_maximum_and_ignores_fixed_defaults(scene, numerics):
    first = api.build_case(scene, numerics)
    other = api.build_case(scene, replace(numerics, n_electrons=7, n_electrons_brem=3))
    assert first.Ne == first.Ne_brem == numerics.precision.max_electrons
    assert first.adaptive_precision == numerics.precision.to_dict()
    assert case_content_key(first) == case_content_key(other)
    assert pickle.loads(pickle.dumps(first)).to_dict().keys() == first.to_dict().keys()
    assert Case(**first.to_dict()).adaptive_precision == first.adaptive_precision


def test_fixed_n_identity_matches_the_pre_precision_baseline():
    sweep = LegacySweep(
        material="hopg", beam=BeamSpec(energy_keV=30), thickness_ang=1e4, tilt_deg=30
    )
    identity = dataset_identity("hopg", "full", Settings(), sweep, bremsstrahlung_model="eedl")
    assert (
        identity["parameter_sha256"]
        == "701329024d40898b1749aabea2d858dc7ed834f859a82df46ffa2ef2668ed9f5"
    )
    case = build_cases(sweep, n_electrons=60, n_electrons_brem=60, bremsstrahlung_model="eedl")[0]
    assert "adaptive_precision" not in case
    assert (
        case_content_key(case) == "a02f8a78e9aa594e074f339c1097974f55ed6894567fd4ee0b6f088c3bb74595"
    )


def test_adaptive_dataset_identity_hashes_policy_not_unused_fixed_counts(scene, precision):
    sweep = LegacySweep(
        material="hopg", beam=scene.beam, target=scene.target, detector=scene.detector
    )
    settings = Settings(precision=precision)
    first = dataset_identity("hopg", "full", settings, sweep, bremsstrahlung_model="eedl")
    other = dataset_identity(
        "hopg",
        "full",
        replace(settings, n_electrons=7, n_electrons_brem=3),
        sweep,
        bremsstrahlung_model="eedl",
    )
    changed = dataset_identity(
        "hopg",
        "full",
        replace(settings, precision=replace(precision, target_rse=0.2)),
        sweep,
        bremsstrahlung_model="eedl",
    )
    assert first["parameter_sha256"] == other["parameter_sha256"]
    assert first["parameter_sha256"] != changed["parameter_sha256"]
    assert first["resolved_parameters"]["adaptive_precision"] == json.loads(
        json.dumps(precision.to_dict())
    )


@pytest.fixture
def adaptive_output(scene, numerics):
    case = api.build_case(scene, numerics)
    with pytest.warns(LineYieldStatisticsWarning, match="max_electrons=60"):
        output = runner.run_case(case, transport_core="per-electron")
    return case, output


@pytest.mark.slow
def test_public_runner_equals_fixed_n_and_records_statistics(adaptive_output):
    case, output = adaptive_output
    statistics = output["adaptive_sampling"]
    assert statistics["statistics_limited"]
    assert statistics["realized_electrons"] == statistics["realized_electrons_brem"] == 60
    assert statistics["precision"] == case.adaptive_precision
    fixed = {key: value for key, value in case.items() if key != "adaptive_precision"}
    reference = runner.run_case(fixed, transport_core="per-electron")
    for key in ("spec", "spec_characteristic", "brem", "brem_wide", "E_grid"):
        np.testing.assert_array_equal(output[key], reference[key])


def test_split_transport_spectrum_path_retains_provenance(scene, numerics):
    case = api.build_case(scene, numerics)
    with pytest.warns(LineYieldStatisticsWarning):
        transport = runner._transport_case(case, transport_core="per-electron")
    output = runner._spectrum_case(case, transport)
    assert output["adaptive_sampling"]["realized_electrons"] == transport["Ne_lines"]
    assert case.Ne == numerics.precision.max_electrons


@pytest.mark.parametrize("split", [False, True])
def test_early_stop_preserves_requested_case_and_matches_realized_fixed_n(scene, split):
    precision = pr.Precision(
        1.0,
        40,
        200,
        20,
        observables=("line", "brem"),
        max_electron_share=1.0,
        min_effective_electrons=0.0,
        stability_blocks=0,
    )
    case = api.build_case(scene, pr.Numerics(precision=precision, bremsstrahlung_model="eedl"))
    key = case_content_key(case)
    if split:
        output = runner._spectrum_case(
            case, runner._transport_case(case, transport_core="per-electron")
        )
    else:
        output = runner.run_case(case, transport_core="per-electron")
    n = output["adaptive_sampling"]["realized_electrons"]
    assert n == 40 < case.Ne == 200
    assert case_content_key(case) == key
    fixed = {k: v for k, v in case.items() if k != "adaptive_precision"}
    fixed.update(Ne=n, Ne_brem=n)
    reference = runner.run_case(fixed, transport_core="per-electron")
    for field in ("spec", "spec_characteristic", "brem", "brem_wide", "E_grid"):
        np.testing.assert_array_equal(output[field], reference[field])


def test_public_batch_means_survive_checkpoint_storage(scene, numerics, tmp_path):
    numerics = replace(
        numerics, precision=replace(numerics.precision, batch_means_band_eV=(1000.0, 3000.0))
    )
    case = api.build_case(scene, numerics)
    with pytest.warns(LineYieldStatisticsWarning):
        output = runner.run_case(case, transport_core="per-electron")
    results = {}
    store_result(results, case, output)
    _checkpoint_store.save("batch", tmp_path, results)
    record = _checkpoint_store.load("batch", tmp_path)[case.name][case.E0_keV]
    batch = record["adaptive_sampling"]["batch_means"]
    assert batch["n_batches"] == 3
    for component in ("spec", "brem_wide"):
        np.testing.assert_array_equal(
            batch[component]["standard_error"],
            output["adaptive_sampling"]["batch_means"][component]["standard_error"],
        )


def test_simulate_carries_requested_identity_and_realized_provenance(scene, numerics):
    with pytest.warns(LineYieldStatisticsWarning):
        result = pr.simulate(scene.beam, scene.target, scene.detector, numerics=numerics)
    sampling = result.provenance["adaptive_sampling"]
    assert sampling["mode"] == "adaptive" and sampling["realized_electrons"] == 60
    assert result.provenance["identity_digest"] == case_content_key(
        result.case, xsgen_tables=result.provenance["xsgen_tables"]
    )
    assert result.case.Ne == numerics.precision.max_electrons


@pytest.mark.slow
def test_checkpoint_components_and_cas_roundtrip_realized_provenance(tmp_path, adaptive_output):
    case, output = adaptive_output
    results = {}
    store_result(results, case, output)
    _checkpoint_store.save("adaptive", tmp_path, results)
    restored = _checkpoint_store.load("adaptive", tmp_path)[case.name][case.E0_keV]
    assert restored["adaptive_sampling"] == output["adaptive_sampling"]
    for component in _checkpoint_store.COMPONENTS:
        artifact = _checkpoint_store.component_path("adaptive", component, tmp_path)
        record = _checkpoint_io.load(str(artifact))[case.name][case.E0_keV]
        assert record["adaptive_sampling"] == output["adaptive_sampling"]
        assert record["case"]["adaptive_precision"] == case.adaptive_precision
    key = case_content_key(case)
    _checkpoint_store.cas_save("hopg", key, tmp_path, output)
    assert (
        _checkpoint_store.cas_load("hopg", key, tmp_path)["adaptive_sampling"]
        == output["adaptive_sampling"]
    )


def test_adaptive_policy_survives_configured_case_lowering(scene, precision):
    legacy = LegacySweep(
        material="hopg", beam=scene.beam, target=scene.target, detector=scene.detector
    )
    cases = api.build_configured_cases(legacy, Settings(precision=precision))
    assert cases[0]["adaptive_precision"] == precision.to_dict()


def test_public_sweep_retains_policy_for_every_case(scene, numerics):
    sweep = pr.Sweep(scene, {"beam.energy_keV": [20.0, 30.0]})
    assert all(
        case["adaptive_precision"] == numerics.precision.to_dict() for case in sweep.cases(numerics)
    )


def test_unsupported_routes_fail_before_transport(scene, numerics, monkeypatch):
    case = api.build_case(scene, numerics)
    monkeypatch.setattr(
        runner, "_transport_case", lambda *_a, **_kw: pytest.fail("transport started")
    )
    with pytest.raises(ValueError, match="observation directions"):
        runner.run_case(case, observation_directions=np.array([[0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="observation directions"):
        runner.run_case_directions(case, [[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="coherent"):
        api.build_case(replace(scene, emission="coherent"), numerics)


def test_adaptive_case_rejects_count_mismatch(scene, numerics):
    case = api.build_case(scene, numerics)
    with pytest.raises(ValueError, match="precision.max_electrons"):
        replace(case, Ne=40)


def test_pre_precision_case_pickle_state_keeps_manual_controls(scene):
    case = api.build_case(
        scene, pr.Numerics(n_electrons=8, n_electrons_brem=4, bremsstrahlung_model="eedl")
    )
    case = replace(case, E_cut_lines_keV=7.0, E_cut_brem_keV=2.0, brem_step_eV=100.0)
    # Frozen/slotted dataclasses pickle a positional list of field values.
    # Precision was appended after all existing fields, so old state is a prefix.
    old_state = case.__getstate__()[:-1]
    restored = Case.__new__(Case)
    restored.__setstate__(old_state)
    assert "adaptive_precision" not in restored
    assert restored.E_cut_lines_keV == 7.0
    assert restored.E_cut_brem_keV == 2.0
    assert restored.brem_step_eV == 100.0
    assert case_content_key(restored) == case_content_key(case)


@pytest.mark.parametrize(
    "route,value",
    [
        ("gdf_source", {}),
        ("groove_spacing_ang", 100.0),
        ("secondary_threshold_eV", 50.0),
        ("pair_production_model", "penelope-2024"),
        ("positron_transport", True),
    ],
)
def test_unsupported_block_routes_are_rejected_at_case_boundary(scene, numerics, route, value):
    case = api.build_case(scene, numerics)
    with pytest.raises(ValueError, match=route):
        replace(case, **{route: value})


def test_physical_detector_is_rejected_at_lowering(scene, numerics):
    detector = pr.PlanarDetector(pr.PlanarPose((0.0, 0.0, 50.0)), size_mm=(1.0, 1.0))
    with pytest.raises(ValueError, match="physical detectors"):
        api.build_case(replace(scene, detector=detector), numerics)


@pytest.mark.slow
def test_mixed_cpu_pool_preserves_fixed_case_and_adaptive_provenance(scene, monkeypatch):
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    precision = pr.Precision(
        1.0, 40, 200, 20, max_electron_share=1.0, min_effective_electrons=0.0, stability_blocks=0
    )
    adaptive = api.build_case(scene, pr.Numerics(precision=precision, bremsstrahlung_model="eedl"))
    fixed = api.build_case(
        scene, pr.Numerics(n_electrons=8, n_electrons_brem=4, bremsstrahlung_model="eedl")
    )
    expected_fixed = runner.run_cases([fixed], max_workers=1, progress=False, engine="cpu")[0]
    expected_adaptive = runner.run_case(adaptive, transport_core="per-electron")
    outputs = runner.run_cases([adaptive, fixed], max_workers=1, progress=False, engine="cpu")
    assert outputs[0]["adaptive_sampling"] == expected_adaptive["adaptive_sampling"]
    assert "adaptive_sampling" not in outputs[1]
    for field in ("spec", "spec_characteristic", "brem", "brem_wide"):
        # Parent/worker BLAS reductions may differ by a few ulps. The
        # adaptive/fixed comparison within one process is exact above.
        np.testing.assert_allclose(
            outputs[0][field], expected_adaptive[field], rtol=2e-14, atol=0.0
        )
        np.testing.assert_array_equal(outputs[1][field], expected_fixed[field])


@pytest.mark.parametrize("inherited", ["auto", "per-electron", "lockstep"])
def test_adaptive_worker_only_overrides_automatic_pool_pin(monkeypatch, inherited):
    from pyrite._env import env_value

    monkeypatch.setattr(runner, "_WORKER_INHERITED_CORE", inherited)
    pinned = "lockstep" if inherited == "auto" else inherited
    monkeypatch.setenv("PYRITE_MC_TRANSPORT_CORE", pinned)
    from pyrite.montecarlo.runner.adaptive import _adaptive_worker_call

    core = _adaptive_worker_call(lambda _case: env_value("PYRITE_MC_TRANSPORT_CORE"), {})
    assert core == ("per-electron" if inherited == "auto" else inherited)
    assert env_value("PYRITE_MC_TRANSPORT_CORE") == pinned


def test_precision_cannot_cross_with_electron_count_sweep(scene, precision):
    sweep = LegacySweep(
        material="hopg",
        beam=scene.beam,
        target=scene.target,
        detector=scene.detector,
        n_electrons=[40, 60],
    )
    with pytest.raises(ValueError, match="electron-count sweep grids"):
        build_cases(sweep, precision=precision)
