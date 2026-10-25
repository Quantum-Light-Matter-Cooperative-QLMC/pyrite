"""Opt-in Physics/Case/Result/checkpoint/replay wiring for ordinary photon hits."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import elementary_charge

import pyrite as pr
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import photon_aperture as photon
from pyrite.montecarlo.transport.photon_histories import PhotonAperture
from pyrite.results.aperture import aperture_dataframe, aperture_from_payload
from tests.montecarlo.test_coupled_radiative_runner import (
    _coupled_case,
)
from tests.montecarlo.test_coupled_radiative_runner import (
    synthetic_si_tables as synthetic_si_tables,
)


@pytest.fixture
def selected(monkeypatch, synthetic_si_tables):
    from pyrite.campaign.profiles import case_content_key

    base = _coupled_case(inelastic_model="continuous", atomic_electron_deflection="none")
    config = pr.PhotonTransport(
        (1000, 3000, 10000), n_samples=32, source_ceiling_eV=20000, seed=341
    )
    monkeypatch.setattr(photon, "case_table_markers", lambda case: {"bremslib:Si": "0" * 64})
    aperture = PhotonAperture((0, 0, 1e7), (1, 0, 0), (0, 1, 0), 1e8, 1e8)
    context = photon.prepare_source(
        base,
        config,
        aperture,
        (),
        base["crystal_width_mm"] * 1e7,
        base["crystal_height_mm"] * 1e7,
        "lockstep",
    )
    digest = case_content_key(base, xsgen_tables=photon.case_table_markers(base))
    case = replace(base, photon_transport=photon.photon_case_record(base, digest, context))
    return SimpleNamespace(base=base, case=case, config=config, context=context)


@pytest.mark.parametrize(
    "options",
    [
        dict(n_samples=0),
        dict(n_samples=True),
        dict(seed=-1),
        dict(seed=2**64),
        dict(edges_eV=(1, 1)),
        dict(edges_eV=(np.nan, 2)),
        dict(source="unknown"),
        dict(source="annihilation", n_samples=1),
        dict(charged_feedback="yes"),
        dict(electron_cutoff_eV=0),
        dict(source_ceiling_eV=1000),
    ],
)
def test_invalid_selector(options):
    with pytest.raises((TypeError, ValueError)):
        pr.PhotonTransport(**dict(dict(edges_eV=(1000, 10000), n_samples=10), **options))


def test_mode_off_case_and_identity_remain_unchanged(selected):
    from pyrite._physics_provenance import resolved_physics_view
    from pyrite.campaign.profiles import case_content_key

    assert pr.Physics().photon_transport is None
    assert "photon_transport" not in selected.base
    assert case_content_key(selected.case) != case_content_key(selected.base)
    off = resolved_physics_view(selected.base, table_markers={})
    on = resolved_physics_view(selected.case, table_markers={})
    assert "observation.photon/cpu-reference" not in off.descriptor_ids
    assert "observation.photon/cpu-reference" in on.descriptor_ids
    with pytest.raises(TypeError, match="PhotonTransport"):
        pr.Physics(photon_transport=True)


@pytest.mark.slow
def test_real_primary_transport_and_aperture_are_replayed_once(selected, monkeypatch, tmp_path):
    from pyrite.montecarlo.runner import case_tables
    from pyrite.montecarlo.runner.artifacts import (
        spectrum_from_artifact,
        stream_spectrum_from_artifact,
    )
    from pyrite.montecarlo.trajectories import (
        TrajectoryArtifactError,
        TrajectoryCapture,
        read_trajectory_artifact,
    )

    monkeypatch.setattr(case_tables, "case_table_markers", photon.case_table_markers)
    capture = TrajectoryCapture(str(tmp_path))
    path = capture.path_for(selected.case)
    calls = []
    original_transport = runner._transport_case

    def transport(*args, **kwargs):
        calls.append(1)
        return original_transport(*args, **kwargs)

    monkeypatch.setattr(runner, "_transport_case", transport)
    output = runner.run_case(
        selected.case,
        transport_core="lockstep",
        trajectory_capture=capture,
    )
    assert calls == [1]
    tally = aperture_from_payload(output["photon_aperture"])
    assert tally.n_primaries == selected.base["Ne_brem"]
    assert tally.n_proposals == 32
    assert tally.provenance["data_sha256"]["epdl_differential"]
    assert tally.detected_per_primary > 0

    def forbidden(*args, **kwargs):
        pytest.fail("full artifact replay must not rerun primary transport")

    monkeypatch.setattr(runner, "_transport_case", forbidden)
    artifact = read_trajectory_artifact(path)
    replay = spectrum_from_artifact(artifact)
    other = aperture_from_payload(replay["photon_aperture"])
    np.testing.assert_array_equal(tally.total_per_primary_eV, other.total_per_primary_eV)
    assert tally.observation_identity_digest == other.observation_identity_digest
    with pytest.raises(TrajectoryArtifactError, match="photon_transport|every segment|bounded"):
        stream_spectrum_from_artifact(path)


def test_public_simulate_returns_separate_aperture_after_one_primary_pass(
    monkeypatch, synthetic_si_tables
):
    from pyrite import api
    from pyrite.montecarlo.runner import case_tables

    monkeypatch.setattr(case_tables, "case_table_markers", lambda case: {"bremslib:Si": "0" * 64})
    monkeypatch.setattr(photon, "case_table_markers", case_tables.case_table_markers)
    scene = pr.Scene(
        pr.Beam(energy_keV=40.0, transverse=pr.PointSpot(), longitudinal=pr.PointBunch()),
        pr.Slab("silicon", thickness_ang=100.0, tilt_deg=5.0, footprint=None),
        pr.PlanarDetector(pr.PlanarPose((0, 0, 10)), size_mm=(1.0, 1.0)),
        physics=pr.Physics(
            elastic_model="mott",
            inelastic_model="continuous",
            atomic_electron_deflection="none",
            bremsstrahlung_model="bremslib",
            radiative_model="bremslib-soft-hard",
            photon_transport=pr.PhotonTransport((1000.0, 3000.0, 10000.0), n_samples=32),
        ),
        numerics=pr.Numerics(
            cxr_trials=4,
            brem_trials=3,
            energy_bins=pr.EnergyBins(
                cxr=np.array([1500.0, 2000.0]),
                brem=np.array([5000.0, 10000.0]),
            ),
        ),
    )
    calls = []

    def transport(case, **kwargs):
        calls.append("primary")
        return dict(
            segs=dict(
                elec_id=np.array([0]),
                L_ang=np.array([1.0]),
                r_mid=np.zeros((1, 3)),
                v_hat=np.array([[0.0, 0.0, 1.0]]),
            ),
            E_grid=np.array([1500.0, 2000.0]),
            E_brem=np.array([5000.0, 10000.0]),
            n_hat=np.array([0.0, 0.0, 1.0]),
            Ne_lines=4,
            Ne_brem=3,
        )

    def spectrum(case, tp, timing):
        return dict(
            E_grid=tp["E_grid"],
            E_grid_brem=tp["E_brem"],
            spec=np.ones(2),
            spec_characteristic=np.zeros(2),
            brem=np.full(2, 2.0),
            brem_wide=np.full(2, 2.0),
        )

    monkeypatch.setattr(runner, "_transport_case", transport)
    monkeypatch.setattr(runner, "_spectrum_case_impl", spectrum)
    # An analytic hit reference isolates public orchestration from sampler tests.
    built_case = api.build_case(scene)
    selected = SimpleNamespace(case=built_case, config=scene.physics.photon_transport)
    payload = _controlled_payload(selected, monkeypatch)
    scores = []

    def score(case, tp):
        scores.append("aperture")
        return payload

    monkeypatch.setattr(photon, "score_photon_aperture", score)
    result = api.simulate(
        scene.beam, scene.target, scene.detector, physics=scene.physics, numerics=scene.numerics
    )
    assert calls == ["primary"] and scores == ["aperture"]
    assert result.aperture.n_primaries == 3
    assert result.aperture.n_proposals == 32
    np.testing.assert_array_equal(result.brem.density, [2.0, 2.0])
    assert result.brem.normalization == "observation-average"
    np.testing.assert_allclose(result.aperture_dataframe()["total_counts"], [8 / 3, 12 / 3])
    assert "photon_aperture_identity_digest" in result.provenance


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(seed=901), "source Case changed"),
        (dict(transport_core="cuda"), "CPU"),
    ],
)
def test_invalid_execution_is_rejected_before_primary_transport(
    selected, monkeypatch, change, match
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid selection reached primary transport")

    monkeypatch.setattr(runner, "simulate_trajectories", forbidden)
    core = change.pop("transport_core", "lockstep")
    case = replace(selected.case, **change)
    with pytest.raises(ValueError, match=match):
        runner.run_case(case, transport_core=core)


def test_data_pin_change_refuses_before_primary_transport(selected, monkeypatch):
    monkeypatch.setattr(photon, "case_table_markers", lambda case: {"bremslib:Si": "f" * 64})
    with pytest.raises(ValueError, match="table pins changed"):
        runner.run_case(selected.case, transport_core="lockstep")


def _controlled_payload(selected, monkeypatch):
    """Analytic weighted ordinary hits, unequal bin widths, and one unbinned hit."""
    from pyrite.montecarlo.runner.photon_emission import PhotonContinuumScore, _score_exits

    histories = SimpleNamespace(
        exits=dict(
            energy_eV=np.array([2000.0, 5000.0, 11000.0]),
            weight=np.array([8.0, 12.0, 4.0]),
            detected=np.ones(3, dtype=bool),
            generation=np.array([0, 1, 0]),
        )
    )

    def score(*args, **options):
        values = _score_exits(histories, np.array(selected.config.edges_eV), options["n_primaries"])
        return PhotonContinuumScore(
            None,
            histories,
            np.array(selected.config.edges_eV),
            *values,
            options["n_primaries"],
            options["n_samples"],
            False,
        )

    monkeypatch.setattr(photon, "score_bremslib_continuum", score)
    segments = dict(
        elec_id=np.array([0]),
        L_ang=np.array([1.0]),
        r_mid=np.zeros((1, 3)),
        v_hat=np.array([[0.0, 0.0, 1.0]]),
    )
    return photon.score_photon_aperture(selected.case, dict(segs=segments))


def test_checkpoint_component_cas_and_unit_views_preserve_aperture(selected, monkeypatch, tmp_path):
    from pyrite.checkpoints import _checkpoint_io
    from pyrite.checkpoints import _checkpoint_store as store
    from pyrite.checkpoints.persistence import _cas_payload_from_record
    from pyrite.results.store import store_result

    payload = _controlled_payload(selected, monkeypatch)
    grid = np.array([1500.0, 2000.0])
    output = dict(E_grid=grid, spec=np.ones(2), brem=np.zeros(2), eta=1.0, photon_aperture=payload)
    results = {}
    store_result(results, selected.case, output)
    store.save("silicon", tmp_path, results)
    assert (tmp_path / "silicon" / "photon.h5").is_file()
    line = _checkpoint_io.load(tmp_path / "silicon" / "line.h5")
    assert "photon_aperture" not in line[selected.case.name][selected.case.E0_keV]
    loaded = store.load("silicon", tmp_path)[selected.case.name][selected.case.E0_keV]
    raw = _cas_payload_from_record(loaded)
    tally = aperture_from_payload(raw["photon_aperture"])
    table = aperture_dataframe(tally)
    n = selected.case.Ne_brem
    np.testing.assert_allclose(table["total_counts"], [8 / n, 12 / n], rtol=1e-14)
    assert tally.unbinned_detected_per_primary == 4 / n
    np.testing.assert_allclose(table["total"], [8 / (n * 2000), 12 / (n * 7000)], rtol=1e-14)
    current = aperture_dataframe(tally, current_nA=2.0)
    np.testing.assert_allclose(
        current["total"], table["total"].to_numpy() * 2e-9 / elementary_charge
    )
    assert current["density_units"][0] == "photons / (s eV)"
    assert table["density_units"][0] == "photons / (electron eV)"
    store.component_path("silicon", "photon", tmp_path).unlink()
    with pytest.raises(ValueError, match="missing its aperture"):
        store.load("silicon", tmp_path)


def test_corrupt_aperture_payload_rejected(selected, monkeypatch):
    payload = deepcopy(_controlled_payload(selected, monkeypatch))
    payload["arrays"]["total_per_primary_eV"][0] *= 2
    with pytest.raises(ValueError, match="checksum"):
        aperture_from_payload(payload)


@pytest.mark.parametrize("component", ["characteristic", "annihilation"])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_event_production_payloads_keep_original_population_and_budgets(
    selected,
    monkeypatch,
    component,
    feedback,
    empty,
):
    from pyrite.montecarlo.runner.photon_emission import PhotonContinuumScore, _score_exits
    from pyrite.montecarlo.transport.photon_event_sources import emission_sources

    base = replace(
        selected.base,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        secondary_threshold_eV=1000.0,
    )
    if component == "annihilation":
        base = replace(base, pair_production_model="penelope-2024", positron_transport=True)
    config = pr.PhotonTransport(
        (1000.0, 3000.0, 10000.0), source=component, charged_feedback=feedback
    )
    stage = object()
    monkeypatch.setattr(
        photon, "_scene_feedback", lambda *args: (stage, {"model": "test-feedback"})
    )
    context = photon.prepare_source(
        base,
        config,
        selected.context["aperture"],
        (),
        selected.context["width"],
        selected.context["height"],
        "lockstep",
    )
    case = replace(base, photon_transport=photon.photon_case_record(base, "1" * 64, context))
    energy = np.array([500.0 if empty else 2000.0])
    emissions = dict(
        position_ang=np.zeros((1, 3)),
        direction=np.array([[0.0, 0.0, 1.0]]),
        energy_eV=energy,
        parent=np.array([1]),
    )
    sources = emission_sources(emissions, np.arange(base.Ne))
    reserved = np.zeros(base.Ne)
    reserved[1] = energy[0]
    relaxation = (
        SimpleNamespace(
            photons=emissions,
            reserved_eV=reserved,
            escaped_eV=np.zeros(base.Ne),
            deposited_eV=np.zeros(base.Ne),
            binding_eV=np.zeros(base.Ne),
        )
        if component == "characteristic"
        else None
    )
    calls = []

    def adapt(segments, source, n, prepared_stage, seed, cutoff):
        assert source == component and n == base.Ne
        assert (
            prepared_stage is stage
            if component == "characteristic" or feedback
            else prepared_stage is None
        )
        return sources, relaxation

    def score(source, layers, aperture, edges, **options):
        calls.append(options)
        histories = SimpleNamespace(
            exits=dict(
                energy_eV=source.energy_eV,
                weight=source.weight,
                detected=np.ones(source.energy_eV.size, dtype=bool),
                generation=np.zeros(source.energy_eV.size, dtype=int),
            )
        )
        edges = np.array(edges)
        values = _score_exits(histories, edges, options["n_primaries"])
        return PhotonContinuumScore(
            source, histories, edges, *values, base.Ne, source.energy_eV.size, feedback
        )

    monkeypatch.setattr(photon, "_event_sources", adapt)
    monkeypatch.setattr(photon, "score_photon_sources", score)
    payload = photon.score_photon_aperture(case, dict(segs={}))
    tally = aperture_from_payload(payload)
    assert len(calls) == 1 and tally.n_primaries == base.Ne
    assert tally.source_component == component
    assert tally.n_proposals == (0 if empty else 1)
    assert tally.charged_feedback is feedback
    assert tally.local_source_energy_eV[1] == (500.0 if empty else 0.0)
    assert tally.tracked_source_energy_eV[1] == (0.0 if empty else 2000.0)
    if relaxation is not None:
        np.testing.assert_array_equal(tally.relaxation.reserved_eV, reserved)
        np.testing.assert_array_equal(tally.relaxation.emitted_eV, reserved)
    if feedback:
        assert calls[0]["charged_stage"] is stage
    else:
        assert calls[0]["electron_cutoff_eV"] == 1000.0


def test_failed_checkpoint_save_preserves_complete_components(selected, monkeypatch, tmp_path):
    from pyrite.checkpoints import _checkpoint_store as store

    payload = _controlled_payload(selected, monkeypatch)
    record = dict(
        case=selected.case,
        E_grid=np.array([1000.0, 2000.0]),
        spec=np.ones(2),
        brem=np.zeros(2),
        photon_aperture=payload,
    )
    results = {selected.case.name: {40.0: record}}
    store.save("original", tmp_path, results)
    paths = list((tmp_path / "original").glob("*.h5"))
    original = {path: path.read_bytes() for path in paths}
    record.pop("photon_aperture")
    with pytest.raises(ValueError, match="missing its aperture"):
        store.save("original", tmp_path, results)
    assert {path: path.read_bytes() for path in paths} == original


def test_partial_projections_refuse_to_drop_aperture_and_full_slim_keeps_bytes(
    selected, monkeypatch
):
    from pyrite.results.selection import project_dataset, slim_results

    payload = _controlled_payload(selected, monkeypatch)
    record = dict(case=selected.case, spec=np.ones(2), photon_aperture=payload)
    results = {selected.case.name: {40.0: record}}
    for component in ("line", "brem"):
        with pytest.raises(ValueError, match="retain photon_aperture"):
            project_dataset(results, component)
    slim = slim_results(results, downcast=True)[selected.case.name][40.0]
    assert slim["spec"].dtype == np.float32
    tally = aperture_from_payload(slim["photon_aperture"])
    assert tally.total_per_primary_eV.dtype == payload["arrays"]["total_per_primary_eV"].dtype
    assert (
        tally.total_per_primary_eV.tobytes() == payload["arrays"]["total_per_primary_eV"].tobytes()
    )


def test_changed_photon_data_pin_refuses_execution(selected):
    record = deepcopy(selected.case.photon_transport)
    record["identity"]["data_sha256"]["epdl_integrated"] = "f" * 64
    with pytest.raises(ValueError, match="identity or data pins changed"):
        photon.prepare_photon_case(replace(selected.case, photon_transport=record))
