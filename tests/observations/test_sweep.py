import os
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.campaign.profiles import case_content_key
from pyrite.checkpoints import _checkpoint_store
from pyrite.instrument import ResolvedObservation
from pyrite.materials import CATALOG
from pyrite.montecarlo import runner
from pyrite.observations import ObservationStore, SweepObservation
from pyrite.runs.run import run_sweep


def _case(name: str, tilt_deg: float) -> dict:
    E_grid = (100.0, 400.0, 31.0)
    return dict(
        name=name,
        crystal="hopg",
        composition=CATALOG.crystal("hopg").composition,
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        E0_keV=30.0,
        thickness_ang=1e4,
        E_grid=E_grid,
        E_grid_line=E_grid,
        E_grid_brem=(0.0, 500.0, 50.0),
        theta_obs_rad=np.deg2rad(90.0),
        dtheta_obs_rad=np.deg2rad(2.0),
        tilt_deg=tilt_deg,
        tilt_azim_deg=0.0,
        domega_sr=1e-4,
        beam_uvw=(0, 0, 1),
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1,
        abs_layers=None,
        layer_radiators=None,
        brem_file=None,
        Ne=10,
        Ne_brem=5,
        seed=42,
        spec_chunk=None,
        brem_chunk=None,
    )


CASES = [_case("tilt30", 30.0), _case("tilt45", 45.0)]


def _observation(**acquisition) -> ResolvedObservation:
    return ResolvedObservation(
        detector=pr.PlanarDetector(
            pose=pr.PlanarPose.from_observation(100.0, 90.0),
            pixels=pr.PixelGrid((4, 6), (0.5, 0.5)),
            response=pr.IdealPhotonCounter(),
        ),
        scorer=pr.PixelScorer((2, 2)),
        filters=(
            pr.FilterPlate(
                "silicon", 0.05, (1.5, 10.0), pr.PlanarPose.from_observation(50.0, 90.0)
            ),
        ),
        acquisition=pr.Acquisition(
            **{"exposure_s": 1.0, "measured_edges_eV": (0.0, 250.0, 500.0), **acquisition}
        ),
    )


def _sweep_observation(tmp_path, **acquisition) -> SweepObservation:
    return SweepObservation(
        observation=_observation(**acquisition),
        store=ObservationStore("hopg", tmp_path / "observations"),
        content_key_fn=case_content_key,
        emission="incoherent",
        provenance_fn=lambda case: {"backend": "test"},
    )


def _run(tmp_path, observation, results=None):
    results = {} if results is None else results
    complete = run_sweep(
        [dict(case) for case in CASES],
        results,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        content_key_fn=case_content_key,
        progress=False,
        max_workers=0,
        observation=observation,
    )
    return complete, results


@pytest.fixture
def transports(monkeypatch) -> list[str]:
    calls: list[str] = []
    transport = runner._transport_case

    def counting(case, *args, **kwargs):
        calls.append(case["name"])
        return transport(case, *args, **kwargs)

    monkeypatch.setattr(runner, "_transport_case", counting)
    return calls


def _no_transport(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("this sweep must not transport")

    monkeypatch.setattr(runner, "_transport_case", refuse)


def test_sweep_produces_each_observation_on_its_scalar_transport(tmp_path, transports) -> None:
    observation = _sweep_observation(tmp_path)
    complete, results = _run(tmp_path, observation)

    assert complete
    assert transports == ["tilt30", "tilt45"]
    cas_root = tmp_path / "checkpoints"
    for case in CASES:
        key = case_content_key(case)
        (digest,) = observation.store.digests(key)
        stored = observation.store.load(digest)
        plan = observation.plan(case)
        separate = plan.assemble(runner.run_case_directions(case, plan.directions_sample), {})
        assert stored.digest == separate.digest
        np.testing.assert_array_equal(
            stored.spatial.line.intrinsic_by_tile, separate.spatial.line.intrinsic_by_tile
        )
        blob = _checkpoint_store.cas_load("hopg", key, str(cas_root))
        assert "directional" not in blob
        assert "directional" not in results[case["name"]][30.0]
    assert observation.store.verify() == ()


def test_rerun_reuses_records_and_observations_without_transport(tmp_path, monkeypatch) -> None:
    observation = _sweep_observation(tmp_path)
    _run(tmp_path, observation)
    _no_transport(monkeypatch)

    complete, _ = _run(tmp_path, observation)

    assert complete


def test_acquisition_change_rescores_stored_factors_without_transport(
    tmp_path, monkeypatch
) -> None:
    first = _sweep_observation(tmp_path)
    _run(tmp_path, first)
    _no_transport(monkeypatch)
    second = _sweep_observation(tmp_path, exposure_s=4.0, hit_threshold_eV=120.0)

    complete, _ = _run(tmp_path, second)

    assert complete
    for case in CASES:
        digests = second.store.digests(case_content_key(case))
        assert len(digests) == 2
        true_digests = {second.store.load(d).identity.true_spatial_digest for d in digests}
        assert len(true_digests) == 1
    assert len(list((second.store.path / "objects" / "true").glob("*.h5"))) == len(CASES)


def test_missing_observation_reruns_only_for_the_observation(tmp_path, transports) -> None:
    observation = _sweep_observation(tmp_path)
    _, before = _run(tmp_path, observation)
    checkpoint = tmp_path / "checkpoints" / "hopg"

    def _record_stamps():
        # cases.json is the per-run content-key pointer table, rewritten by
        # every cached run; the records themselves must stay untouched.
        return {
            path: path.stat().st_mtime_ns
            for path in checkpoint.rglob("*")
            if path.is_file() and path.name != "cases.json"
        }

    stamps = _record_stamps()
    for path in observation.store.path.rglob("*"):
        if path.is_file():
            path.unlink()
    transports.clear()

    complete, after = _run(tmp_path, _sweep_observation(tmp_path))

    assert complete
    assert transports == ["tilt30", "tilt45"]
    assert stamps == _record_stamps()
    for case in CASES:
        np.testing.assert_array_equal(
            after[case["name"]][30.0]["spec"], before[case["name"]][30.0]["spec"]
        )
        assert len(observation.store.digests(case_content_key(case))) == 1


def test_geometry_change_transports_again(tmp_path, transports) -> None:
    _run(tmp_path, _sweep_observation(tmp_path))
    transports.clear()
    moved = _sweep_observation(tmp_path)
    moved = replace(
        moved,
        observation=replace(moved.observation, scorer=pr.PixelScorer((1, 3))),
    )

    complete, _ = _run(tmp_path, moved)

    assert complete
    assert transports == ["tilt30", "tilt45"]


def test_sweeps_without_an_observation_write_no_observation_store(tmp_path) -> None:
    run_sweep(
        [dict(case) for case in CASES],
        {},
        checkpoint_dir=str(tmp_path / "checkpoints"),
        content_key_fn=case_content_key,
        progress=False,
        max_workers=0,
    )
    assert not os.path.exists(tmp_path / "observations")


@pytest.mark.parametrize("resolved", [False, True], ids=["scalar-profile", "counting-profile"])
def test_scan_places_the_observation_store_beside_the_checkpoint_root(
    tmp_path, monkeypatch, resolved
) -> None:
    from types import SimpleNamespace

    from pyrite.campaign import observation as campaign_observation
    from pyrite.runs import scan

    monkeypatch.setattr(
        campaign_observation,
        "resolve_profile_observation",
        lambda _catalog, _profile: _observation() if resolved else None,
    )
    args = SimpleNamespace(checkpoint_dir=str(tmp_path / "ckpts"))
    identity = {"catalog_profile": "standard", "resolved_parameters": {}}
    settings = SimpleNamespace(emission="both")

    built = scan._sweep_observation(args, identity, settings, "hopg-abc", case_content_key)

    if not resolved:
        assert built is None
        return
    assert built.store.path == (tmp_path / "observations" / "hopg-abc").resolve()
    assert built.emission == "both"
    assert built.provenance_fn(CASES[0])["backend"]
    assert built.provenance(CASES[0])["case"] == {
        "name": CASES[0]["name"],
        "E0_keV": float(CASES[0]["E0_keV"]),
    }
