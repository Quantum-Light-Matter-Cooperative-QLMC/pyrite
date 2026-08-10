from __future__ import annotations

import json
from datetime import timedelta

import pytest

from pyrite.checkpoints import campaign_lock
from pyrite.energy_grid import artifacts, gc


def _identity(material: str, stop: float = 100.0) -> dict:
    return artifacts.artifact_identity(
        material,
        [{"energy_keV": 30, "start_eV": 10, "stop_eV": 50, "num": 5}],
        {"start_eV": 0, "stop_eV": stop, "step_eV": 10},
        [30],
    )


def _catalog(path, *digests: str) -> None:
    refs = ", ".join(f'material_{index} = "{digest}"' for index, digest in enumerate(digests))
    path.write_text(f"[profiles.standard]\nenergy_grid_refs = {{ {refs} }}\n")


def _lock(path, digest: str, material: str = "material") -> None:
    campaign_lock.write_lock(
        path,
        profile="standard",
        material=material,
        dataset_identity={"digest": "case"},
        energy_grid_digest=digest,
    )


def _setup(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    catalog = tmp_path / "materials.toml"
    checkpoints = tmp_path / "checkpoints"
    store = tmp_path / "energy-grid-artifacts"
    catalog.write_text("[profiles.standard]\n")
    return catalog, checkpoints, store


def test_roots_include_profile_active_and_archive_locks(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    profile = artifacts.write_artifact(store, _identity("profile"))
    active = artifacts.write_artifact(store, _identity("active"))
    archived = artifacts.write_artifact(store, _identity("archived"))
    _catalog(catalog, profile.digest)
    _lock(checkpoints / "active", active.digest, "active")
    _lock(checkpoints / "archive" / "run" / "archived", archived.digest, "archived")

    assert gc.reachability_roots(catalog, checkpoints) == tuple(
        sorted((profile.digest, active.digest, archived.digest))
    )
    assert gc.plan_gc(catalog, checkpoints, store_root=store, clock=lambda: 100).candidates == ()


def test_roots_include_legacy_pickle_sibling_lock(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    locked = artifacts.write_artifact(store, _identity("legacy"))
    _lock(checkpoints / "legacy.pkl", locked.digest, "legacy")

    assert gc.reachability_roots(catalog, checkpoints) == (locked.digest,)


def test_first_orphan_is_retained_then_grace_and_prune_all_select_it(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    orphan = artifacts.write_artifact(store, _identity("orphan"))

    first = gc.plan_gc(catalog, checkpoints, store_root=store, clock=lambda: 100)
    assert first.candidates == ()
    assert first.retained == (orphan.digest,)
    assert (
        gc.plan_gc(
            catalog, checkpoints, store_root=store, clock=lambda: 100 + 13 * 86400
        ).candidates
        == ()
    )
    aged = gc.plan_gc(
        catalog,
        checkpoints,
        store_root=store,
        grace=timedelta(days=14),
        clock=lambda: 100 + 14 * 86400,
    )
    assert [candidate.digest for candidate in aged.candidates] == [orphan.digest]

    fresh_catalog, fresh_checkpoints, fresh_store = _setup(tmp_path / "fresh")
    immediate = artifacts.write_artifact(fresh_store, _identity("immediate"))
    forced = gc.plan_gc(
        fresh_catalog,
        fresh_checkpoints,
        store_root=fresh_store,
        prune_all=True,
        clock=lambda: 1,
    )
    assert [candidate.digest for candidate in forced.candidates] == [immediate.digest]


def test_reachable_artifact_clears_existing_orphan_state(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    artifact = artifacts.write_artifact(store, _identity("reachable"))
    gc.plan_gc(catalog, checkpoints, store_root=store, clock=lambda: 100)
    _catalog(catalog, artifact.digest)

    plan = gc.plan_gc(catalog, checkpoints, store_root=store, clock=lambda: 101)
    assert plan.retained == (artifact.digest,)
    assert json.loads(gc.metadata_path(store).read_text()) == {
        "schema": gc.METADATA_SCHEMA,
        "orphans": {},
    }


def test_corrupt_lock_fails_closed(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    artifacts.write_artifact(store, _identity("orphan"))
    lock = checkpoints / "cxr.lock.json"
    lock.parent.mkdir()
    lock.write_text("not-json")

    with pytest.raises(gc.ArtifactGCError, match="invalid campaign lock"):
        gc.plan_gc(catalog, checkpoints, store_root=store)


def test_execute_fails_closed_when_catalog_or_metadata_changes(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    artifact = artifacts.write_artifact(store, _identity("orphan"))
    gc.plan_gc(catalog, checkpoints, store_root=store, clock=lambda: 0)
    plan = gc.plan_gc(catalog, checkpoints, store_root=store, prune_all=True, clock=lambda: 1)
    _catalog(catalog, artifact.digest)

    with pytest.raises(gc.ArtifactGCError, match="changed after GC preview"):
        gc.execute_gc(plan)
    assert artifact.path.exists()

    _catalog(catalog)
    plan = gc.plan_gc(catalog, checkpoints, store_root=store, prune_all=True, clock=lambda: 2)
    gc.metadata_path(store).write_text('{"schema":"cxr.energy-grid-gc-metadata.v1","orphans":{}}\n')
    with pytest.raises(gc.ArtifactGCError, match="metadata changed"):
        gc.execute_gc(plan)
    assert artifact.path.exists()


def test_execute_deletes_only_explicit_revalidated_candidates(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    orphan = artifacts.write_artifact(store, _identity("orphan"))
    plan = gc.plan_gc(catalog, checkpoints, store_root=store, prune_all=True, clock=lambda: 0)

    assert gc.execute_gc(plan) == (orphan.path,)
    assert not orphan.path.exists()


def test_verify_reports_missing_referenced_and_corrupt_stored_artifacts(tmp_path):
    catalog, checkpoints, store = _setup(tmp_path)
    missing = artifacts.artifact_digest(_identity("missing"))
    corrupt = artifacts.write_artifact(store, _identity("corrupt"))
    corrupt.path.write_bytes(b"corrupt")
    _catalog(catalog, missing)

    report = gc.verify_artifacts(catalog, checkpoints, store_root=store)
    assert not report.ok
    assert {(issue.digest, issue.source) for issue in report.issues} == {
        (missing, "catalog"),
        (corrupt.digest, "store"),
    }
