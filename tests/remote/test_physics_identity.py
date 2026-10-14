"""Staged resolution must agree before accepting checkpoint/CAS reuse (#485)."""

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from pyrite.remote import _queue_scripts as scripts
from pyrite.remote import physics_identity as parity


def test_inventory_uses_dataset_union_without_lowering_cases_or_execution(monkeypatch):
    from pyrite import api, materials
    from pyrite.runs import scan

    monkeypatch.setattr(
        materials,
        "CATALOG",
        SimpleNamespace(profile_detector_set=lambda profile: {"a": {}, "b": {}}),
    )
    calls = []

    def resolve(args, material):
        calls.append((material, args.detector_id, args.quick, args.high_energy_floor_map))
        return (
            None,
            args.detector_id,
            {
                "parameter_sha256": "dataset-" + args.detector_id,
                "resolved_parameters": {"xsgen_tables": {"union": "digest"}},
            },
            "stem",
        )

    monkeypatch.setattr(scan, "_resolved_run", resolve)
    monkeypatch.setattr(
        api,
        "build_configured_cases",
        lambda *args: pytest.fail("preflight must not lower a complete sweep"),
    )
    expected = parity.expected_inventory(["hopg"], True, "full", "profile")
    assert set(expected["inventory"]) == {"hopg/a", "hopg/b"}
    assert all(
        row["table_markers"] == {"union": "digest"} for row in expected["inventory"].values()
    )
    assert (
        expected["inventory"]["hopg/a"]["dataset_digest"]
        != (expected["inventory"]["hopg/b"]["dataset_digest"])
    )
    assert calls == [("hopg", "a", True, {}), ("hopg", "b", True, {})]
    parity.verify_inventory(json.loads(json.dumps(expected)))


@pytest.mark.parametrize("changed", ["dataset_digest", "table_markers", "stem"])
def test_host_mismatch_is_refused_with_material_and_fix(monkeypatch, changed):
    expected = {
        "request": {},
        "inventory": {
            "hopg/default": {
                "dataset_digest": "local",
                "table_markers": {"union": "local"},
                "stem": "hopg",
            }
        },
    }
    current = copy.deepcopy(expected["inventory"])
    current["hopg/default"][changed] = "different"
    monkeypatch.setattr(parity, "resolve_inventory", lambda request: current)
    with pytest.raises(ValueError, match="hopg/default.*sync.*resubmit"):
        parity.verify_inventory(expected)


@pytest.mark.parametrize("chunked", [False, True])
def test_job_script_verifies_staged_identity_before_runs_on_every_allocation(chunked):
    expected = {"request": {"catalog_profile": "quote'profile"}, "inventory": {}}
    script = (
        scripts._chunked_queue_script("j", ["hopg"], False, None, 10, expected_physics=expected)
        if chunked
        else scripts._queue_script("j", ["hopg"], False, None, expected_physics=expected)
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    verify = script.index("python -m pyrite.remote.physics_identity")
    assert script.index("pyrite tables verify") < verify < script.index("mats=(")
    assert "FAILED (physics identity)" in script
    assert ".physics_ok" not in script


def test_submission_embeds_expected_inventory_in_staged_script(monkeypatch):
    from pyrite.remote import queue, transport

    expected = {"request": {}, "inventory": {"hopg/default": {"dataset_digest": "same"}}}
    monkeypatch.setattr(parity, "expected_inventory", lambda *args: expected)
    monkeypatch.setattr(transport, "_check_materials", lambda materials: None)
    monkeypatch.setattr(queue, "_refuse_if_busy", lambda *args: None)
    monkeypatch.setattr(scripts, "_stems", lambda *args: ["hopg"])
    staged = []
    monkeypatch.setattr(queue, "_stage_job_script", lambda *args: staged.append(args[-1]))
    monkeypatch.setattr(queue, "_submit_staged_job", lambda *args, **kwargs: "123")
    queue.start_queue(["hopg"], no_sync=True)
    assert len(staged) == 1
    assert "python -m pyrite.remote.physics_identity" in staged[0]
    assert '"dataset_digest":"same"' in staged[0]


def test_staged_table_digest_change_is_detected_by_real_scan_resolution(monkeypatch):
    from pyrite.xsgen import store

    expected = parity.expected_inventory(["hopg"], True, "full", "standard")
    parity.verify_inventory(expected)
    identity_markers = store.identity_markers

    def changed(tables):
        markers = identity_markers(tables)
        return {key: "f" * 64 for key in markers}

    monkeypatch.setattr(store, "identity_markers", changed)
    with pytest.raises(ValueError, match="hopg/"):
        parity.verify_inventory(expected)
