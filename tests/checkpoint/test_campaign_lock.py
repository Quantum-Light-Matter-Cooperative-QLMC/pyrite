import json

import pytest
from click.testing import CliRunner

from pyrite.checkpoints import campaign_lock
from pyrite.materials import CATALOG
from pyrite.runs import scan


def _identity():
    return {
        "schema": "cxr.dataset-identity.v1",
        "catalog_profile": "campaign",
        "parameter_sha256": "b" * 64,
        "parameters": {"energy_keV": [30.0, 100.0]},
    }


def test_completed_campaign_lock_is_deterministic_and_readable(tmp_path):
    digest = "a" * 64
    checkpoint = tmp_path / "hopg--full-deadbeef"

    path = campaign_lock.write_lock(
        checkpoint,
        profile="campaign",
        material="hopg",
        dataset_identity=_identity(),
        energy_grid_digest=digest,
    )
    first = path.read_bytes()
    campaign_lock.write_lock(
        checkpoint,
        profile="campaign",
        material="hopg",
        dataset_identity=_identity(),
        energy_grid_digest=digest,
    )

    assert path == checkpoint / "cxr.lock.json"
    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert campaign_lock.read_lock(path)["artifacts"]["energy_grid"] == {"hopg": digest}


def test_legacy_run_lock_is_explicit_and_legacy_pickle_path_is_stable(tmp_path):
    checkpoint = tmp_path / "hopg.pkl"
    path = campaign_lock.write_lock(
        checkpoint,
        profile="standard",
        material="hopg",
        dataset_identity=_identity(),
        energy_grid_digest=None,
    )

    payload = json.loads(path.read_text())
    assert path == tmp_path / "hopg.lock.json"
    assert payload["legacy_energy_grid"] is True
    assert payload["artifacts"]["energy_grid"] == {}


def _fake_cases():
    return [
        {
            "name": "cfg0",
            "E0_keV": 30,
            "crystal": "hopg",
            "thickness_ang": 1e4,
            "tilt_deg": 0.0,
            "beam_uvw": (0, 0, 1),
            "hkl_list": [(0, 0, 2)],
        }
    ]


def _run_scan(monkeypatch, tmp_path, *, ref):
    """Drive one completed ``cxr run`` with a stubbed sweep and profile ref."""
    captured = {}

    def _run_sweep(*args, **kwargs):
        captured["dataset_identity"] = kwargs["dataset_identity"]
        return True

    monkeypatch.setattr(scan, "run_sweep", _run_sweep)
    monkeypatch.setattr(scan, "build_cases", lambda *a, **kw: _fake_cases())
    monkeypatch.setattr(scan, "gate_cases_by_penetration", lambda cases, **kw: (cases, []))
    monkeypatch.setattr(type(CATALOG), "profile_energy_grid_ref", lambda self, name, material: ref)
    result = CliRunner().invoke(
        scan.command,
        ["standard", "-m", "hopg", "--checkpoint-dir", str(tmp_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    locks = list(tmp_path.rglob("cxr.lock.json"))
    assert len(locks) == 1
    return campaign_lock.read_lock(locks[0]), captured["dataset_identity"]


def test_completed_run_lock_records_dataset_identity_and_artifact_digest(monkeypatch, tmp_path):
    digest = "c" * 64

    payload, identity = _run_scan(monkeypatch, tmp_path, ref=digest)

    assert payload["profile"] == identity["catalog_profile"]
    assert payload["material"] == "hopg"
    assert payload["artifacts"]["energy_grid"] == {"hopg": digest}
    assert payload["legacy_energy_grid"] is False
    # The lock stores the same identity the sweep ran under, JSON-normalized.
    assert payload["dataset_identity"] == json.loads(json.dumps(identity))


def test_completed_run_lock_marks_legacy_grid_when_profile_has_no_ref(monkeypatch, tmp_path):
    payload, identity = _run_scan(monkeypatch, tmp_path, ref=None)

    assert payload["legacy_energy_grid"] is True
    assert payload["artifacts"]["energy_grid"] == {}
    assert payload["dataset_identity"] == json.loads(json.dumps(identity))


def test_campaign_lock_rejects_invalid_digest_and_schema(tmp_path):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        campaign_lock.lock_payload(
            profile="standard",
            material="hopg",
            dataset_identity=_identity(),
            energy_grid_digest="BAD",
        )

    path = tmp_path / "cxr.lock.json"
    path.write_text('{"schema":"wrong"}\n')
    with pytest.raises(ValueError, match="expected schema"):
        campaign_lock.read_lock(path)
