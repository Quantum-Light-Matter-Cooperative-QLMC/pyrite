"""Diagnostic energy accounting includes terminal hard-event debits."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def score():
    script = Path(__file__).resolve().parents[2] / "checks/straggling_defaults/study.py"
    namespace = runpy.run_path(str(script))
    return namespace["primary_scores"]


def test_terminal_hard_losses_are_debited(score, monkeypatch):
    from pyrite.montecarlo.transport import events

    # Event-shape checks have their own tests; isolate the energy ledger here.
    monkeypatch.setattr(events, "check_segment_event_contract", lambda result: None)
    result = {
        "electron_id": np.array([0, 0, 1, 2]),
        "event_kind": np.array([8, 7, 7, 5]),
        "E_start_keV": np.array([30.0, 20.0, 30.0, 30.0]),
        "E_end_keV": np.array([25.0, 8.0, 5.5, 29.0]),
        "hard_W_keV": np.array([5.0, 3.0, 0.0, 0.0]),
        "hard_radiative_k_eV": np.array([0.0, 0.0, 1000.0, 0.0]),
        "L_ang": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    scores, error = score(result, 3, 30.0)
    assert error == 0
    np.testing.assert_array_equal(scores["final_keV"], [5, 4.5, 29])
    np.testing.assert_array_equal(scores["transmitted"], [0, 0, 1])
    np.testing.assert_array_equal(scores["path_ang"], [3, 3, 4])


def test_missing_primary_is_rejected(score, monkeypatch):
    from pyrite.montecarlo.transport import events

    monkeypatch.setattr(events, "check_segment_event_contract", lambda result: None)
    with pytest.raises(ValueError, match="incomplete"):
        score({"electron_id": np.array([0]), "event_kind": np.array([5])}, 2, 30.0)


def test_evidence_rejects_corruption_and_duplicate_seeds(tmp_path):
    script = Path(__file__).resolve().parents[2] / "checks/straggling_defaults/analyze.py"
    read_groups = runpy.run_path(str(script))["read_groups"]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"histories": 2, "seeds": {"off": [1, 2, 3]}}))
    raw = tmp_path / "1.npz"
    np.savez_compressed(raw, transmitted=[1.0, 0.0])
    record = dict(
        schema="pyrite.straggling-defaults.sample.v1",
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        raw_sha256="bad",
        straggling=False,
        seed=1,
        histories=2,
        material="silicon",
        range_fraction=0.8,
        core="cuda",
        max_de_frac=0.0,
        inelastic_cutoff_eV=50.0,
        tables={},
    )
    sample = tmp_path / "1.json"
    sample.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        read_groups(tmp_path, protocol)
    record["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    sample.write_text(json.dumps(record))
    _, groups = read_groups(tmp_path, protocol)
    assert len(groups) == 1
    (tmp_path / "2.npz").write_bytes(raw.read_bytes())
    (tmp_path / "2.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate seed"):
        read_groups(tmp_path, protocol)


@pytest.mark.parametrize(
    ("complete", "se", "delta", "expected"),
    [
        (False, 0.001, 0.0, "unresolved"),
        (True, 0.0, 0.0, "unresolved"),
        (True, 0.001, 0.05, "outside-diagnostic-margin"),
    ],
)
def test_incomplete_or_zero_variance_evidence_cannot_qualify(complete, se, delta, expected):
    script = Path(__file__).resolve().parents[2] / "checks/straggling_defaults/analyze.py"
    compare = runpy.run_path(str(script))["comparisons"]
    shared = dict(
        material="silicon",
        fraction=0.8,
        core="cuda",
        cap=0.0,
        cutoff_eV=50.0,
        tables={},
        complete=complete,
        transmitted_energy=None,
        warmed_transport_s={"mean": 1.0},
    )
    groups = [
        dict(
            shared,
            straggling=switch,
            observables={
                k: {"mean": mean, "se": se} for k in ["transmitted", "backscattered", "stopped"]
            },
        )
        for switch, mean in [("off", 0.1), ("on", 0.1 + delta)]
    ]
    protocol = {"diagnostic_margins": {"terminal_fraction_absolute": 0.02}}
    result = compare(groups, protocol)
    assert result[0]["differences"]["transmitted"]["outcome"] == expected
