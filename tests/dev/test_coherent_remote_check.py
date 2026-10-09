"""Remote #350 checks preserve incident counts and one transport across slices."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from checks import coherent_physical_population_remote as check


@pytest.mark.parametrize("mode, samples", [("ladder", 40), ("smoke", 200)])
def test_remote_case_overrides_profile_sample_counts(mode, samples):
    case = check._case(SimpleNamespace(mode=mode, energy=30.0, max_points=20000000, charge_pc=1.0))
    assert case["Ne"] == samples and case["Ne_brem"] == samples
    assert case["bunch_charge_pc"] == 1.0
    assert case["line_grid_policy"]["resolution"]["max_points"] == 20000000


def test_scheduler_slice_reuses_persisted_transport(tmp_path, monkeypatch):
    import json

    from pyrite._backend import BACKEND, xp
    from pyrite.energy_grid import convergence, convergence_case
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.coherent_population import (
        CoherentSamplingError,
        require_resolved_power,
    )

    args = SimpleNamespace(
        mode="ladder",
        energy=30.0,
        charge_pc=1.0,
        max_points=20000000,
        reference_points=40000000,
        reference_divisor=3.0,
        max_minutes=1.0,
        out=str(tmp_path / "result.json"),
    )
    monkeypatch.setattr(check, "_stamp", lambda: {"code_digest": "one-code"})
    monkeypatch.setattr(check, "_require_gpu", lambda: None)
    monkeypatch.setattr(check, "_gpu_limits", lambda: {"fixture": True})
    monkeypatch.setattr(check, "_case", lambda args: {"Ne": 2, "bunch_charge_pc": 1.0})
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        return {
            "Ne_lines": 2,
            "E_grid": np.array([900.0, 901.0, 902.0]),
            "segs": {"L_ang": np.array([1.0])},
            "diagnostic_grid": {
                "window_plan": {
                    "start_eV": 900.0,
                    "stop_eV": 902.0,
                    "backbone_spacing_eV": 0.5,
                    "seeds": [],
                },
                "coherent_windows": {
                    "rows": [{"points": 3, "step_electron_eV": 0.7, "step_all_eV": 0.7}]
                },
            },
        }

    evaluated_grids = []

    class Ladder:
        fail_sampling = False

        def __init__(self, case, *, transport, coherent):
            np.testing.assert_array_equal(transport["segs"]["L_ang"], [1.0])
            self.fingerprint = {"n_segments": 1, "digest": "one-transport"}

        def lines(self, energy):
            if self.fail_sampling:
                require_resolved_power(xp.asarray([-1.0, np.nan, -np.inf]))
            evaluated_grids.append(energy.copy())
            return np.ones_like(energy)

    monkeypatch.setattr(runner, "_transport_case", transport)
    monkeypatch.setattr(convergence_case, "CaseLadder", Ladder)
    monkeypatch.setattr(
        convergence,
        "spectrum_observables",
        lambda *args, **kwargs: {"yield": 2.0, "centroid_eV": 901.0, "fwhm_eV": 1.0},
    )
    monkeypatch.setattr(BACKEND, "release_memory", lambda: None)
    args.max_minutes = -1.0  # Force a scheduler handoff after the first completed rung.
    assert check.run(args) == 75
    before = json.loads((tmp_path / "result.json").read_text())
    assert set(before["evals"]) == {"auto"}
    args.max_minutes = 1000.0
    monkeypatch.setattr(
        convergence,
        "spectrum_observables",
        lambda *args, **kwargs: {"yield": 2.0, "centroid_eV": 901.0, "fwhm_eV": float("nan")},
    )
    with pytest.raises(ArithmeticError, match="unresolved finer observables"):
        check.run(args)
    failed = json.loads((tmp_path / "result.json").read_text())
    assert failed["state"] == "failed"
    assert set(failed["evals"]) == {"auto"}
    monkeypatch.setattr(
        convergence,
        "spectrum_observables",
        lambda *args, **kwargs: {"yield": 2.0, "centroid_eV": 901.0, "fwhm_eV": 1.0},
    )
    Ladder.fail_sampling = True
    with pytest.raises(CoherentSamplingError):
        check.run(args)
    sampling_failure = json.loads((tmp_path / "result.json").read_text())
    assert sampling_failure["state"] == "failed"
    assert sampling_failure["sampling_diagnostics"]["negative_count"] == 1
    assert sampling_failure["sampling_diagnostics"]["nonfinite_count"] == 2
    assert sampling_failure["sampling_diagnostics"]["minimum_finite_raw_power"] == -1.0
    assert set(sampling_failure["evals"]) == {"auto"}
    Ladder.fail_sampling = False
    assert check.run(args) == 0
    after = json.loads((tmp_path / "result.json").read_text())
    assert set(after["evals"]) == {"auto", "finer", "reference"}
    for key, value in before["evals"]["auto"].items():
        assert after["evals"]["auto"][key] == value
    assert len(calls) == 1
    for grid in evaluated_grids:
        assert grid[0] == 900.0 and grid[-1] == 902.0
    assert np.max(np.diff(evaluated_grids[-1])) <= 0.7 / args.reference_divisor
    assert after["state"] == "done"
    assert "sampling_diagnostics" not in after and "error" not in after
    monkeypatch.setattr(check, "_stamp", lambda: {"code_digest": "changed-code"})
    with pytest.raises(RuntimeError, match="changed inputs, code or table digest"):
        check.run(args)


def test_unfinished_slice_checks_backend_before_using_cached_gpu_limits(tmp_path, monkeypatch):
    import json

    output = tmp_path / "result.json"
    args = SimpleNamespace(out=str(output), max_minutes=10.0)
    check._atomic(
        output,
        {
            "config": {},
            "stamp": {"code_digest": "same-code"},
            "gpu_limits": {"previous_cuda_success": True},
            "state": "between-rungs",
        },
    )
    monkeypatch.setattr(check, "_stamp", lambda: {"code_digest": "same-code"})

    def reject_backend():
        raise RuntimeError("requires CUDA float64")

    monkeypatch.setattr(check, "_require_gpu", reject_backend)
    with pytest.raises(RuntimeError, match="requires CUDA float64"):
        check.run(args)
    assert json.loads(output.read_text())["state"] == "failed"


@pytest.fixture
def replay_source(tmp_path):
    from pyrite.energy_grid.convergence import segment_fingerprint

    path = tmp_path / "prior.json"
    config = {"charge_pc": 1.0}
    stamp = {"code_digest": "old-code", "code_tables_digest": "same-tables"}
    case = {"Ne": 2, "bunch_charge_pc": 1.0}
    transport = {
        "Ne_lines": 2,
        "E_grid": np.array([900.0, 901.0, 902.0]),
        "segs": {"L_ang": np.array([1.0])},
    }
    record = {
        "config": config,
        "stamp": stamp,
        "state": "failed",
        "incident_samples": 2,
        "fingerprint": segment_fingerprint(transport["segs"]),
        "auto_record": {"num": 3, "start_eV": 900.0, "stop_eV": 902.0},
    }
    saved = {
        "config": config,
        "digest": "old-code",
        "tables_digest": "same-tables",
        "case": case,
        "transport": transport,
    }
    check._atomic(path, record)
    check._atomic(path.with_suffix(".transport.pkl"), saved, binary=True)
    return path, record, saved


def test_replay_import_preserves_source_and_records_both_code_generations(replay_source):
    path, record, saved = replay_source
    original = path.read_bytes(), path.with_suffix(".transport.pkl").read_bytes()
    case, transport, provenance = check._replay_transport(
        path, record["config"], {"code_digest": "new-code", "code_tables_digest": "same-tables"}
    )
    assert case == saved["case"]
    np.testing.assert_array_equal(transport["E_grid"], saved["transport"]["E_grid"])
    assert provenance["stamp"]["code_digest"] == "old-code"
    assert len(provenance["snapshot_sha256"]) == len(provenance["record_sha256"]) == 64
    assert original == (path.read_bytes(), path.with_suffix(".transport.pkl").read_bytes())


@pytest.mark.parametrize("change", ["inputs", "tables", "code", "segments", "population", "axis"])
def test_replay_refuses_mismatched_source_evidence(replay_source, change):
    path, record, saved = replay_source
    config = dict(record["config"])
    stamp = {"code_digest": "new-code", "code_tables_digest": "same-tables"}
    if change == "inputs":
        config["charge_pc"] = 2.0
    elif change == "tables":
        stamp["code_tables_digest"] = "other-tables"
    elif change == "code":
        saved["digest"] = "unrecorded-code"
    elif change == "segments":
        saved["transport"]["segs"]["L_ang"][0] = 2.0
    elif change == "population":
        saved["transport"]["Ne_lines"] = 3
    else:
        saved["transport"]["E_grid"][0] = 899.0
    check._atomic(path.with_suffix(".transport.pkl"), saved, binary=True)
    with pytest.raises((RuntimeError, ValueError)):
        check._replay_transport(path, config, stamp)


def test_smoke_replay_never_generates_transport_and_retains_source_on_resume(
    replay_source, monkeypatch
):
    import json

    from pyrite.energy_grid import convergence_case
    from pyrite.montecarlo import runner

    path, record, saved = replay_source
    args = SimpleNamespace(
        mode="smoke",
        charge_pc=1.0,
        max_minutes=10.0,
        out=str(path.with_name("replayed.json")),
        transport_record=str(path),
    )
    record["config"]["mode"] = saved["config"]["mode"] = "smoke"
    check._atomic(path, record)
    check._atomic(path.with_suffix(".transport.pkl"), saved, binary=True)
    monkeypatch.setattr(
        check,
        "_stamp",
        lambda: {
            "code_digest": "new-code",
            "code_tables_digest": "same-tables",
        },
    )
    monkeypatch.setattr(check, "_require_gpu", lambda: None)
    monkeypatch.setattr(check, "_gpu_limits", lambda: {})

    def reject_new_transport(*args, **kwargs):
        raise AssertionError("replay must never generate new transport")

    monkeypatch.setattr(runner, "_transport_case", reject_new_transport)
    monkeypatch.setattr(check, "_case", reject_new_transport)
    monkeypatch.setattr(
        convergence_case,
        "CaseLadder",
        lambda *args, **kwargs: SimpleNamespace(
            fingerprint=record["fingerprint"],
        ),
    )
    monkeypatch.setattr(
        runner,
        "_spectrum_case",
        lambda case, transport: {
            "E_grid": transport["E_grid"],
            "spec": np.ones(3),
            "spec_coherent": np.ones(3),
        },
    )
    assert check.run(args) == 0
    result = json.loads(Path(args.out).read_text())
    assert result["stamp"]["code_digest"] == "new-code"
    assert result["transport_source"]["stamp"]["code_digest"] == "old-code"
    assert check.run(args) == 0
    args.transport_record = None
    with pytest.raises(RuntimeError, match="requires --transport-record"):
        check.run(args)
    args.transport_record = str(path)
    # The same stem with another suffix would overwrite the source snapshot.
    args.out = str(path.with_suffix(".txt"))
    with pytest.raises(RuntimeError, match="source record and snapshot"):
        check.run(args)
    args.out = str(path.with_name("replayed.json"))
    record["extra"] = "changed source"
    check._atomic(path, record)
    with pytest.raises(RuntimeError, match="source changed"):
        check.run(args)
