"""Remote #350 checks preserve incident counts and one transport across slices."""

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
