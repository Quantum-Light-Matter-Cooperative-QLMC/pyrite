"""CPU-only contract tests for the scheduled hardware/ledger probes (#344)."""

from pathlib import Path

import numpy as np
import pytest


def _script(name):
    import importlib.util

    path = Path(__file__).parents[2] / "checks" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_metrics_reject_mismatch_and_zero_reference():
    parity = _script("scheduled_cuda_parity.py")
    base = np.array([1.0, 2.0, 3.0])
    assert parity.compare(base, base + 1e-6, bound=1e-5)["pass"]
    assert not parity.compare(base, base * 1.04, bound=0.03)["pass"]
    with pytest.raises(ValueError, match="reference spectrum"):
        parity.compare(np.zeros(3), np.zeros(3), bound=1e-5)
    with pytest.raises(ValueError, match="nonfinite"):
        parity.compare(base, np.array([1.0, np.nan, 3.0]), bound=1e-5)


def test_parity_case_seeds_are_distinct():
    parity = _script("scheduled_cuda_parity.py")
    assert len({case[0] for case in parity.CASES}) == 3
    assert parity.CHANNELS == ("line", "brem", "characteristic", "coherent")


def test_validation_refresh_rejects_unregistered_checks():
    refresh = _script("refresh_validation_evidence.py")
    with pytest.raises(ValueError, match="not a mapped check"):
        refresh.selected_checks(["../../tmp/evil.py"], all_mapped=False)
    checks = refresh.selected_checks(["checks/brem_angular_comparison.py"], all_mapped=False)
    assert checks == list(refresh.DEFAULT_CPU_CHECKS)
    assert "checks/kinematic_validity_check.py" not in checks


@pytest.mark.parametrize("state", ["idle", "busy", "unavailable"])
def test_cuda_scheduler_submission_and_fail_closed(tmp_path, state):
    import os
    import shutil
    import subprocess

    root = tmp_path / "checkout"
    (root / "checks").mkdir(parents=True)
    script = root / "checks" / "schedule_cuda_parity.sh"
    shutil.copy2(Path(__file__).parents[2] / "checks" / script.name, script)
    commands = tmp_path / "bin"
    commands.mkdir()
    queue = commands / "squeue"
    queue.write_text(
        "#!/bin/bash\n"
        + {"idle": "exit 0\n", "busy": "echo 123\n", "unavailable": "exit 1\n"}[state]
    )
    submit = commands / "sbatch"
    submit.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > submitted.args\necho 456\n')
    queue.chmod(0o755)
    submit.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    submissions = root / "evidence" / "validation-344" / "submissions.tsv"
    assert result.returncode == (1 if state == "unavailable" else 0)
    assert submissions.exists() == (state == "idle")
    assert (root / "submitted.args").exists() == (state == "idle")
    if state == "idle":
        assert submissions.read_text().endswith("\t456\n")
        assert "--dependency=singleton" in (root / "submitted.args").read_text()


def test_exhaustive_diagnostics_cannot_be_published(tmp_path, capsys):
    refresh = _script("refresh_validation_evidence.py")
    output = tmp_path / "evidence"
    with pytest.raises(SystemExit) as error:
        refresh.main(["--output-dir", str(output), "--all-mapped", "--publish"])
    assert error.value.code == 2
    assert "without scientific failure gates" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("publish", [False, True])
def test_refresh_refuses_diagnostic_recipe_before_execution(tmp_path, monkeypatch, publish):
    refresh = _script("refresh_validation_evidence.py")

    def unexpected_run(*args, **kwargs):
        pytest.fail("diagnostic producer was executed")

    monkeypatch.setattr(refresh.subprocess, "run", unexpected_run)
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="no reviewed CPU recipe"):
        refresh.refresh(["checks/kinematic_validity_check.py"], output, publish=publish)
    assert not output.exists()


def test_refresh_records_exact_reviewed_invocation(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from pyrite.validation.check_records import records_for_exit, write_records

    refresh = _script("refresh_validation_evidence.py")
    check = "checks/brem_source_comparison.py"

    def collect(command, **kwargs):
        # Emulate a missing reference, preserving the collector's real JSONL contract.
        assert command[-2:] == [check, "--download"]
        output = Path(command[command.index("--output-dir") + 1])
        write_records(output / "brem_source_comparison.jsonl", records_for_exit(check, 2, "abc"))
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(refresh.subprocess, "run", collect)
    monkeypatch.setattr(refresh, "current_revision", lambda: "abc")
    output = tmp_path / "evidence"
    assert refresh.refresh([check], output) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checks"][0]["verdict"] == "skip"
    assert manifest["commands"][0]["args"] == ["--download"]
    assert manifest["commands"][0]["elapsed_seconds"] >= 0


def test_refresh_requires_ledger_inputs_before_running_producers(tmp_path, monkeypatch):
    refresh = _script("refresh_validation_evidence.py")

    def unexpected_run(*args, **kwargs):
        pytest.fail("producer ran before ledger preflight")

    monkeypatch.setattr(refresh.subprocess, "run", unexpected_run)
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="ledger input is required"):
        refresh.refresh(["checks/brem_source_comparison.py"], output, repo=tmp_path)
    assert not output.exists()


def test_optional_oracles_report_skip_when_dependencies_are_missing(monkeypatch):
    import builtins

    dans = _script("dans_diffraction_oracle.py")

    def unavailable(*args, **kwargs):
        raise dans.DansDiffractionUnavailableError("optional dependency missing")

    monkeypatch.setattr(dans, "validate_dans_crystal", unavailable)
    assert dans.main() == 2

    cascade = _script("xraylib_cascade_oracle.py")
    original_import = builtins.__import__

    def missing_xraylib(name, *args, **kwargs):
        if name == "xraylib":
            raise ImportError("optional dependency missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_xraylib)
    assert cascade.main() == 2


def test_cascade_oracle_cannot_pass_without_gated_reference_rows(monkeypatch):
    import sys
    from types import SimpleNamespace

    cascade = _script("xraylib_cascade_oracle.py")

    def undefined(*args):
        raise ValueError("reference is undefined")

    oracle = SimpleNamespace(
        SymbolToAtomicNumber=lambda _: 14,
        CS_Photo_Partial=undefined,
        CS_FluorLine_Kissel_Cascade=undefined,
        **{f"{name}_SHELL": i for i, name in enumerate(cascade.SHELLS)},
        **{f"{name}_LINE": i for i, name in enumerate(cascade.LINES)},
    )
    monkeypatch.setitem(sys.modules, "xraylib", oracle)
    monkeypatch.setattr(cascade, "_pyrite_cascade", lambda *args: ({}, {}))
    assert cascade.main() == 1


def test_quantum_budget_rejects_nonfinite_rates(monkeypatch):
    budget = _script("quantum_decoherence_budget.py")
    monkeypatch.setattr(budget, "planar_profile", lambda *args: object())
    monkeypatch.setattr(budget, "UNAVAILABLE_CELLS", ())
    monkeypatch.setattr(budget, "_print_row", lambda *args: None)
    monkeypatch.setattr(
        budget,
        "_row",
        lambda label, profile, energy, electronic: {
            "label": label,
            "energy_eV": energy,
            "tds": np.full(11, np.nan),
            "electronic": np.zeros(11),
            "beta": 1.0,
        },
    )
    assert budget.main([]) == 1


def test_quantum_budget_material_table_matches_catalog():
    from pyrite.materials.channeling import ChannelingDataError, planar_profile

    budget = _script("quantum_decoherence_budget.py")
    # MoS2 is computed for TDS only; its electronic rate stays unsupported (#541).
    assert [(c[1], c[3]) for c in budget.CELLS] == [
        ("silicon", True),
        ("diamond", True),
        ("mos2", False),
    ]
    for _, crystal, hkl, _ in budget.CELLS:
        planar_profile(crystal, hkl, 293.15)
    for crystal, hkl in budget.UNAVAILABLE_CELLS:
        with pytest.raises(ChannelingDataError):
            planar_profile(crystal, hkl, 293.15)


def _cheap_budget(monkeypatch, budget, profile):
    monkeypatch.setattr(budget, "planar_profile", profile)
    monkeypatch.setattr(budget, "_print_row", lambda *args: None)
    rate = np.array([0.0, 1.0, 1.0])
    monkeypatch.setattr(
        budget,
        "_row",
        lambda label, profile, energy, electronic: {
            "label": label,
            "energy_eV": energy,
            "tds": rate,
            "electronic": rate if electronic else None,
            "beta": 1.0,
        },
    )


def test_quantum_budget_flags_material_coverage_changes(monkeypatch, capsys):
    from pyrite.materials.channeling import ChannelingDataError

    budget = _script("quantum_decoherence_budget.py")
    _cheap_budget(monkeypatch, budget, lambda *args: object())
    assert budget.main([]) == 1
    assert "hopg unexpectedly has channeling data" in capsys.readouterr().out

    def lose_mos2(crystal, *args):
        if crystal in {"mos2", "hopg", "hbn", "wse2"}:
            raise ChannelingDataError(crystal)
        return object()

    _cheap_budget(monkeypatch, budget, lose_mos2)
    assert budget.main([]) == 1
    assert "MoS2(001) lost its channeling data" in capsys.readouterr().out


def test_quantum_budget_reports_mos2_outside_its_thermal_window(monkeypatch, capsys):
    from pyrite.materials.channeling import ChannelingDataError

    budget = _script("quantum_decoherence_budget.py")
    assert budget._outside_thermal_window("mos2", 293.15) == ""
    assert budget._outside_thermal_window("silicon", 77.0) == ""

    def computable(crystal, *args):
        if crystal in {"hopg", "hbn", "wse2"}:
            raise ChannelingDataError(crystal)
        return object()

    _cheap_budget(monkeypatch, budget, computable)
    assert budget.main(["--temperature-K", "77"]) == 0
    assert "MoS2(001): not computable at 77 K (Mo thermal row holds only at 295 +- 10 K)" in (
        capsys.readouterr().out
    )


def test_quantum_budget_prints_tds_only_rows_as_bounds(capsys):
    budget = _script("quantum_decoherence_budget.py")
    nan = float("nan")
    row = {
        "label": "MoS2(001)",
        "energy_eV": 3.0e4,
        "deltas": np.zeros(2),
        "tds": np.zeros(2),
        "electronic": None,
        "n_bound": 3,
        "l_cr_ang": nan,
        "l_pxr_ang": 1.0,
        "ell_half_ang": 2.0,
        "ell_one_ang": 3.0,
        "diffusion_inv_ang3": 4.0,
        "gamma_l_cr": nan,
    }
    budget._print_row(row)
    out = capsys.readouterr().out
    assert "G_el[1/A]  unsupported" in out
    assert "1/G(d/2) <= 2 A" in out and "2G/D^2 >= 4 1/A^3" in out


def test_channeling_overlap_rejects_nonfinite_density():
    from types import SimpleNamespace

    emission = _script("channeling_modulated_emission.py")
    solver = SimpleNamespace(
        basis=SimpleNamespace(vectors=np.ones((1, 1, 1))),
        orders=np.array([0]),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        emission.overlap_factor(solver, np.full((1, 1, 1), np.nan), np.ones(1))


def test_channeling_limits_skip_when_comb_gate_is_never_exercised(monkeypatch):
    from types import SimpleNamespace

    emission = _script("channeling_modulated_emission.py")
    # Exercise verdict control flow without running the expensive density solver.
    profile = SimpleNamespace(
        period_ang=1.0,
        nuclear_density=np.ones(1),
        electron_density=np.ones(1),
        potential=lambda x: np.cos(2 * np.pi * x),
        nuclear=np.ones_like,
        extrema=lambda: (0.0, 1.0),
    )
    solver = SimpleNamespace(entry_state=lambda *args: (np.ones((1, 1, 1)), np.zeros(1), 0.0))
    monkeypatch.setattr(emission, "planar_profile", lambda *args: profile)
    monkeypatch.setattr(emission, "tds_kernel", lambda *args, **kwargs: 0)
    monkeypatch.setattr(emission, "electronic_kernel", lambda *args, **kwargs: 0)
    monkeypatch.setattr(emission, "DensityMatrixSolver", lambda *args: solver)
    monkeypatch.setattr(emission, "comb_state", lambda *args: (np.full((1, 1, 1), 0.9), 0.1))
    monkeypatch.setattr(emission, "overlap_factor", lambda *args: (1.0, 1.0))
    monkeypatch.setattr(
        emission,
        "depth_profile",
        lambda *args: (np.array([0.0, 1000.0, 2000.0]), np.ones(3), np.ones(3)),
    )
    assert emission.main(["--energies-eV", "300000"]) == 2


def test_channeling_yield_rejects_nonfinite_profile():
    emission = _script("channeling_modulated_emission.py")
    with pytest.raises(ValueError, match="nonfinite"):
        emission.yield_factor(np.array([0.0, 1.0]), np.array([1.0, np.nan]), np.ones(2))


@pytest.mark.parametrize("absorption,depth", [(False, True), (True, False), (True, True)])
def test_multilayer_analytic_requires_both_gates(monkeypatch, absorption, depth):
    analytic = _script("multilayer_validation_check.py")
    evaluated = []

    def absorption_gate():
        evaluated.append("absorption")
        return absorption

    def depth_gate():
        evaluated.append("depth")
        return depth

    monkeypatch.setattr(analytic, "check_absorption_closed_form", absorption_gate)
    monkeypatch.setattr(analytic, "check_depth_dose_kanaya_okayama", depth_gate)
    if absorption and depth:
        analytic.main()
    else:
        with pytest.raises(SystemExit) as error:
            analytic.main()
        assert error.value.code == 1
    assert evaluated == ["absorption", "depth"]


@pytest.mark.parametrize(
    "failing", [None, "substrate", "crystalline_sum", "amorphous_sum", "boost"]
)
def test_multilayer_slice3_fails_on_any_assertion(monkeypatch, failing):
    slice3 = _script("multilayer_slice3_check.py")
    grid = np.array([0.0, 1.0])
    segments = {"L_ang": np.ones(2)}

    def case(substrate):
        return {
            "substrate": substrate,
            "layer_radiators": [{}, None if substrate == "sio2" else {}],
        }

    def spectrum(case, tp):
        boosted = case["substrate"] == "silicon" and failing != "boost"
        flux = 2.0 if boosted else 1.0
        return {"spec": np.full(2, flux), "spec_characteristic": np.ones(2)}

    def radiate(segs, E_grid, n_hat, case, tp, L):
        silent = failing == "substrate" and case["substrate"] == "silicon" and L == 1
        return np.zeros(2) if silent else np.ones(2), np.ones(2)

    errors = iter(
        [
            [1.0 if failing == "crystalline_sum" else 0.0, 0.0],
            [0.0, 1.0 if failing == "amorphous_sum" else 0.0],
        ]
    )
    monkeypatch.setattr(slice3, "_case", case)
    monkeypatch.setattr(
        slice3, "_transport_case", lambda case: {"segs": segments, "n_hat": None, "E_grid": grid}
    )
    monkeypatch.setattr(slice3, "_spectrum_case", spectrum)
    monkeypatch.setattr(slice3, "_segments_in_layer", lambda segs, L: segs)
    monkeypatch.setattr(slice3, "_radiate_layer", radiate)
    monkeypatch.setattr(slice3, "_component_errors", lambda *args: next(errors))
    if failing is None:
        slice3.main()
        return
    with pytest.raises(SystemExit) as error:
        slice3.main()
    assert error.value.code == 1


@pytest.mark.parametrize(("tail", "code"), [(1.0, 0), (0.0, 1), (np.nan, 1)])
def test_channeling_handoff_gates_chunk_distance(monkeypatch, tail, code):
    """Separable chunks pass; a shifted (TV 1/6) or nonfinite marginal fails."""
    handoff = _script("channeling_handoff.py")
    second = np.ones(64)
    second[32:] = tail
    marginals = iter([np.ones(64), second])

    class Profile:
        def extrema(self):
            return 0.0, 20.0

    class Solver:
        cutoff_eV, gamma0, beta0 = 0.0, 10.0, 1.0
        basis = type("Basis", (), {"energies": np.array([1.0, 30.0])})

        def __init__(self, *args):
            pass

        def propagate(self, *args, **kwargs):
            flux = np.zeros(1)
            return type(
                "Result",
                (),
                {
                    "final_density": np.ones((1, 1, 1)),
                    "entry_flux": 0.0,
                    "free_flux_by_energy": next(marginals),
                    "free_flux": flux,
                    "reprojection_flux": flux,
                    "overflow_flux": flux,
                },
            )

    monkeypatch.setattr(handoff, "planar_profile", lambda *args: Profile())
    monkeypatch.setattr(handoff, "tds_kernel", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(handoff, "electronic_kernel", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(handoff, "DensityMatrixSolver", Solver)
    assert handoff.main(["--chunks", "2"]) == code
