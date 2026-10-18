"""Small analytic regressions for cheap numerical-validation evidence."""

import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "checks/equal_precision"


def test_table_coverage_uses_entire_electron_path_and_explicit_units():
    check = runpy.run_path(str(ROOT / "benchmark_validation.py"))["grid_coverage"]
    assert check([1000, 10000, 1000000], [10, 800], True)["covered"]
    assert not check([10000, 100000, 500000], [10, 800], True)["covered"]
    assert not check([20, 100, 800], [10, 300])["covered"]
    with pytest.raises(ValueError):
        check([1, np.nan, 3], [1, 3])
    with pytest.raises(ValueError):
        check([1, 3, 2], [1, 3])


def test_fixed_request_grid_respects_small_budget_and_independent_studies():
    import json

    module = runpy.run_path(str(ROOT / "run_cheap_validation.py"))
    protocol = json.loads((ROOT / "cheap-validation.json").read_bytes())
    cpu = module["requests"](protocol, "cpu")
    gpu = module["requests"](protocol, "gpu")
    assert len(cpu) == 18 and len(gpu) == 48
    assert sum(r["core"] == "geant4" for r in cpu) == 6
    assert max(r["histories"] for r in cpu + gpu) == 500
    assert len({tuple(sorted(r.items())) for r in cpu + gpu}) == 66
    backend = [r for r in gpu if r["histories"] == 64]
    assert {(r["scene"], r["seed"]) for r in backend} == {
        (r["scene"], r["seed"]) for r in cpu if r["core"] == "per-electron"
    }
    assert set(protocol["convergence"]["seeds"]).isdisjoint(protocol["backend"]["seeds"])


def test_geant4_material_cut_dump_keeps_production_and_tracking_controls_distinct():
    parse = runpy.run_path(str(ROOT / "analyze_cheap_validation.py"))["parse_cut_couples"]
    text = """ Material : Silicon
 Range cuts : gamma 10 Ang e- 1 cm e+ 10 Ang proton 10 Ang
 Energy thresholds : gamma 990 eV e- 4.04011 MeV e+ 990 eV proton 100 meV
G4WT0 >  Material : Silicon
G4WT0 >  Range cuts : gamma 10 Ang e- 1 cm e+ 10 Ang proton 10 Ang
G4WT0 >  Energy thresholds : gamma 990 eV e- 4.04011 MeV e+ 990 eV proton 100 meV
"""
    result = parse(text)
    assert result["Silicon"]["gamma_production_threshold_keV"] == 0.99
    assert result["Silicon"]["electron_production_threshold_keV"] == pytest.approx(4040.11)
    assert "electron_tracking_threshold_keV" not in result["Silicon"]
    with pytest.raises(ValueError):
        parse("no threshold dump")


def evidence():
    import json

    protocol = json.loads((ROOT / "cheap-validation.json").read_bytes())
    scene = json.loads((ROOT / "manifest.json").read_bytes())["scenes"][0]
    names = runpy.run_path(str(ROOT / "analyze_cheap_validation.py"))["OBSERVABLES"]
    scores = {name: np.zeros(4) for name in names}
    scores["transmitted"][:] = 1
    scores["transmitted_mu_bin_3"][:] = 1
    scores["transmitted_energy_keV_per_primary"][:] = 300
    scores["terminal_energy_keV_per_primary"][:] = 300
    controls = dict(
        E_cut_keV=10,
        energy_model="midpoint",
        max_dE_frac=0.05,
        elastic_model="elsepa",
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000,
        transport_core="per-electron",
        crystal_width_mm=10,
        crystal_height_mm=10,
    )
    record = dict(
        validation_protocol_sha256="frozen",
        validation_sources=protocol["validation_sources"],
        arm="baseline",
        scene=scene,
        core="per-electron",
        histories=4,
        seed=1,
        resolved_controls=controls,
        max_de_frac=0.05,
        cutoff_eV=1000,
        cache_state="warmed-in-process",
        warmup=dict(
            repeats=1,
            histories=4,
            seed=1000001,
            same_scene_tables_numerics=True,
            device_synchronized=False,
        ),
        cxr_execution_audit=dict(enabled=True, calls=[]),
        trajectory_blocks=[
            dict(
                primaries=4,
                energy_ledger_max_error_keV=0,
                photon_debit_max_error_keV=0,
                direction_norm_max_error=0,
            )
        ],
        table_coverage={
            k: dict(covered=True, requested_keV=[10, 300], min_keV=1, max_keV=1000)
            for k in ["elsepa-0-0", "sbethe-0", "bremslib-Si"]
        },
    )
    record["trajectory_blocks"].append(dict(record["trajectory_blocks"][0]))
    return record, scores, protocol, scene


def test_complete_audited_scores_are_accepted():
    record, scores, protocol, scene = evidence()
    runpy.run_path(str(ROOT / "analyze_cheap_validation.py"))["check_record"](
        record, scores, protocol, "frozen", scene
    )


@pytest.mark.parametrize(
    "corruption", ["source", "controls", "warmup", "audit", "table", "shape", "nan"]
)
def test_loader_rejects_inconsistent_numerical_validation_evidence(corruption):
    record, scores, protocol, scene = evidence()
    if corruption == "source":
        record["validation_sources"] = {}
    elif corruption == "controls":
        record["resolved_controls"]["max_dE_frac"] = 0.1
    elif corruption == "warmup":
        record["warmup"]["histories"] = 2
    elif corruption == "audit":
        record["trajectory_blocks"][0]["energy_ledger_max_error_keV"] = 1
    elif corruption == "table":
        record["table_coverage"]["bremslib-Si"]["min_keV"] = 20
    elif corruption == "shape":
        scores["transmitted"] = np.zeros(3)
    else:
        scores["transmitted"][0] = np.nan
    with pytest.raises(ValueError):
        runpy.run_path(str(ROOT / "analyze_cheap_validation.py"))["check_record"](
            record, scores, protocol, "frozen", scene
        )


@pytest.mark.parametrize("corruption", ["fraction", "partition", "ledger", "blocks"])
def test_score_and_audit_bookkeeping_cannot_be_faked_by_valid_shapes(corruption):
    record, scores, protocol, scene = evidence()
    if corruption == "fraction":
        scores["transmitted"][0] = 0.5
    elif corruption == "partition":
        scores["stopped"][0] = 1
    elif corruption == "ledger":
        scores["terminal_energy_keV_per_primary"][0] = 200
    else:
        record["trajectory_blocks"] = [dict(record["trajectory_blocks"][0], primaries=8)]
    with pytest.raises(ValueError):
        runpy.run_path(str(ROOT / "analyze_cheap_validation.py"))["check_record"](
            record, scores, protocol, "frozen", scene
        )


def test_activity_counts_only_energy_cap_events_not_hard_photons():
    activity = runpy.run_path(str(ROOT / "probe_step_activity.py"))["substep_activity"]
    result = activity(
        dict(
            event_kind=np.array([0, 1, 1, 9, 5, 1, 7]), electron_id=np.array([0, 0, 0, 0, 0, 1, 1])
        )
    )
    assert result == dict(rows=7, energy_limited_rows=3, primaries_with_energy_substeps=2)


def test_activity_ladder_keeps_all_scenes_and_fresh_planning_population():
    import json

    protocol = json.loads((ROOT / "activity-ladder.json").read_bytes())
    previous = json.loads((ROOT / "cheap-validation.json").read_bytes())
    rows = runpy.run_path(str(ROOT / "run_activity_ladder.py"))["requests"](protocol)
    assert len(rows) == len(set(rows)) == 24
    assert set(protocol["scenes"]) == set(previous["scenes"])
    assert protocol["caps"] == [0.05, 0.001, 0.00025, 0.0000625]
    assert protocol["seed"] not in previous["convergence"]["seeds"] + [68001]


@pytest.mark.parametrize(
    "corruption", [None, "seed", "cap", "warmup", "audit", "counts", "capacity"]
)
def test_activity_ladder_rejects_incomplete_or_mismatched_planning_samples(corruption):
    import json

    protocol = json.loads((ROOT / "activity-ladder.json").read_bytes())
    record, _, _, scene = evidence()
    record.update(
        schema="pyrite.equal-precision.step-activity.v1",
        histories=64,
        seed=69001,
        activity={
            phase: dict(
                primaries=64, rows=1000, energy_limited_rows=100, primaries_with_energy_substeps=20
            )
            for phase in ["warmup", "measurement"]
        },
    )
    record["warmup"].update(histories=64, seed=1000000 + 69001)
    if corruption == "seed":
        record["seed"] = 68001
    elif corruption == "cap":
        record["max_de_frac"] = 0.001
    elif corruption == "warmup":
        record["activity"]["warmup"]["primaries"] = 32
    elif corruption == "audit":
        record["cxr_execution_audit"]["enabled"] = False
    elif corruption == "counts":
        record["activity"]["measurement"]["primaries_with_energy_substeps"] = 65
    elif corruption == "capacity":
        record["max_steps"] = 80000
    check = runpy.run_path(str(ROOT / "run_activity_ladder.py"))["check_sample"]
    if corruption is None:
        check(record, scene["id"], 0.05, protocol)
    else:
        with pytest.raises(ValueError):
            check(record, scene["id"], 0.05, protocol)


def test_incomplete_fixed_grid_never_produces_selected_subset_intervals(tmp_path):
    import json
    import subprocess
    import sys

    raw = tmp_path / "evidence"
    source = raw / "source"
    source.mkdir(parents=True)
    for name in ["cheap-validation.json", "manifest.json"]:
        (source / name).write_bytes((ROOT / name).read_bytes())
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analyze_cheap_validation.py"),
            str(raw),
            "--protocol",
            str(source / "cheap-validation.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_bytes())
    assert not result["complete_fixed_design"]
    assert len(result["missing"]) == 66
    assert result["comparisons"] == []
    assert not result["qualified_numerical_convergence"]
