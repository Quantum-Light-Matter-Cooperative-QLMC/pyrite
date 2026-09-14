import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.energy_grid import derive as analyze


def test_geometry_plan_is_full_curated_tilt_azimuth_product():
    # Curated production grid (issue_notes.md #1): tilt in {5, 45}, azimuth in
    # {100, 140, 180}; polar=0 and azim=90 are excluded upstream.
    scan = SimpleNamespace(
        tilt_deg=(5.0, 45.0),
        tilt_azim_deg=(100.0, 140.0, 180.0),
    )

    coarse, spot_check = analyze._geometry_plan(["a", "b"], scan)

    # The full quantized tilt x azimuth product, every geometry sampled; the
    # legacy large-tilt spot-check set is empty (the product already spans it).
    # Specs are now 4-tuples carrying the (default) diagnostic thickness.
    assert spot_check == []
    assert len(coarse) == 2 * 2 * 3
    assert coarse[:3] == [
        ("a", 5.0, 100.0, analyze.DIAGNOSTIC_THICKNESS_ANG),
        ("a", 5.0, 140.0, analyze.DIAGNOSTIC_THICKNESS_ANG),
        ("a", 5.0, 180.0, analyze.DIAGNOSTIC_THICKNESS_ANG),
    ]
    assert {spec[1] for spec in coarse} == {5.0, 45.0}
    assert {spec[2] for spec in coarse} == {100.0, 140.0, 180.0}


def test_geometry_plan_defaults_to_profile_angles_single_thickness():
    class Scan:
        tilt_deg = [5.0, 45.0]
        tilt_azim_deg = [100.0, 180.0]

    coarse, _ = analyze._geometry_plan(["hopg"], Scan())
    # 2 tilts x 2 azimuths x 1 default thickness, all 4-tuples ending in 1e7
    assert len(coarse) == 4
    assert all(len(spec) == 4 for spec in coarse)
    assert {spec[3] for spec in coarse} == {analyze.DIAGNOSTIC_THICKNESS_ANG}


def test_geometry_plan_overrides_expand_thickness_axis():
    class Scan:
        tilt_deg = [5.0]
        tilt_azim_deg = [180.0]

    coarse, _ = analyze._geometry_plan(
        ["hopg"], Scan(), tilts=[5.0], azimuths=[180.0], thicknesses=[1.0e6, 1.0e7]
    )
    assert len(coarse) == 2
    assert sorted(spec[3] for spec in coarse) == [1.0e6, 1.0e7]


def test_subset_resume_preserves_unrequested_checkpoint_rows(monkeypatch):
    existing = [{"energy_keV": 30.0, "raw_eV": 1.0}]
    monkeypatch.setattr(
        analyze,
        "_scan_specs",
        lambda *args, **kwargs: [analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)],
    )
    monkeypatch.setattr(
        analyze,
        "_run_specs",
        lambda *args, **kwargs: [analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)],
    )
    snapshots = []

    rows, complete = analyze.derive_bounds(
        ["hopg"],
        [200.0],
        existing_rows=existing,
        on_progress=lambda value: snapshots.append(value),
    )

    assert complete is True
    assert [row["energy_keV"] for row in rows] == [30.0, 200.0]
    assert [row["energy_keV"] for row in snapshots[-1]] == [30.0, 200.0]


def test_new_energy_not_in_reference_table_falls_back_to_start_eV_convention(monkeypatch):
    # `--energy 25` etc. (not one of hopg's existing E_grid_line_by_energy
    # rows) must not KeyError on the missing table entry; it should fall back
    # to the same start_eV convention `energy_grid.apply` uses for brand-new
    # rows (10 eV floor at <=60 keV beam energy).
    monkeypatch.setattr(
        analyze,
        "_scan_specs",
        lambda *args, **kwargs: [analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)],
    )
    monkeypatch.setattr(
        analyze,
        "_run_specs",
        lambda *args, **kwargs: [analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)],
    )

    rows, complete = analyze.derive_bounds(["hopg"], [25.0])

    assert complete is True
    assert rows[0]["energy_keV"] == 25.0
    assert rows[0]["start_eV"] == analyze.line_start_eV(25.0) == 10.0


def test_budget_expiry_returns_tempfail_without_starting_next_energy(monkeypatch):
    monkeypatch.setattr(
        analyze,
        "_scan_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scan started")),
    )

    rows, complete = analyze.derive_bounds(
        ["hopg"],
        [200.0],
        existing_rows=[{"energy_keV": 30.0, "raw_eV": 1.0}],
        max_seconds=0.0,
        time_fn=lambda: 10.0,
    )

    assert complete is False
    assert rows == [{"energy_keV": 30.0, "raw_eV": 1.0}]


def test_budget_expiry_checkpoints_completed_phase(monkeypatch):
    candidate = analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)
    monkeypatch.setattr(analyze, "_scan_specs", lambda *args, **kwargs: [candidate])
    clock = iter([0.0, 0.0, 11.0])
    phases = []

    rows, complete = analyze.derive_bounds(
        ["hopg"],
        [200.0],
        max_seconds=10.0,
        time_fn=lambda: next(clock),
        on_phase_progress=lambda value: phases.append(value),
    )

    assert complete is False
    assert rows == []
    assert phases[-1]["energy_keV"] == 200.0
    assert phases[-1]["near_zero"] == [analyze.asdict(candidate)]


def test_resume_after_near_zero_does_not_rerun_completed_phase(monkeypatch):
    candidate = analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)
    calls = []
    monkeypatch.setattr(
        analyze,
        "_scan_specs",
        lambda *args, **kwargs: calls.append("spot") or [candidate],
    )
    monkeypatch.setattr(analyze, "_run_specs", lambda *args, **kwargs: [candidate])

    rows, complete = analyze.derive_bounds(
        ["hopg"],
        [200.0],
        phase_state={"energy_keV": 200.0, "near_zero": [analyze.asdict(candidate)]},
    )

    assert complete is True
    # near_zero was resumed from checkpoint (not rerun); spot_check is empty under
    # the curated plan, so the coarse scanner is never called again this energy.
    assert calls == []
    assert [row["energy_keV"] for row in rows] == [200.0]


def test_partial_phase_resume_starts_at_saved_geometry_cursor(monkeypatch):
    candidate = analyze.Candidate("hopg", 1.0, 90.0, 1000.0, 1.0, 500.0, 0.5)
    seen = []

    values, timed_out = analyze._resume_phase(
        "near_zero",
        [("hopg", 1.0, float(index)) for index in range(120)],
        200.0,
        200,
        None,
        "auto",
        {
            "energy_keV": 200.0,
            "near_zero": [analyze.asdict(candidate)],
            "near_zero_cursor": 50,
        },
        None,
        0.0,
        None,
        lambda: 0.0,
        lambda specs, *args: seen.append(specs) or [candidate],
    )

    assert timed_out is False
    assert [len(specs) for specs in seen] == [10] * 7
    assert seen[0][0][2] == 50.0
    assert len(values) == 8


def test_refine_phase_checkpoints_each_expensive_geometry():
    calls = []
    snapshots = []

    def runner(specs, *_args):
        calls.append(specs)
        material, tilt, azim = specs[0]
        return [analyze.Candidate(material, tilt, azim, 1000.0, 1.0, 500.0, 0.5)]

    values, timed_out = analyze._resume_phase(
        "refined",
        [("hopg", 1.0, float(index)) for index in range(3)],
        250.0,
        2000,
        None,
        "auto",
        {"energy_keV": 250.0},
        lambda value: snapshots.append(dict(value)),
        0.0,
        None,
        lambda: 0.0,
        runner,
    )

    assert timed_out is False
    assert [len(specs) for specs in calls] == [1, 1, 1]
    assert [snapshot["refined_cursor"] for snapshot in snapshots] == [1, 2, 3]
    assert len(values) == 3


def test_refine_phase_hands_off_after_one_case_overshoots_soft_budget():
    calls = []
    snapshots = []

    def runner(specs, *_args):
        calls.append(specs)
        material, tilt, azim = specs[0]
        return [analyze.Candidate(material, tilt, azim, 1000.0, 1.0, 500.0, 0.5)]

    values, timed_out = analyze._resume_phase(
        "refined",
        [("hopg", 1.0, float(index)) for index in range(3)],
        250.0,
        2000,
        None,
        "auto",
        {"energy_keV": 250.0},
        lambda value: snapshots.append(dict(value)),
        0.0,
        600.0,
        lambda: 601.0,
        runner,
    )

    assert timed_out is True
    assert len(calls) == 1
    assert snapshots[-1]["refined_cursor"] == 1
    assert len(values) == 1


def test_reduced_sampling_defaults_match_handoff():
    assert analyze.COARSE_NE == 200
    assert analyze.REFINE_NE == 2000
    assert analyze.COARSE_ENGINE == "auto"


def test_main_returns_tempfail_when_budget_leaves_work(monkeypatch):
    # main() now drives derive_all_materials, not derive_bounds directly; an
    # incomplete run (complete=False) must still surface the exit-75 contract.
    monkeypatch.setattr(analyze, "derive_all_materials", lambda *args, **kwargs: ({}, False))
    monkeypatch.setattr(analyze, "_print_report", lambda rows: None)

    assert analyze.main(["--materials", "hopg", "--energies", "200"]) == 75


def test_main_uses_persistent_materials_and_energies(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        analyze.lg_defaults,
        "load_defaults",
        lambda: {
            **analyze.lg_defaults.FALLBACK,
            "materials": ["wse2", "mose2"],
            "energies": [40.0, 60.0],
        },
    )
    monkeypatch.setattr(
        analyze,
        "derive_all_materials",
        lambda materials, energies, *args, **kwargs: (
            seen.update(materials=materials, energies=energies) or {},
            True,
        ),
    )

    assert analyze.main([]) == 0
    assert seen == {"materials": ["wse2", "mose2"], "energies": [40.0, 60.0]}


def test_build_case_passes_wide_brem_grid_to_diagnostic_sweep(monkeypatch):
    # The brem coverage channel must be measured on the WIDE diagnostic brem
    # grid, not the profile's 30 keV E_grid_brem -- otherwise a per-material
    # bespoke brem stop would be silently clipped to the narrow profile ceiling.
    captured = {}
    real_sweep = analyze.material_sweep

    def spy(material, **kwargs):
        captured.update(kwargs)
        return real_sweep(material, **kwargs)

    monkeypatch.setattr(analyze, "material_sweep", spy)
    monkeypatch.setattr(analyze, "build_cases", lambda sweep, **kw: [{"sweep": sweep}])

    analyze._build_case("hopg", 100.0, 5.0, 100.0, analyze.DIAGNOSTIC_THICKNESS_ANG, 10)

    assert "E_grid_brem" in captured
    np.testing.assert_array_equal(captured["E_grid_brem"], analyze.WIDE_BREM_EV)


def test_build_case_marks_grid_for_post_transport_sinc_resolution(monkeypatch):
    monkeypatch.setattr(analyze, "material_sweep", lambda *args, **kwargs: object())
    monkeypatch.setattr(analyze, "build_cases", lambda *args, **kwargs: [{"name": "case"}])

    case = analyze._build_case("hopg", 100.0, 5.0, 100.0, 1.0e7, 10)

    assert case["_diagnostic_line_grid"] == {
        "start_eV": analyze.WIDE_GRID_START_EV,
        "stop_eV": analyze.WIDE_GRID_STOP_EV,
        "aliased_weight_limit": analyze.ALIASED_WEIGHT_LIMIT,
        "backend_safety_ulps": analyze.BACKEND_SAFETY_ULPS,
        "maximum_spacing_eV": analyze.MAX_DIAGNOSTIC_SPACING_EV,
    }


def test_candidate_rejects_result_above_alias_budget():
    E = np.array([10.0, 20.0, 30.0])
    result = {
        "E_grid": E,
        "spec": np.array([1.0, 1.0, 0.0]),
        "E_grid_brem": E,
        "brem_wide": np.array([1.0, 1.0, 0.0]),
        "line_grid_diagnostic": {
            "target_spacing_eV": 1.0,
            "actual_spacing_eV": 1.0,
            "aliased_weight_fraction": 0.02,
            "aliased_weight_limit": 0.01,
            "backend_dtype": "float32",
            "observable_class": "intrinsic_source",
        },
    }

    with pytest.raises(ValueError, match="alias budget"):
        analyze._candidate_from_result("hopg", 5.0, 100.0, 1.0e7, result)


def test_diagnostic_transport_resolves_grid_before_spectrum(monkeypatch):
    from pyrite.montecarlo import runner

    case = analyze._build_case("hopg", 100.0, 5.0, 100.0, 1.0e7, 1)
    segments = {
        "E_keV": np.array([100.0]),
        "L_ang": np.array([1000.0]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "elec_id": np.array([0]),
    }
    monkeypatch.setattr(runner, "simulate_trajectories", lambda *args, **kwargs: segments)

    transport = runner._transport_case(case, transport_core="lockstep")

    assert transport["E_grid"].size > 2
    assert np.all(np.diff(transport["E_grid"].astype(np.float32)) > 0.0)
    assert transport["diagnostic_grid"]["observable_class"] == "intrinsic_source"


def test_candidate_brem_channel_refuses_silent_truncation():
    # The incoherent (brem) coverage call must keep allow_shortfall=False: a brem
    # spectrum whose 95% mass sits in the final bin means the true coverage lies
    # beyond the diagnostic ceiling and must raise, never clamp (issue_notes.md #1).
    from pyrite.energy_grid.bounds import CoverageGridTooNarrow

    E = np.arange(0.0, 100.0, 10.0)
    coherent = np.ones_like(E)
    truncated = np.zeros_like(E)
    truncated[-1] = 1.0
    result = {"E_grid": E, "spec": coherent, "E_grid_brem": E, "brem_wide": truncated}

    with pytest.raises(CoverageGridTooNarrow):
        analyze._candidate_from_result("hopg", 5.0, 100.0, analyze.DIAGNOSTIC_THICKNESS_ANG, result)


def test_derive_all_materials_produces_independent_per_material_output(monkeypatch):
    # Approach A: each material's rows come from its OWN single-material
    # derive_bounds call, so the driver is that material's own worst geometry --
    # not a single worst-case shared across all materials.

    def fake_derive_bounds(materials, energies, *args, **kwargs):
        m = materials[0]
        rows = [
            {"energy_keV": e, "driver_material": m, "brem_stop_eV": 100.0 * e} for e in energies
        ]
        return rows, True

    monkeypatch.setattr(analyze, "derive_bounds", fake_derive_bounds)
    monkeypatch.setattr(
        analyze,
        "_brem_grid_for_rows",
        lambda rows, step_eV: {
            "stop_eV": max(r["brem_stop_eV"] for r in rows),
            "raw_eV": 0.0,
            "step_eV": step_eV,
        },
    )

    combined, complete = analyze.derive_all_materials(
        ["hopg", "diamond"], [100.0, 200.0], brem_step_eV=12.5
    )

    assert complete is True
    assert set(combined) == {"hopg", "diamond"}
    assert [r["driver_material"] for r in combined["hopg"]["line_rows"]] == ["hopg", "hopg"]
    assert [r["driver_material"] for r in combined["diamond"]["line_rows"]] == [
        "diamond",
        "diamond",
    ]
    assert combined["hopg"]["brem"]["stop_eV"] == 200.0 * 100.0
    assert combined["hopg"]["brem"]["step_eV"] == 12.5


def test_derive_all_materials_skips_material_with_complete_checkpoint(monkeypatch, tmp_path):
    # A per-material checkpoint that already holds every requested energy must be
    # loaded, not recomputed: the coarse/refine scanners are never called.
    monkeypatch.setattr(
        analyze,
        "_scan_specs",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("coarse scan started")),
    )
    monkeypatch.setattr(
        analyze,
        "_run_specs",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("refine started")),
    )
    monkeypatch.setattr(
        analyze,
        "_brem_grid_for_rows",
        lambda rows, step_eV: {"stop_eV": 0.0, "raw_eV": 0.0, "step_eV": step_eV},
    )
    json_out = str(tmp_path / "bounds.json")
    seeded = [{"energy_keV": 200.0, "raw_eV": 1.0, "brem_stop_eV": 1.0}]
    with open(f"{json_out}.hopg.json", "w") as f:
        json.dump(seeded, f)

    combined, complete = analyze.derive_all_materials(["hopg"], [200.0], json_out=json_out)

    assert complete is True
    assert combined["hopg"]["line_rows"] == seeded


def test_brem_grid_for_rows_covers_worst_energy_brem_stop():
    # One E_grid_brem per material must clear the widest-energy brem tail: the
    # max over the per-energy brem_stop_eV, reporting the raw of the row that set
    # it, at the production brem spacing.
    rows = [
        {"energy_keV": 100.0, "brem_stop_eV": 5000.0, "brem_raw_eV": 4700.0},
        {"energy_keV": 300.0, "brem_stop_eV": 12000.0, "brem_raw_eV": 11500.0},
        {"energy_keV": 200.0, "brem_stop_eV": 9000.0, "brem_raw_eV": 8600.0},
    ]

    grid = analyze._brem_grid_for_rows(rows)

    assert grid["stop_eV"] == 12000.0
    assert grid["raw_eV"] == 11500.0
    assert grid["step_eV"] == analyze.WIDE_BREM_STEP_EV


def test_brem_grid_for_rows_uses_requested_output_step():
    rows = [{"brem_stop_eV": 12000.0, "brem_raw_eV": 11500.0}]

    grid = analyze._brem_grid_for_rows(rows, step_eV=12.5)

    assert grid == {"stop_eV": 12000.0, "raw_eV": 11500.0, "step_eV": 12.5}
