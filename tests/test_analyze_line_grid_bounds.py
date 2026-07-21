import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_plan_is_full_curated_tilt_azimuth_product():
    analyze = _load_script("analyze_line_grid_bounds")
    # Curated production grid (issue_notes.md #1): tilt in {5, 45}, azimuth in
    # {100, 140, 180}; polar=0 and azim=90 are excluded upstream.
    scan = SimpleNamespace(
        tilt_deg=(5.0, 45.0),
        tilt_azim_deg=(100.0, 140.0, 180.0),
    )

    coarse, spot_check = analyze._geometry_plan(["a", "b"], scan)

    # The full quantized tilt x azimuth product, every geometry sampled; the
    # legacy large-tilt spot-check set is empty (the product already spans it).
    assert spot_check == []
    assert len(coarse) == 2 * 2 * 3
    assert coarse[:3] == [("a", 5.0, 100.0), ("a", 5.0, 140.0), ("a", 5.0, 180.0)]
    assert {spec[1] for spec in coarse} == {5.0, 45.0}
    assert {spec[2] for spec in coarse} == {100.0, 140.0, 180.0}


def test_subset_resume_preserves_unrequested_checkpoint_rows(monkeypatch):
    analyze = _load_script("analyze_line_grid_bounds")
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


def test_budget_expiry_returns_tempfail_without_starting_next_energy(monkeypatch):
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
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
    analyze = _load_script("analyze_line_grid_bounds")
    assert analyze.COARSE_NE == 200
    assert analyze.REFINE_NE == 2000
    assert analyze.COARSE_ENGINE == "auto"


def test_main_returns_tempfail_when_budget_leaves_work(monkeypatch):
    analyze = _load_script("analyze_line_grid_bounds")
    monkeypatch.setattr(analyze, "derive_bounds", lambda *args, **kwargs: ([], False))
    monkeypatch.setattr(analyze, "_print_report", lambda rows: None)

    assert analyze.main(["--materials", "hopg", "--energies", "200"]) == 75
