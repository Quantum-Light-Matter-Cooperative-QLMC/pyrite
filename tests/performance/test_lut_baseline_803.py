"""The #803 baseline must pin every control and observe real transport sites."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
HARNESS = ROOT / "checks/equal_precision/baseline_803.py"
DRIVER = ROOT / "checks/equal_precision/run_baseline_803.py"
PROTOCOL = json.loads((ROOT / "checks/equal_precision/baseline-803.json").read_text())
CONTROLS = ("core", "elastic", "radiative", "inelastic", "lut", "moments", "schedule")


@pytest.mark.parametrize("core", ["per-electron", "cuda"])
def test_every_sample_names_all_controls_once(core):
    plan = runpy.run_path(str(DRIVER))["samples"](PROTOCOL, core)
    stems = [stem for stem, _, _ in plan]
    assert len(stems) == len(set(stems))
    harness = runpy.run_path(str(HARNESS))
    for _, _, argv in plan:
        args = harness["parse_args"]([*argv, "--output", "x.json"])
        assert all(f"--{name}" in argv for name in CONTROLS)
        assert args.core == core
        assert core == "cuda" or args.schedule == "full-history"


def test_schedule_arm_pairs_with_full_history_detailed_arm():
    plan = runpy.run_path(str(DRIVER))["samples"](PROTOCOL, "cuda")
    stems = {stem for stem, _, _ in plan}
    for stem in stems:
        for schedule in ("-continuation", "-auto"):
            if stem.endswith(schedule):
                assert stem.removesuffix(schedule) + "-full-history" in stems


def test_layer_stack_is_contiguous_and_keeps_catalog_keys():
    harness = runpy.run_path(str(HARNESS))
    _, scene = harness["load_scene"]("w2um-on-si100um-800")
    layers, thickness, keys = harness["layer_stack"](scene, lambda key: [(key, 1.0)])
    assert keys == ["tungsten", "silicon"]
    assert [(top, bot) for top, bot, _ in layers] == [(0.0, 2e4), (2e4, 102e4)]
    assert thickness == 102e4


def test_observed_sites_exist_on_current_transport():
    from pyrite.montecarlo.transport import hard_radiative, host_pass

    for owner, name in (
        (host_pass, "_transport_lut"),
        (host_pass, "resolve_schedule"),
        (hard_radiative, "radiative_core_args"),
    ):
        assert callable(getattr(owner, name))


def test_head_from_git_dir_resolves_loose_and_packed_refs(tmp_path):
    harness = runpy.run_path(str(HARNESS))
    git = tmp_path / ".git"
    (git / "refs/heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/topic\n")
    (git / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/topic\n")
    assert harness["head_from_git_dir"](tmp_path) == "a" * 40
    (git / "refs/heads/topic").write_text("b" * 40 + "\n")
    assert harness["head_from_git_dir"](tmp_path) == "b" * 40


def _record(tmp_path, stem, *, wall, digest="d", scores=None, **controls):
    import numpy as np

    base = dict(
        core="cuda",
        elastic="elsepa",
        radiative="bremslib-soft-hard",
        inelastic="continuous",
        lut_requested="off",
        moments_requested="table",
        schedule_requested="full-history",
    )
    record = {
        "schema": "pyrite.lut-baseline-803.sample.v1",
        "status": "ok",
        "scene": {"id": "s"},
        "seed": 1,
        "controls": base | controls,
        "resolved": {},
        "trajectory_sha256": digest,
        "segments": 1,
        "timings_s": {
            "warm_median": wall,
            "warm_per_history": wall,
            "cold_request": wall,
            "warm_requests": [wall],
        },
        "observed_warm": [[{"site": "radiative_core_args", "wall_s": wall / 4}]],
        "memory": {"peak_rss_kib": 1, "device_pool_reserved_bytes_after_warm": None},
        "source": {"sha": "x"},
    }
    (tmp_path / f"{stem}.json").write_text(json.dumps(record))
    np.savez(tmp_path / f"{stem}.npz", **(scores or {"t": np.ones(4)}))


def test_analysis_pairs_one_control_and_screens_against_tolerance(tmp_path):
    import numpy as np

    analyze = runpy.run_path(str(ROOT / "checks/equal_precision/analyze_baseline_803.py"))
    rng = np.random.default_rng(0)
    base = rng.normal(100.0, 1.0, 400)
    _record(tmp_path, "exact", wall=2.0, moments_requested="exact", scores={"e": base})
    _record(tmp_path, "table", wall=1.0, scores={"e": base + 1e-6})
    _record(tmp_path, "cont", wall=0.5, digest="other", schedule_requested="continuation")
    records, _ = analyze["load"](tmp_path)
    by_name = {c["comparison"]: c for c in analyze["compare"](records, 1e-3)}
    assert by_name["moments"]["speedup_geomean"] == pytest.approx(2.0)
    assert by_name["moments"]["screen"] == {"equivalent": ["e"]}
    assert by_name["continuation"]["bitwise_equal"] is False
    assert analyze["preparation"](records)[0]["warm_moment_prep_fraction"] == pytest.approx(0.25)


def test_stage_timer_splits_nested_ranges_exclusively(monkeypatch):
    harness = runpy.run_path(str(HARNESS))
    clock = iter([0.0, 1.0, 3.0, 6.0])
    timer = harness["StageTimer"]()
    with monkeypatch.context() as patch:
        patch.setattr(harness["time"], "perf_counter", lambda: next(clock))
        timer.push("outer")
        timer.push("inner")
        timer.pop()
        timer.pop()
    totals = timer.take()
    assert totals["inner"] == {"calls": 1, "inclusive_s": 2.0, "exclusive_s": 2.0}
    assert totals["outer"] == {"calls": 1, "inclusive_s": 6.0, "exclusive_s": 4.0}
    timer.push("open")
    with pytest.raises(RuntimeError, match="unbalanced"):
        timer.take()


def test_stage_timer_hooks_the_production_range_entry_points():
    from pyrite.montecarlo import runner

    assert callable(runner._nsys_push) and callable(runner._nsys_pop)


def test_stage_split_shares_sum_to_one_with_unranged_remainder(tmp_path):
    analyze = runpy.run_path(str(ROOT / "checks/equal_precision/analyze_baseline_803.py"))
    record = {
        "split_stage_synchronized": {
            "wall_s": 4.0,
            "stages": {"a": {"exclusive_s": 1.0}, "b": {"exclusive_s": 2.0}},
            "kernel": {"kernel_s": 1.0, "launches": 3},
        }
    }
    split = analyze["stage_split"](record)
    assert split["exclusive_share"] == {"a": 0.25, "b": 0.5, "unranged": 0.25}
    assert split["kernel_share"] == 0.25 and split["launches"] == 3
    assert analyze["stage_split"]({}) is None


def test_head_from_git_dir_follows_a_linked_worktree(tmp_path):
    harness = runpy.run_path(str(HARNESS))
    common = tmp_path / "main/.git"
    (common / "refs/heads").mkdir(parents=True)
    (common / "refs/heads/topic").write_text("c" * 40 + "\n")
    linked = common / "worktrees/wt"
    linked.mkdir(parents=True)
    (linked / "commondir").write_text("../..\n")
    (linked / "HEAD").write_text("ref: refs/heads/topic\n")
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {linked}\n")
    assert harness["head_from_git_dir"](tree) == "c" * 40
    (linked / "HEAD").write_text("d" * 40 + "\n")
    assert harness["head_from_git_dir"](tree) == "d" * 40
