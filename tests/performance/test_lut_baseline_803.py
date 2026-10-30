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
