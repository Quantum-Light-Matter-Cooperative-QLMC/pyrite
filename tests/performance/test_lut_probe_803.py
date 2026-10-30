"""The #803 component probe must anchor on current transport source and stay inert."""

import ast
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PROBE = runpy.run_path(str(ROOT / "checks/equal_precision/probe_components_803.py"))
TRANSPORT = ROOT / "src/pyrite/montecarlo/transport"


@pytest.mark.parametrize("core", ["per-electron", "cuda"])
@pytest.mark.parametrize("component", sorted(PROBE["COMPONENTS"]))
def test_every_component_anchors_once_and_transforms_to_valid_python(core, component):
    source = (TRANSPORT / f"{PROBE['MODULES'][core]}.py").read_text()
    assert PROBE["transform"](source, component, core, 0) == source
    amplified = PROBE["transform"](source, component, core, 3)
    ast.parse(amplified)
    added = len(amplified.splitlines()) - len(source.splitlines())
    assert added > 0
    # The only side effect is the impossible-value sink.
    assert amplified.count(PROBE["SINK"]) == 1


def test_changed_anchor_is_refused():
    with pytest.raises(ValueError, match="anchor inventory"):
        PROBE["transform"]("no anchors here\n", "rate", "cuda", 1)


def test_plan_runs_reference_first_and_rotates_components_by_seed():
    driver = runpy.run_path(str(ROOT / "checks/equal_precision/run_probe_803.py"))
    protocol = json.loads((ROOT / "checks/equal_precision/probe-803.json").read_text())
    plan = driver["samples"](protocol, ["s"], "cuda")
    stems = [stem for stem, _ in plan]
    assert len(stems) == len(set(stems))
    seed_a, seed_b = protocol["seeds"][:2]
    start_b = next(i for i, stem in enumerate(stems) if f"-{seed_b}-" in stem)
    assert stems[0] == f"s-{seed_a}-cuda-continuous-rate-k0"
    assert stems[start_b] == f"s-{seed_b}-cuda-continuous-rate-k0"
    assert plan[1][1][0] != plan[start_b + 1][1][0]
    assert all("--precision" in argv for _, argv in plan)
    assert any(stem.endswith("mixed_math-angle-k4") for stem in stems)


def _record(path, component, k, kernel, digest="d", seed=1):
    record = {
        "schema": "pyrite.lut-probe-803.sample.v1",
        "status": "ok",
        "scene": {"id": "s"},
        "seed": seed,
        "histories": 8,
        "controls": dict.fromkeys(
            (
                "core",
                "elastic",
                "radiative",
                "inelastic",
                "lut_requested",
                "moments_requested",
                "schedule_requested",
            ),
            "x",
        ),
        "probe": {"component": component, "amplification": k},
        "trajectory_sha256": digest,
        "timings_s": {"warm_median": 2 * kernel},
        "split_warm": [{"kernel": {"kernel_s": kernel}}],
        "source": {"sha": "a"},
    }
    path.joinpath(f"{component}-{k}.json").write_text(json.dumps(record))


def test_analysis_costs_per_evaluation_and_rejects_changed_digests(tmp_path):
    analyze = runpy.run_path(str(ROOT / "checks/equal_precision/analyze_probe_803.py"))
    _record(tmp_path, "rate", 0, 1.0)
    _record(tmp_path, "rate", 4, 1.4)
    _record(tmp_path, "angle", 1, 1.05, digest="other")
    rows, broken, missing, _ = analyze["summarize"](analyze["load"]([tmp_path]))
    (row,) = rows
    assert row["component"] == "rate" and row["K"] == 4
    assert row["kernel_share_mean"] == pytest.approx(0.1)
    assert row["wall_share_mean"] == pytest.approx(0.05)
    assert broken == [
        {"group": ["s", *["x"] * 7, "exact", 8], "seed": 1, "component": "angle", "K": 1}
    ]
    assert missing == []
    (rem,) = analyze["remainders"](rows)
    assert rem["unattributed"] == pytest.approx(0.9)
