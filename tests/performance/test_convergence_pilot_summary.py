"""Pilot records become planner rows only from their own effective settings."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
SUMMARY = runpy.run_path(str(HERE / "summarize_convergence_pilot.py"))
PLANNER = runpy.run_path(str(HERE / "prepare_convergence_resources.py"))
DESIGN = HERE / "convergence-design-v1.json"
IDS = dict(cpus=2, hardware="test-host", runtime="test-runtime", memory_margin_factor=1.5)
SCENE = "si-300keV-100um"


def bench_record(root, name, seed, *, cutoff=10.0, dE=0.05, level=None, mean=0.8, n=200):
    rng = np.random.default_rng(seed)
    scores = {
        "transmitted": (rng.random(n) < mean).astype(float),
        "backscattered": (rng.random(n) < 0.1).astype(float),
        "stopped": (rng.random(n) < 0.05).astype(float),
        "transmitted_energy_keV": rng.random(n) * 300,
    }
    npz = root / f"{name}.npz"
    np.savez_compressed(npz, **scores)
    policy = None
    if level is not None:
        f = 2.0**level
        policy = dict(
            rtol=2e-4 / f,
            atol=[1e-10 / f, 1e-14 / f, 1e-10 / f],
            max_nodes=32768,
            held_out_count=8191,
        )
    record = dict(
        schema="pyrite.geant4-performance.sample.v1",
        scene=dict(id=SCENE),
        seed=seed,
        histories=n,
        estimator="discrete",
        cone_scoring=dict(mode="sampled-photons"),
        controls=dict(
            electron_cutoff_keV=cutoff,
            radiative_cutoff_eV=1000.0,
            max_dE_frac=dE,
            elsepa_level=0,
            mixed_c1=None,
            mixed_c2=None,
            mixed_max_step_um=None,
            transport_precision="fp64",
            moment_policy=policy,
        ),
        timings_s=dict(warm_request=2.0, process_total=12.0),
        peak_rss_kib=1024,
        vram_telemetry=dict(peak_device_used_bytes=4096),
        device="fake-gpu",
        raw_scores_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
    )
    path = root / f"{name}.json"
    path.write_text(json.dumps(record))
    return path


def pilot_set(root):
    paths = []
    for seed in (301, 302):
        paths.append(bench_record(root, f"nominal-{seed}", seed))
        paths.append(bench_record(root, f"cut3-{seed}", seed, cutoff=1.25, mean=0.82))
        paths.append(bench_record(root, f"moment3-{seed}", seed, level=3))
    return paths


def test_summary_feeds_planner_with_derived_settings(tmp_path):
    summary = SUMMARY["summarize"](DESIGN, pilot_set(tmp_path), tmp_path, IDS)
    rows = {r["id"]: r for r in summary["rows"]}
    design = json.loads(DESIGN.read_text())
    expected = {
        r["id"]
        for r in design["expected_rows"]
        if r["config"] == f"{SCENE}|pyrite-ss-cuda-fp64"
        and r["contrast"] == "nominal:L3"
        and r["control"] in ("electron_cutoff_keV", "moment_policy")
        and r["workload"] == "transport"
    }
    assert set(rows) == expected and len(expected) == 8
    row = next(iter(rows.values()))
    assert row["sampling"] == "independent" and row["variance"] > 0
    assert [a["histories"] for a in row["arms"]] == [400, 400]
    assert row["arms"][0]["elapsed_seconds"] == 4.0 and row["arms"][0]["overhead_seconds"] == 10.0
    assert summary["seeds"] == [301, 302] and summary["numerical_gate"] == "unmeasured"
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary))
    plan = PLANNER["make_plan"](DESIGN, out)
    assert plan["headline"] is False and plan["cost_complete"] is False


@pytest.mark.parametrize("damage", ["two-controls", "off-ladder", "seed", "hash", "duplicate"])
def test_unqualified_pilot_records_refused(tmp_path, damage):
    paths = pilot_set(tmp_path)
    if damage == "two-controls":
        paths.append(bench_record(tmp_path, "both", 303, cutoff=1.25, dE=0.00625))
    elif damage == "off-ladder":
        paths.append(bench_record(tmp_path, "odd", 303, cutoff=3.0))
    elif damage == "seed":
        paths.append(bench_record(tmp_path, "held-out", 401))
    elif damage == "hash":
        np.savez_compressed(tmp_path / "nominal-301.npz", transmitted=np.zeros(200))
    else:
        paths.append(bench_record(tmp_path, "again", 301))
    with pytest.raises(ValueError):
        SUMMARY["summarize"](DESIGN, paths, tmp_path, IDS)


def test_unavailable_controls_and_corners_stay_missing(tmp_path):
    paths = [bench_record(tmp_path, "nominal-301", 301)]
    record = json.loads(paths[0].read_text())
    record["controls"]["elsepa_level"] = 3
    elsepa = tmp_path / "elsepa.json"
    elsepa.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="no expected row"):
        SUMMARY["summarize"](DESIGN, [*paths, elsepa], tmp_path, IDS)
