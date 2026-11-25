"""Paired variances need an audited per-primary stream mapping, and v1.1 derives from v1."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
AUDIT = runpy.run_path(str(HERE / "audit_primary_pairing.py"))
SUMMARY = runpy.run_path(str(HERE / "summarize_convergence_pilot.py"))
PLANNER = runpy.run_path(str(HERE / "prepare_convergence_resources.py"))
AMEND = runpy.run_path(str(HERE / "amend_convergence_design.py"))
DESIGN = HERE / "convergence-design-v1.json"
IDS = dict(cpus=2, hardware="test-host", runtime="test-runtime", memory_margin_factor=1.5)
N = 400


def primaries(seed, shift=0.0):
    """Scores as a pure function of (seed, primary); ``shift`` mimics a control change."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    energy = rng.random(N) * 300 + shift
    fate = np.digitize(u, [0.1, 0.15, 0.17])  # back, stopped, side, transmitted
    return dict(
        backscattered=(fate == 0).astype(float),
        stopped=(fate == 1).astype(float),
        side_exited=(fate == 2).astype(float),
        transmitted=(fate == 3).astype(float),
        transmitted_energy_keV=np.where(fate == 3, energy, 0.0),
    )


def write(root, name, seed, scores, *, cutoff=10.0, block=8121, rows=256):
    npz = root / f"{name}.npz"
    np.savez_compressed(npz, **scores)
    record = dict(
        schema="pyrite.geant4-performance.sample.v1",
        scene=dict(id="si-300keV-100um"),
        seed=seed,
        histories=N,
        block=block,
        estimator="discrete",
        cone_scoring=dict(mode="sampled-photons"),
        controls=dict(
            electron_cutoff_keV=cutoff,
            radiative_cutoff_eV=1000.0,
            max_dE_frac=0.05,
            elsepa_level=0,
            mixed_c1=None,
            mixed_c2=None,
            mixed_max_step_um=None,
            transport_precision="fp64",
            moment_policy=None,
            rows_per_round=rows,
            seg_capacity=16384,
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


def evidence(root, *, replay_shift=0.0, second_seed=302):
    production = [
        write(root, "nominal-301", 301, primaries(301)),
        write(root, "nominal-302", second_seed, primaries(second_seed)),
        write(root, "cut3-301", 301, primaries(301, 0.5), cutoff=1.25),
    ]
    replay = [
        write(root, "nominal-301-r", 301, primaries(301, replay_shift), block=97, rows=64),
        write(root, "cut3-301-r", 301, primaries(301, 0.5), cutoff=1.25, block=97, rows=64),
    ]
    return production, replay


def test_audited_pairs_feed_planner_with_difference_variance(tmp_path):
    production, replay = evidence(tmp_path)
    audit = AUDIT["audit"](DESIGN, production + replay, tmp_path)
    assert not audit["failed"] and len(audit["verified"]) == 2 * 4
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    nominal_only = [p for p in production if "302" not in p.name]
    summary = SUMMARY["summarize"](DESIGN, nominal_only, tmp_path, IDS, path)
    rows = {r["id"]: r for r in summary["rows"]}
    assert all(r["sampling"] == "crn-paired" and len(r["arms"]) == 2 for r in rows.values())
    energy = next(r for r in rows.values() if r["variance"] > 0)
    d = primaries(301, 0.5)["transmitted_energy_keV"] - primaries(301)["transmitted_energy_keV"]
    assert energy["variance"] == pytest.approx(np.var(d, ddof=1), rel=1e-12)
    assert energy["independent_variance"] > 1e4 * energy["variance"]
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary))
    plan = PLANNER["make_plan"](DESIGN, out)
    assert plan["pairing_audit_sha256"] == summary["pairing_audit_sha256"]
    summary["pairing_audit_sha256"] = None
    out.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="pairing audit"):
        PLANNER["make_plan"](DESIGN, out)


def test_v1_1_summary_counts_discordances_and_plans_exact_agreement(tmp_path):
    design_path = tmp_path / "design-v1.1.json"
    design_path.write_text(json.dumps(AMEND["derive"](12.0)))
    production, replay = evidence(tmp_path)
    audit = AUDIT["audit"](design_path, production + replay, tmp_path)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    nominal_only = [p for p in production if "302" not in p.name]
    summary = SUMMARY["summarize"](design_path, nominal_only, tmp_path, IDS, path)
    fates = [r for r in summary["rows"] if r["variance"] == 0]
    energy = [r for r in summary["rows"] if r["variance"] > 0]
    assert fates and all(r["discordant"] == 0 for r in fates)
    assert energy and all(r["discordant"] > 0 for r in energy)
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary))
    plan = {r["id"]: r for r in PLANNER["make_plan"](design_path, out)["rows"]}
    assert {plan[r["id"]]["sampling"] for r in fates} == {"exact-agreement"}
    assert all(plan[r["id"]]["sampling"] == "crn-paired" for r in energy)


def test_score_above_frozen_support_is_refused(tmp_path):
    design = AMEND["derive"](12.0)
    row = next(r for r in design["expected_rows"] if r["observable"] == "transmitted")
    with pytest.raises(ValueError, match="score bound"):
        SUMMARY["discordance"](design, row, np.array([0.0, 2.0]))


def test_layout_dependent_scores_are_not_verified(tmp_path):
    production, replay = evidence(tmp_path, replay_shift=1e-9)
    audit = AUDIT["audit"](DESIGN, production + replay, tmp_path)
    failed = {(f["setting"].get("electron_cutoff_keV"), f["observable"]) for f in audit["failed"]}
    assert (None, "transmitted_energy_keV") in failed
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    summary = SUMMARY["summarize"](
        DESIGN, [p for p in production if "302" not in p.name], tmp_path, IDS, path
    )
    energy = [r for r in summary["rows"] if r["arms"][0]["mean"] > 1]
    assert energy and all(r["sampling"] == "independent" for r in energy)


def test_reused_seed_streams_fail_separation(tmp_path):
    production, replay = evidence(tmp_path)
    clone = write(tmp_path, "nominal-303", 303, primaries(301))
    audit = AUDIT["audit"](DESIGN, production + replay + [clone], tmp_path)
    assert not audit["verified"]
    assert any(f.get("reason") == "seed separation" for f in audit["failed"])


def test_missing_terminal_fate_refused(tmp_path):
    scores = primaries(301)
    scores["transmitted"][0] = scores["backscattered"][0] = 0.0
    scores["stopped"][0] = scores["side_exited"][0] = 0.0
    path = write(tmp_path, "nominal-301", 301, scores)
    with pytest.raises(ValueError, match="terminal fate"):
        AUDIT["audit"](DESIGN, [path], tmp_path)


def test_v1_1_amendment_derives_from_frozen_v1():
    design = AMEND["derive"](12.0)
    frozen = json.loads(DESIGN.read_text())
    assert design["multiplicity"] == frozen["multiplicity"] == len(design["expected_rows"])
    assert design["row_alpha"] == frozen["row_alpha"] and design["target_rse"] == 0.05
    for old, new in zip(frozen["expected_rows"], design["expected_rows"], strict=True):
        assert new["id"] == old["id"]
        assert new["allowance"] == pytest.approx(2.5 * old["allowance"], rel=1e-14)
        if old["control"] == "elsepa_energy_density":
            assert new["support"] == "unsupported" and "#1426" in new["reason"]
        else:
            assert new["support"] == old["support"]
    assert design["resources"]["confirmation_gpu_hours"] == 12.0
    assert design["exact_agreement"]["score_bounds"]["transmitted_energy_keV"] == (
        "beam_energy_keV"
    )
    assert design["amendment"]["amends_sha256"] == hashlib.sha256(DESIGN.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="extend"):
        AMEND["derive"](4.0)
    with pytest.raises(ValueError, match="original v1"):
        AMEND["amend_v1_1"](design, "x", 12.0)
