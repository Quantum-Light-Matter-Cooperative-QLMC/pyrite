"""Summarize #1041 PyRITE CUDA pilot records for the confirmation resource planner.

Planning only: never runs histories or certifies convergence. Each record's
scene, arm, workload and control levels are derived from the bench record
itself and checked against the frozen design, not taken from file names.
Without a pairing audit, arms at different levels are independent populations
and their per-primary variances are summed. With an ``audit_primary_pairing.py``
result that verifies both arms' (config, setting, observable) cells, rows whose
arms share seeds and histories use the common-random-number paired variance
Var(Y_right - Y_left) and still charge both arms. Rows with missing arms, or
design support other than ``available``, are omitted and stay missing.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyrite._util.hashing import file_sha256

HERE = Path(__file__).resolve().parent
PILOT_SEEDS = (301, 302, 303)
RECORD_FIELDS = {
    "electron_cutoff_keV": "electron_cutoff_keV",
    "radiative_separator_eV": "radiative_cutoff_eV",
    "max_dE_frac": "max_dE_frac",
    "C1": "mixed_c1",
    "C2": "mixed_c2",
    "max_step_um": "mixed_max_step_um",
    "elsepa_energy_density": "elsepa_level",
}
CONTRASTS = {"nominal:L3": (0, 3), "L1:L2": (1, 2), "L2:L3": (2, 3)}


def _same(a, b):
    return isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def _policy_level(policy, levels):
    if policy is None:
        return 0  # omitted flag: nominal production policy
    for index, level in enumerate(levels):
        if (
            _same(policy["rtol"], level["rtol"])
            and all(map(_same, policy["atol"], level["atols"]))
            and policy["max_nodes"] == level["max_nodes"]
            and policy["held_out_count"] == level["probe_count"]
        ):
            return index
    raise ValueError("moment policy is not a design level")


def control_level(control, controls):
    """Return the design level index a record's effective control matches."""
    if control["id"] == "moment_policy":
        return _policy_level(controls["moment_policy"], control["levels"])
    value = controls[RECORD_FIELDS[control["id"]]]
    if control["id"] == "elsepa_energy_density":
        if type(value) is not int or value not in range(len(control["levels"])):
            raise ValueError("ELSEPA level is not a design level")
        return value  # nested level n has 2**n energy density
    matches = [i for i, level in enumerate(control["levels"]) if _same(value, level)]
    if len(matches) != 1:
        raise ValueError(f"{control['id']}={value!r} is not a design level")
    return matches[0]


def classify(record, configs):
    """Return (config id, workload, setting) for one PyRITE CUDA bench record."""
    if record["schema"] != "pyrite.geant4-performance.sample.v1":
        raise ValueError("not a bench sample record")
    controls = record["controls"]
    algorithm = "ss" if controls["mixed_c1"] is None else "mixed"
    arm = f"pyrite-{algorithm}-cuda-{controls['transport_precision']}"
    config = configs.get(f"{record['scene']['id']}|{arm}")
    if config is None:
        raise ValueError(f"no design configuration for {record['scene']['id']}|{arm}")
    if record["seed"] not in PILOT_SEEDS:
        raise ValueError("pilot records must use declared planning seeds 301-303")
    if record["estimator"] == "discrete":
        workload = "transport"
    elif record["cone_scoring"]["mode"] == "point":
        workload = "cone"
    else:
        raise ValueError("only discrete transport or nominal point-cone records are summarized")
    setting = {}
    for control in config["controls"]:
        if control["id"] in RECORD_FIELDS or control["id"] == "moment_policy":
            level = control_level(control, controls)
            if level:
                setting[control["id"]] = level
    if len(setting) > 1:
        raise ValueError("pilot records refine one primitive control at a time")
    return config["id"], workload, tuple(sorted(setting.items()))


def load_groups(paths, configs):
    groups = defaultdict(list)
    for path in paths:
        record = json.loads(path.read_text())
        groups[classify(record, configs)].append((path, record))
    for key, runs in groups.items():
        seeds = [r["seed"] for _, r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed in pilot group {key}")
    return groups


def arm_scores(runs, observable):
    """Per-seed checked score arrays, in seed order."""
    out = []
    for path, record in sorted(runs, key=lambda item: item[1]["seed"]):
        npz = path.with_suffix(".npz")
        if file_sha256(npz) != record["raw_scores_sha256"]:
            raise ValueError(f"raw score hash mismatch: {npz}")
        with np.load(npz) as data:
            values = np.asarray(data[observable], dtype=np.float64)
        if values.size != record["histories"] or not np.all(np.isfinite(values)):
            raise ValueError(f"incomplete or nonfinite scores: {npz}")
        out.append((record["seed"], values))
    return out


def verified_cells(audit):
    if audit is None:
        return set()
    if audit["schema"] != "pyrite.geant4-performance.primary-pairing-audit.v1":
        raise ValueError("not a primary pairing audit")
    return {
        (c["config"], tuple(sorted(c["setting"].items())), c["observable"])
        for c in audit["verified"]
    }


def paired_variance(sides, observable):
    """Var(Y_right - Y_left) over primaries aligned by seed and global index."""
    left, right = (arm_scores(runs, observable) for runs in sides)
    if [(s, v.size) for s, v in left] != [(s, v.size) for s, v in right]:
        return None
    d = np.concatenate([b - a for (_, a), (_, b) in zip(left, right, strict=True)])
    return float(d.var(ddof=1))


def arm_summary(runs, observable, root, identities):
    scores = [v for _, v in arm_scores(runs, observable)]
    files = []
    for path, _ in runs:
        npz = path.with_suffix(".npz")
        files += [path.relative_to(root).as_posix(), npz.relative_to(root).as_posix()]
    pooled = np.concatenate(scores)
    timings = [r["timings_s"] for _, r in runs]
    arm = dict(
        histories=int(pooled.size),
        elapsed_seconds=float(sum(t["warm_request"] for t in timings)),
        overhead_seconds=float(max(t["process_total"] - t["warm_request"] for t in timings)),
        cpus=identities["cpus"],
        gpus=1,
        peak_memory_bytes=max(r["peak_rss_kib"] for _, r in runs) * 1024,
        memory_margin_factor=identities["memory_margin_factor"],
        peak_vram_bytes=max(r["vram_telemetry"]["peak_device_used_bytes"] for _, r in runs),
        hardware_identity=f"{identities['hardware']}; device {runs[0][1]['device']}",
        runtime_identity=identities["runtime"],
        seeds=sorted(r["seed"] for _, r in runs),
        mean=float(pooled.mean()),
        nonzero=int(np.count_nonzero(pooled)),
    )
    return arm, float(pooled.var(ddof=1)), files


def summarize(design_path, paths, root, identities, audit_path=None):
    design = json.loads(design_path.read_text())
    configs = {c["id"]: c for c in design["configurations"]}
    groups = load_groups(paths, configs)
    audit = None if audit_path is None else json.loads(audit_path.read_text())
    cells = verified_cells(audit)
    rows, evidence, seeds = [], {}, set()
    for row in design["expected_rows"]:
        if row["support"] != "available" or row["contrast"] not in CONTRASTS:
            continue
        sides, settings = [], []
        for level in CONTRASTS[row["contrast"]]:
            setting = ((row["control"], level),) if level else ()
            settings.append(setting)
            sides.append(groups.get((row["config"], row["workload"], setting)))
        if not all(sides):
            continue
        paired = None
        if all((row["config"], s, row["observable"]) in cells for s in settings):
            paired = paired_variance(sides, row["observable"])
        arms, variance, files = [], 0.0, []
        for runs in sides:
            arm, var, used = arm_summary(runs, row["observable"], root, identities)
            arms.append(arm)
            variance += var
            files += used
            seeds.update(arm["seeds"])
        for name in files:
            evidence[name] = file_sha256(root / name)
        rows.append(
            dict(
                id=row["id"],
                sampling="independent" if paired is None else "crn-paired",
                variance=variance if paired is None else paired,
                independent_variance=variance,
                evidence_files=sorted(set(files)),
                arms=arms,
                difference=arms[1]["mean"] - arms[0]["mean"],
            )
        )
    if not rows:
        raise ValueError("no expected row has both pilot arms")
    return dict(
        schema="pyrite.geant4-performance.convergence-pilot-summary.v1",
        design_sha256=file_sha256(design_path),
        phase="pilot",
        seeds=sorted(seeds),
        evidence_files_sha256=dict(sorted(evidence.items())),
        rows=rows,
        summarizer_sha256=file_sha256(Path(__file__)),
        pairing_audit_sha256=None if audit_path is None else file_sha256(audit_path),
        numerical_gate="unmeasured",
        headline=False,
        submission_ready=False,
        limitations=[
            "Pilot differences are planning evidence, never convergence verdicts",
            "Paired rows rest only on the cited pairing audit; others are independent arms",
            "VRAM is a sampled device-wide upper bound",
            "Overhead is process total minus warm request, charged per future shard",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+", help="bench JSON records under --root")
    parser.add_argument("--root", type=Path, required=True, help="summary directory")
    parser.add_argument("--design", type=Path, default=HERE / "convergence-design-v1.json")
    parser.add_argument("--cpus", type=int, required=True, help="CPUs allocated per pilot job")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--memory-margin", type=float, default=1.5)
    parser.add_argument(
        "--pairing-audit", type=Path, default=None, help="audit_primary_pairing.py output"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists() or args.output.resolve().parent != root:
        parser.error("output must be new and sit in --root beside its evidence")
    paths = [p.resolve() for p in args.records]
    if any(root not in p.parents for p in paths):
        parser.error("every record must sit under --root")
    identities = dict(
        cpus=args.cpus,
        hardware=args.hardware,
        runtime=args.runtime,
        memory_margin_factor=args.memory_margin,
    )
    result = summarize(args.design, paths, root, identities, args.pairing_audit)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=1)
        stream.write("\n")
    paired = sum(r["sampling"] == "crn-paired" for r in result["rows"])
    print(json.dumps(dict(rows=len(result["rows"]), paired=paired, seeds=result["seeds"])))


if __name__ == "__main__":
    main()
