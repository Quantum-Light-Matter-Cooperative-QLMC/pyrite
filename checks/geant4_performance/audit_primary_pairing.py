"""Audit #1041 PyRITE CUDA per-primary pairing before paired variances are admitted.

The design forbids paired (common-random-number) variances on a common seed
alone. A configuration's per-primary scores are pairable across control levels
only if each primary's score is a function of ``(seed, global primary index,
configuration)`` and nothing else. This audit checks that empirically from
bench records, never by running histories:

* replay invariance: the same configuration, seed and histories run with a
  different block split and launch layout (``--block``, ``--rows-per-round``,
  ``--seg-capacity``) give bitwise-identical per-primary arrays;
* alignment: every array has one finite entry per primary in index order, and
  every primary has exactly one terminal fate (zero scores retained);
* seed separation: nominal scores for different seeds are not identical and
  their per-primary correlation stays inside a Bonferroni normal bound.

A verified (config, setting, observable) cell admits paired variances only when
both arms of a contrast are verified and use identical seeds and histories.
"""

import argparse
import json
import math
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from pyrite._util.hashing import file_sha256

HERE = Path(__file__).resolve().parent
SUMMARY = runpy.run_path(str(HERE / "summarize_convergence_pilot.py"))
LAYOUT = ("block", "rows_per_round", "seg_capacity")
FATES = ("transmitted", "backscattered", "side_exited", "stopped")
SCHEMA = "pyrite.geant4-performance.primary-pairing-audit.v1"


def layout(record):
    controls = record["controls"]
    return (record["block"], controls["rows_per_round"], controls["seg_capacity"])


def load_scores(path, record):
    """Return checked per-primary arrays; refuse misaligned or partial records."""
    npz = path.with_suffix(".npz")
    if file_sha256(npz) != record["raw_scores_sha256"]:
        raise ValueError(f"raw score hash mismatch: {npz}")
    with np.load(npz) as data:
        arrays = {k: np.asarray(data[k], dtype=np.float64) for k in data.files}
    n = record["histories"]
    for name, values in arrays.items():
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{npz}: {name} is not one finite score per primary")
    if not np.array_equal(sum(arrays[f] for f in FATES), np.ones(n)):
        raise ValueError(f"{npz}: every primary needs exactly one terminal fate")
    return arrays


def replay_cells(runs, observables):
    """Bitwise per-observable comparison across layouts of one (config, setting, seed)."""
    layouts = {layout(r) for _, r, _ in runs}
    if len(layouts) < 2:
        return None
    base = runs[0][2]
    return {
        name: all(np.array_equal(base[name], other[name]) for _, _, other in runs[1:])
        for name in observables
    }


def seed_separation(by_seed, observables, z):
    """Per-observable correlation bound between nominal scores of different seeds."""
    seeds = sorted(by_seed)
    result = {}
    for name in observables:
        worst, identical = 0.0, False
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                x, y = by_seed[a][name], by_seed[b][name]
                identical |= bool(np.array_equal(x, y) and np.ptp(x) > 0)
                if x.size == y.size and np.ptp(x) > 0 and np.ptp(y) > 0:
                    r = float(np.corrcoef(x, y)[0, 1])
                    worst = max(worst, abs(r) * math.sqrt(x.size))
        result[name] = dict(max_abs_r_sqrt_n=worst, bound=z, passed=worst <= z and not identical)
    return result


def audit(design_path, paths, root):
    design = json.loads(design_path.read_text())
    configs = {c["id"]: c for c in design["configurations"]}
    observables = sorted(
        {s["observable"] for s in design["scales"] if s["workload"] == "transport"}
    )
    cells, files = defaultdict(list), {}
    for path in paths:
        record = json.loads(path.read_text())
        config, workload, setting = SUMMARY["classify"](record, configs)
        if workload != "transport":
            raise ValueError("pairing audit covers discrete transport records only")
        key = (config, setting, record["seed"], record["histories"])
        if any(layout(r) == layout(record) for _, r, _ in cells[key]):
            raise ValueError(f"duplicate layout for {key}")
        cells[key].append((path, record, load_scores(path, record)))
        for name in (path, path.with_suffix(".npz")):
            files[name.relative_to(root).as_posix()] = file_sha256(name)
    replay = []
    for (config, setting, seed, n), runs in sorted(cells.items()):
        result = replay_cells(runs, observables)
        if result is not None:
            replay.append(
                dict(
                    config=config,
                    setting=dict(setting),
                    seed=seed,
                    histories=n,
                    layouts=sorted(layout(r) for _, r, _ in runs),
                    bitwise_equal=result,
                )
            )
    nominal = defaultdict(dict)
    for (config, setting, seed, _), runs in cells.items():
        if not setting:
            nominal[config][seed] = runs[0][2]
    tests = sum(len(v) * (len(v) - 1) // 2 for v in nominal.values()) * len(observables)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * max(tests, 1)))
    separation = {
        config: seed_separation(by_seed, observables, z)
        for config, by_seed in sorted(nominal.items())
        if len(by_seed) > 1
    }
    verified = []
    for row in replay:
        sep = separation.get(row["config"])
        for name, equal in row["bitwise_equal"].items():
            if equal and sep is not None and sep[name]["passed"]:
                verified.append(dict(config=row["config"], setting=row["setting"], observable=name))
    failed = [
        dict(config=r["config"], setting=r["setting"], observable=k)
        for r in replay
        for k, ok in r["bitwise_equal"].items()
        if not ok
    ] + [
        dict(config=c, setting={}, observable=k, reason="seed separation")
        for c, sep in separation.items()
        for k, v in sep.items()
        if not v["passed"]
    ]
    return dict(
        schema=SCHEMA,
        design_sha256=file_sha256(design_path),
        auditor_sha256=file_sha256(Path(__file__)),
        evidence_files_sha256=dict(sorted(files.items())),
        replay=replay,
        seed_separation=separation,
        verified=verified,
        failed=failed,
        headline=False,
        numerical_gate="unmeasured",
        limitations=[
            "Covers the audited configurations, settings, observables and code identity only",
            "Pairing also requires both confirmation arms to use the same seed and global "
            "primary offsets; the per-shard map is audited separately",
            "Seed separation is a correlation screen, not a full stream-independence proof",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+", help="bench JSON records under --root")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=HERE / "convergence-design-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists() or args.output.resolve().parent != root:
        parser.error("output must be new and sit in --root beside its evidence")
    paths = [p.resolve() for p in args.records]
    if any(root not in p.parents for p in paths):
        parser.error("every record must sit under --root")
    result = audit(args.design, paths, root)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=1)
        stream.write("\n")
    print(json.dumps(dict(verified=len(result["verified"]), failed=len(result["failed"]))))


if __name__ == "__main__":
    main()
