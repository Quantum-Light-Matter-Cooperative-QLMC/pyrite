"""Geant4 arm of the matched-bounds benchmark (#821): run g4bench, reduce records.

Runs one (scene, seed, threads) process of the single-file app in ``g4bench/``
and converts its per-thread binary per-primary records to the same observables
and per-primary score arrays as ``pyrite_bench.py``. The app's own steady-clock
intervals (initialization, warmup run, measured run) are the timings; the
process wall is recorded separately. Refuses incomplete or duplicate events,
any electron process outside the matched set, and cut tables that would allow
delta rays or move the photon threshold.
"""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from pyrite.perf.precision import primary_moments

HERE = Path(__file__).parent
RECORD = np.dtype(
    [
        ("event", "<i4"),
        ("terminal", "<i4"),
        ("exit_keV", "<f8"),
        ("path_um", "<f8"),
        ("photons", "<i4", (4,)),
        ("photon_keV_ge10", "<f8"),
        ("steps_coulomb", "<i4"),
        ("steps_ioni", "<i4"),
        ("steps_brem", "<i4"),
        ("steps_other", "<i4"),
        ("cone_photons", "<i4", (3,)),
    ]
)
EXPECTED_PROCESSES = {"Transportation", "eIoni", "eBrem", "CoulombScat"}
CONE_BANDS = tuple(f"cone_source_photons_{b}" for b in ("10_50kev", "50_100kev", "ge100kev"))
BANDS = ("photons_1_10keV", "photons_10_50keV", "photons_50_100keV", "photons_ge100keV")


def audit_log(text, material):
    """Check matched bounds from the app's own dumps; return parsed facts."""
    processes = set(re.findall(r"^G4BENCH_PROCESS e- (\S+)$", text, re.M))
    if processes != EXPECTED_PROCESSES:
        raise RuntimeError(f"unexpected electron processes {sorted(processes)}")
    block = text.split(f"Material : Bench{material}", 1)[1]
    thresholds = re.search(r"Energy thresholds :\s+gamma\s+(\S+ \S+)\s+e-\s+(\S+ \S+)", block)
    if thresholds.group(1) != "1 keV":
        raise RuntimeError(f"photon threshold {thresholds.group(1)} is not 1 keV")
    if not thresholds.group(2).endswith(("GeV", "MeV")):
        raise RuntimeError(f"electron production threshold {thresholds.group(2)} too low")
    for pattern in [
        r"Lowest e\+e- kinetic energy\s+10 keV",
        r"Fluctuations of dE/dx are enabled\s+0",
    ]:
        if not re.search(pattern, text):
            raise RuntimeError(f"missing bound: {pattern}")
    step = re.search(r"Step function for e\+-\s+(\(.*?\))", text).group(1)
    timing = dict(
        (k, float(v)) for k, v in re.findall(r"(\w+_s)=([0-9.]+)", text.split("G4BENCH_TIMING")[1])
    )
    return dict(
        processes=sorted(processes),
        gamma_threshold=thresholds.group(1),
        electron_threshold=thresholds.group(2),
        step_function=step,
    ), timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--step-dr", type=float, default=0.0)
    parser.add_argument("--step-final-um", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / "scenes.json").read_text())
    scene = next(s for s in spec["scenes"] if s["id"] == args.scene)
    work = args.output.with_suffix("")
    work.mkdir(parents=True, exist_ok=False)
    cmd = [
        str(args.executable.resolve()),
        "--material",
        scene["material"],
        "--energy-kev",
        str(scene["energy_keV"]),
        "--thickness-um",
        str(scene["thickness_um"]),
        "--histories",
        str(args.histories),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--output",
        str(work / "rec"),
        "--cone-center-deg",
        str(spec["cone"]["center_deg"]),
        "--cone-half-deg",
        str(spec["cone"]["half_angle_deg"]),
    ]
    if args.step_dr > 0:
        cmd += ["--step-dr", str(args.step_dr), "--step-final-um", str(args.step_final_um)]
    start = time.perf_counter()
    with (work / "raw.log").open("w") as log:
        done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - start
    text = (work / "raw.log").read_text()
    if done.returncode:
        raise RuntimeError(f"g4bench exit {done.returncode}; see {work / 'raw.log'}")
    bounds, timing = audit_log(text, scene["material"])
    rec = np.concatenate([np.fromfile(p, dtype=RECORD) for p in sorted(work.glob("rec.t*.bin"))])
    rec = rec[np.argsort(rec["event"], kind="stable")]
    if not np.array_equal(rec["event"], np.arange(args.histories)) or np.any(rec["terminal"] < 4):
        raise RuntimeError("incomplete, duplicate or unterminated Geant4 events")
    t = rec["terminal"]
    scores = {
        "backscattered": (t == 4).astype(float),
        "transmitted": (t == 5).astype(float),
        "side_exited": (t == 6).astype(float),
        "stopped": (t == 7).astype(float),
        "transmitted_energy_keV": np.where(t == 5, rec["exit_keV"], 0.0),
        "photon_keV_ge10keV": rec["photon_keV_ge10"].astype(float),
        "path_um": rec["path_um"].astype(float),
        "steps": (
            rec["steps_coulomb"] + rec["steps_ioni"] + rec["steps_brem"] + rec["steps_other"]
        ).astype(float),
        "steps_coulomb": rec["steps_coulomb"].astype(float),
        "steps_ioni": rec["steps_ioni"].astype(float),
        "steps_brem": rec["steps_brem"].astype(float),
        "steps_other": rec["steps_other"].astype(float),
    }
    for i, name in enumerate(BANDS):
        scores[name] = rec["photons"][:, i].astype(float)
    for i, name in enumerate(CONE_BANDS):
        scores[name] = rec["cone_photons"][:, i].astype(float)
    scores["cone_source_photons_ge10kev"] = rec["cone_photons"].sum(axis=1).astype(float)
    npz = args.output.with_suffix(".npz")
    np.savez_compressed(npz, **scores)
    for p in work.glob("rec.t*.bin"):
        p.unlink()
    record = dict(
        schema="pyrite.geant4-performance.sample.v1",
        engine="geant4",
        scene=scene,
        histories=args.histories,
        warmup_histories=args.warmup,
        seed=args.seed,
        threads=args.threads,
        cone=spec["cone"],
        bounds=bounds,
        timings_s=dict(
            init=timing["init_s"],
            warmup_run=timing["warmup_s"],
            warm_request=timing["measured_s"],
            process_wall=wall,
        ),
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((HERE / "g4bench/g4bench.cc").read_bytes()).hexdigest(),
        raw_log_sha256=hashlib.sha256((work / "raw.log").read_bytes()).hexdigest(),
        observables={k: primary_moments(v) for k, v in scores.items()},
        raw_scores_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "timings_s": record["timings_s"]}))


if __name__ == "__main__":
    main()
