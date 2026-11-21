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
import math
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
BASE_PROCESSES = {"Transportation", "eIoni", "eBrem"}
EXPECTED_PROCESSES = {
    "ss": BASE_PROCESSES | {"CoulombScat"},
    "wvi": BASE_PROCESSES | {"msc", "CoulombScat"},
    "gs": BASE_PROCESSES | {"msc"},
}
CONE_BANDS = tuple(f"cone_source_photons_{b}" for b in ("10_50kev", "50_100kev", "ge100kev"))
BANDS = ("photons_1_10keV", "photons_10_50keV", "photons_50_100keV", "photons_ge100keV")


def audit_log(text, material, elastic="ss", expected=None):
    """Check matched bounds from the app's own dumps; return parsed facts."""
    processes = set(re.findall(r"^G4BENCH_PROCESS e- (\S+)$", text, re.MULTILINE))
    if processes != EXPECTED_PROCESSES[elastic]:
        raise RuntimeError(f"unexpected electron processes {sorted(processes)}")
    block = text.split(f"Material : Bench{material}", 1)[1]
    thresholds = re.search(r"Energy thresholds :\s+gamma\s+(\S+ \S+)\s+e-\s+(\S+ \S+)", block)
    if thresholds.group(1) != "1 keV":
        raise RuntimeError(f"photon threshold {thresholds.group(1)} is not 1 keV")
    if not thresholds.group(2).endswith(("GeV", "MeV")):
        raise RuntimeError(f"electron production threshold {thresholds.group(2)} too low")
    for pattern in [
        r"Fluctuations of dE/dx are enabled\s+0",
    ]:
        if not re.search(pattern, text):
            raise RuntimeError(f"missing bound: {pattern}")
    step = re.search(r"Step function for e\+-\s+(\(.*?\))", text).group(1)
    effective = re.findall(r"^G4BENCH_NUMERICS (.+)$", text, re.M)
    if len(effective) != 1:
        raise RuntimeError("missing or duplicate effective numerical audit")
    numerics = {k: float(v) for k, v in re.findall(r"(\w+)=([^ ]+)", effective[0])}
    step_values = re.fullmatch(r"\(\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\s*(nm|um|mm|cm|m)\s*\)", step)
    if step_values is None:
        raise RuntimeError(f"unrecognized effective step function: {step}")
    numerics.update(
        step_dr=float(step_values[1]),
        step_final_um=float(step_values[2])
        * dict(nm=0.001, um=1, mm=1000, cm=10000, m=1e6)[step_values[3]],
    )
    required = {
        "electron_cutoff_keV",
        "bins_per_decade",
        "msc_range_factor",
        "msc_theta_limit_rad",
        "msc_skin",
    }
    if not required <= numerics.keys() or not all(math.isfinite(v) for v in numerics.values()):
        raise RuntimeError("incomplete or nonfinite effective numerical audit")
    pinned = dict(msc_theta_limit_rad=0.15 if elastic == "wvi" else 0.0)
    if elastic == "gs":
        pinned["msc_skin"] = 3.0
    for key, value in dict(pinned, **(expected or {})).items():
        if key not in numerics or not math.isclose(
            numerics[key], value, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"effective {key}={numerics.get(key)} differs from requested {value}"
            )
    timing = {
        k: float(v) for k, v in re.findall(r"(\w+_s)=([0-9.]+)", text.split("G4BENCH_TIMING")[1])
    }
    return dict(
        processes=sorted(processes),
        gamma_threshold=thresholds.group(1),
        electron_threshold=thresholds.group(2),
        step_function=step,
        numerics=numerics,
    ), timing


def numerical_controls(args, energy_keV):
    """Validate invocation-local controls before creating output or launching Geant4."""
    values = dict(
        step_dr=args.step_dr,
        step_final_um=args.step_final_um,
        electron_cutoff_keV=args.electron_cutoff_kev,
        bins_per_decade=args.bins_per_decade,
    )
    if not all(math.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError("numerical controls must be finite and positive")
    if args.step_dr > 1 or args.electron_cutoff_kev >= energy_keV:
        raise ValueError("step fraction must be <=1 and electron cutoff below beam energy")
    if not 5 <= args.bins_per_decade < 1000000:
        raise ValueError("bins per decade outside Geant4 supported range [5, 1000000)")
    factor = args.msc_range_factor
    if factor is not None and args.elastic == "ss":
        raise ValueError("MSC range factor is inapplicable to single scattering")
    if factor is not None and (not math.isfinite(factor) or not 0 < factor < 1):
        raise ValueError("MSC range factor must be finite and in (0, 1)")
    if args.elastic != "ss":
        values["msc_range_factor"] = (
            factor if factor is not None else (0.08 if args.elastic == "gs" else 0.04)
        )
    return values


def accuracy_only(controls, elastic, requested=False):
    """Keep refined settings out of frozen nominal timing populations."""
    nominal = dict(step_dr=0.2, step_final_um=1000.0, electron_cutoff_keV=10.0, bins_per_decade=7)
    if elastic != "ss":
        nominal["msc_range_factor"] = 0.08 if elastic == "gs" else 0.04
    return requested or controls != nominal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--step-dr", type=float, default=0.2)
    parser.add_argument("--step-final-um", type=float, default=1000.0)
    parser.add_argument("--electron-cutoff-kev", type=float, default=10.0)
    parser.add_argument("--bins-per-decade", type=int, default=7)
    parser.add_argument("--msc-range-factor", type=float, default=None)
    parser.add_argument(
        "--elastic",
        choices=sorted(EXPECTED_PROCESSES),
        default="ss",
        help="ss: DPWA single scattering (#821); wvi: WentzelVI plus single Coulomb; "
        "gs: Goudsmit-Saunderson (#894 mixed-scheme arms)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accuracy-only", action="store_true", help="exclude this sample from T(eps)"
    )
    args = parser.parse_args()
    spec = json.loads((HERE / "scenes.json").read_text())
    scene = next(s for s in spec["scenes"] if s["id"] == args.scene)
    controls = numerical_controls(args, scene["energy_keV"])
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
    cmd += ["--elastic", args.elastic]
    for option, value in controls.items():
        cmd += ["--" + option.replace("_", "-").replace("keV", "kev"), str(value)]
    start = time.perf_counter()
    with (work / "raw.log").open("w") as log:
        done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - start
    text = (work / "raw.log").read_text()
    if done.returncode:
        raise RuntimeError(f"g4bench exit {done.returncode}; see {work / 'raw.log'}")
    bounds, timing = audit_log(text, scene["material"], args.elastic, expected=controls)
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
        elastic=args.elastic,
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
    if accuracy_only(controls, args.elastic, args.accuracy_only):
        record["purpose"] = "accuracy-only"
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "timings_s": record["timings_s"]}))


if __name__ == "__main__":
    main()
